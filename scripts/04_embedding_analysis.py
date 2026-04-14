"""
Step 2-4: Embedding Analysis — UMAP, classifier probing, feature attribution
============================================================================
Run after 02_hippie_train_and_embed.sh has produced the embedding CSV.

Loads:
  /home/jesus/results_hippie_rebuttals/FA_T4/.../pretraining_multi_embeddings.csv
  /home/jesus/datasets_hippie/FA_T4/metadata.csv

Produces:
  /mnt/d/datasets_hippie/Roy_shruti_folic_data/embedding_results/
    umap_*.png                     — UMAP colored by various metadata
    classifier_probing.csv         — classifier accuracy by condition
    latent_dim_correlations.csv    — latent dim × metric correlations
    well_aggregated_embeddings.csv — mean embedding per well (for trajectory plots)
    well_trajectories.png          — embedding path over timepoints per mouse

Usage:
  /home/jesus/hippie_rebuttals/hip-hip-hippie/hippie_venv/bin/python 03_embedding_analysis.py \
      --config full --z-dim 32 --beta 4 --epochs 10

  Optional: --epochs 1 (or 20) to analyze different epoch checkpoints
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─── Arguments ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config",  default="full")
parser.add_argument("--z-dim",   type=int, default=32)
parser.add_argument("--beta",    type=float, default=4)
parser.add_argument("--epochs",  type=int, default=10)
parser.add_argument("--run-suffix", type=str, default="",
                    help="Suffix appended to config dir name (e.g. _well_cond)")
parser.add_argument("--residualize", type=str, default=None,
                    help="Comma-separated metadata columns to regress out of embeddings "
                         "before UMAP/classifier (e.g. 'div' or 'div,mouse_id')")
args = parser.parse_args()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import METADATA_CSV, embedding_dir, results_embedding_dir

EMB_DIR      = embedding_dir(args.config, args.z_dim, args.beta, args.run_suffix)
META_PATH    = METADATA_CSV
_resid_tag = ("_resid_" + args.residualize.replace(",", "_")) if args.residualize else ""
OUT_DIR      = results_embedding_dir(args.epochs, args.run_suffix, _resid_tag)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load embeddings + metadata ──────────────────────────────────────────────
print(f"Loading metadata from {META_PATH}")
meta = pd.read_csv(META_PATH)

# Epoch-specific parsed CSV (preferred) → generic → raw
parsed_ep   = EMB_DIR / f"embeddings_parsed_ep{args.epochs}.csv"
parsed_path = EMB_DIR / "embeddings_parsed.csv"    # legacy / 10-epoch
raw_path    = EMB_DIR / "pretraining_multi_embeddings.csv"

if parsed_ep.exists():
    emb_path = parsed_ep
elif parsed_path.exists():
    emb_path = parsed_path
else:
    emb_path = None

if emb_path is not None:
    print(f"Loading pre-parsed embeddings from {emb_path}")
    emb_df   = pd.read_csv(emb_path)
    latent_cols = [c for c in emb_df.columns if c.startswith("z")]
    Z = emb_df[latent_cols].values
else:
    print(f"Parsing raw embeddings from {raw_path}")
    raw  = pd.read_csv(raw_path)
    def parse_emb(s):
        return np.fromstring(str(s).strip().strip("[]"), sep=" ")
    Z = np.vstack([parse_emb(row) for row in raw["embeddings"]])
    # Save epoch-specific parsed file
    pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])]).to_csv(
        parsed_ep, index=False)
    print(f"Saved parsed embeddings to {parsed_ep}")
print(f"Embedding matrix: {Z.shape}")

# Align metadata rows (assume same row order as preprocessing output)
if len(meta) != len(Z):
    print(f"WARNING: metadata rows ({len(meta)}) ≠ embedding rows ({len(Z)}). "
          f"Truncating to min.")
    n = min(len(meta), len(Z))
    meta = meta.iloc[:n].reset_index(drop=True)
    Z    = Z[:n]

# ─── Post-hoc residualization ────────────────────────────────────────────────
if args.residualize:
    resid_cols = [c.strip() for c in args.residualize.split(",")]
    print(f"Residualizing embeddings on: {resid_cols}")
    # Exact per-group mean subtraction: for each unique combination of confound
    # values, subtract the group mean embedding. This is non-parametric and makes
    # no linearity assumption across groups.
    group_key = meta[resid_cols].astype(str).agg("_".join, axis=1)
    Z_resid = Z.copy()
    for grp, idx in group_key.groupby(group_key).groups.items():
        Z_resid[idx] -= Z[idx].mean(axis=0)
    Z = Z_resid
    print(f"  Residualized {group_key.nunique()} groups — embedding matrix: {Z.shape}")

# Date ordering
date_order  = sorted(meta["date"].unique())
meta["timepoint"] = meta["date"].map({d: i for i, d in enumerate(date_order)})

mice    = sorted(meta["mouse_id"].unique())
dates   = sorted(meta["date"].unique())
n_mice  = len(mice)
n_dates = len(dates)

# ═══════════════════════════════════════════════════════════════════════════
# UMAP (or PCA fallback if umap-learn not installed)
# ═══════════════════════════════════════════════════════════════════════════
try:
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                        random_state=42, verbose=False)
    Z2 = reducer.fit_transform(Z)
    dim_method = "UMAP"
except ImportError:
    print("umap-learn not found — using PCA for 2D projection")
    pca = PCA(n_components=2, random_state=42)
    Z2  = pca.fit_transform(Z)
    dim_method = "PCA"

print(f"{dim_method} projection: {Z2.shape}")

def scatter_colored(Z2, color_values, title, cmap, fname,
                    categorical=True, alpha=0.3, s=4):
    """Scatter Z2 with colour-coding. categorical=True → legend; False → colorbar."""
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        unique_vals = sorted(set(color_values))
        palette     = cm.tab10(np.linspace(0, 0.9, len(unique_vals)))
        val_to_col  = dict(zip(unique_vals, palette))
        colors      = [val_to_col[v] for v in color_values]
        sc          = ax.scatter(Z2[:, 0], Z2[:, 1], c=colors, s=s, alpha=alpha,
                                 linewidths=0, rasterized=True)
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=7, label=v)
                   for v, c in val_to_col.items()]
        ax.legend(handles=handles, title=title.split(" by ")[-1],
                  bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    else:
        sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=color_values, cmap=cmap,
                        s=s, alpha=alpha, linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, label=title.split(" by ")[-1])

    ax.set_title(title)
    ax.set_xlabel(f"{dim_method} 1")
    ax.set_ylabel(f"{dim_method} 2")
    plt.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")

# UMAP colored by mouse
scatter_colored(Z2, meta["mouse_id"].values,
                f"{dim_method} by mouse ID (epochs={args.epochs})",
                "tab10", "umap_by_mouse.png")

# UMAP colored by date (timepoint)
scatter_colored(Z2, meta["date"].values,
                f"{dim_method} by recording date (epochs={args.epochs})",
                "viridis", "umap_by_date.png")

# UMAP colored by condition (if known)
if (meta["condition"] != "unknown").any():
    scatter_colored(Z2, meta["condition"].values,
                    f"{dim_method} by FA condition (epochs={args.epochs})",
                    "tab10", "umap_by_condition.png")
else:
    print("NOTE: condition labels unknown — skipping condition UMAP.")
    print("      Update MOUSE_CONDITION in preprocess_FA_kilosort.py once mapping is confirmed.")

# UMAP colored by sex
if "sex" in meta.columns:
    scatter_colored(Z2, meta["sex"].values,
                    f"{dim_method} by sex (epochs={args.epochs})",
                    "tab10", "umap_by_sex.png")

# UMAP colored by DIV (continuous)
if "div" in meta.columns and (meta["div"] >= 0).any():
    div_vals = meta["div"].replace(-1, np.nan).fillna(meta["div"][meta["div"] >= 0].median()).values
    scatter_colored(Z2, div_vals, f"{dim_method} by DIV",
                    "plasma", "umap_by_div.png", categorical=False)

# UMAP colored by SNR (continuous)
if "snr" in meta.columns:
    snr_vals = meta["snr"].fillna(meta["snr"].median()).values
    scatter_colored(Z2, snr_vals, f"{dim_method} by SNR",
                    "plasma", "umap_by_snr.png", categorical=False)

# UMAP colored by half_width
if "half_width" in meta.columns:
    hw_vals = np.clip(meta["half_width"].fillna(meta["half_width"].median()).values,
                      *np.percentile(meta["half_width"].dropna(), [1, 99]))
    scatter_colored(Z2, hw_vals, f"{dim_method} by half-width",
                    "plasma", "umap_by_halfwidth.png", categorical=False)

# ═══════════════════════════════════════════════════════════════════════════
# Well-level aggregated embeddings
# ═══════════════════════════════════════════════════════════════════════════
group_cols = ["date", "timepoint", "mouse_id", "run", "well", "condition"]
# include sex and div if available
for _c in ["sex", "div"]:
    if _c in meta.columns:
        group_cols.append(_c)
latent_df  = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
full_df    = pd.concat([meta[group_cols].reset_index(drop=True), latent_df], axis=1)

z_cols    = [f"z{i}" for i in range(Z.shape[1])]
well_emb  = full_df.groupby(group_cols)[z_cols].mean().reset_index()
well_emb.to_csv(OUT_DIR / "well_aggregated_embeddings.csv", index=False)
print(f"Saved well_aggregated_embeddings.csv  ({len(well_emb)} wells)")

# Well-level 2D projection
Zw       = well_emb[z_cols].values
try:
    Zw2  = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=42).fit_transform(Zw)
except Exception:
    Zw2  = PCA(n_components=2, random_state=42).fit_transform(Zw)

# Well-level UMAP by mouse + date
fig, ax = plt.subplots(figsize=(8, 6))
palette = cm.tab10(np.linspace(0, 0.9, n_mice))
markers = ["o", "s", "^", "D", "v", "P"]
for i, mouse in enumerate(mice):
    sub = well_emb[well_emb["mouse_id"] == mouse]
    idx = sub.index.values
    sc  = ax.scatter(Zw2[idx, 0], Zw2[idx, 1],
                     c=[palette[i]] * len(idx),
                     marker=markers[i % len(markers)],
                     s=60, alpha=0.85, edgecolors="k", linewidths=0.3,
                     label=mouse)
ax.set_title(f"Well-level {dim_method} (mean embedding per well, epochs={args.epochs})")
ax.set_xlabel(f"{dim_method} 1"); ax.set_ylabel(f"{dim_method} 2")
ax.legend(title="Mouse ID", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
fig.savefig(OUT_DIR / "umap_well_level_by_mouse.png", dpi=150)
plt.close()
print("Saved umap_well_level_by_mouse.png")

# Well-level UMAP by condition
if "condition" in well_emb.columns and (well_emb["condition"] != "unknown").any():
    conds   = sorted(well_emb["condition"].unique())
    pal_c   = cm.tab10(np.linspace(0, 0.9, len(conds)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for ic, cond in enumerate(conds):
        sub = well_emb[well_emb["condition"] == cond]
        idx = sub.index.values
        ax.scatter(Zw2[idx, 0], Zw2[idx, 1], c=[pal_c[ic]]*len(idx),
                   s=70, alpha=0.85, edgecolors="k", linewidths=0.3, label=cond)
    ax.set_title(f"Well-level {dim_method} by FA condition (epochs={args.epochs})")
    ax.set_xlabel(f"{dim_method} 1"); ax.set_ylabel(f"{dim_method} 2")
    ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "umap_well_level_by_condition.png", dpi=150)
    plt.close()
    print("Saved umap_well_level_by_condition.png")

# Well-level UMAP by sex (if available)
if "sex" in well_emb.columns:
    sexes   = sorted(well_emb["sex"].unique())
    pal_s   = {"Male": "#4878d0", "Female": "#ee854a", "unknown": "grey"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for sex in sexes:
        sub = well_emb[well_emb["sex"] == sex]
        idx = sub.index.values
        ax.scatter(Zw2[idx, 0], Zw2[idx, 1], c=pal_s.get(sex, "grey"),
                   s=70, alpha=0.85, edgecolors="k", linewidths=0.3, label=sex)
    ax.set_title(f"Well-level {dim_method} by sex (epochs={args.epochs})")
    ax.set_xlabel(f"{dim_method} 1"); ax.set_ylabel(f"{dim_method} 2")
    ax.legend(title="Sex", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "umap_well_level_by_sex.png", dpi=150)
    plt.close()
    print("Saved umap_well_level_by_sex.png")

# Trajectory plot per mouse over dates
fig, ax = plt.subplots(figsize=(8, 6))
for i, mouse in enumerate(mice):
    sub = well_emb[well_emb["mouse_id"] == mouse].sort_values("timepoint")
    # Mean per timepoint
    mean_by_tp = sub.groupby("timepoint")[z_cols].mean()
    tp_order   = mean_by_tp.index.values
    try:
        pts = umap.UMAP(n_neighbors=min(5, len(mean_by_tp)-1),
                        random_state=42).fit_transform(mean_by_tp.values) \
              if len(mean_by_tp) > 2 else Zw2[sub.index.values[:len(tp_order)]]
    except Exception:
        pts = PCA(n_components=2).fit_transform(mean_by_tp.values)

    ax.plot(pts[:, 0], pts[:, 1], "-o", color=palette[i], lw=2, ms=8,
            label=mouse, alpha=0.8)
    for j, tp in enumerate(tp_order):
        ax.annotate(f"T{tp+1}", (pts[j, 0], pts[j, 1]),
                    fontsize=7, ha="center", va="bottom",
                    color=palette[i])
ax.set_title(f"Embedding trajectories per mouse over recording dates (epochs={args.epochs})")
ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
ax.legend(title="Mouse ID", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
fig.savefig(OUT_DIR / "embedding_trajectories.png", dpi=150)
plt.close()
print("Saved embedding_trajectories.png")

# ═══════════════════════════════════════════════════════════════════════════
# Classifier probing (condition, skip if all unknown)
# ═══════════════════════════════════════════════════════════════════════════
if (meta["condition"] != "unknown").all():
    print("\n── Classifier probing: condition from well-level embeddings ──")
    le = LabelEncoder()
    y  = le.fit_transform(well_emb["condition"].values)

    # Leave-one-mouse-out cross-validation
    mice_arr = well_emb["mouse_id"].values
    results  = []

    for test_mouse in mice:
        train_mask = mice_arr != test_mouse
        test_mask  = mice_arr == test_mouse
        if test_mask.sum() == 0:
            continue
        X_train = well_emb[z_cols].values[train_mask]
        X_test  = well_emb[z_cols].values[test_mask]
        y_train = y[train_mask]
        y_test  = y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        scaler  = StandardScaler().fit(X_train)
        clf     = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(scaler.transform(X_train), y_train)
        pred    = clf.predict(scaler.transform(X_test))
        ba      = balanced_accuracy_score(y_test, pred)
        results.append({"left_out_mouse": test_mouse, "balanced_accuracy": round(ba, 3),
                        "n_test": int(test_mask.sum())})

    clf_df = pd.DataFrame(results)
    print(clf_df.to_string(index=False))
    clf_df.to_csv(OUT_DIR / "classifier_probing_lomo.csv", index=False)
    print("Saved classifier_probing_lomo.csv")
    print(f"Mean balanced accuracy: {clf_df['balanced_accuracy'].mean():.3f}")
else:
    print("\nNOTE: Skipping classifier probing (condition labels not yet set).")
    print("      Run again after updating MOUSE_CONDITION in preprocess_FA_kilosort.py.")

# ═══════════════════════════════════════════════════════════════════════════
# Feature attribution: latent dim correlation with quality metrics
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Latent dimension × quality metric correlations ──")
CORR_METRICS = ["snr", "amplitude_median", "half_width", "peak_to_valley",
                "peak_trough_ratio", "firing_range", "sync_spike_4",
                "isi_violations_ratio", "presence_ratio"]
CORR_METRICS = [m for m in CORR_METRICS if m in meta.columns]

corr_rows = []
for zdim in range(Z.shape[1]):
    z_vals = Z[:, zdim]
    row    = {"latent_dim": zdim}
    for m in CORR_METRICS:
        vals = meta[m].fillna(meta[m].median()).values
        r, p = stats.spearmanr(z_vals, vals)
        row[f"rho_{m}"]  = round(r, 3)
        row[f"p_{m}"]    = round(p, 4)
    corr_rows.append(row)

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT_DIR / "latent_dim_metric_correlations.csv", index=False)
print("Saved latent_dim_metric_correlations.csv")

# Find top 3 most condition-informative dims (proxy: highest cross-mouse variance)
mouse_means = np.array([Z[meta["mouse_id"] == m].mean(axis=0) for m in mice])
between_mouse_var = mouse_means.var(axis=0)
top_dims = np.argsort(between_mouse_var)[::-1][:6]
print(f"Top 6 dims with highest between-mouse variance: {top_dims}")

# Plot top dim distributions per mouse
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, dim in zip(axes.ravel(), top_dims):
    data_by_mouse = [Z[meta["mouse_id"] == m, dim] for m in mice]
    parts = ax.violinplot(data_by_mouse, positions=range(n_mice),
                          showmedians=True)
    for body, col in zip(parts["bodies"], cm.tab10(np.linspace(0, 0.9, n_mice))):
        body.set_facecolor(col); body.set_alpha(0.7)
    ax.set_xticks(range(n_mice))
    ax.set_xticklabels(mice, rotation=30, fontsize=8)
    ax.set_ylabel(f"z{dim}")
    ax.set_title(f"Latent dim z{dim}")
plt.suptitle("Top latent dims by between-mouse variance")
plt.tight_layout()
fig.savefig(OUT_DIR / "top_latent_dims_by_mouse.png", dpi=150)
plt.close()
print("Saved top_latent_dims_by_mouse.png")

print(f"\n=== Analysis complete. Results in {OUT_DIR} ===")
