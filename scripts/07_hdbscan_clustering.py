"""
05 — HDBSCAN cluster analysis of HIPPIE embeddings
====================================================
1. Load epoch-specific 32-dim embeddings + metadata
2. UMAP projection (cached per epoch)
3. UMAP colored by quality metrics (waveform + network)
4. HDBSCAN clustering on UMAP coordinates
5. 4-panel UMAP: labels | overlaid waveforms | overlaid ISIs | overlaid ACGs
6. Per-cluster profile grid: waveform × ACG × ISI

Usage:
  python 05_hdbscan_cluster_analysis.py --epochs 20 [--min-cluster-size 500] [--min-samples 50]
  python 05_hdbscan_cluster_analysis.py --epochs 100
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import umap
import hdbscan

# ─── Config ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",           type=int,   default=20)
parser.add_argument("--min-cluster-size", type=int,   default=200)
parser.add_argument("--min-samples",      type=int,   default=50)
parser.add_argument("--umap-n-neighbors", type=int,   default=30)
parser.add_argument("--umap-min-dist",    type=float, default=0.1)
parser.add_argument("--config",           type=str,   default="full_architecture")
parser.add_argument("--z-dim",            type=int,   default=32)
parser.add_argument("--beta",             type=float, default=4.0)
args = parser.parse_args()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (embedding_dir, embedding_file,
                    METADATA_CSV_T3 as METADATA_CSV,
                    WAVEFORMS_CSV_T3 as WAVEFORMS_CSV,
                    ACG_CSV_T3 as ACG_CSV,
                    ISI_CSV_T3 as ISI_CSV,
                    results_hdbscan_dir)

EMB_BASE  = embedding_dir(args.config, args.z_dim, args.beta)
EMB_PATH  = embedding_file(args.epochs, args.config, args.z_dim, args.beta)
META_PATH = METADATA_CSV
WAVE_PATH = WAVEFORMS_CSV
ACG_PATH  = ACG_CSV
ISI_PATH  = ISI_CSV

OUT_DIR    = results_hdbscan_dir(args.epochs)
OUT_DIR.mkdir(parents=True, exist_ok=True)
UMAP_CACHE = OUT_DIR / "umap_coords.npy"

cmap20 = plt.get_cmap("tab20")

cond_palette = {
    "0mg_deficient":      "#e74c3c",
    "2mg_control":        "#2ecc71",
    "20mg_super_excess":  "#f39c12",
    "folinic_acid_excess":"#9b59b6",
    "10mg_excess":        "#3498db",
}

# ─── Load embeddings + metadata ───────────────────────────────────────────────
print(f"Loading embeddings (ep{args.epochs}) and metadata...")
emb  = pd.read_csv(EMB_PATH).values.astype(np.float32)
meta = pd.read_csv(META_PATH)
assert len(emb) == len(meta), "Embedding / metadata length mismatch"
print(f"  Embedding matrix: {emb.shape}")

# ─── UMAP ─────────────────────────────────────────────────────────────────────
if UMAP_CACHE.exists():
    print(f"Loading cached UMAP from {UMAP_CACHE}")
    umap_xy = np.load(UMAP_CACHE)
else:
    print(f"Running UMAP (n_neighbors={args.umap_n_neighbors}, min_dist={args.umap_min_dist})...")
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        random_state=42,
        low_memory=True,
    )
    umap_xy = reducer.fit_transform(StandardScaler().fit_transform(emb))
    np.save(UMAP_CACHE, umap_xy)
    print(f"  Saved UMAP cache → {UMAP_CACHE}")

x, y = umap_xy[:, 0], umap_xy[:, 1]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def draw_base_scatter(ax, cluster_labels, alpha=0.35, s=0.3):
    """Draw UMAP scatter colored by cluster (background layer)."""
    noise_mask = cluster_labels == -1
    ax.scatter(x[noise_mask], y[noise_mask], c="lightgrey", s=0.15, alpha=0.15,
               rasterized=True, linewidths=0)
    for ci in sorted(set(cluster_labels) - {-1}):
        mask = cluster_labels == ci
        ax.scatter(x[mask], y[mask], c=[cmap20(ci % 20)], s=s, alpha=alpha,
                   rasterized=True, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])


def scatter_umap_continuous(ax, values, title, cmap="viridis", vmin=None, vmax=None):
    sc = ax.scatter(x, y, c=values, cmap=cmap, s=0.3, alpha=0.4,
                    rasterized=True, linewidths=0, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def scatter_umap_categorical(ax, labels, title, palette=None):
    unique = sorted(set(labels))
    if palette is None:
        palette = {u: cmap20(i / max(len(unique) - 1, 1)) for i, u in enumerate(unique)}
    colors = [palette[l] for l in labels]
    ax.scatter(x, y, c=colors, s=0.3, alpha=0.4, rasterized=True, linewidths=0)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=palette[u], markersize=5, label=str(u))
               for u in unique]
    ax.legend(handles=handles, fontsize=5, loc="best", framealpha=0.6, ncol=2)


# ─── Fig A: UMAP colored by continuous quality metrics ────────────────────────
print("Plotting metric-colored UMAPs...")

continuous_metrics = [
    ("half_width",           "Half-width (ms)",       "plasma"),
    ("peak_to_valley",       "Peak-to-valley (ms)",   "plasma"),
    ("amplitude_median",     "Amplitude median (µV)", "viridis"),
    ("repolarization_slope", "Repolarization slope",  "RdBu_r"),
    ("recovery_slope",       "Recovery slope",        "RdBu_r"),
    ("firing_range",         "Firing range",          "hot"),
    ("sync_spike_2",         "Synchrony (2 ms)",      "hot"),
    ("sync_spike_4",         "Synchrony (4 ms)",      "hot"),
    ("snr",                  "SNR",                   "viridis"),
    ("div",                  "DIV",                   "coolwarm"),
]

ncols = 5
nrows = int(np.ceil(len(continuous_metrics) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
axes = axes.flatten()
for i, (col, label, cmap) in enumerate(continuous_metrics):
    ax = axes[i]
    if col not in meta.columns:
        ax.set_visible(False); continue
    vals = meta[col].values.astype(float)
    lo, hi = np.nanpercentile(vals, 1), np.nanpercentile(vals, 99)
    sc = scatter_umap_continuous(ax, vals, label, cmap=cmap, vmin=lo, vmax=hi)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f"UMAP colored by quality metrics (epochs={args.epochs})", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "umap_metrics_continuous.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved umap_metrics_continuous.png")

# ─── Fig B: UMAP colored by categorical metadata ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
scatter_umap_categorical(axes[0], meta["condition"].values, "Condition", cond_palette)
scatter_umap_categorical(axes[1], meta["mouse_id"].values,  "Mouse ID")
scatter_umap_categorical(axes[2], meta["sex"].values,       "Sex")
fig.suptitle(f"UMAP — categorical metadata (epochs={args.epochs})", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "umap_categorical.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved umap_categorical.png")

# ─── HDBSCAN ──────────────────────────────────────────────────────────────────
print(f"\nRunning HDBSCAN (min_cluster_size={args.min_cluster_size}, "
      f"min_samples={args.min_samples})...")
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=args.min_cluster_size,
    min_samples=args.min_samples,
    core_dist_n_jobs=-1,
)
cluster_labels = clusterer.fit_predict(umap_xy)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise    = (cluster_labels == -1).sum()
print(f"  {n_clusters} clusters found, {n_noise} noise points "
      f"({100 * n_noise / len(cluster_labels):.1f}%)")

np.save(OUT_DIR / "hdbscan_labels.npy", cluster_labels)
meta["hdbscan_cluster"] = cluster_labels
valid_clusters = sorted(set(cluster_labels) - {-1})

# ─── Load waveforms / ACG / ISI ───────────────────────────────────────────────
print("\nLoading waveforms, ACG, ISI...")
waves = pd.read_csv(WAVE_PATH, header=0).values.astype(np.float32)   # (N, 30)
acgs  = pd.read_csv(ACG_PATH,  header=0).values.astype(np.float32)   # (N, 201)
isis  = pd.read_csv(ISI_PATH,  header=0).values.astype(np.float32)   # (N, 100)

def norm_rows(arr):
    rng = arr.max(axis=1, keepdims=True) - arr.min(axis=1, keepdims=True)
    rng[rng == 0] = 1
    return (arr - arr.min(axis=1, keepdims=True)) / rng * 2 - 1

def prob_rows(arr):
    s = arr.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return arr / s

waves_n = norm_rows(waves)
acgs_p  = prob_rows(acgs)
isis_p  = prob_rows(isis)

acg_lags = np.array([float(c) for c in pd.read_csv(ACG_PATH, header=0, nrows=0).columns])
isi_bins = np.arange(100)
wave_t   = np.arange(30)

# Pre-compute per-cluster means
cl_profiles = {}
for ci in valid_clusters:
    mask = cluster_labels == ci
    cl_profiles[ci] = {
        "n":      mask.sum(),
        "w_mean": waves_n[mask].mean(axis=0),
        "w_std":  waves_n[mask].std(axis=0),
        "a_mean": acgs_p[mask].mean(axis=0),
        "a_std":  acgs_p[mask].std(axis=0),
        "s_mean": isis_p[mask].mean(axis=0),
        "s_std":  isis_p[mask].std(axis=0),
        "cx":     x[mask].mean(),
        "cy":     y[mask].mean(),
    }

# ─── Fig C: 4-panel UMAP with overlaid profiles ───────────────────────────────
# Panel 1: cluster labels
# Panel 2: UMAP + mean waveform inset per cluster
# Panel 3: UMAP + mean ISI inset per cluster
# Panel 4: UMAP + mean ACG inset per cluster

def add_cluster_insets(ax, profile_key, plot_fn, x_range, y_range,
                       inset_frac_x=0.08, inset_frac_y=0.08):
    """
    For each cluster, add a small inset axes at the cluster centroid showing
    the profile given by plot_fn(inset_ax, profile_dict, color).
    inset size is expressed as fraction of the UMAP axis extent.
    """
    dx = (x_range[1] - x_range[0]) * inset_frac_x
    dy = (y_range[1] - y_range[0]) * inset_frac_y
    for ci in valid_clusters:
        p  = cl_profiles[ci]
        cx = p["cx"]
        cy = p["cy"]
        col = cmap20(ci % 20)
        # bbox in data coordinates: [x0, y0, width, height]
        ins = ax.inset_axes(
            [cx - dx / 2, cy - dy / 2, dx, dy],
            transform=ax.transData,
        )
        ins.set_xticks([]); ins.set_yticks([])
        for spine in ins.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(1.2)
        ins.patch.set_alpha(0.85)
        plot_fn(ins, p, col)
        # cluster label above inset
        ax.text(cx, cy + dy * 0.58, str(ci),
                fontsize=6, ha='center', va='bottom',
                color=col, weight='bold')

def plot_wave(ins, p, col):
    ins.plot(wave_t, p["w_mean"], color=col, lw=1.0)
    ins.fill_between(wave_t, p["w_mean"] - p["w_std"], p["w_mean"] + p["w_std"],
                     color=col, alpha=0.2)
    ins.axhline(0, color="grey", lw=0.4, ls="--")
    ins.set_ylim(-1.5, 1.5)

def plot_isi(ins, p, col):
    ins.bar(isi_bins, p["s_mean"], color=col, alpha=0.8, width=0.9)
    ins.set_xlim(0, 99)

def plot_acg(ins, p, col):
    ins.plot(acg_lags, p["a_mean"], color=col, lw=1.0)
    ins.fill_between(acg_lags, p["a_mean"] - p["a_std"], p["a_mean"] + p["a_std"],
                     color=col, alpha=0.2)
    ins.axvline(0, color="grey", lw=0.4, ls="--")

x_range = (x.min(), x.max())
y_range = (y.min(), y.max())
pad_x = (x_range[1] - x_range[0]) * 0.06
pad_y = (y_range[1] - y_range[0]) * 0.06

fig, axes = plt.subplots(1, 4, figsize=(26, 6.5))
panel_titles = [
    f"HDBSCAN clusters (n={n_clusters})",
    "Overlaid mean waveform",
    "Overlaid mean ISI distribution",
    "Overlaid mean ACG",
]
plot_fns = [None, plot_wave, plot_isi, plot_acg]

for pi, ax in enumerate(axes):
    draw_base_scatter(ax, cluster_labels, alpha=0.25 if pi > 0 else 0.4, s=0.25)
    ax.set_xlim(x_range[0] - pad_x, x_range[1] + pad_x)
    ax.set_ylim(y_range[0] - pad_y, y_range[1] + pad_y)
    ax.set_title(panel_titles[pi], fontsize=9)

    if pi == 0:
        # Just labels at centroids
        for ci in valid_clusters:
            p = cl_profiles[ci]
            ax.text(p["cx"], p["cy"], str(ci),
                    fontsize=7, ha='center', va='center',
                    color=cmap20(ci % 20), weight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, pad=1, linewidth=0))
    else:
        add_cluster_insets(ax, None, plot_fns[pi], x_range, y_range,
                           inset_frac_x=0.10, inset_frac_y=0.10)

fig.suptitle(f"HDBSCAN clusters with per-cluster profiles "
             f"(ep{args.epochs}, min_cs={args.min_cluster_size})", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "umap_hdbscan_4panel.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved umap_hdbscan_4panel.png")

# ─── Fig D: per-cluster composition ───────────────────────────────────────────
if valid_clusters:
    conditions  = ["0mg_deficient", "2mg_control", "20mg_super_excess",
                   "folinic_acid_excess", "10mg_excess"]
    cond_colors = [cond_palette.get(c, "grey") for c in conditions]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(valid_clusters) * 0.6 + 2), 5))
    ax = axes[0]
    bottom = np.zeros(len(valid_clusters))
    for cond, col in zip(conditions, cond_colors):
        fracs = [(meta.loc[meta["hdbscan_cluster"] == ci, "condition"] == cond).sum()
                 / max((meta["hdbscan_cluster"] == ci).sum(), 1)
                 for ci in valid_clusters]
        ax.bar(valid_clusters, fracs, bottom=bottom, color=col, label=cond, width=0.8)
        bottom += np.array(fracs)
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Fraction")
    ax.set_title("Condition composition per cluster")
    ax.legend(fontsize=7); ax.set_xticks(valid_clusters)

    ax = axes[1]
    mean_divs = [meta.loc[meta["hdbscan_cluster"] == ci, "div"].mean() for ci in valid_clusters]
    ax.bar(valid_clusters, mean_divs, color="steelblue", width=0.8)
    ax.set_xlabel("Cluster ID"); ax.set_ylabel("Mean DIV")
    ax.set_title("Mean DIV per cluster"); ax.set_xticks(valid_clusters)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cluster_composition.png")

# ─── Fig E: per-cluster metric boxplots ───────────────────────────────────────
# Columns that should be converted from seconds → ms for display
MS_COLS = {"half_width", "peak_to_valley"}
# E/I threshold on half-width: ~0.20 ms (conservative for extracellular MEA recordings)
EI_HW_THRESHOLD_MS = 0.20   # ms

metric_cols = [
    # Waveform shape (E/I relevant)
    ("half_width",           "Half-width (ms)\n← narrow=inh | broad=exc →", True),
    ("peak_to_valley",       "Peak-to-valley (ms)",                          False),
    ("repolarization_slope", "Repolarization slope",                         False),
    ("recovery_slope",       "Recovery slope",                               False),
    ("peak_trough_ratio",    "Peak/trough ratio",                            False),
    ("amplitude_median",     "Amplitude median (µV)",                        False),
    # Firing properties
    ("firing_range",         "Firing range",                                 False),
    ("sync_spike_2",         "Synchrony (2 ms)",                             False),
    ("sync_spike_4",         "Synchrony (4 ms)",                             False),
    ("snr",                  "SNR",                                          False),
    # Development
    ("div",                  "DIV",                                          False),
]
metric_cols = [(c, l, ei) for c, l, ei in metric_cols if c in meta.columns]

if valid_clusters:
    nmet = len(metric_cols)
    ncols_box = min(6, nmet)
    nrows_box  = int(np.ceil(nmet / ncols_box))
    fig, axes = plt.subplots(nrows_box, ncols_box,
                             figsize=(ncols_box * 2.5, nrows_box * 4))
    axes = np.array(axes).flatten()

    colors = [cmap20(ci % 20) for ci in valid_clusters]
    for ax_i, (col, label, is_ei) in enumerate(metric_cols):
        ax = axes[ax_i]
        scale = 1000.0 if col in MS_COLS else 1.0
        data_by_cluster = [meta.loc[meta["hdbscan_cluster"] == ci, col].dropna().values * scale
                           for ci in valid_clusters]
        bp = ax.boxplot(data_by_cluster, labels=valid_clusters,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, col_c in zip(bp["boxes"], colors):
            patch.set_facecolor(col_c)
            patch.set_alpha(0.7)
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Cluster", fontsize=6)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        if is_ei:
            ax.axhline(EI_HW_THRESHOLD_MS, color="red", lw=1.2, ls="--",
                       label=f"E/I threshold ({EI_HW_THRESHOLD_MS} ms)")
            ax.legend(fontsize=5, loc="upper right")

    for j in range(ax_i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Quality metrics per HDBSCAN cluster (ep{args.epochs}, "
                 f"min_cs={args.min_cluster_size})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_metrics_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cluster_metrics_boxplot.png")

# ─── Fig E2: E/I scatter — half-width × peak-to-valley per cluster ─────────────
if valid_clusters and "half_width" in meta.columns and "peak_to_valley" in meta.columns:
    hw_ms  = meta["half_width"].values   * 1000.0
    ptv_ms = meta["peak_to_valley"].values * 1000.0

    # Subsample for speed (max 3000 pts per cluster)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 6))

    for ci in valid_clusters:
        mask = cluster_labels == ci
        idx  = np.where(mask)[0]
        if len(idx) > 3000:
            idx = rng.choice(idx, 3000, replace=False)
        col = cmap20(ci % 20)
        ax.scatter(hw_ms[idx], ptv_ms[idx], c=[col], s=1.5, alpha=0.25,
                   rasterized=True, linewidths=0)

    # Cluster centroids + labels
    for ci in valid_clusters:
        mask = cluster_labels == ci
        cx   = hw_ms[mask].mean()
        cy   = ptv_ms[mask].mean()
        col  = cmap20(ci % 20)
        ax.scatter(cx, cy, c=[col], s=60, marker="D", edgecolors="black",
                   linewidths=0.8, zorder=5)
        ax.text(cx, cy + 0.005, str(ci), fontsize=7, ha="center", va="bottom",
                color=col, weight="bold")

    # E/I reference lines
    ax.axvline(EI_HW_THRESHOLD_MS, color="red", lw=1.2, ls="--",
               label=f"Half-width threshold ({EI_HW_THRESHOLD_MS} ms)")
    ptv_threshold = 0.40  # ms — approximate; broad = excitatory
    ax.axhline(ptv_threshold, color="blue", lw=1.2, ls="--",
               label=f"Peak-to-valley threshold ({ptv_threshold} ms)")

    # Quadrant labels
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    pad = 0.02
    ax.text(EI_HW_THRESHOLD_MS - pad, ptv_threshold + pad,
            "narrow spike\nbroad P2V", fontsize=7, ha="right", va="bottom",
            color="grey", style="italic")
    ax.text(EI_HW_THRESHOLD_MS + pad, ptv_threshold + pad,
            "broad spike\nbroad P2V\n(putative Exc)",
            fontsize=7, ha="left", va="bottom", color="green", style="italic")
    ax.text(EI_HW_THRESHOLD_MS - pad, ptv_threshold - pad,
            "narrow spike\nnarrow P2V\n(putative Inh/FS)",
            fontsize=7, ha="right", va="top", color="purple", style="italic")
    ax.text(EI_HW_THRESHOLD_MS + pad, ptv_threshold - pad,
            "broad spike\nnarrow P2V", fontsize=7, ha="left", va="top",
            color="grey", style="italic")

    # Legend for clusters
    handles = [plt.Line2D([0], [0], marker="D", color="w",
                          markerfacecolor=cmap20(ci % 20),
                          markeredgecolor="black", markersize=6, label=f"Cl {ci}")
               for ci in valid_clusters]
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              framealpha=0.7, ncol=3)

    ax.set_xlabel("Half-width (ms)", fontsize=10)
    ax.set_ylabel("Peak-to-valley (ms)", fontsize=10)
    ax.set_title(f"E/I scatter: half-width × peak-to-valley per cluster\n"
                 f"(ep{args.epochs}, min_cs={args.min_cluster_size})", fontsize=10)
    ax.legend(handles=handles, fontsize=6, loc="upper left",
              framealpha=0.7, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ei_scatter_hw_ptv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved ei_scatter_hw_ptv.png")

# ─── Fig F: per-cluster profile grid (waveform × ACG × ISI) ───────────────────
if valid_clusters:
    display_clusters = valid_clusters[:20]
    n_cl = len(display_clusters)
    fig, axes = plt.subplots(n_cl, 3, figsize=(10, n_cl * 1.8 + 1))
    if n_cl == 1:
        axes = axes[np.newaxis, :]

    for row_i, ci in enumerate(display_clusters):
        p = cl_profiles[ci]
        col = cmap20(ci % 20)

        ax = axes[row_i, 0]
        ax.plot(wave_t, p["w_mean"], color=col, lw=1.5)
        ax.fill_between(wave_t, p["w_mean"] - p["w_std"], p["w_mean"] + p["w_std"],
                        color=col, alpha=0.25)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylabel(f"Cl {ci}\n(n={p['n']})", fontsize=7)
        if row_i == 0: ax.set_title("Mean waveform", fontsize=8)
        ax.set_xticks([]); ax.tick_params(axis='y', labelsize=6)

        ax = axes[row_i, 1]
        ax.plot(acg_lags, p["a_mean"], color=col, lw=1.2)
        ax.fill_between(acg_lags, p["a_mean"] - p["a_std"], p["a_mean"] + p["a_std"],
                        color=col, alpha=0.25)
        ax.axvline(0, color="grey", lw=0.5, ls="--")
        if row_i == 0: ax.set_title("Mean ACG", fontsize=8)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.tick_params(axis='both', labelsize=6)

        ax = axes[row_i, 2]
        ax.bar(isi_bins, p["s_mean"], color=col, alpha=0.7, width=0.9)
        if row_i == 0: ax.set_title("Mean ISI distribution", fontsize=8)
        ax.set_xticks([0, 25, 50, 75, 99])
        ax.tick_params(axis='both', labelsize=6)

    axes[-1, 0].set_xlabel("Sample", fontsize=7)
    axes[-1, 1].set_xlabel("Lag (ms)", fontsize=7)
    axes[-1, 2].set_xlabel("ISI bin", fontsize=7)
    fig.suptitle(f"Per-cluster profiles (ep{args.epochs}, min_cs={args.min_cluster_size})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cluster_profiles_wave_acg_isi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cluster_profiles_wave_acg_isi.png")

print(f"\n=== Done. Results in {OUT_DIR} ===")
