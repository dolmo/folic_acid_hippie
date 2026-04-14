"""
06 — E/I proportion analysis across FA conditions
==================================================
Using HDBSCAN clusters from epochs_100 run:
  Excitatory clusters: 2, 3, 4, 12
  Inhibitory clusters: 0, 1, 5, 6, 7, 8, 9, 10, 11
  Noise (excluded): -1

Approach:
  1. Assign each unit to E / I / noise
  2. Compute per-well E fraction = E / (E + I)  (noise excluded)
  3. Compare E fraction across FA conditions
     - Kruskal-Wallis omnibus test
     - Mann-Whitney U pairwise post-hoc (Bonferroni corrected)
  4. Visualise:
     a. Boxplot of E fraction per condition
     b. E fraction trajectory over DIV, coloured by condition
     c. Per-condition E fraction heatmap: DIV × condition
     d. Strip plot (per-well dots) with condition summary
     e. Stacked bar: absolute E / I unit counts per condition

Output: embedding_results/epochs_100_hdbscan/ei_proportion/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from itertools import combinations

# ─── Config (from config.py) ──────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import METADATA_CSV, results_hdbscan_dir

HDBSCAN_DIR = results_hdbscan_dir(100)
META_PATH   = METADATA_CSV
OUT_DIR     = HDBSCAN_DIR / "ei_proportion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXC_CLUSTERS = {2, 3, 4, 12}
INH_CLUSTERS = {0, 1, 5, 6, 7, 8, 9, 10, 11}

COND_PALETTE = {
    "0mg_deficient":       "#e74c3c",
    "2mg_control":         "#2ecc71",
    "20mg_super_excess":   "#f39c12",
    "folinic_acid_excess": "#9b59b6",
    "10mg_excess":         "#3498db",
}
COND_ORDER = ["0mg_deficient", "2mg_control", "10mg_excess",
              "20mg_super_excess", "folinic_acid_excess"]

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading metadata and HDBSCAN labels...")
meta   = pd.read_csv(META_PATH)
labels = np.load(HDBSCAN_DIR / "hdbscan_labels.npy")
assert len(meta) == len(labels), "Length mismatch"

meta["hdbscan_cluster"] = labels

# Assign cell type
def assign_type(cl):
    if cl in EXC_CLUSTERS:
        return "Excitatory"
    elif cl in INH_CLUSTERS:
        return "Inhibitory"
    else:
        return "Noise"

meta["cell_type"] = meta["hdbscan_cluster"].apply(assign_type)

# ─── Per-well E fraction ──────────────────────────────────────────────────────
print("Computing per-well E/I fractions...")

# Unique well identifier
meta["well_id"] = (meta["date"].astype(str) + "_" +
                   meta["mouse_id"] + "_" +
                   meta["well"])

well_stats = []
for well_id, grp in meta.groupby("well_id"):
    ei = grp[grp["cell_type"] != "Noise"]
    n_exc = (ei["cell_type"] == "Excitatory").sum()
    n_inh = (ei["cell_type"] == "Inhibitory").sum()
    n_tot = n_exc + n_inh
    if n_tot == 0:
        continue
    row = grp.iloc[0]
    well_stats.append({
        "well_id":   well_id,
        "condition": row["condition"],
        "mouse_id":  row["mouse_id"],
        "div":       row["div"],
        "sex":       row["sex"],
        "n_exc":     n_exc,
        "n_inh":     n_inh,
        "n_total":   n_tot,
        "e_frac":    n_exc / n_tot,
        "i_frac":    n_inh / n_tot,
        "ei_ratio":  n_exc / n_inh if n_inh > 0 else np.nan,
    })

ws = pd.DataFrame(well_stats)
ws.to_csv(OUT_DIR / "per_well_ei.csv", index=False)
print(f"  {len(ws)} wells with at least one labelled unit")
print(ws.groupby("condition")[["e_frac", "n_exc", "n_inh"]].agg(["mean","std","count"]))

# ─── Statistics ───────────────────────────────────────────────────────────────
print("\n=== Statistical tests ===")

present_conds = [c for c in COND_ORDER if c in ws["condition"].values]
groups = [ws.loc[ws["condition"] == c, "e_frac"].dropna().values
          for c in present_conds]

# Kruskal-Wallis
kw_stat, kw_p = stats.kruskal(*groups)
print(f"Kruskal-Wallis: H={kw_stat:.3f}, p={kw_p:.4e}")

# Pairwise Mann-Whitney with Bonferroni
pairs = list(combinations(range(len(present_conds)), 2))
n_pairs = len(pairs)
print(f"\nPairwise Mann-Whitney U (Bonferroni corrected, n_comparisons={n_pairs}):")
pw_results = []
for i, j in pairs:
    u, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
    p_adj = min(p * n_pairs, 1.0)
    sig = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else ("*" if p_adj < 0.05 else "ns"))
    print(f"  {present_conds[i]:22s} vs {present_conds[j]:22s}: "
          f"U={u:.0f}, p={p:.4e}, p_adj={p_adj:.4e}  {sig}")
    pw_results.append({
        "cond_a": present_conds[i], "cond_b": present_conds[j],
        "U": u, "p_raw": p, "p_bonferroni": p_adj, "sig": sig
    })

pw_df = pd.DataFrame(pw_results)
pw_df.to_csv(OUT_DIR / "pairwise_mannwhitney.csv", index=False)

# ─── Summary stats table ──────────────────────────────────────────────────────
summary = ws.groupby("condition")["e_frac"].agg(
    n_wells="count",
    mean_e_frac="mean",
    median_e_frac="median",
    std_e_frac="std",
    sem_e_frac=lambda x: x.sem()
).reset_index()
summary.to_csv(OUT_DIR / "ei_summary_by_condition.csv", index=False)
print("\nSummary:\n", summary.to_string(index=False))

# ─── Fig 1: Boxplot + strip — E fraction per condition ───────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

rng = np.random.default_rng(0)
y_maxes = []
for xi, cond in enumerate(present_conds):
    vals = ws.loc[ws["condition"] == cond, "e_frac"].values
    col  = COND_PALETTE.get(cond, "grey")
    jitter = rng.uniform(-0.15, 0.15, len(vals))
    ax.scatter(xi + jitter, vals, color=col, alpha=0.5, s=18, linewidths=0,
               zorder=3)
    bp = ax.boxplot(vals, positions=[xi], widths=0.35,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=2.0),
                    boxprops=dict(facecolor=col, alpha=0.3),
                    whiskerprops=dict(color=col),
                    capprops=dict(color=col))
    y_maxes.append(vals.max() if len(vals) else 0)
    # n= and median annotation below each box
    med = np.median(vals)
    ax.text(xi, vals.min() - 0.008, f"n={len(vals)}\nmed={med:.3f}",
            ha="center", va="top", fontsize=7, color="dimgrey")

# Significance brackets — show all pairs where p_raw < 0.15 (borderline + sig)
# Use p_adj for label; show p_raw for context
notable = [(i, j, r) for (i, j), r in zip(pairs, pw_results)
           if r["p_raw"] < 0.15]
y_top = max(y_maxes)
bracket_gap = 0.025
for k, (i, j, r) in enumerate(notable):
    y = y_top + 0.03 + k * bracket_gap
    col_br = "black" if r["sig"] != "ns" else "dimgrey"
    lw_br  = 1.5 if r["sig"] != "ns" else 0.8
    ls_br  = "-" if r["sig"] != "ns" else "--"
    ax.plot([i, i, j, j], [y - 0.006, y, y, y - 0.006],
            lw=lw_br, color=col_br, ls=ls_br)
    label = (f"{r['sig']}\np_adj={r['p_bonferroni']:.3f}"
             if r["sig"] != "ns"
             else f"p_adj={r['p_bonferroni']:.2f}\n(p_raw={r['p_raw']:.3f})")
    ax.text((i + j) / 2, y + 0.002, label,
            ha="center", va="bottom", fontsize=6.5, color=col_br)

ax.set_xticks(range(len(present_conds)))
ax.set_xticklabels([c.replace("_", "\n") for c in present_conds], fontsize=9)
ax.set_ylabel("Excitatory fraction (E / E+I)", fontsize=10)
kw_sig = "***" if kw_p < 0.001 else ("**" if kw_p < 0.01 else
          ("*" if kw_p < 0.05 else "ns"))
ax.set_title(f"E/I proportion per FA condition (per-well)\n"
             f"Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.3e} [{kw_sig}]  "
             f"| Brackets: pairs with p_raw < 0.15 (Bonferroni-adjusted labels)",
             fontsize=9)
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
ax.grid(axis="y", lw=0.5, alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_boxplot_e_frac_by_condition.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved fig1_boxplot_e_frac_by_condition.png")

# ─── Fig 2: E fraction trajectory over DIV, one line per condition ────────────
div_cond = (ws.groupby(["condition", "div"])["e_frac"]
              .agg(mean="mean", sem=lambda x: x.sem(), n="count")
              .reset_index())

# Per-DIV Kruskal-Wallis across conditions
divs_sorted = sorted(ws["div"].unique())
div_kw = {}
for d in divs_sorted:
    sub = ws[ws["div"] == d]
    grp_d = [sub.loc[sub["condition"] == c, "e_frac"].dropna().values
              for c in present_conds if (sub["condition"] == c).sum() >= 2]
    if len(grp_d) >= 2:
        h, p = stats.kruskal(*grp_d)
        div_kw[d] = (h, p)
    else:
        div_kw[d] = (np.nan, np.nan)

fig, (ax, ax_stat) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={"height_ratios": [4, 1]},
                                   sharex=True)
for cond in present_conds:
    sub = div_cond[div_cond["condition"] == cond].sort_values("div")
    col = COND_PALETTE.get(cond, "grey")
    ax.plot(sub["div"], sub["mean"], marker="o", color=col,
            label=cond.replace("_", " "), lw=2, markersize=5)
    ax.fill_between(sub["div"],
                    sub["mean"] - sub["sem"],
                    sub["mean"] + sub["sem"],
                    color=col, alpha=0.15)
    # n wells per DIV label
    for _, row in sub.iterrows():
        ax.text(row["div"], row["mean"] + row["sem"] + 0.003,
                f"n={int(row['n'])}", ha="center", va="bottom",
                fontsize=5.5, color=col, alpha=0.8)

ax.set_ylabel("Mean excitatory fraction ± SEM", fontsize=10)
ax.set_title(f"E fraction trajectory over development\n"
             f"Overall KW: H={kw_stat:.2f}, p={kw_p:.3e} [{kw_sig}]", fontsize=10)
ax.legend(fontsize=8, loc="best")
ax.grid(lw=0.5, alpha=0.4)

# Per-DIV p-value panel
ax_stat.axhline(0.05, color="red", lw=0.8, ls="--", alpha=0.7, label="p=0.05")
ax_stat.axhline(0.01, color="darkorange", lw=0.8, ls="--", alpha=0.7, label="p=0.01")
for d in divs_sorted:
    h, p = div_kw[d]
    if not np.isnan(p):
        col_pt = "red" if p < 0.05 else "steelblue"
        ax_stat.scatter(d, p, color=col_pt, s=40, zorder=3)
        ax_stat.text(d, p * 1.15, f"{p:.3f}", ha="center", va="bottom",
                     fontsize=6.5, color=col_pt)
ax_stat.set_yscale("log")
ax_stat.set_ylabel("KW p-value\n(per DIV)", fontsize=8)
ax_stat.set_xlabel("DIV", fontsize=10)
ax_stat.legend(fontsize=7, loc="upper right")
ax_stat.grid(lw=0.5, alpha=0.4)
ax_stat.invert_yaxis()

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_trajectory_e_frac_by_div.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig2_trajectory_e_frac_by_div.png")

# ─── Fig 3: Heatmap — mean E fraction, rows=DIV, cols=condition ──────────────
pivot      = div_cond.pivot(index="div", columns="condition", values="mean")
pivot_n    = div_cond.pivot(index="div", columns="condition", values="n")
pivot_sem  = div_cond.pivot(index="div", columns="condition", values="sem")
for piv in (pivot, pivot_n, pivot_sem):
    piv.columns.name = None
pivot      = pivot[[c for c in present_conds if c in pivot.columns]]
pivot_n    = pivot_n[[c for c in present_conds if c in pivot_n.columns]]
pivot_sem  = pivot_sem[[c for c in present_conds if c in pivot_sem.columns]]

# Per-row (per-DIV) KW p-values already in div_kw
fig, (ax, ax_kw) = plt.subplots(
    1, 2, figsize=(len(pivot.columns) * 1.6 + 2.5, len(pivot) * 0.85 + 2),
    gridspec_kw={"width_ratios": [len(pivot.columns), 0.9]})

im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
               vmin=pivot.values[~np.isnan(pivot.values)].min() - 0.02,
               vmax=pivot.values[~np.isnan(pivot.values)].max() + 0.02,
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="Mean E fraction", fraction=0.04, pad=0.02)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([c.replace("_", "\n") for c in pivot.columns], fontsize=8)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"DIV {d}" for d in pivot.index], fontsize=8)
ax.set_title("Mean E fraction ± SEM\n(n wells per cell)", fontsize=9)

for r, div in enumerate(pivot.index):
    for c, cond in enumerate(pivot.columns):
        v   = pivot.loc[div, cond]   if cond in pivot.columns   else np.nan
        n   = pivot_n.loc[div, cond] if cond in pivot_n.columns else np.nan
        sem = pivot_sem.loc[div, cond] if cond in pivot_sem.columns else np.nan
        if not np.isnan(v):
            ax.text(c, r - 0.15, f"{v:.3f}",
                    ha="center", va="center", fontsize=7, color="black", weight="bold")
            ax.text(c, r + 0.22, f"±{sem:.3f}  n={int(n)}",
                    ha="center", va="center", fontsize=5.5, color="#333333")

# KW p-value column
ax_kw.set_ylim(-0.5, len(pivot) - 0.5)
ax_kw.invert_yaxis()
for r, d in enumerate(pivot.index):
    h, p = div_kw.get(d, (np.nan, np.nan))
    if not np.isnan(p):
        sig_label = ("***" if p < 0.001 else ("**" if p < 0.01 else
                     ("*" if p < 0.05 else "ns")))
        col_kw = "red" if p < 0.05 else "steelblue"
        ax_kw.barh(r, -np.log10(max(p, 1e-10)), color=col_kw, alpha=0.6, height=0.6)
        ax_kw.text(0.05, r, f"p={p:.3f} {sig_label}",
                   va="center", ha="left", fontsize=7, color=col_kw)
ax_kw.axvline(-np.log10(0.05), color="red", lw=0.8, ls="--", alpha=0.7)
ax_kw.set_yticks(range(len(pivot.index)))
ax_kw.set_yticklabels([])
ax_kw.set_xlabel("−log₁₀(p)\nKW per DIV", fontsize=8)
ax_kw.set_title("DIV-level\nKW test", fontsize=8)

fig.suptitle("Mean excitatory fraction — DIV × condition", fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_heatmap_e_frac_div_cond.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig3_heatmap_e_frac_div_cond.png")

# ─── Fig 4: Stacked bar — unit counts per condition (E vs I) ─────────────────
cond_counts = (meta[meta["cell_type"] != "Noise"]
               .groupby(["condition", "cell_type"])
               .size()
               .unstack(fill_value=0))

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

exc_vals = [cond_counts.loc[c, "Excitatory"] if c in cond_counts.index else 0
            for c in present_conds]
inh_vals = [cond_counts.loc[c, "Inhibitory"] if c in cond_counts.index else 0
            for c in present_conds]
xs = np.arange(len(present_conds))

# Absolute counts
ax = axes[0]
bars_e = ax.bar(xs, exc_vals, color="#e74c3c", alpha=0.8, label="Excitatory")
bars_i = ax.bar(xs, inh_vals, bottom=exc_vals, color="#3498db", alpha=0.8,
                label="Inhibitory")
# Count labels inside each segment
for xi, (ev, iv) in enumerate(zip(exc_vals, inh_vals)):
    if ev > 0:
        ax.text(xi, ev / 2, f"{ev:,}", ha="center", va="center",
                fontsize=7, color="white", weight="bold")
    if iv > 0:
        ax.text(xi, ev + iv / 2, f"{iv:,}", ha="center", va="center",
                fontsize=7, color="white", weight="bold")
    ax.text(xi, ev + iv + 100, f"tot={ev+iv:,}", ha="center", va="bottom",
            fontsize=6.5, color="dimgrey")
ax.set_xticks(xs)
ax.set_xticklabels([c.replace("_", "\n") for c in present_conds], fontsize=8)
ax.set_ylabel("Unit count", fontsize=9)
ax.set_title("Absolute unit counts", fontsize=10)
ax.legend(fontsize=8)

# Normalised fractions + pairwise p-values
ax = axes[1]
totals = np.array(exc_vals, dtype=float) + np.array(inh_vals, dtype=float)
totals[totals == 0] = 1
e_fracs = np.array(exc_vals) / totals
i_fracs = np.array(inh_vals) / totals
ax.bar(xs, e_fracs, color="#e74c3c", alpha=0.8, label="Excitatory")
ax.bar(xs, i_fracs, bottom=e_fracs, color="#3498db", alpha=0.8, label="Inhibitory")

# Fraction labels inside segments
for xi, (ef, inf_) in enumerate(zip(e_fracs, i_fracs)):
    ax.text(xi, ef / 2,    f"{ef:.2f}", ha="center", va="center",
            fontsize=8, color="white", weight="bold")
    ax.text(xi, ef + inf_/2, f"{inf_:.2f}", ha="center", va="center",
            fontsize=8, color="white", weight="bold")

# Add pairwise p-values from per-well analysis as a text table below
ax.set_xticks(xs)
ax.set_xticklabels([c.replace("_", "\n") for c in present_conds], fontsize=8)
ax.set_ylabel("Fraction", fontsize=9)
ax.set_title(f"Normalised E/I composition\n"
             f"KW (per-well): H={kw_stat:.2f}, p={kw_p:.3e} [{kw_sig}]",
             fontsize=9)
ax.legend(fontsize=8)

# Pairwise stats annotation box
notable_pairs_str = "\n".join(
    f"{r['cond_a'].replace('_',' ')} vs {r['cond_b'].replace('_',' ')}: "
    f"p_raw={r['p_raw']:.3f}, p_adj={r['p_bonferroni']:.3f} {r['sig']}"
    for r in pw_results if r["p_raw"] < 0.15
)
if notable_pairs_str:
    ax.text(0.01, -0.28, "Notable pairwise comparisons (p_raw<0.15):\n" + notable_pairs_str,
            transform=ax.transAxes, fontsize=6, va="top", ha="left",
            bbox=dict(facecolor="lightyellow", edgecolor="grey",
                      boxstyle="round,pad=0.3", alpha=0.9))

fig.suptitle("E/I unit composition per FA condition", fontsize=11)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig4_stacked_bar_ei_counts.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig4_stacked_bar_ei_counts.png")

# ─── Fig 5: E fraction by condition × sex ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
sex_styles  = {"Male": "o", "Female": "s"}
sex_offsets = {"Male": -0.15, "Female": 0.15}
sex_colors  = {"Male": "#2980b9", "Female": "#c0392b"}

well_max = {}  # track per-condition top y for bracket placement
for xi, cond in enumerate(present_conds):
    col = COND_PALETTE.get(cond, "grey")
    top_y = 0
    for sex, marker in sex_styles.items():
        vals = ws.loc[(ws["condition"] == cond) & (ws["sex"] == sex),
                      "e_frac"].values
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.06, 0.06, len(vals))
        xpos = xi + sex_offsets[sex] + jitter
        sc = ax.scatter(xpos, vals, color=sex_colors[sex], alpha=0.55, s=22,
                        marker=marker, linewidths=0.5, edgecolors="grey", zorder=3)
        mean_v = vals.mean()
        ax.plot([xi + sex_offsets[sex] - 0.07,
                 xi + sex_offsets[sex] + 0.07],
                [mean_v, mean_v], color=sex_colors[sex], lw=2.5, zorder=4)
        ax.text(xi + sex_offsets[sex], vals.min() - 0.007,
                f"n={len(vals)}", ha="center", va="top",
                fontsize=6, color=sex_colors[sex])
        top_y = max(top_y, vals.max())
    well_max[xi] = top_y

# Within-condition Male vs Female Mann-Whitney U
y_bracket_base = max(well_max.values()) + 0.015
for xi, cond in enumerate(present_conds):
    m_vals = ws.loc[(ws["condition"] == cond) & (ws["sex"] == "Male"),
                    "e_frac"].dropna().values
    f_vals = ws.loc[(ws["condition"] == cond) & (ws["sex"] == "Female"),
                    "e_frac"].dropna().values
    if len(m_vals) >= 2 and len(f_vals) >= 2:
        u_s, p_s = stats.mannwhitneyu(m_vals, f_vals, alternative="two-sided")
        sig_s = "***" if p_s < 0.001 else ("**" if p_s < 0.01 else
                ("*" if p_s < 0.05 else "ns"))
        col_br = "black" if sig_s != "ns" else "dimgrey"
        lw_br  = 1.4 if sig_s != "ns" else 0.8
        y = y_bracket_base
        ax.plot([xi - 0.15, xi - 0.15, xi + 0.15, xi + 0.15],
                [y - 0.005, y, y, y - 0.005],
                lw=lw_br, color=col_br)
        label = f"{sig_s}\np={p_s:.3f}" if sig_s != "ns" else f"p={p_s:.3f}"
        ax.text(xi, y + 0.002, label, ha="center", va="bottom",
                fontsize=6.5, color=col_br)

# Legend for sex markers
handles_sex = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=sex_colors["Male"],
               markeredgecolor="grey", markersize=7, label="Male"),
    plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=sex_colors["Female"],
               markeredgecolor="grey", markersize=7, label="Female"),
]
ax.legend(handles=handles_sex, fontsize=9, loc="upper right")
ax.set_xticks(range(len(present_conds)))
ax.set_xticklabels([c.replace("_", "\n") for c in present_conds], fontsize=9)
ax.set_ylabel("Excitatory fraction", fontsize=10)
ax.set_title("E fraction by condition and sex\n"
             "(bar = mean per sex; bracket = Male vs Female MWU within condition)",
             fontsize=9)
ax.grid(axis="y", lw=0.5, alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig5_e_frac_by_condition_sex.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig5_e_frac_by_condition_sex.png")

print(f"\n=== Done. All results in {OUT_DIR} ===")
