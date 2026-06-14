"""
Step 5: Condition × DIV trajectory EDA
=======================================
Compares unit-level and network-level metrics across FA conditions and over
developmental time (DIV), framing results as "before vs after" FA exposure.

Design:
  - Conditions A/B/D/E share the same 4 mice (M07137/M07708/M07865/M08092)
    recorded at DIV 3, 6, 9, 13, 16, 20, 23 → clean longitudinal comparison
  - Condition C (10 mg, DIV 5/8 only, different mice) treated separately
  - "Early" = DIV 6-8  (low FA exposure time)
  - "Late"  = DIV 24-28 (maximal FA exposure time)

Outputs saved to:
  /mnt/d/datasets_hippie/Roy_shruti_folic_data/condition_EDA/

Usage:
  /home/jesus/hippie_rebuttals/hip-hip-hippie/hippie_venv/bin/python 05_condition_trajectory_EDA.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ── Paths (from config.py) ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import METADATA_CSV_T3 as METADATA_CSV, RESULTS_NETWORK, RESULTS_COND

META_PATH = METADATA_CSV
NET_PATH  = RESULTS_NETWORK / "network_metrics_all_wells.csv"
OUT_DIR   = RESULTS_COND
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Condition display settings ────────────────────────────────────────────────
COND_ORDER  = ["0mg_deficient", "2mg_control", "10mg_excess", "20mg_super_excess"]
COND_LABELS = {
    "0mg_deficient":      "0 mg FA (deficient)",
    "2mg_control":        "2 mg FA (control)",
    "10mg_excess":        "10 mg FA (excess)",
    "20mg_super_excess":  "20 mg FA (super-excess)",
}

COND_COLORS = {
    "0mg_deficient":      "#d62728",   # red
    "2mg_control":        "#2ca02c",   # green (control)
    "10mg_excess":        "#ff7f0e",   # orange
    "20mg_super_excess":  "#9467bd",   # purple
}

MAIN_DIVS = [6, 8, 13, 15, 16, 17, 21, 24, 28, 31, 36]
EARLY_DIV = [6, 8]
LATE_DIV  = [24, 28]

# Waveform / electrophysiology metrics to analyse
UNIT_METRICS = [
    ("half_width",           "Half-width (ms)",            False),
    ("peak_to_valley",       "Peak-to-valley (ms)",        False),
    ("repolarization_slope", "Repolarisation slope",       False),
    ("recovery_slope",       "Recovery slope",             False),
    ("snr",                  "SNR",                        False),
    ("amplitude_median",     "Amplitude median (µV)",      False),
    ("firing_range",         "Firing range",               False),
    ("sync_spike_2",         "Synchrony (2 ms)",           False),
    ("sync_spike_4",         "Synchrony (4 ms)",           False),
    ("isi_violations_ratio", "ISI violations ratio",       True),   # log scale
]
NET_METRICS = [
    ("mean_firing_rate_hz",   "Mean firing rate (Hz)"),
    ("burst_rate_per_min",    "Burst rate (bursts / min)"),
    ("network_synchrony",     "Network synchrony (Fano factor)"),
    ("burst_fraction",        "Burst fraction"),
    ("mean_ibi_s",            "Mean IBI (s)"),
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading metadata …")
meta = pd.read_csv(META_PATH)

# Aggregate unit-level metrics to well-level medians (removes within-well correlation)
print("Aggregating unit-level metrics to well medians …")
unit_metrics_names = [m for m, _, _ in UNIT_METRICS]
agg_dict = {m: "median" for m in unit_metrics_names if m in meta.columns}
agg_dict["cluster_id"] = "count"

well_unit = (
    meta.groupby(["date", "mouse_id", "run", "well", "condition", "sex", "div"])
        .agg(agg_dict)
        .reset_index()
        .rename(columns={"cluster_id": "n_units"})
)

# Load network metrics and attach condition/div from metadata
print("Loading network metrics …")
net_raw = pd.read_csv(NET_PATH, dtype={"date": str, "run": str})
net_raw = net_raw.drop(columns=["condition"], errors="ignore")  # was all "unknown"

# Build (mouse_id, well) → (condition, div) lookup (constant across dates)
wc = meta.drop_duplicates(["mouse_id", "well"])[["mouse_id", "well", "condition", "sex"]]
# For div we need per (date, mouse_id) → use full mapping
div_lookup = meta.drop_duplicates(["mouse_id", "div"])[["mouse_id", "div"]].copy()
# Actually div already attached via (mouse_id, date) but network metrics has "date" as folder string
# Safest: join condition from (mouse_id, well), then div from metadata using date string
meta_dates = meta[["date", "mouse_id", "div"]].drop_duplicates().copy()
meta_dates["date_str"] = meta_dates["date"].astype(str)

# net_raw "date" column is a string (from directory names, e.g. "250228")
net = net_raw.merge(wc, on=["mouse_id", "well"], how="left")
net = net.merge(meta_dates[["date_str", "mouse_id", "div"]],
                left_on=["date", "mouse_id"],
                right_on=["date_str", "mouse_id"],
                how="left")
net = net.drop(columns=["date_str"], errors="ignore")

print(f"  Unit-level well rows: {len(well_unit)}  |  Network metric rows: {len(net)}")

# ── Helper: Kruskal-Wallis + pairwise Mann-Whitney with Bonferroni ─────────────
def kruskal_pairwise(groups_dict, alpha=0.05):
    """groups_dict: {label: array_like}. Returns (kw_p, pairwise_df)."""
    labels = list(groups_dict.keys())
    arrays = [np.asarray(v) for v in groups_dict.values()]
    valid  = [(l, a) for l, a in zip(labels, arrays) if len(a) >= 3]
    if len(valid) < 2:
        return np.nan, pd.DataFrame()
    labs, arrs = zip(*valid)
    try:
        _, kw_p = stats.kruskal(*arrs)
    except ValueError:
        return np.nan, pd.DataFrame()
    n_pairs = len(list(combinations(labs, 2)))
    rows = []
    for (l1, a1), (l2, a2) in combinations(zip(labs, arrs), 2):
        _, p = stats.mannwhitneyu(a1, a2, alternative="two-sided")
        p_adj = min(p * n_pairs, 1.0)
        rows.append({"cond_a": l1, "cond_b": l2, "p_raw": p, "p_bonf": p_adj,
                     "sig": "***" if p_adj < 0.001 else "**" if p_adj < 0.01
                            else "*" if p_adj < 0.05 else "ns"})
    return kw_p, pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Trajectory plots — unit-level metrics over DIV, per condition
# (main 4 conditions, DIV 3-23)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Unit-level metric trajectories …")
n_metrics = len(UNIT_METRICS)
ncols = 2
nrows = (n_metrics + 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
axes = axes.ravel()

for ax, (metric, ylabel, log_scale) in zip(axes, UNIT_METRICS):
    if metric not in well_unit.columns:
        ax.set_visible(False)
        continue

    for cond in COND_ORDER:
        sub = well_unit[
            (well_unit["condition"] == cond) &
            (well_unit["div"].isin(MAIN_DIVS))
        ]
        if sub.empty:
            continue
        gp = sub.groupby("div")[metric]
        means  = gp.mean()
        sems   = gp.sem()
        col    = COND_COLORS[cond]
        label  = COND_LABELS[cond]
        ax.plot(means.index, means.values, "-o", color=col, lw=2, ms=6, label=label)
        ax.fill_between(means.index,
                        means.values - sems.values,
                        means.values + sems.values,
                        color=col, alpha=0.18)

    # Shade "early" / "late" bands
    ax.axvspan(EARLY_DIV[0] - 0.5, EARLY_DIV[-1] + 0.5, color="steelblue", alpha=0.07,
               label="_early" if ax != axes[0] else "Early (DIV 6-8)")
    ax.axvspan(LATE_DIV[0] - 0.5, LATE_DIV[-1] + 0.5, color="tomato", alpha=0.07,
               label="_late" if ax != axes[0] else "Late (DIV 24-28)")

    ax.set_xticks(MAIN_DIVS)
    ax.set_xlabel("DIV")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    if log_scale:
        ax.set_yscale("log")

# Shared legend on first axis
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.01), frameon=True)
for ax in axes[n_metrics:]:
    ax.set_visible(False)

plt.suptitle("Unit-level metric trajectories over DIV by FA condition\n"
             "(well-level medians, shaded ±SEM across wells/mice)", fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_unit_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_unit_trajectories.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Network metric trajectories over DIV, per condition
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Network metric trajectories …")
fig, axes = plt.subplots(len(NET_METRICS), 1, figsize=(11, 3.8 * len(NET_METRICS)))
if len(NET_METRICS) == 1:
    axes = [axes]

for ax, (metric, ylabel) in zip(axes, NET_METRICS):
    if metric not in net.columns:
        ax.set_visible(False)
        continue

    for cond in COND_ORDER:
        sub = net[
            (net["condition"] == cond) &
            (net["div"].isin(MAIN_DIVS))
        ]
        if sub.empty:
            continue
        gp = sub.groupby("div")[metric]
        means = gp.mean()
        sems  = gp.sem()
        col   = COND_COLORS[cond]
        ax.plot(means.index, means.values, "-o", color=col, lw=2.5, ms=7,
                label=COND_LABELS[cond])
        ax.fill_between(means.index,
                        means.values - sems.values,
                        means.values + sems.values,
                        color=col, alpha=0.18)

    ax.axvspan(EARLY_DIV[0] - 0.5, EARLY_DIV[-1] + 0.5, color="steelblue", alpha=0.07)
    ax.axvspan(LATE_DIV[0] - 0.5, LATE_DIV[-1] + 0.5, color="tomato", alpha=0.07)
    ax.set_xticks(MAIN_DIVS)
    ax.set_xlabel("DIV")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Network: {ylabel}")
    ax.legend(loc="upper left", fontsize=8, ncol=2)

# Add condition C (10mg) as dashed reference lines at DIV 5/8
#net_c = net[net["condition"] == "10mg_excess"]
#for ax, (metric, _) in zip(axes, NET_METRICS):
 #   if metric not in net.columns or net_c.empty:
  #      continue
   # for div_val in [5, 8]:
    #    sub_c = net_c[net_c["div"] == div_val][metric].dropna()
     #   if len(sub_c):
      #      ax.axhline(sub_c.mean(), color=COND_COLORS["10mg_excess"],
       #                lw=1.5, ls="--", alpha=0.7,
        #               label=f"10mg DIV{div_val} (confounded)" if metric == NET_METRICS[0][0] else "")

plt.suptitle("Network metric trajectories over DIV by FA condition\n"
             "Blue band = early (DIV 6-8), Red band = late (DIV 24-28)\n"
             "Dashed brown lines = 10 mg FA at DIV 5/8 (different mice, confounded)",
             fontsize=11, y=1.005)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig2_network_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_network_trajectories.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: "Before vs After" — violin plots comparing early vs late DIV
# by condition, for each unit-level metric
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Before vs after violin plots (unit-level) …")
well_unit["epoch"] = well_unit["div"].apply(
    lambda d: "Early\n(DIV 6-8)" if d in EARLY_DIV else
              ("Late\n(DIV 24-28)" if d in LATE_DIV else None)
)
bva = well_unit[well_unit["epoch"].notna()].copy()

ncols = 2
nrows = (len(UNIT_METRICS) + 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3))
axes = axes.ravel()

for ax, (metric, ylabel, log_scale) in zip(axes, UNIT_METRICS):
    if metric not in bva.columns:
        ax.set_visible(False)
        continue

    x_positions = []
    x_labels    = []
    pos = 0
    group_centers = {}

    for cond in COND_ORDER:
        group_center = pos + 0.5
        group_centers[cond] = group_center
        for epoch in ["Early\n(DIV 6-8)", "Late\n(DIV 24-28)"]:
            vals = bva[(bva["condition"] == cond) & (bva["epoch"] == epoch)][metric].dropna().values
            if len(vals) == 0:
                pos += 1
                x_positions.append(pos - 0.5)
                x_labels.append("")
                continue
            col   = COND_COLORS[cond]
            alpha = 0.5 if epoch.startswith("Early") else 0.9
            vp    = ax.violinplot(vals, positions=[pos], widths=0.7,
                                  showmedians=True, showextrema=False)
            for pc in vp["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(alpha)
                pc.set_edgecolor("k")
                pc.set_linewidth(0.5)
            vp["cmedians"].set_color("k")
            vp["cmedians"].set_linewidth(1.5)
            x_positions.append(pos)
            x_labels.append("E" if epoch.startswith("E") else "L")
            pos += 1
        pos += 0.6   # gap between conditions

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)

    # Condition label below group — use axis-fraction y so bbox_inches="tight" stays sane
    for cond in COND_ORDER:
        if cond in group_centers:
            short_lbl = COND_LABELS[cond].replace(" (", "\n(")  # max 2 lines
            ax.text(group_centers[cond], -0.08,
                    short_lbl,
                    ha="center", va="top", fontsize=6, color=COND_COLORS[cond],
                    transform=ax.get_xaxis_transform(), clip_on=False)
    if log_scale:
        ax.set_yscale("log")

# Legend for epoch fill
legend_els = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
           alpha=0.5, markersize=10, label="Early (DIV 6-8)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
           alpha=0.9, markersize=10, label="Late (DIV 24-28)"),
]
for ax in axes[len(UNIT_METRICS):]:
    ax.set_visible(False)

plt.suptitle("Before vs After FA exposure — Unit-level metrics\n"
             "E = Early (DIV 6-8), L = Late (DIV 24-28)  |  Violin fill = condition colour",
             fontsize=12)
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.01))
fig.savefig(OUT_DIR / "fig3_before_after_unit_violin.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved fig3_before_after_unit_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Before vs After — network metrics violin
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Before vs after violin plots (network metrics) …")
net["epoch"] = net["div"].apply(
    lambda d: "Early\n(DIV 6-8)" if d in EARLY_DIV else
              ("Late\n(DIV 24-28)" if d in LATE_DIV else None)
)
net_bva = net[net["epoch"].notna()].copy()

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
axes = axes.ravel()

for ax, (metric, ylabel) in zip(axes, NET_METRICS):
    if metric not in net_bva.columns:
        ax.set_visible(False)
        continue
    x_positions = []
    x_labels    = []
    group_centers = {}
    pos = 0
    for cond in COND_ORDER:
        group_centers[cond] = pos + 0.5
        for epoch in ["Early\n(DIV 6-8)", "Late\n(DIV 24-28)"]:
            vals = net_bva[(net_bva["condition"] == cond) & (net_bva["epoch"] == epoch)][metric].dropna().values
            col   = COND_COLORS[cond]
            alpha = 0.45 if epoch.startswith("E") else 0.90
            if len(vals) >= 2:
                vp = ax.violinplot(vals, positions=[pos], widths=0.7,
                                   showmedians=True, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(col); pc.set_alpha(alpha)
                    pc.set_edgecolor("k"); pc.set_linewidth(0.5)
                vp["cmedians"].set_color("k"); vp["cmedians"].set_linewidth(1.8)
            x_positions.append(pos)
            x_labels.append("E" if epoch.startswith("E") else "L")
            pos += 1
        pos += 0.6

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    for cond in COND_ORDER:
        ax.text(group_centers[cond], -0.10,
                COND_LABELS[cond].split("(")[0].strip(),
                ha="center", va="top", fontsize=7, color=COND_COLORS[cond],
                transform=ax.get_xaxis_transform(), clip_on=False)

for ax in axes[len(NET_METRICS):]:
    ax.set_visible(False)

fig.legend(handles=legend_els, loc="lower right", fontsize=10)
plt.suptitle("Before vs After FA exposure — Network-level metrics\n"
             "E = Early (DIV 6-8), L = Late (DIV 24-28)", fontsize=12, y=1.005)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_before_after_network_violin.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_before_after_network_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Delta (late − early) per condition — within-mouse effect size
# Shows how much each condition changes from DIV 6-8 to DIV 24-28
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Delta (late − early) per condition …")
ALL_METRICS = [(m, lbl, ls) for m, lbl, ls in UNIT_METRICS if m in well_unit.columns]
NET_FOR_DELTA = [(m, lbl) for m, lbl in NET_METRICS if m in net.columns]

def compute_delta(df, metric, early_divs, late_divs, cond_order):
    """Compute (late median − early median) per mouse per condition."""
    rows = []
    for mouse in df["mouse_id"].unique():
        for cond in cond_order:
            sub = df[(df["mouse_id"] == mouse) & (df["condition"] == cond)]
            early_vals = sub[sub["div"].isin(early_divs)][metric].dropna()
            late_vals  = sub[sub["div"].isin(late_divs)][metric].dropna()
            if len(early_vals) > 0 and len(late_vals) > 0:
                rows.append({
                    "mouse_id":  mouse,
                    "condition": cond,
                    "early":     early_vals.mean(),
                    "late":      late_vals.mean(),
                    "delta":     late_vals.mean() - early_vals.mean(),
                    "pct_change": 100 * (late_vals.mean() - early_vals.mean()) / (abs(early_vals.mean()) + 1e-9),
                })
    return pd.DataFrame(rows)

# Combine unit-level and network-level deltas
delta_records = []
for metric, label, _ in ALL_METRICS:
    delta_df = compute_delta(well_unit, metric, EARLY_DIV, LATE_DIV, COND_ORDER)
    delta_df["metric"] = label
    delta_records.append(delta_df)
for metric, label in NET_FOR_DELTA:
    delta_df = compute_delta(net, metric, EARLY_DIV, LATE_DIV, COND_ORDER)
    delta_df["metric"] = f"NET: {label}"
    delta_records.append(delta_df)
all_deltas = pd.concat(delta_records, ignore_index=True)

# Bar chart of mean Δ per condition × metric
all_metric_labels = [lbl for _, lbl, _ in ALL_METRICS] + [f"NET: {lbl}" for _, lbl in NET_FOR_DELTA]
n_m = len(all_metric_labels)
fig, axes = plt.subplots(2, (n_m + 1) // 2, figsize=(4 * ((n_m + 1) // 2), 9))
axes = axes.ravel()

for ax, mlabel in zip(axes, all_metric_labels):
    sub = all_deltas[all_deltas["metric"] == mlabel]
    if sub.empty:
        ax.set_visible(False)
        continue
    means = sub.groupby("condition")["delta"].mean().reindex(COND_ORDER)
    sems  = sub.groupby("condition")["delta"].sem().reindex(COND_ORDER)
    cols  = [COND_COLORS[c] for c in COND_ORDER]
    bars  = ax.bar(range(len(COND_ORDER)), means.values, yerr=sems.values,
                   color=cols, alpha=0.85, edgecolor="k", linewidth=0.5,
                   error_kw={"linewidth": 1.5, "capsize": 4})
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([COND_LABELS[c].split("(")[0].strip() for c in COND_ORDER],
                        rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Δ (late − early)")
    ax.set_title(mlabel, fontsize=9)

    # Kruskal-Wallis p-value annotation
    groups = {c: sub[sub["condition"] == c]["delta"].dropna().values for c in COND_ORDER}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) >= 2:
        try:
            _, kw_p = stats.kruskal(*groups.values())
            sig = "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns"
            ax.set_xlabel(f"KW p={kw_p:.3f} {sig}", fontsize=7)
        except Exception:
            pass

for ax in axes[n_m:]:
    ax.set_visible(False)

plt.suptitle("Δ (late DIV 24-28 − early DIV 6-8) per condition\n"
             "Bars = mean ± SEM across mice; KW = Kruskal-Wallis across conditions",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig5_delta_late_minus_early.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig5_delta_late_minus_early.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Heatmap — condition × DIV for each network metric
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: Condition × DIV heatmaps (network metrics) …")
fig, axes = plt.subplots(1, len(NET_METRICS), figsize=(4 * len(NET_METRICS), 5))
if len(NET_METRICS) == 1:
    axes = [axes]

for ax, (metric, ylabel) in zip(axes, NET_METRICS):
    if metric not in net.columns:
        ax.set_visible(False)
        continue

    pivot = (
        net[net["condition"].isin(COND_ORDER) & net["div"].isin(MAIN_DIVS)]
        .groupby(["condition", "div"])[metric]
        .mean()
        .unstack("div")
        .reindex(COND_ORDER)
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(MAIN_DIVS)))
    ax.set_xticklabels(MAIN_DIVS, fontsize=9)
    ax.set_yticks(range(len(COND_ORDER)))
    ax.set_yticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=9)
    ax.set_xlabel("DIV")
    ax.set_title(ylabel, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.7)

plt.suptitle("Network metrics: mean per condition × DIV\n(heatmap, yellow=low, red=high)",
             fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig6_heatmap_condition_div.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig6_heatmap_condition_div.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Heatmap — condition × DIV for unit-level metrics
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 7: Unit-level condition × DIV heatmaps …")
n_um = len(UNIT_METRICS)
fig, axes = plt.subplots(2, (n_um + 1) // 2, figsize=(4.5 * ((n_um + 1) // 2), 8))
axes = axes.ravel()

for ax, (metric, ylabel, _) in zip(axes, UNIT_METRICS):
    if metric not in well_unit.columns:
        ax.set_visible(False)
        continue
    pivot = (
        well_unit[well_unit["condition"].isin(COND_ORDER) & well_unit["div"].isin(MAIN_DIVS)]
        .groupby(["condition", "div"])[metric]
        .median()
        .unstack("div")
        .reindex(COND_ORDER)
    )
    # z-score across DIV for each condition to highlight trajectory shape
    pivot_z = pivot.subtract(pivot.mean(axis=1), axis=0).divide(
        pivot.std(axis=1).replace(0, 1), axis=0)
    im = ax.imshow(pivot_z.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(MAIN_DIVS)))
    ax.set_xticklabels(MAIN_DIVS, fontsize=8)
    ax.set_yticks(range(len(COND_ORDER)))
    ax.set_yticklabels([COND_LABELS[c] for c in COND_ORDER], fontsize=7)
    ax.set_xlabel("DIV")
    ax.set_title(ylabel, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.6, label="z-score")

for ax in axes[n_um:]:
    ax.set_visible(False)

plt.suptitle("Unit-level metrics: z-scored trajectory per condition × DIV\n"
             "(z-score = deviation from each condition's own mean across DIVs)",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_heatmap_unit_zscore.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig7_heatmap_unit_zscore.png")


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TABLE — per metric, KW test at each DIV + early vs late
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding statistical summary table …")
stat_rows = []
for metric, label, _ in ALL_METRICS:
    for div in MAIN_DIVS:
        sub_div = well_unit[(well_unit["div"] == div) & (well_unit["condition"].isin(COND_ORDER))]
        if sub_div.empty:
            continue
        groups = {c: sub_div[sub_div["condition"] == c][metric].dropna().values for c in COND_ORDER}
        kw_p, pairs = kruskal_pairwise(groups)
        stat_rows.append({
            "level": "unit", "metric": label, "div": div,
            "kruskal_p": round(kw_p, 4) if not np.isnan(kw_p) else np.nan,
            "sig": "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns"
        })

for metric, label in NET_FOR_DELTA:
    for div in MAIN_DIVS:
        sub_div = net[(net["div"] == div) & (net["condition"].isin(COND_ORDER))]
        if sub_div.empty:
            continue
        groups = {c: sub_div[sub_div["condition"] == c][metric].dropna().values for c in COND_ORDER}
        kw_p, pairs = kruskal_pairwise(groups)
        stat_rows.append({
            "level": "network", "metric": label, "div": div,
            "kruskal_p": round(kw_p, 4) if not np.isnan(kw_p) else np.nan,
            "sig": "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns"
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(OUT_DIR / "stats_kruskal_by_div.csv", index=False)
print("  Saved stats_kruskal_by_div.csv")

# Print significant findings
sig_df = stat_df[stat_df["sig"].isin(["*", "**", "***"])].sort_values("kruskal_p")
print(f"\n  Significant condition effects across DIVs ({len(sig_df)} / {len(stat_df)} tests):")
if len(sig_df):
    print(sig_df[["level","metric","div","kruskal_p","sig"]].to_string(index=False))
else:
    print("  None at α=0.05 after Bonferroni correction.")

# Delta table
delta_stat_rows = []
for metric_label in all_deltas["metric"].unique():
    sub = all_deltas[all_deltas["metric"] == metric_label]
    groups = {c: sub[sub["condition"] == c]["delta"].dropna().values for c in COND_ORDER}
    kw_p, pairs = kruskal_pairwise(groups)
    delta_stat_rows.append({"metric": metric_label,
                             "kruskal_p_delta": round(kw_p, 4) if not np.isnan(kw_p) else np.nan})
delta_stat_df = pd.DataFrame(delta_stat_rows).sort_values("kruskal_p_delta")
delta_stat_df.to_csv(OUT_DIR / "stats_delta_kruskal.csv", index=False)

print("\n  Delta (late - early) KW test across conditions:")
print(delta_stat_df.to_string(index=False))

print(f"\n=== Done. Results in {OUT_DIR} ===")
