"""
Step 4d: Raw Quality-Metric EDA — Folic Acid MEA data
======================================================
Compares per-unit and per-well metrics across mouse IDs before any HIPPIE
embedding. Serves as a positive-control / sanity check:
  - Are there already measurable electrophysiological differences between mice?
  - Which metrics are most variable across recording dates?
  - What is the data composition (units per mouse, date, well)?

Run after preprocess_FA_kilosort.py has completed.

Usage:
  /home/jesus/hippie_rebuttals/hip-hip-hippie/hippie_venv/bin/python 01_raw_metric_EDA.py

Outputs saved to:
  /mnt/d/datasets_hippie/Roy_shruti_folic_data/EDA_results/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy import stats

# ─── Paths (from config.py) ──────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
#from config import METADATA_CSV, RESULTS_EDA
from config import METADATA_CSV_T3, RESULTS_EDA

META_PATH = METADATA_CSV_T3
#META_PATH = METADATA_CSV
OUT_DIR   = RESULTS_EDA
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load metadata ───────────────────────────────────────────────────────────
print("Loading metadata …")
df = pd.read_csv(META_PATH)

# Convert date to ordered DIV-proxy (days since first recording)
date_order = sorted(df["date"].unique())
df["timepoint"] = df["date"].map({d: i for i, d in enumerate(date_order)})
df["date_label"] = df["date"].map(
    lambda d: f"T{date_order.index(d)+1}\n({d})"
)

print(f"  {len(df):,} units  |  {df['mouse_id'].nunique()} mice  |  "
      f"{df['date'].nunique()} dates  |  {df['well'].nunique()} well IDs")
print(df.groupby(["date", "mouse_id"]).size().unstack(fill_value=0).to_string())

# Numeric quality metrics available
NUMERIC_METRICS = [
    "snr", "amplitude_median", "half_width", "peak_to_valley",
    "peak_trough_ratio", "recovery_slope", "repolarization_slope",
    "presence_ratio", "isi_violations_ratio", "firing_range",
    "sync_spike_2", "sync_spike_4", "sync_spike_8",
    "noise_ratio", "amplitude_cv_median",
]
NUMERIC_METRICS = [m for m in NUMERIC_METRICS if m in df.columns]
print(f"\nMetrics available: {NUMERIC_METRICS}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Unit count per mouse × date
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
pivot = df.groupby(["date", "mouse_id"]).size().unstack(fill_value=0)
pivot.plot(kind="bar", ax=ax, width=0.75)
ax.set_title("QC-passing units per mouse per recording date")
ax.set_xlabel("Recording date")
ax.set_ylabel("Number of units")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Mouse ID", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
fig.savefig(OUT_DIR / "unit_counts_per_mouse_date.png", dpi=150)
plt.close()
print("Saved unit_counts_per_mouse_date.png")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Distribution of key metrics across mice (violin plots)
# ═══════════════════════════════════════════════════════════════════════════
PLOT_METRICS = ["snr", "amplitude_median", "half_width", "peak_to_valley",
                "firing_range", "isi_violations_ratio", "sync_spike_4"]
PLOT_METRICS = [m for m in PLOT_METRICS if m in df.columns]

mice    = sorted(df["mouse_id"].unique())
n_mice  = len(mice)
colors  = cm.tab10(np.linspace(0, 0.9, n_mice))

fig, axes = plt.subplots(len(PLOT_METRICS), 1, figsize=(12, 3 * len(PLOT_METRICS)))
if len(PLOT_METRICS) == 1:
    axes = [axes]

for ax, metric in zip(axes, PLOT_METRICS):
    data_by_mouse = [df.loc[df["mouse_id"] == m, metric].dropna().values for m in mice]
    parts = ax.violinplot(data_by_mouse, positions=range(n_mice),
                          showmedians=True, showextrema=True)
    for i, (body, col) in enumerate(zip(parts["bodies"], colors)):
        body.set_facecolor(col)
        body.set_alpha(0.7)
    ax.set_xticks(range(n_mice))
    ax.set_xticklabels(mice, rotation=30)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} distribution by mouse")

plt.tight_layout()
fig.savefig(OUT_DIR / "metric_distributions_by_mouse.png", dpi=150)
plt.close()
print("Saved metric_distributions_by_mouse.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Metric evolution over time (per mouse) — well-level aggregates
# ═══════════════════════════════════════════════════════════════════════════
TREND_METRICS = ["snr", "amplitude_median", "half_width", "firing_range", "sync_spike_4"]
TREND_METRICS = [m for m in TREND_METRICS if m in df.columns]

# Aggregate to well level (median per well)
well_df = df.groupby(["date", "timepoint", "mouse_id", "well", "run"])[
    TREND_METRICS + ["spike_count"]
].median().reset_index()

fig, axes = plt.subplots(len(TREND_METRICS), 1, figsize=(12, 3.5 * len(TREND_METRICS)))
if len(TREND_METRICS) == 1:
    axes = [axes]

for ax, metric in zip(axes, TREND_METRICS):
    for i, (mouse, col) in enumerate(zip(mice, colors)):
        sub = well_df[well_df["mouse_id"] == mouse].sort_values("timepoint")
        # Plot individual wells as dots, mean as line
        ax.scatter(sub["timepoint"], sub[metric], c=[col], alpha=0.4, s=20, zorder=2)
        mean_per_tp = sub.groupby("timepoint")[metric].mean()
        ax.plot(mean_per_tp.index, mean_per_tp.values, color=col,
                lw=2, marker="o", ms=5, label=mouse, zorder=3)
    ax.set_xticks(range(len(date_order)))
    ax.set_xticklabels([f"T{i+1}\n{d}" for i, d in enumerate(date_order)], rotation=30)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over recording dates")
    ax.legend(title="Mouse", loc="upper left", ncol=3, fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "metric_trends_over_time.png", dpi=150)
plt.close()
print("Saved metric_trends_over_time.png")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Kruskal-Wallis test: do mice differ significantly on each metric?
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Kruskal-Wallis test: metric differences across mice ──")
kw_results = []
for metric in NUMERIC_METRICS:
    groups = [df.loc[df["mouse_id"] == m, metric].dropna().values for m in mice]
    groups = [g for g in groups if len(g) > 10]
    if len(groups) < 2:
        continue
    stat, p = stats.kruskal(*groups)
    kw_results.append({"metric": metric, "H_stat": round(stat, 2), "p_value": p})

kw_df = pd.DataFrame(kw_results).sort_values("p_value")
kw_df["significant"] = kw_df["p_value"] < 0.05
print(kw_df.to_string(index=False))
kw_df.to_csv(OUT_DIR / "kruskal_wallis_by_mouse.csv", index=False)
print("Saved kruskal_wallis_by_mouse.csv")

# ═══════════════════════════════════════════════════════════════════════════
# 5. Correlation heatmap between metrics (all units pooled)
# ═══════════════════════════════════════════════════════════════════════════
corr_data = df[NUMERIC_METRICS].dropna()
if len(corr_data) > 100:
    corr = corr_data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(NUMERIC_METRICS)))
    ax.set_xticklabels(NUMERIC_METRICS, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(NUMERIC_METRICS)))
    ax.set_yticklabels(NUMERIC_METRICS, fontsize=8)
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Spearman correlation between unit-level quality metrics")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "metric_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved metric_correlation_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# 6. Well-level summary statistics (to be used for condition analysis later)
# ═══════════════════════════════════════════════════════════════════════════
all_metrics = [m for m in NUMERIC_METRICS if m in df.columns]
well_summary = df.groupby(
    ["date", "timepoint", "mouse_id", "run", "well", "condition"]
)[all_metrics + ["spike_count"]].agg(["mean", "median", "std"]).reset_index()

# Flatten multi-level columns
well_summary.columns = [
    "_".join(c).strip("_") if isinstance(c, tuple) else c
    for c in well_summary.columns
]
well_summary.to_csv(OUT_DIR / "well_level_summary.csv", index=False)
print("Saved well_level_summary.csv")

print("\n=== EDA complete. All figures saved to", OUT_DIR, "===")
