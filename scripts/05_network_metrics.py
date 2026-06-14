"""
Step 5: Network-level activity metrics per well
===============================================
Computes burst statistics and population-level synchrony directly from
spike_times.npy + spike_clusters.npy without HIPPIE embeddings.
These provide independent biological validation of condition differences.

Metrics computed per well:
  - mean_firing_rate_hz      : mean spikes/s across all QC units
  - population_rate_peak_hz  : peak of population firing rate (50ms bins)
  - burst_rate_per_min       : bursts per minute (population burst detection)
  - mean_ibi_s               : mean inter-burst interval (s)
  - cv_ibi                   : coefficient of variation of IBI
  - burst_fraction           : fraction of total spikes in bursts
  - network_synchrony        : Fano factor of population rate (burst proxy)
  - n_units                  : number of QC-passing units

Results saved to:
  /mnt/d/datasets_hippie/Roy_shruti_folic_data/network_metrics/
    network_metrics_all_wells.csv
    network_metrics_plots.png

Usage:
  /home/jesus/hippie_rebuttals/hip-hip-hippie/hippie_venv/bin/python 04_network_metrics.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.signal import find_peaks
from scipy import stats

# ─── Paths (from config.py) ──────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW_T3 as DATA_RAW, METADATA_CSV_T3 as METADATA_CSV, RESULTS_NETWORK

DATA_ROOT = DATA_RAW
META_PATH = METADATA_CSV
OUT_DIR   = RESULTS_NETWORK
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE   = 10000.0   # Hz
BURST_BIN_MS  = 50.0      # ms population-rate bin for burst detection
BURST_THRESH  = 3.0       # z-score threshold above mean for burst detection
MIN_UNITS     = 5         # skip wells with fewer good units

# ─── Load QC unit list from metadata ─────────────────────────────────────────
print("Loading QC unit metadata …")
meta = pd.read_csv(META_PATH)

# Build lookup: (date, mouse_id, run, well) → set of good cluster IDs
# Run dirs are zero-padded 6-digit strings (e.g., "000005"); date dirs like "250228".
def make_key(row):
    return (str(row["date"]), row["mouse_id"], f"{int(row['run']):06d}", row["well"])

meta["_key"] = meta.apply(make_key, axis=1)
good_clusters = meta.groupby("_key")["cluster_id"].apply(set).to_dict()
print(f"  {len(good_clusters)} wells with QC units")

# ─── Population-rate burst detector ─────────────────────────────────────────
def compute_network_metrics(spike_times_all, good_ids, recording_duration_s):
    """Compute well-level network metrics using only QC-passing units.

    spike_times_all : 1-D int array (samples at SAMPLE_RATE)
    good_ids        : set of cluster IDs that pass QC
    recording_duration_s : float

    Returns dict of metric → value.
    """
    n_units = len(good_ids)
    if n_units < MIN_UNITS:
        return None

    # ── Firing rates per unit ──────────────────────────────────────────────
    fr_list = []
    spikes_in_good = []
    for cid in good_ids:
        mask  = spike_clusters == cid
        n_sp  = int(mask.sum())
        if recording_duration_s > 0:
            fr_list.append(n_sp / recording_duration_s)
        spikes_in_good.append(spike_times_all[mask])

    mean_fr = float(np.mean(fr_list)) if fr_list else 0.0

    # ── Population rate (all QC-unit spikes binned at BURST_BIN_MS) ────────
    all_good_spikes = np.concatenate(spikes_in_good) if spikes_in_good else np.array([])
    if len(all_good_spikes) < 10:
        return {
            "mean_firing_rate_hz": mean_fr, "population_rate_peak_hz": 0,
            "burst_rate_per_min": 0, "mean_ibi_s": np.nan, "cv_ibi": np.nan,
            "burst_fraction": 0, "network_synchrony": 0, "n_units": n_units,
        }

    bin_size_s   = BURST_BIN_MS / 1000.0
    n_bins       = int(np.ceil(recording_duration_s / bin_size_s))
    bin_edges    = np.linspace(0, recording_duration_s, n_bins + 1)
    pop_spikes_s = all_good_spikes.astype(float) / SAMPLE_RATE
    pop_rate, _  = np.histogram(pop_spikes_s, bins=bin_edges)   # spikes per bin

    # Convert to population firing rate (Hz) = spikes/(bin_s * n_units)
    pop_rate_hz = pop_rate / (bin_size_s * n_units)

    peak_pop_rate = float(pop_rate_hz.max())
    synchrony     = float(np.var(pop_rate) / (np.mean(pop_rate) + 1e-9))  # Fano factor

    # ── Burst detection ─────────────────────────────────────────────────────
    mean_r = pop_rate_hz.mean()
    std_r  = pop_rate_hz.std()
    thresh = mean_r + BURST_THRESH * std_r

    burst_starts, _ = find_peaks(pop_rate_hz,
                                 height=thresh,  
                                 distance=int(100 / BURST_BIN_MS))  # min 100ms apart

    n_bursts    = len(burst_starts)
    duration_min = recording_duration_s / 60.0
    burst_rate  = n_bursts / duration_min if duration_min > 0 else 0.0

    # Inter-burst intervals
    if n_bursts > 1:
        burst_times_s = bin_edges[burst_starts]
        ibis          = np.diff(burst_times_s)
        mean_ibi      = float(ibis.mean())
        cv_ibi        = float(ibis.std() / (ibis.mean() + 1e-9))
    else:
        mean_ibi = np.nan
        cv_ibi   = np.nan

    # Burst fraction: spikes occurring within ±100ms of a burst peak
    if n_bursts > 0:
        burst_peak_s = bin_edges[burst_starts] + bin_size_s / 2
        near_burst   = np.zeros(len(pop_spikes_s), dtype=bool)
        for bp in burst_peak_s:
            near_burst |= np.abs(pop_spikes_s - bp) < 0.1  # 100ms window
        burst_frac = float(near_burst.mean())
    else:
        burst_frac = 0.0

    return {
        "mean_firing_rate_hz"  : round(mean_fr, 4),
        "population_rate_peak_hz": round(peak_pop_rate, 4),
        "burst_rate_per_min"   : round(burst_rate, 4),
        "mean_ibi_s"           : round(mean_ibi, 4) if not np.isnan(mean_ibi) else np.nan,
        "cv_ibi"               : round(cv_ibi, 4)   if not np.isnan(cv_ibi)   else np.nan,
        "burst_fraction"       : round(burst_frac, 4),
        "network_synchrony"    : round(synchrony, 4),
        "n_units"              : n_units,
    }


# ─── Main loop ───────────────────────────────────────────────────────────────
results = []
dates   = sorted(DATA_ROOT.iterdir())

for date_dir in dates:
    if not date_dir.is_dir():
        continue
    date_str = date_dir.name

    for mouse_dir in sorted(date_dir.iterdir()):
        if not mouse_dir.is_dir():
            continue
        mouse_id = mouse_dir.name
        net_dir  = mouse_dir / "Network"
        if not net_dir.is_dir():
            continue

        for run_dir in sorted(net_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_id = run_dir.name

            for well_dir in sorted(run_dir.iterdir()):
                if not well_dir.is_dir():
                    continue
                well_id = well_dir.name
                phy_dir = well_dir / "phy_output"
                if not phy_dir.is_dir():
                    continue

                key = (date_str, mouse_id, run_id, well_id)
                if key not in good_clusters:
                    continue  # well had no QC-passing units
                good_ids = good_clusters[key]

                try:
                    spike_times    = np.load(phy_dir / "spike_times.npy").ravel()
                    spike_clusters = np.load(phy_dir / "spike_clusters.npy").ravel()
                except Exception as e:
                    print(f"  [SKIP] {key}: {e}")
                    continue

                rec_dur_s = float(spike_times.max()) / SAMPLE_RATE

                metrics = compute_network_metrics(spike_times, good_ids, rec_dur_s)
                if metrics is None:
                    continue

                # Retrieve condition from metadata
                sub = meta[
                    (meta["date"] == date_str) &
                    (meta["mouse_id"] == mouse_id) &
                    (meta["run"] == run_id) &
                    (meta["well"] == well_id)
                ]
                condition = sub["condition"].iloc[0] if len(sub) > 0 else "unknown"

                row = {
                    "date": date_str, "mouse_id": mouse_id,
                    "run": run_id, "well": well_id,
                    "condition": condition, **metrics,
                }
                results.append(row)
                print(f"  {date_str}/{mouse_id}/{run_id}/{well_id}: "
                      f"FR={metrics['mean_firing_rate_hz']:.2f} Hz  "
                      f"bursts={metrics['burst_rate_per_min']:.1f}/min  "
                      f"n={metrics['n_units']} units")

net_df = pd.DataFrame(results)
net_df.to_csv(OUT_DIR / "network_metrics_all_wells.csv", index=False)
print(f"\nSaved network_metrics_all_wells.csv  ({len(net_df)} wells)")

if net_df.empty:
    print("No wells processed — check key matching between metadata and data directory.")
    raise SystemExit(1)

# ─── Plots ───────────────────────────────────────────────────────────────────
date_order = sorted(net_df["date"].unique())
net_df["timepoint"] = net_df["date"].map({d: i for i, d in enumerate(date_order)})
mice   = sorted(net_df["mouse_id"].unique())
colors = cm.tab10(np.linspace(0, 0.9, len(mice)))

PLOT_METRICS = [
    "mean_firing_rate_hz", "burst_rate_per_min",
    "mean_ibi_s", "network_synchrony", "burst_fraction",
]
PLOT_METRICS = [m for m in PLOT_METRICS if m in net_df.columns]

fig, axes = plt.subplots(len(PLOT_METRICS), 1, figsize=(12, 3.5 * len(PLOT_METRICS)))
if len(PLOT_METRICS) == 1:
    axes = [axes]

for ax, metric in zip(axes, PLOT_METRICS):
    for i, (mouse, col) in enumerate(zip(mice, colors)):
        sub = net_df[net_df["mouse_id"] == mouse].sort_values("timepoint")
        ax.scatter(sub["timepoint"], sub[metric], c=[col], alpha=0.5, s=25, zorder=2)
        mean_per_tp = sub.groupby("timepoint")[metric].mean()
        ax.plot(mean_per_tp.index, mean_per_tp.values, color=col,
                lw=2, marker="o", ms=5, label=mouse, zorder=3)
    ax.set_xticks(range(len(date_order)))
    ax.set_xticklabels([f"T{i+1}\n{d}" for i, d in enumerate(date_order)], rotation=30)
    ax.set_ylabel(metric)
    ax.set_title(f"Network: {metric} over time")
    ax.legend(title="Mouse", loc="upper left", ncol=3, fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "network_metrics_over_time.png", dpi=150)
plt.close()
print("Saved network_metrics_over_time.png")

# Kruskal-Wallis by mouse
print("\n── Kruskal-Wallis test: network metrics across mice ──")
for metric in PLOT_METRICS:
    groups = [net_df.loc[net_df["mouse_id"] == m, metric].dropna().values for m in mice]
    groups = [g for g in groups if len(g) > 2]
    if len(groups) < 2:
        continue
    stat, p = stats.kruskal(*groups)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {metric:30s}  H={stat:.2f}  p={p:.4f}  {sig}")

print(f"\n=== Network metrics done. Results in {OUT_DIR} ===")
