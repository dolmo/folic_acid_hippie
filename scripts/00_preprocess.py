"""
Preprocess Folic Acid MEA / Kilosort outputs → HIPPIE-format CSVs.

For each well in the raw data directory:
  - Extracts primary-channel mean waveform from templates.npy
  - Computes ISI histogram (100 bins, 0-100ms) from spike_times.npy
  - Computes autocorrelogram (201 bins, ±100ms at 1ms) from spike_times.npy
  - Joins all cluster_*.tsv quality metrics
  - Applies quality filters (SNR, presence ratio, ISI violations)
  - Saves per-well CSVs and one combined dataset for HIPPIE training

Usage:
  python scripts/00_preprocess.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import warnings
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW, HIPPIE_INPUT, DATA_PROCESSED

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_ROOT  = DATA_RAW
OUT_HIPPIE = HIPPIE_INPUT
OUT_WELLS  = DATA_PROCESSED

# ─── Quality-filter thresholds ──────────────────────────────────────────────
MIN_PRESENCE_RATIO   = 0.5
MAX_ISI_VIOL_RATIO   = 0.10
MIN_SNR              = 3.0
MIN_SPIKE_COUNT      = 50   # need enough spikes for ISI/ACG

# ─── ACG / ISI parameters ───────────────────────────────────────────────────
SAMPLE_RATE   = 10000.0   # Hz
ACG_WINDOW_MS = 100.0     # ± ms
ACG_BIN_MS    = 1.0       # ms per bin → 201 bins total
ISI_MAX_MS    = 100.0     # ms
ISI_BINS      = 100       # bins

# Well-level condition + sex mapping from FolicAcid_T4_02252025_SA.tsv
# Legend: A=2mg_control, B=0mg_deficient, C=10mg_excess, D=20mg_super_excess, E=folinic_acid_excess
# Key: (mouse_id, well_index)  where well_index = int("well000"[4:]) = 0..5
# Value: (condition_label, sex)
WELL_CONDITION = {
    # M07708: A_Male, B_Male, D_Male, A_Female, B_Female, D_Female
    ("M07708", 0): ("2mg_control",         "Male"),
    ("M07708", 1): ("0mg_deficient",       "Male"),
    ("M07708", 2): ("20mg_super_excess",   "Male"),
    ("M07708", 3): ("2mg_control",         "Female"),
    ("M07708", 4): ("0mg_deficient",       "Female"),
    ("M07708", 5): ("20mg_super_excess",   "Female"),
    # M07137: A_Male, B_Male, D_Male, E_Male, A_Female, B_Female
    ("M07137", 0): ("2mg_control",         "Male"),
    ("M07137", 1): ("0mg_deficient",       "Male"),
    ("M07137", 2): ("20mg_super_excess",   "Male"),
    ("M07137", 3): ("folinic_acid_excess", "Male"),
    ("M07137", 4): ("2mg_control",         "Female"),
    ("M07137", 5): ("0mg_deficient",       "Female"),
    # M08092: A_Male, B_Male, D_Male, E_Male, D_Female, E_Female
    ("M08092", 0): ("2mg_control",         "Male"),
    ("M08092", 1): ("0mg_deficient",       "Male"),
    ("M08092", 2): ("20mg_super_excess",   "Male"),
    ("M08092", 3): ("folinic_acid_excess", "Male"),
    ("M08092", 4): ("20mg_super_excess",   "Female"),
    ("M08092", 5): ("folinic_acid_excess", "Female"),
    # M07865: E_Male, A_Female, B_Female, D_Female, E_Female, E_Female
    ("M07865", 0): ("folinic_acid_excess", "Male"),
    ("M07865", 1): ("2mg_control",         "Female"),
    ("M07865", 2): ("0mg_deficient",       "Female"),
    ("M07865", 3): ("20mg_super_excess",   "Female"),
    ("M07865", 4): ("folinic_acid_excess", "Female"),
    ("M07865", 5): ("folinic_acid_excess", "Female"),
    # M08068: C_Male, E_Female, C_Male, C_Male, E_Female, E_Male
    ("M08068", 0): ("10mg_excess",         "Male"),
    ("M08068", 1): ("folinic_acid_excess", "Female"),
    ("M08068", 2): ("10mg_excess",         "Male"),
    ("M08068", 3): ("10mg_excess",         "Male"),
    ("M08068", 4): ("folinic_acid_excess", "Female"),
    ("M08068", 5): ("folinic_acid_excess", "Male"),
    # M08032: C_Male, E_Male, C_Male, C_Male, E_Male, E_Female
    ("M08032", 0): ("10mg_excess",         "Male"),
    ("M08032", 1): ("folinic_acid_excess", "Male"),
    ("M08032", 2): ("10mg_excess",         "Male"),
    ("M08032", 3): ("10mg_excess",         "Male"),
    ("M08032", 4): ("folinic_acid_excess", "Male"),
    ("M08032", 5): ("folinic_acid_excess", "Female"),
}


# ─── Helper functions ────────────────────────────────────────────────────────

def get_primary_channel(templates):
    """Return the index (0-3) of the highest-amplitude channel per cluster.
    templates: (n_clusters, n_timepoints, n_channels)
    """
    ptp = templates.max(axis=1) - templates.min(axis=1)   # (n_clusters, n_channels)
    return np.argmax(ptp, axis=1)                          # (n_clusters,)


def extract_waveforms(templates):
    """Extract primary-channel waveform for each cluster.
    Returns ndarray (n_clusters, n_timepoints).
    """
    primary = get_primary_channel(templates)
    return np.array([templates[i, :, primary[i]] for i in range(len(templates))])


def compute_isi_distributions(spike_times, spike_clusters, cluster_ids):
    """Compute ISI histogram per cluster.
    spike_times : 1-D int array (samples at SAMPLE_RATE)
    spike_clusters : 1-D int array
    Returns ndarray (n_clusters, ISI_BINS), spike_counts (n_clusters,)
    """
    isi_dists    = np.zeros((len(cluster_ids), ISI_BINS), dtype=np.float32)
    spike_counts = np.zeros(len(cluster_ids),             dtype=np.int32)
    bin_edges    = np.linspace(0, ISI_MAX_MS, ISI_BINS + 1)

    for idx, cid in enumerate(cluster_ids):
        mask  = spike_clusters == cid
        times = np.sort(spike_times[mask]).astype(np.float64) / SAMPLE_RATE * 1000.0  # ms
        spike_counts[idx] = len(times)
        if len(times) > 1:
            isi = np.diff(times)
            hist, _ = np.histogram(isi[isi < ISI_MAX_MS], bins=bin_edges)
            isi_dists[idx] = hist.astype(np.float32)

    return isi_dists, spike_counts


def compute_acg_single(spike_times_ms):
    """Compute autocorrelogram for one spike train.
    Returns 1-D array of length 201 (bin centres −100 to +100 ms at 1ms).
    """
    n_bins  = int(2 * ACG_WINDOW_MS / ACG_BIN_MS) + 1   # 201
    acg     = np.zeros(n_bins, dtype=np.float32)
    n       = len(spike_times_ms)
    if n < 2:
        return acg

    st = np.sort(spike_times_ms)
    bin_edges = np.linspace(
        -ACG_WINDOW_MS - ACG_BIN_MS / 2,
         ACG_WINDOW_MS + ACG_BIN_MS / 2,
        n_bins + 1,
    )

    for i in range(n):
        lo = np.searchsorted(st, st[i] - ACG_WINDOW_MS)
        hi = np.searchsorted(st, st[i] + ACG_WINDOW_MS + 1e-6)
        lags = st[lo:hi] - st[i]
        lags = lags[lags != 0.0]
        if len(lags):
            hist, _ = np.histogram(lags, bins=bin_edges)
            acg += hist.astype(np.float32)

    acg[n_bins // 2] = 0.0   # zero-lag (self-coincidence) removed
    return acg


def compute_acgs(spike_times, spike_clusters, cluster_ids):
    """Compute ACG for every cluster.
    Returns ndarray (n_clusters, 201).
    """
    n_bins = int(2 * ACG_WINDOW_MS / ACG_BIN_MS) + 1
    acgs   = np.zeros((len(cluster_ids), n_bins), dtype=np.float32)

    for idx, cid in enumerate(cluster_ids):
        mask         = spike_clusters == cid
        times_ms     = (spike_times[mask].astype(np.float64) / SAMPLE_RATE * 1000.0)
        acgs[idx]    = compute_acg_single(times_ms)

    return acgs


def load_quality_metrics(phy_dir):
    """Join all cluster_*.tsv files into a single DataFrame indexed by cluster_id."""
    metrics = {}
    for tsv_path in sorted(Path(phy_dir).glob("cluster_*.tsv")):
        metric_name = tsv_path.stem.replace("cluster_", "")
        try:
            df  = pd.read_csv(tsv_path, sep="\t")
            col = [c for c in df.columns if c != "cluster_id"]
            if col:
                metrics[metric_name] = df.set_index("cluster_id")[col[0]]
        except Exception:
            pass
    if not metrics:
        return pd.DataFrame()
    return pd.DataFrame(metrics)


def apply_quality_filter(metrics_df, spike_counts):
    """Return boolean mask (cluster_ids as index) for quality-passing clusters."""
    mask = pd.Series(True, index=metrics_df.index)

    if "presence_ratio" in metrics_df.columns:
        mask &= metrics_df["presence_ratio"].fillna(0) >= MIN_PRESENCE_RATIO

    if "isi_violations_ratio" in metrics_df.columns:
        mask &= metrics_df["isi_violations_ratio"].fillna(1) <= MAX_ISI_VIOL_RATIO

    if "snr" in metrics_df.columns:
        mask &= metrics_df["snr"].fillna(0) >= MIN_SNR

    # enforce minimum spike count
    spike_series = pd.Series(spike_counts, index=metrics_df.index)
    mask &= spike_series >= MIN_SPIKE_COUNT

    return mask


# ─── Per-well processing ─────────────────────────────────────────────────────

def process_well(phy_dir, well_id, run_id, mouse_id, date_str):
    """Process a single well. Returns dict of arrays or None on failure."""
    phy_dir = Path(phy_dir)

    # Load arrays
    try:
        templates      = np.load(phy_dir / "templates.npy")         # (n, 30, 4)
        spike_times    = np.load(phy_dir / "spike_times.npy").ravel()
        spike_clusters = np.load(phy_dir / "spike_clusters.npy").ravel()
    except Exception as e:
        print(f"  [SKIP] {phy_dir}: {e}", flush=True)
        return None

    n_clusters   = templates.shape[0]
    cluster_ids  = np.arange(n_clusters)

    # Load and join quality metrics
    metrics_df = load_quality_metrics(phy_dir)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(index=pd.Index(cluster_ids, name="cluster_id"))
    else:
        # Ensure all cluster IDs present
        metrics_df = metrics_df.reindex(cluster_ids)

    # Compute spike counts per cluster
    spike_count_arr = np.array([np.sum(spike_clusters == c) for c in cluster_ids])

    # Quality filter
    q_mask       = apply_quality_filter(metrics_df, spike_count_arr)
    good_ids     = cluster_ids[q_mask.values]
    n_good       = len(good_ids)

    if n_good == 0:
        print(f"  [WARN] {date_str}/{mouse_id}/{run_id}/{well_id}: 0 clusters pass QC", flush=True)
        return None

    # Extract features for good clusters only
    waveforms = extract_waveforms(templates[good_ids])                           # (n_good, 30)
    isi_dists, spike_counts = compute_isi_distributions(
        spike_times, spike_clusters, good_ids
    )
    acgs = compute_acgs(spike_times, spike_clusters, good_ids)                  # (n_good, 201)

    # Build metadata row per cluster
    well_idx          = int(well_id.replace("well", ""))
    condition, sex    = WELL_CONDITION.get((mouse_id, well_idx), ("unknown", "unknown"))
    meta_rows = []
    for local_idx, cid in enumerate(good_ids):
        row = {
            "global_idx"     : None,         # filled when concatenating
            "cluster_id"     : int(cid),
            "well"           : well_id,
            "run"            : run_id,
            "mouse_id"       : mouse_id,
            "date"           : date_str,
            "condition"      : condition,
            "sex"            : sex,
            "spike_count"    : int(spike_count_arr[cid]),
        }
        # Add all quality metrics
        for col in metrics_df.columns:
            row[col] = metrics_df.loc[cid, col] if cid in metrics_df.index else np.nan
        meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)

    return {
        "waveforms" : waveforms,
        "isi_dists" : isi_dists,
        "acgs"      : acgs,
        "metadata"  : meta_df,
        "n_total"   : n_clusters,
        "n_good"    : n_good,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    OUT_HIPPIE.mkdir(parents=True, exist_ok=True)
    OUT_WELLS.mkdir(parents=True, exist_ok=True)

    all_waveforms = []
    all_isi       = []
    all_acg       = []
    all_meta      = []

    global_idx = 0
    well_count = 0
    skip_count = 0

    # Walk the directory tree: date → mouse → Network → run → well
    dates = sorted(DATA_ROOT.iterdir())
    for date_dir in dates:
        if not date_dir.is_dir():
            continue
        date_str = date_dir.name

        for mouse_dir in sorted(date_dir.iterdir()):
            if not mouse_dir.is_dir():
                continue
            mouse_id = mouse_dir.name

            net_dir = mouse_dir / "Network"
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

                    tag = f"{date_str}/{mouse_id}/{run_id}/{well_id}"
                    print(f"Processing {tag} ...", end=" ", flush=True)

                    result = process_well(phy_dir, well_id, run_id, mouse_id, date_str)
                    if result is None:
                        skip_count += 1
                        continue

                    n_good = result["n_good"]
                    print(f"{n_good}/{result['n_total']} units passed QC", flush=True)

                    # Assign global indices
                    result["metadata"]["global_idx"] = range(global_idx, global_idx + n_good)
                    global_idx += n_good
                    well_count += 1

                    # Accumulate for combined dataset
                    all_waveforms.append(result["waveforms"])
                    all_isi.append(result["isi_dists"])
                    all_acg.append(result["acgs"])
                    all_meta.append(result["metadata"])

                    # Save per-well outputs
                    well_tag   = f"{date_str}_{mouse_id}_{run_id}_{well_id}"
                    well_out   = OUT_WELLS / well_tag
                    well_out.mkdir(parents=True, exist_ok=True)

                    pd.DataFrame(result["waveforms"]).to_csv(
                        well_out / "waveforms.csv", index=False)
                    pd.DataFrame(result["isi_dists"]).to_csv(
                        well_out / "isi_dist.csv",  index=False)
                    pd.DataFrame(result["acgs"]).to_csv(
                        well_out / "acg.csv",        index=False)
                    result["metadata"].to_csv(
                        well_out / "metadata.csv",   index=False)

    if not all_waveforms:
        print("No data processed — check paths.", file=sys.stderr)
        return

    # ── Concatenate and save combined HIPPIE dataset ──────────────────────────
    print(f"\nCombining {well_count} wells, {global_idx} units total …", flush=True)

    wf_all   = np.vstack(all_waveforms)
    isi_all  = np.vstack(all_isi)
    acg_all  = np.vstack(all_acg)
    meta_all = pd.concat(all_meta, ignore_index=True)

    # ACG bin-centre column headers (−100 to +100 ms)
    acg_cols = [f"{v:.2f}" for v in np.arange(
        -ACG_WINDOW_MS,
         ACG_WINDOW_MS + ACG_BIN_MS,
         ACG_BIN_MS,
    )]

    pd.DataFrame(wf_all).to_csv(
        OUT_HIPPIE / "waveforms.csv",  index=False)
    pd.DataFrame(isi_all).to_csv(
        OUT_HIPPIE / "isi_dist.csv",   index=False)
    pd.DataFrame(acg_all, columns=acg_cols[:acg_all.shape[1]]).to_csv(
        OUT_HIPPIE / "acg.csv",        index=False)

    # Labels: placeholder (all "FA_unit") — update once condition map is confirmed
    labels_df = pd.DataFrame({"label": meta_all["condition"].values})
    labels_df.to_csv(OUT_HIPPIE / "labels.csv", index=False)

    meta_all.to_csv(OUT_HIPPIE / "metadata.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Wells processed : {well_count}  (skipped: {skip_count})")
    print(f"Total units     : {global_idx}")
    print(f"  waveforms.csv : {wf_all.shape}")
    print(f"  isi_dist.csv  : {isi_all.shape}")
    print(f"  acg.csv       : {acg_all.shape}")
    print(f"\nUnits per date:")
    print(meta_all.groupby("date").size().to_string())
    print(f"\nUnits per mouse:")
    print(meta_all.groupby("mouse_id").size().to_string())
    print(f"\nOutputs written to:")
    print(f"  {OUT_HIPPIE}")
    print(f"  {OUT_WELLS}")


if __name__ == "__main__":
    main()
