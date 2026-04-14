"""
Preprocess FolicAcid T3 experiment → HIPPIE-format CSVs.

Same pipeline as 00_preprocess.py but for the T3 experiment (Oct-Nov 2024).
T3 has different mice, conditions (0mg, 2mg, 10mg, 20mg — no folinic acid),
and no sex labels in the metadata.

Usage:
  python scripts/00b_preprocess_t3.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import warnings
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_RAW_T3, HIPPIE_INPUT_T3, DATA_PROCESSED_T3

# ─── Paths ─────────────────────────────────────────────────────────────────
DATA_ROOT  = DATA_RAW_T3
OUT_HIPPIE = HIPPIE_INPUT_T3
OUT_WELLS  = DATA_PROCESSED_T3

# ─── Quality-filter thresholds (same as T4) ────────────────────────────────
MIN_PRESENCE_RATIO   = 0.5
MAX_ISI_VIOL_RATIO   = 0.10
MIN_SNR              = 3.0
MIN_SPIKE_COUNT      = 50

# ─── ACG / ISI parameters ───────────────────────────────────────────────────
SAMPLE_RATE   = 10000.0   # Hz
ACG_WINDOW_MS = 100.0     # ± ms
ACG_BIN_MS    = 1.0       # ms per bin → 201 bins total
ISI_MAX_MS    = 100.0     # ms
ISI_BINS      = 100       # bins

# ─── T3 condition mapping ──────────────────────────────────────────────────
# From FolicAcid_T3.xlsx — wells 1-6 in TSV → well000-well005 in directory
# T3 has NO sex info, so sex is set to "unknown".
# Conditions: 0mg_deficient, 2mg_control, 10mg_excess, 20mg_super_excess
WELL_CONDITION = {
    # M07037: 0mg,0mg,0mg,2mg,2mg,2mg
    ("M07037", 0): ("0mg_deficient",     "unknown"),
    ("M07037", 1): ("0mg_deficient",     "unknown"),
    ("M07037", 2): ("0mg_deficient",     "unknown"),
    ("M07037", 3): ("2mg_control",       "unknown"),
    ("M07037", 4): ("2mg_control",       "unknown"),
    ("M07037", 5): ("2mg_control",       "unknown"),
    # M07896: 0mg,0mg,0mg,2mg,2mg,2mg
    ("M07896", 0): ("0mg_deficient",     "unknown"),
    ("M07896", 1): ("0mg_deficient",     "unknown"),
    ("M07896", 2): ("0mg_deficient",     "unknown"),
    ("M07896", 3): ("2mg_control",       "unknown"),
    ("M07896", 4): ("2mg_control",       "unknown"),
    ("M07896", 5): ("2mg_control",       "unknown"),
    # M07420: 2mg,2mg,2mg,2mg,2mg,2mg (all control)
    ("M07420", 0): ("2mg_control",       "unknown"),
    ("M07420", 1): ("2mg_control",       "unknown"),
    ("M07420", 2): ("2mg_control",       "unknown"),
    ("M07420", 3): ("2mg_control",       "unknown"),
    ("M07420", 4): ("2mg_control",       "unknown"),
    ("M07420", 5): ("2mg_control",       "unknown"),
    # M08032: 10mg,10mg,10mg,20mg,20mg,20mg
    ("M08032", 0): ("10mg_excess",       "unknown"),
    ("M08032", 1): ("10mg_excess",       "unknown"),
    ("M08032", 2): ("10mg_excess",       "unknown"),
    ("M08032", 3): ("20mg_super_excess", "unknown"),
    ("M08032", 4): ("20mg_super_excess", "unknown"),
    ("M08032", 5): ("20mg_super_excess", "unknown"),
    # M07297: 10mg,20mg,20mg,20mg,20mg,20mg
    ("M07297", 0): ("10mg_excess",       "unknown"),
    ("M07297", 1): ("20mg_super_excess", "unknown"),
    ("M07297", 2): ("20mg_super_excess", "unknown"),
    ("M07297", 3): ("20mg_super_excess", "unknown"),
    ("M07297", 4): ("20mg_super_excess", "unknown"),
    ("M07297", 5): ("20mg_super_excess", "unknown"),
}

# ─── T3 DIV mapping ────────────────────────────────────────────────────────
# From FolicAcid_T3.xlsx — DIV depends on (date, mouse) because M08032/M07297
# were plated later (same pattern as T4's M08032/M08068).
_MAIN_COHORT = {"M07037", "M07896", "M07420"}
_LATE_COHORT = {"M08032", "M07297"}

DIV_TABLE = {
    # Main cohort DIVs
    ("241028", "main"):  6,
    ("241030", "main"):  8,
    ("241104", "main"): 13,
    ("241107", "main"): 16,
    ("241112", "main"): 21,
    ("241115", "main"): 24,
    ("241119", "main"): 28,
    ("241122", "main"): 31,
    ("241127", "main"): 36,   # extrapolated: 241122=DIV31 + 5 days
    # Late cohort DIVs
    ("241104", "late"):  6,
    ("241106", "late"):  8,
    ("241113", "late"): 15,
    ("241115", "late"): 17,
    ("241119", "late"): 21,
    ("241122", "late"): 24,
    ("241126", "late"): 28,
}


def get_div(date_str, mouse_id):
    """Look up DIV for a (date, mouse) pair."""
    cohort = "late" if mouse_id in _LATE_COHORT else "main"
    return DIV_TABLE.get((date_str, cohort), -1)


# ─── Helper functions (same as 00_preprocess.py) ───────────────────────────

def get_primary_channel(templates):
    ptp = templates.max(axis=1) - templates.min(axis=1)
    return np.argmax(ptp, axis=1)


def extract_waveforms(templates):
    primary = get_primary_channel(templates)
    return np.array([templates[i, :, primary[i]] for i in range(len(templates))])


def compute_isi_distributions(spike_times, spike_clusters, cluster_ids):
    isi_dists    = np.zeros((len(cluster_ids), ISI_BINS), dtype=np.float32)
    spike_counts = np.zeros(len(cluster_ids),             dtype=np.int32)
    bin_edges    = np.linspace(0, ISI_MAX_MS, ISI_BINS + 1)
    for idx, cid in enumerate(cluster_ids):
        mask  = spike_clusters == cid
        times = np.sort(spike_times[mask]).astype(np.float64) / SAMPLE_RATE * 1000.0
        spike_counts[idx] = len(times)
        if len(times) > 1:
            isi = np.diff(times)
            hist, _ = np.histogram(isi[isi < ISI_MAX_MS], bins=bin_edges)
            isi_dists[idx] = hist.astype(np.float32)
    return isi_dists, spike_counts


def compute_acg_single(spike_times_ms):
    n_bins = int(2 * ACG_WINDOW_MS / ACG_BIN_MS) + 1
    acg    = np.zeros(n_bins, dtype=np.float32)
    n      = len(spike_times_ms)
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
    acg[n_bins // 2] = 0.0
    return acg


def compute_acgs(spike_times, spike_clusters, cluster_ids):
    n_bins = int(2 * ACG_WINDOW_MS / ACG_BIN_MS) + 1
    acgs   = np.zeros((len(cluster_ids), n_bins), dtype=np.float32)
    for idx, cid in enumerate(cluster_ids):
        mask     = spike_clusters == cid
        times_ms = (spike_times[mask].astype(np.float64) / SAMPLE_RATE * 1000.0)
        acgs[idx] = compute_acg_single(times_ms)
    return acgs


def load_quality_metrics(phy_dir):
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
    mask = pd.Series(True, index=metrics_df.index)
    if "presence_ratio" in metrics_df.columns:
        mask &= metrics_df["presence_ratio"].fillna(0) >= MIN_PRESENCE_RATIO
    if "isi_violations_ratio" in metrics_df.columns:
        mask &= metrics_df["isi_violations_ratio"].fillna(1) <= MAX_ISI_VIOL_RATIO
    if "snr" in metrics_df.columns:
        mask &= metrics_df["snr"].fillna(0) >= MIN_SNR
    spike_series = pd.Series(spike_counts, index=metrics_df.index)
    mask &= spike_series >= MIN_SPIKE_COUNT
    return mask


# ─── Per-well processing ─────────────────────────────────────────────────

def process_well(phy_dir, well_id, run_id, mouse_id, date_str):
    phy_dir = Path(phy_dir)
    try:
        templates      = np.load(phy_dir / "templates.npy")
        spike_times    = np.load(phy_dir / "spike_times.npy").ravel()
        spike_clusters = np.load(phy_dir / "spike_clusters.npy").ravel()
    except Exception as e:
        print(f"  [SKIP] {phy_dir}: {e}", flush=True)
        return None

    n_clusters  = templates.shape[0]
    cluster_ids = np.arange(n_clusters)

    metrics_df = load_quality_metrics(phy_dir)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(index=pd.Index(cluster_ids, name="cluster_id"))
    else:
        metrics_df = metrics_df.reindex(cluster_ids)

    spike_count_arr = np.array([np.sum(spike_clusters == c) for c in cluster_ids])

    q_mask   = apply_quality_filter(metrics_df, spike_count_arr)
    good_ids = cluster_ids[q_mask.values]
    n_good   = len(good_ids)

    if n_good == 0:
        print(f"  [WARN] {date_str}/{mouse_id}/{run_id}/{well_id}: 0 clusters pass QC", flush=True)
        return None

    waveforms = extract_waveforms(templates[good_ids])
    isi_dists, spike_counts = compute_isi_distributions(
        spike_times, spike_clusters, good_ids
    )
    acgs = compute_acgs(spike_times, spike_clusters, good_ids)

    well_idx       = int(well_id.replace("well", ""))
    condition, sex = WELL_CONDITION.get((mouse_id, well_idx), ("unknown", "unknown"))
    div            = get_div(date_str, mouse_id)

    meta_rows = []
    for local_idx, cid in enumerate(good_ids):
        row = {
            "global_idx"  : None,
            "cluster_id"  : int(cid),
            "well"        : well_id,
            "run"         : run_id,
            "mouse_id"    : mouse_id,
            "date"        : date_str,
            "condition"   : condition,
            "sex"         : sex,
            "div"         : div,
            "experiment"  : "T3",
            "spike_count" : int(spike_count_arr[cid]),
        }
        for col in metrics_df.columns:
            row[col] = metrics_df.loc[cid, col] if cid in metrics_df.index else np.nan
        meta_rows.append(row)

    return {
        "waveforms": waveforms,
        "isi_dists": isi_dists,
        "acgs":      acgs,
        "metadata":  pd.DataFrame(meta_rows),
        "n_total":   n_clusters,
        "n_good":    n_good,
    }


# ─── Main ────────────────────────────────────────────────────────────────

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

                    result["metadata"]["global_idx"] = range(global_idx, global_idx + n_good)
                    global_idx += n_good
                    well_count += 1

                    all_waveforms.append(result["waveforms"])
                    all_isi.append(result["isi_dists"])
                    all_acg.append(result["acgs"])
                    all_meta.append(result["metadata"])

                    # Save per-well outputs
                    well_tag = f"{date_str}_{mouse_id}_{run_id}_{well_id}"
                    well_out = OUT_WELLS / well_tag
                    well_out.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(result["waveforms"]).to_csv(well_out / "waveforms.csv", index=False)
                    pd.DataFrame(result["isi_dists"]).to_csv(well_out / "isi_dist.csv",  index=False)
                    pd.DataFrame(result["acgs"]).to_csv(well_out / "acg.csv",            index=False)
                    result["metadata"].to_csv(well_out / "metadata.csv",                 index=False)

    if not all_waveforms:
        print("No data processed — check paths.", file=sys.stderr)
        return

    print(f"\nCombining {well_count} wells, {global_idx} units total …", flush=True)

    wf_all  = np.vstack(all_waveforms)
    isi_all = np.vstack(all_isi)
    acg_all = np.vstack(all_acg)
    meta_all = pd.concat(all_meta, ignore_index=True)

    acg_cols = [f"{v:.2f}" for v in np.arange(
        -ACG_WINDOW_MS, ACG_WINDOW_MS + ACG_BIN_MS, ACG_BIN_MS,
    )]

    pd.DataFrame(wf_all).to_csv(OUT_HIPPIE / "waveforms.csv",  index=False)
    pd.DataFrame(isi_all).to_csv(OUT_HIPPIE / "isi_dist.csv",  index=False)
    pd.DataFrame(acg_all, columns=acg_cols[:acg_all.shape[1]]).to_csv(
        OUT_HIPPIE / "acg.csv", index=False)

    labels_df = pd.DataFrame({"label": meta_all["condition"].values})
    labels_df.to_csv(OUT_HIPPIE / "labels.csv", index=False)
    meta_all.to_csv(OUT_HIPPIE / "metadata.csv", index=False)

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Wells processed : {well_count}  (skipped: {skip_count})")
    print(f"Total units     : {global_idx}")
    print(f"  waveforms.csv : {wf_all.shape}")
    print(f"  isi_dist.csv  : {isi_all.shape}")
    print(f"  acg.csv       : {acg_all.shape}")
    print(f"\nUnits per condition:")
    print(meta_all.groupby("condition").size().to_string())
    print(f"\nUnits per date:")
    print(meta_all.groupby("date").size().to_string())
    print(f"\nUnits per mouse:")
    print(meta_all.groupby("mouse_id").size().to_string())
    print(f"\nOutputs written to:")
    print(f"  {OUT_HIPPIE}")
    print(f"  {OUT_WELLS}")


if __name__ == "__main__":
    main()
