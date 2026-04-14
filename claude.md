# Roy & Shruti — Folic Acid MEA / HIPPIE Analysis

## Project Context

Collaboration with Roy Ben-Shalom and Shruti Shah (UC Davis) to analyze multi-electrode array (MEA)
recordings from a **Folic Acid (FA) dose experiment** using the HIPPIE neural embedding pipeline.

**Biological question:** Do different folic acid doses produce distinct electrophysiological
phenotypes in mouse neuronal cultures, and can HIPPIE latent embeddings detect them?

**Three FA conditions:**
- `2mg` = Control (FA control)
- `10mg` = Excess folic acid
- `0mg` = Deficient folic acid

**Problem:** Their initial clustering analysis did not yield convincing clusters. The goal is to find
a better workflow that produces more stable, interpretable clusters and to understand what the model
focuses on.

---

## Data Structure

```
FolicAcid_T4_02252025_SA-selected/
└── {DATE}/              # 7 recording dates: 250228, 250303, 250306, 250310, 250313, 250317, 250320
    └── {MOUSE_ID}/      # 6 mice: M07137, M07708, M07865, M08032*, M08068*, M08092
        └── Network/
            └── {RUN_ID}/           # 2 runs per mouse per date (e.g. 000005, 000006)
                └── {WELL}/         # well000-well005 (up to 6 wells; well004 often absent)
                    └── phy_output/ # Kilosort-style spike-sorted outputs
```

*M08032 and M08068 only present from 250317 onward (later addition to experiment).

**Total:** ~370 wells, ~400-525 Kilosort clusters per well (all labeled "unsorted" — no manual curation).

**Recording parameters:** 991 channels, 10 kHz, float32, HP filtered (Maxwell Biosystems MCS).

### Files per well (phy_output/)
| File | Description |
|------|-------------|
| `templates.npy` | Mean waveform per cluster (n_clusters x n_timepoints x n_channels) |
| `spike_times.npy` | All spike times (samples) |
| `spike_clusters.npy` | Cluster assignment per spike |
| `amplitudes.npy` | Per-spike amplitude |
| `channel_positions.npy` | XY positions of channels |
| `cluster_*.tsv` | Per-cluster quality metrics (SNR, ISI violations, amplitude, half-width, etc.) |

### Available per-cluster quality metrics (TSVs)
`amplitude_cutoff`, `amplitude_cv_median`, `amplitude_cv_range`, `amplitude_median`,
`firing_range`, `half_width`, `isi_violations_count`, `isi_violations_ratio`,
`noise_cutoff`, `noise_ratio`, `num_negative_peaks`, `num_positive_peaks`,
`peak_to_valley`, `peak_trough_ratio`, `presence_ratio`, `recovery_slope`,
`repolarization_slope`, `rp_contamination`, `rp_violations`, `sd_ratio`,
`sliding_rp_violation`, `snr`, `sync_spike_2`, `sync_spike_4`, `sync_spike_8`,
`loc_x`, `loc_y`

---

## Mouse to Condition Mapping

**TODO: Confirm mapping from Roy/Shruti.**
Expected: each of the 6 mice belongs to one of the 3 FA conditions (likely 2 mice per condition).

---

## Key Decisions from Email Exchange

1. **Skip finetuning** (or treat it as optional/ablation). Finetuning on the small labeled subset
   collapses embeddings toward those labels and reduces exploratory structure.
2. **Use full HIPPIE model**, not baseline (richer regularization, augmentation, conditional encoder).
3. **Aggregate** embeddings at the well or mouse level for condition-level comparisons — single-unit
   resolution is too noisy and statistically fragile.
4. **Find what drives clustering** — identify which latent dimensions or input features differentiate
   the three FA conditions.

---

## Planned Experiments

### Step 0 — Data Preprocessing Pipeline

Convert Kilosort outputs to HIPPIE-format CSVs for each well:

- **waveform.csv** — Extract primary channel waveform from `templates.npy` for each cluster.
  Shape: (n_clusters x n_waveform_timepoints).
- **isi_dist.csv** — Compute interspike interval distribution histogram per cluster from
  `spike_times.npy` + `spike_clusters.npy`.
- **acg.csv** — Compute autocorrelogram per cluster.
- **labels.csv** — No ground truth; fill with dummy/placeholder column.
- **metadata.tsv** — Per-cluster table: `cluster_id`, `well`, `run`, `mouse_id`, `date`,
  `condition` (0mg/2mg/10mg), plus all quality metrics. Join all cluster_*.tsv files here.

**Quality filtering** before HIPPIE ingestion:
- Keep clusters with `isi_violations_ratio < 0.1` AND `presence_ratio > 0.5` AND `snr > 3`.
  Standard thresholds for well-isolated single units.
- Record how many clusters survive per well/condition.

### Step 1 — HIPPIE Pretraining (Core Experiment)

**No finetuning.** Use the full HIPPIE model (cVAE).

Train three variants to observe embedding collapse over epochs:
- **Exp-1a:** 1 epoch
- **Exp-1b:** 10 epochs  ← primary
- **Exp-1c:** 20 epochs

Training data: all FA wells (all conditions, all timepoints pooled).
Also include public HIPPIE pretraining datasets for better transfer representations:
`hausser_cell_type`, `hull_cell_type`, `lissberger_labeled_cell_type`,
`mouse_organoids_cell_line`, `juxtacellular_mouse_s1_area`

Prediction/embedding: every well in the FA dataset.

**Expected outcome:** 1 epoch = spread-out clusters; 20 epochs may collapse to lines.
This sweep identifies the sweet spot.

### Step 2 — Well-Level Aggregated Embeddings

After extracting per-unit embeddings from Step 1:

1. **Aggregate per well:** Compute mean (and optionally median) embedding across all
   quality-filtered units in each well.
   Result: one vector per (date x mouse x well).

2. **UMAP/t-SNE** of well-level vectors, colored by:
   - FA condition (0mg / 2mg / 10mg)
   - Recording date (DIV proxy — developmental time)
   - Mouse ID (to detect subject-level confounds)
   - Well index

3. **Trajectory analysis:** Plot each mouse's embedding path across the 7 recording dates.
   Do conditions diverge over development? When does the divergence begin?

### Step 3 — Condition Classification (Classifier Probing)

Using well-level aggregated embeddings from Step 2:

- Train **logistic regression** (+ SVM, random forest) to classify FA condition
  (0mg vs 2mg vs 10mg).
- Use **leave-one-mouse-out cross-validation** to prevent data leakage.
- Metrics: balanced accuracy, AUROC per condition.
- If classifier works, embeddings capture condition-relevant information.

### Step 4 — Feature Attribution (What Drives Clustering)

**4a. Latent dimension correlation:**
- Correlate each HIPPIE latent dimension with FA condition (ANOVA), date, mouse ID.
- Find the top condition-predictive dimensions.
- Plot per-dimension distributions grouped by condition.

**4b. Input feature importance:**
- Correlate raw input features (amplitude_median, half_width, peak_to_valley, firing_range,
  sync_spike_2/4/8, isi_violations_ratio, etc.) with the condition-predictive latent dims.
- Maps latent dimensions back to interpretable electrophysiology.

**4c. Reconstruction perturbation (optional):**
- Perturb individual latent dimensions in the cVAE decoder.
- Observe which waveform/ISI properties change — identifies what each dimension encodes
  (e.g., spike width, firing regularity, network synchrony).

**4d. Direct metric comparison (sanity/positive control):**
- Before embedding: test whether raw per-well mean metrics already differ across FA conditions
  using ANOVA/Kruskal-Wallis (firing_range, amplitude_median, half_width, sync_spike_4, etc.).
- If they separate: positive control confirming the biology is detectable.
- If they don't separate: HIPPIE's joint representation may still find subtle patterns.

### Step 5 — Network-Level Metrics (Validation)

Compute from `spike_times.npy` + `spike_clusters.npy` directly — bypasses single-unit embeddings:
- **Mean firing rate** (spikes/s, averaged across units per well)
- **Network burst rate** (bursts/min via population rate threshold)
- **Inter-burst interval (IBI)** — mean and CV
- **Synchrony index** — fraction of spikes in bursts, or pairwise correlation
- Compare across FA conditions and timepoints
- Correlate with HIPPIE cluster assignments and embedding coordinates for biological validation

---

## Recommended Workflow Priority

```
Step 0   Preprocessing pipeline (for all wells)
  |
  v
Step 4d  Raw metric comparison (quick sanity check)
  |
  v
Step 1b  10-epoch pretrain, full model (primary embedding run)
  |
  v
Step 2   Well-level UMAP (does condition separate?)
  |
  v
Step 3   Classifier probing (quantify separability)
  |
  v
Step 4a-b Feature attribution (what drives the separation?)
  |
  v
Step 5   Network metrics (biological validation)
  |
  v
Step 1a,c Epoch ablation (1 vs 20 epochs comparison)
Step 4c  Latent perturbation (interpretability)
```

---

## Open Questions / TODOs

- [ ] **Confirm mouse to FA condition mapping** (which of M07137, M07708, M07865, M08032,
      M08068, M08092 is 0mg / 2mg / 10mg). Critical for all downstream analysis.
- [ ] Ask Shruti for the Jupyter notebook used to convert .npy to csv — need exact waveform
      extraction parameters (which channel selected, normalization, window size).
- [ ] Confirm DIV for each recording date (e.g., 250228 = DIV X) — needed for developmental axis.
- [ ] Clarify: are the two runs per mouse per date (e.g. 000005, 000006) technical replicates
      or different stimulation conditions? Both used or just one?
- [ ] Confirm role of M08032 and M08068 — same conditions as existing mice or new replicates?

---

## Notes on HIPPIE Pipeline

- Full model (cVAE with regularization + augmentation) >> baseline for this task.
- Pretrain-only embeddings = more exploratory and spread out.
- Finetuned embeddings = more structured but biased toward finetune label space — avoid unless
  finetuning on FA-relevant labels.
- Aggregating by well is statistically robust; per-unit clustering with ~500 unsorted units
  per well will be noisy without manual curation.
- The CSV format Shruti already generated (waveform, ISI, ACG) should be reusable with updated
  pipeline settings — just confirm the extraction parameters match HIPPIE expectations.
- All clusters are currently "unsorted" (no Kilosort manual curation) — apply quality metric
  filtering before embedding rather than relying on cluster_group.tsv.
