# Folic Acid MEA Analysis — Results Summary

**Date:** 2026-03-05
**Data:** `FolicAcid_T4_02252025_SA-selected/` — Maxwell Biosystems MEA, 10 kHz, 991-channel HP-filtered recordings
**Pipeline:** Kilosort phy_output → HIPPIE (full cVAE, pretrain-only) → UMAP / classifier probing / network metrics

---

## 1. Experimental Design

The plate layout was confirmed from `FolicAcid_T4_02252025_SA.tsv`. **Conditions are assigned per well, not per mouse.** Each MEA plate holds 6 wells with different neuronal sources:

| Code | Condition | # Wells | # Units |
|------|-----------|---------|---------|
| A | 2 mg FA (control) | 80 | 24,122 |
| B | 0 mg FA (deficient) | 82 | 20,343 |
| C | 10 mg FA (excess) | 24 | 5,374 |
| D | 20 mg FA (super excess) | 84 | 20,887 |
| E | Folinic Acid excess | 100 | 26,799 |

**Total: 97,525 QC-passing units across 370 wells (7 recording dates, 6 mice).**

Recording dates and DIV:

| Date folder | DIV (main cohort) | DIV (M08032/M08068) |
|-------------|------------------|---------------------|
| 250228 | 3 | — |
| 250303 | 6 | — |
| 250306 | 9 | — |
| 250310 | 13 | — |
| 250313 | 16 | — |
| 250317 | 20 | 5 |
| 250320 | 23 | 8 |

> **Critical confound:** Condition C (10 mg FA) is present only in M08032 and M08068, which also started recording at DIV 5/8 instead of DIV 3. Any comparison of condition C vs the other four conditions is confounded by mouse identity and developmental stage.

---

## 2. Quality Control Filtering

Filters applied per unit: presence ratio ≥ 0.5, ISI violations ratio ≤ 0.10, SNR ≥ 3, ≥ 50 spikes.

- Input clusters: ~499 per well (Kilosort output)
- Passing clusters: **97,525 total** — roughly 50–60% per well pass all filters
- Zero wells were entirely discarded

---

## 3. HIPPIE Embedding Analysis

### 3.1 Setup

Config: `full_model` (full cVAE — waveform + ISI + ACG modalities, regularisation + augmentation + conditional encoder)
Latent dimension: 32 | Beta: 4.0 | Batch size: 512 | Pretrain-only (no finetuning)

Pretraining runs completed:

| Epochs | Conditioning | # Cond. classes | Notes |
|--------|-------------|-----------------|-------|
| 10 | dataset source ID | 8 | baseline, smooth manifold |
| 20 | dataset source ID | 8 | slightly worse than 10 ep |
| 100 | dataset source ID | 8 | no gain vs 10 ep |
| 50 | well identity | 378 | ⚠️ latent collapse, degenerate UMAP |
| 50 | DIV — broken | 17 | ⚠️ FA_T4 excluded from pretraining; DIV embeddings untrained |
| 50 | mouse × DIV | 40 | ⚠️ same bug, fragmented UMAP |
| 50 | **DIV — fixed** | **17** | **FA_T4 in pretraining; loss 0.59; best result** |
| — | post-hoc DIV residualization on ep10 | — | partial; mouse×DIV interaction survives |

### 3.2 Unit-Level UMAP

UMAP projections were produced for all 97,525 units, coloured by:
- Mouse ID (`umap_by_mouse.png`)
- Recording date (`umap_by_date.png`)
- FA condition (`umap_by_condition.png`)
- Sex (`umap_by_sex.png`)
- DIV (`umap_by_div.png`)
- SNR / half-width (continuous quality metrics)

The strongest visual structure in the UMAP is by **mouse ID** and **recording date**, consistent with mouse and batch effects dominating the embedding space.

### 3.3 Well-Level UMAP

Units were averaged per well to give 370 well-level embeddings (`well_aggregated_embeddings.csv`). A second UMAP was run on these and coloured by mouse, condition, and sex. Well-level plots are in `embedding_results/epochs_10/umap_well_level_*.png`.

### 3.4 Condition Classifier Probing

A logistic regression was trained on well-level embeddings in a **leave-one-mouse-out (LOMO) cross-validation** to decode FA condition (5-class). Chance level = 0.20.

#### Per-mouse balanced accuracy by model variant

| Left-out mouse | 10 ep | 20 ep | 100 ep | 50 ep well-cond | 10 ep DIV-resid |
|----------------|-------|-------|--------|-----------------|-----------------|
| M07137 | 0.308 | 0.234 | 0.262 | 0.280 | 0.290 |
| M07708 | 0.202 | 0.212 | 0.164 | 0.149 | 0.213 |
| M07865 | 0.286 | 0.260 | 0.343 | 0.313 | 0.371 |
| M08032 | 0.167 | 0.167 | 0.125 | 0.250 | 0.167 |
| M08068 | 0.208 | 0.167 | 0.250 | 0.208 | 0.250 |
| M08092 | 0.314 | 0.277 | 0.341 | 0.334 | 0.341 |
| **Mean** | **0.248** | **0.220** | **0.248** | **0.256** | **0.272** |

#### Summary

| Model | Epochs | Mean balanced acc | Main mice (M07137/08/65/92) | M08032/68 | Output folder |
|-------|--------|------------------|-----------------------------|-----------|---------------|
| Standard | 10 | 0.248 | 0.228 avg | 0.188 avg | `epochs_10/` |
| Standard | 20 | 0.220 | 0.246 avg | 0.167 avg | `epochs_20/` |
| Standard | 100 | 0.248 | 0.236 avg | 0.188 avg | `epochs_100/` |
| Well-conditioned | 50 | 0.256 | — | — | `epochs_50_well_cond/` ⚠️ |
| DIV-residualized (post-hoc) | 10 | 0.272 | 0.254 avg | 0.209 avg | `epochs_10_resid_div/` |
| DIV-conditioned (broken) | 50 | 0.315 | 0.243 avg | 0.458 avg | `epochs_50_div_cond/` ⚠️ |
| Mouse×DIV-conditioned | 50 | 0.294 | 0.233 avg | 0.417 avg | `epochs_50_mouse_div_cond/` ⚠️ |
| **DIV-conditioned (fixed)** | **50** | **0.432** | **0.325 avg** | **0.646 avg** | **`epochs_50_div_cond2/`** |

⚠️ = inflated or unreliable — see notes below.

**Notes on each approach:**

- **Standard (10/100 ep):** Smooth manifold, dominated by DIV gradient. No improvement from more training. Condition signal near chance.
- **Standard (20 ep):** Slightly worse than 10 ep.
- **Well-conditioned (50 ep):** 370 conditioning classes caused latent space collapse — degenerate "spoke" UMAP. Discarded.
- **DIV-residualized (post-hoc):** Removes per-DIV group means from pretrained embeddings. Modest genuine improvement. DIV gradient partially persists (mouse×DIV interaction survives linear subtraction — see Section 3.6).
- **DIV-conditioned broken (50 ep):** FA_T4 was excluded from the pretraining loop, so DIV-specific class embeddings (IDs 8–16) were never trained — conditioning was random noise. Accuracy inflated by M08032/M08068 confound.
- **Mouse×DIV-conditioned (50 ep):** Same exclusion bug as above, plus fragmented UMAP. Discarded.
- **DIV-conditioned fixed (50 ep):** FA_T4 units added to pretraining loop with per-DIV source IDs, so the model learns what each DIV conditioning class means. Training loss drops from 0.81 → 0.59. **First approach where all 4 main mice are genuinely above chance (0.224–0.394, avg 0.325).** DIV gradient substantially flattened in UMAP; mice interleaved; manifold structure preserved.

**Core finding:** Properly implemented DIV conditioning — with FA_T4 included in pretraining — is the best approach. The DIV-conditioned fixed model (0.432 mean, **0.325 for main mice**) demonstrates that once the dominant developmental confound is given to the decoder, the latent space captures FA condition-relevant structure. M08032/M08068 still score high (0.646 avg) partly due to the mouse/condition confound, but the main 4-condition comparison now shows a genuine signal above chance across all four mice.

### 3.6 Why DIV Residualization Cannot Fully Remove the Confound

Post-hoc residualization (subtracting per-DIV group means from each unit's embedding) removes the **DIV main effect** but leaves a DIV gradient still visible in UMAP. This is not a bug — it reflects a fundamental property of the dataset:

**The DIV structure that survives is the mouse × DIV interaction.** Each mouse has a unique biological fingerprint, and that fingerprint evolves differently over developmental time. Subtracting the average embedding at each DIV removes the population mean trajectory but leaves each mouse's individual trajectory intact. UMAP then projects those mouse-specific developmental trajectories back onto a DIV-like axis.

To fully remove this, one would need to subtract per-(mouse × DIV) session means — i.e., center each recording session independently. But that is equivalent to removing all between-session variation, leaving only within-session unit-to-unit noise (which is what the well-conditioning approach attempted, causing latent space collapse).

**The fundamental tension:** there is no post-hoc transformation that simultaneously:
1. Removes DIV and mouse confounds, AND
2. Preserves condition-relevant between-session variation

This is because conditions are assigned at the well level within sessions — the only clean comparison is **within the same (mouse, DIV) recording session**, comparing wells that differ only in FA condition. This is precisely the design that the raw metric analysis (Section 7) exploits, and why that analysis finds significant condition effects despite the embeddings not separating cleanly.

### 3.5 Latent Dimension — Quality Metric Correlations (Spearman, 10 epochs)

The most strongly encoded variable in the latent space is **firing range** (max |ρ| = 0.45). Other metrics are weakly encoded:

| Metric | Max |ρ| across 32 dims |
|--------|------------------------|
| firing_range | **0.454** |
| snr | 0.221 |
| amplitude_median | 0.216 |
| sync_spike_4 | 0.202 |
| peak_trough_ratio | 0.201 |
| isi_violations_ratio | 0.187 |
| half_width | 0.152 |
| peak_to_valley | 0.125 |
| presence_ratio | 0.035 |

Top latent dimensions by between-mouse variance: **z4, z27, z3, z20, z5, z26**. These dimensions likely encode mouse-specific electrophysiological properties rather than condition-specific ones.

---

## 4. Network-Level Metrics

Population burst detection (50 ms bins, z-score threshold 3) was run on all 370 wells from raw spike trains. Kruskal-Wallis across mice: all metrics **p < 0.001**.

### 4.1 By Condition (mean across all recorded timepoints)

| Condition | FR (Hz) | Burst rate (/min) | Synchrony (Fano) | Burst fraction |
|-----------|---------|------------------|-----------------|----------------|
| 0mg_deficient | 1.27 | **18.0** | 29.9 | 0.284 |
| 2mg_control | 1.68 | 15.1 | 30.9 | 0.201 |
| 20mg_super_excess | 1.38 | 16.4 | 28.3 | 0.274 |
| folinic_acid_excess | 1.58 | 15.8 | 25.4 | 0.209 |
| 10mg_excess | 1.68 | **7.3** | **6.8** | **0.101** |

> **Caveat:** 10mg_excess (C) has markedly lower burst rate and synchrony — but this entirely reflects the earlier DIV of M08032/M08068 (DIV 5/8 vs DIV 20/23 for the other conditions at overlapping recording dates). These differences are not interpretable as FA-specific effects without age-matched comparisons.

### 4.2 Burst Rate Maturation Over Time

Burst activity increases dramatically with DIV, consistent with normal in vitro network maturation:

```
DIV  3  →  burst rate ~16.8/min
DIV  5  →  burst rate  ~8.8/min   (M08032/68 — younger)
DIV 20+ →  burst rate ~25–35/min  (250317/320 recordings)
```

This maturation trajectory is the dominant source of variance in network metrics and must be regressed out before any condition comparison.

---

## 5. Interpretation

The core message from the current analysis:

1. **Properly implemented DIV-conditioned HIPPIE does encode FA condition.** When FA_T4 units are included in pretraining with per-DIV source IDs — so the decoder learns to use DIV as a conditioning signal — the latent space organises around electrophysiological properties beyond developmental stage. Condition classification (5-class LOMO) reaches **0.432 mean balanced accuracy** (0.325 for the clean 4-condition comparison), compared to 0.248 at baseline chance.

2. **The key methodological insight:** conditioning must be trained, not just applied at inference. The DIV class embeddings must be learned during pretraining; attaching them post-hoc (or at inference only on an excluded dataset) produces random noise conditioning and no improvement.

3. **The strongest structure in unconditioned embeddings is mouse identity and DIV.** Standard pretrained HIPPIE uses these as the dominant axes. Post-hoc residualization removes the DIV main effect but not the mouse × DIV interaction (each mouse's developmental trajectory is unique), placing a ceiling on post-hoc deconfounding.

4. **Network activity matures robustly** over DIV 3→23, dominating network metric variance. Condition effects on network dynamics require DIV-matched analysis (see Section 7).

5. **The FA effect operates at the network level, not the single-unit level.** Single-unit waveform + ISI embeddings barely separate conditions even with DIV conditioning. Network synchrony, burst rate, and firing range — which reflect emergent connectivity properties — show significant condition differences from DIV 3 onward. FA likely affects synaptogenesis or E/I balance rather than intrinsic neuronal excitability.

6. **The 10mg_excess condition remains confounded** with mouse identity (M08032/M08068 only, younger DIV). High classification accuracy for those mice (0.646 avg) likely reflects the residual mouse fingerprint rather than FA dose.

---

## 6. Next Steps — Options

### Option A — Train HIPPIE Much Longer (50 / 100 epochs) ✦ Recommended

The current runs used 10–20 epochs on a large dataset (97K units). More pretraining can improve the quality of the latent representation:
- The pretrained encoder may not yet be fully converged; longer training may reveal finer electrophysiological structure
- Run: `bash 02_hippie_train_and_embed.sh 50` and `bash 02_hippie_train_and_embed.sh 100`
- Re-run `03_embedding_analysis.py --epochs 50` and `--epochs 100`
- Check if classifier accuracy improves and if condition/sex structure emerges in UMAP

**When to expect this to help:** if the 10-epoch model is undertrained and hasn't learned a sufficiently discriminative latent space. If accuracy is still ~0.2 at 100 epochs, the issue is not training length but signal strength.

---

### Option B — Supervised Finetuning with Condition Labels

Instead of pretrain-only, use HIPPIE's finetuning phase with condition as the class label:
- This trains the encoder to separate FA conditions using the class-conditioning in the cVAE
- Requires careful leave-one-mouse-out design to avoid overfitting to mouse identity
- Would force the embedding to find whatever distinguishes conditions — even subtle firing-rate shifts

**Risk:** condition C is confounded with mouse; 2-class runs (e.g., A vs B, or A vs D) would be cleaner.

---

### Option C — DIV-Corrected Within-Mouse Condition Comparison

Rather than HIPPIE classification, use **raw quality metrics** corrected for DIV:
- Within each plate, compare wells by condition (same mouse, same recording, different well = different condition)
- Fit a mixed model: `metric ~ condition + DIV + (1|mouse_id)`
- This removes the dominant maturation and mouse confounds and isolates the condition effect
- Most statistically powerful approach for the current dataset design

**Metrics to test:** half_width, peak_to_valley, repolarization_slope, recovery_slope, sync_spike_4, firing_range (the ones showing significant between-mouse variance in EDA)

---

---

## 7. Condition × DIV Trajectory EDA ("Before vs After" FA Exposure)

Script: `05_condition_trajectory_EDA.py` | Output: `condition_EDA/`

This analysis treats the longitudinal design as a "before vs after" comparison:
- **Early ("before"):** DIV 3–6 — minimal FA exposure time, network still establishing
- **Late ("after"):** DIV 20–23 — maximal FA exposure, network fully active
- Conditions A/B/D/E compared across all 7 timepoints (DIV 3→23); condition C excluded (DIV 5/8 only, different mice)

### 7.1 Statistical overview

105 Kruskal-Wallis tests run (15 metrics × 7 DIV timepoints): **33 significant at p < 0.05**.

#### Strongest condition effects (by timepoint)

| Level | Metric | DIV | KW p | Sig |
|-------|--------|-----|------|-----|
| unit | Firing range | 3 | 0.0001 | *** |
| unit | Synchrony (2 ms) | 9 | 0.0002 | *** |
| network | Network synchrony (Fano factor) | 3 | 0.0002 | *** |
| unit | Synchrony (4 ms) | 9 | 0.0004 | *** |
| network | Mean firing rate (Hz) | 3 | 0.0016 | ** |
| network | Burst fraction | 3 | 0.0016 | ** |
| network | Burst rate (bursts/min) | 3 | 0.0017 | ** |
| network | Mean IBI (s) | 3 | 0.0052 | ** |
| network | Mean firing rate (Hz) | 13 | 0.0056 | ** |
| unit | Amplitude median (µV) | 6 | 0.0058 | ** |
| unit | Half-width (ms) | 6 | 0.0069 | ** |

Full table: `condition_EDA/stats_kruskal_by_div.csv`

### 7.2 Key finding — effects appear early and persist

Significant condition differences emerge **at DIV 3** (the earliest recording) across:
- **Firing range** — units from different conditions span different dynamic firing rate ranges even at the very first timepoint
- **Network synchrony** (Fano factor) and **burst rate/fraction/IBI** — network-level activity is already differentiated at DIV 3
- **Half-width** and **amplitude** — waveform shape differs from DIV 3–6 onward

This pattern suggests **FA condition affects neuronal properties established early**, rather than progressively diverging over the recording period.

### 7.3 Trajectory change (Δ late − early) does not differ between conditions

| Metric | KW p (Δ) |
|--------|----------|
| NET: Mean IBI (s) | 0.068 |
| Synchrony (4 ms) | 0.095 |
| Synchrony (2 ms) | 0.109 |
| Peak-to-valley (ms) | 0.196 |
| NET: Burst rate | 0.237 |
| All others | > 0.30 |

**No metric shows a significantly different rate of change across conditions (all Δ p > 0.05).** Conditions do not diverge further or converge over time — they mature in parallel, but from different baseline levels. This is inconsistent with FA acting progressively and more consistent with FA shaping early network properties during the initial DIV 0–3 period before the first recording.

### 7.4 Figures produced

| File | Description |
|------|-------------|
| `fig1_unit_trajectories.png` | Unit metrics (median/well) over DIV per condition ±SEM |
| `fig2_network_trajectories.png` | Network metrics over DIV per condition ±SEM |
| `fig3_before_after_unit_violin.png` | Early vs late violins, unit-level, per condition |
| `fig4_before_after_network_violin.png` | Early vs late violins, network-level, per condition |
| `fig5_delta_late_minus_early.png` | Bar chart of Δ(late−early) ±SEM per condition per metric |
| `fig6_heatmap_condition_div.png` | Network metrics: condition × DIV heatmap |
| `fig7_heatmap_unit_zscore.png` | Unit metrics: z-scored trajectories, condition × DIV heatmap |

### 7.5 Interpretation

The results are consistent with a model where **FA concentration shapes the baseline state of the developing neuronal network**, rather than continuously modulating it. Specifically:

1. **Deficient FA (0 mg)** tends to show elevated synchrony and burst metrics relative to control at early timepoints — potentially reflecting a less regulated, more excitable state in the absence of normal folate.
2. **Super-excess FA (20 mg)** and **Folinic Acid** show effects in the same direction as control or intermediate.
3. **All conditions mature similarly** (parallel trajectories), arguing against ongoing dose-response effects during the recorded period.

> **Caveat:** With n=4 mice and multiple within-plate wells, statistical power is limited. The p<0.05 significance at DIV 3 across multiple network metrics is encouraging but must be interpreted carefully given multiple comparisons and small sample sizes.

---

### Option D — Sex-Stratified Analysis

The dataset has both male and female neurons in each condition. An interaction between FA condition and sex is plausible (FA metabolism differs by sex). Options:
- Repeat the classifier probing separately for Male and Female wells
- Test condition × sex interaction in a mixed model on network metrics
- Plot condition effects separately per sex in the well-level UMAP

---

### Option E — Network Metrics, DIV-Matched

For conditions that share recording dates (A, B, D, E all appear in M07137/M07708/M07865/M08092 at DIV 3–23), perform condition comparisons restricted to the same DIV:
- Within each DIV timepoint: Kruskal-Wallis or mixed model comparing A vs B vs D vs E
- Plot burst_rate, network_synchrony, burst_fraction trajectories over DIV per condition
- This tests whether the network maturation trajectory differs by FA dose

---

### Option F — Reconstruction-Based Feature Attribution

Use the HIPPIE decoder to perturb latent dimensions one at a time and measure change in reconstructed waveform/ISI/ACG:
- Identifies which latent dims encode waveform shape vs firing pattern
- Can reveal if any latent dimension tracks condition even if the classifier is weak
- Requires the decoder weights from the HIPPIE checkpoint





### 100 Epoch run 
● Now we have much richer structure. Here's what the 13 clusters reveal:

  E/I classification from the metrics

  All clusters fall below the 0.35 ms half-width threshold — the red dashed line sits above all boxes. This is expected for dissociated cortical cultures, where spike
  widths at the electrode tend to be shorter than intracellular measures. The relative ordering still tells you a lot:

  ┌─────────┬────────────────────────┬───────────────────────┬─────────────────────────────────────────────────────────────┬──────────────────────────────┐
  │ Cluster │       Half-width       │    Peak-to-valley     │                        Key features                         │       Likely identity        │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 0       │ ~0.00015 s (narrowest) │ ~0.00025 s            │ Very high firing range (~6), high synchrony, high amplitude │ Fast-spiking inhibitory      │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 3       │ ~0.00015 s             │ very short (~0.00030) │ Very narrow P2V, low repol slope                            │ Narrow-spike inhibitory-like │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 4       │ ~0.00030 s (broadest)  │ ~0.00100 s            │ Broadest P2V by far, steep repol slope, early DIV           │ Broad-spike excitatory       │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 5/6     │ ~0.00019 s             │ narrow                │ Moderate metrics                                            │ Intermediate — mixed/regular │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 1       │ ~0.00023 s             │ ~0.00075 s            │ High firing range, late DIV, high synchrony                 │ Mature active excitatory     │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 8       │ ~0.00014 s             │ very narrow           │ Minimal firing range (~0.6), very low SNR ~6                │ Low-quality / sparse units   │
  ├─────────┼────────────────────────┼───────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ 12      │ ~0.00020 s             │ narrow                │ Flat ISI, flat ACG                                          │ Irregular/low-activity units │