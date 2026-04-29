# Folic Acid MEA Analysis with HIPPIE

**Can different folic acid doses change how neurons behave?**

This project analyzes multi-electrode array (MEA) recordings from mouse neuronal cultures
grown under different folic acid (FA) conditions. We use the
[HIPPIE](https://github.com/braingeneers/HIPPIE) neural embedding pipeline to find patterns
in single-neuron activity, then ask whether those patterns differ between FA doses.

---

## Table of Contents

1. [Background](#background)
2. [Data Overview](#data-overview)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Pipeline: Step by Step](#pipeline-step-by-step)
6. [Key Findings](#key-findings)
7. [Glossary](#glossary)

---

## Background

Folic acid (vitamin B9) is essential for neural development. Too little or too much may alter
how neurons grow and fire. This experiment tests **5 folic acid conditions** on mouse neuronal
cultures grown on multi-electrode arrays (MEAs):

| Code | Condition | Description |
|------|-----------|-------------|
| A | 2 mg (Control) | Standard folic acid dose |
| B | 0 mg (Deficient) | No folic acid |
| C | 10 mg (Excess) | Moderate excess |
| D | 20 mg (Super Excess) | High excess |
| E | Folinic Acid Excess | Alternative form of folate |

Each culture was recorded at **7 time points** (DIV 3 through DIV 23) to track how neural
activity develops over time.

### What is HIPPIE?

HIPPIE is a deep learning model (specifically a conditional variational autoencoder, or cVAE)
that takes three features of a neuron's activity — its **waveform shape**, **interspike interval
distribution**, and **autocorrelogram** — and compresses them into a 32-number summary called
an "embedding." Neurons with similar embeddings have similar electrophysiological properties.

---

## Data Overview

### Recording setup
- **Platform:** Maxwell Biosystems MEA (991 channels per well)
- **Sampling rate:** 10 kHz
- **Spike sorting:** Kilosort (automated, no manual curation)

### Animals and wells
- **6 mice:** M07137, M07708, M07865, M08092 (full timeline), M08032, M08068 (late addition)
- **~370 wells** total across all recording dates
- **Conditions are per-well** (not per-mouse) — each mouse's plate has wells with different FA doses

### Timeline (DIV = Days In Vitro)

| Recording Date | DIV (main cohort) | DIV (M08032/68) |
|---|---|---|
| 250228 | 3 | — |
| 250303 | 6 | — |
| 250306 | 9 | — |
| 250310 | 13 | — |
| 250313 | 16 | — |
| 250317 | 20 | 5 |
| 250320 | 23 | 8 |

### After preprocessing
- **97,525 quality-filtered units** across all wells
- Each unit has: waveform (50 timepoints), ISI histogram (100 bins), autocorrelogram (201 bins)
- Plus quality metrics: SNR, amplitude, half-width, firing range, ISI violations, etc.

---

## Project Structure

```
Roy_shruti_folic_data/
|
|-- README.md                    <-- You are here
|-- config.py                    <-- All file paths in one place
|-- requirements.txt             <-- Python dependencies
|
|-- scripts/                     <-- Analysis pipeline (run in order 00 -> 09)
|   |-- 00_preprocess.py         <-- Raw data -> HIPPIE-format CSVs
|   |-- 01_patch_metadata.py     <-- Fix condition/sex labels in metadata
|   |-- 02_raw_metric_eda.py     <-- Quality metric exploration (before HIPPIE)
|   |-- 03_train_hippie.sh       <-- Train HIPPIE model & extract embeddings
|   |-- 04_embedding_analysis.py <-- UMAP, classifiers, feature attribution
|   |-- 05_network_metrics.py    <-- Population-level burst/synchrony metrics
|   |-- 06_condition_trajectory.py  <-- Condition x DIV trajectory analysis
|   |-- 07_hdbscan_clustering.py <-- Cluster neurons by embedding similarity
|   |-- 08_ei_proportion.py      <-- Excitatory/inhibitory fraction per condition
|   |-- 09_umap_animation.py     <-- Animated UMAP across time points
|   '-- __init__.py
|
|-- results/                     <-- All outputs from the scripts
|   |-- eda/                     <-- From 02_raw_metric_eda.py
|   |-- embeddings/              <-- From 04-09 (UMAP plots, classifiers, HDBSCAN)
|   |-- network/                 <-- From 05_network_metrics.py
|   |-- condition_eda/           <-- From 06_condition_trajectory.py
|   '-- figures/                 <-- Key figures for presentations
|
|-- data/
|   |-- raw -> FolicAcid_.../    <-- Symlink to raw Kilosort outputs
|   '-- processed -> per_well_processed/  <-- Symlink to per-well CSVs
|
|-- FolicAcid_T4_02252025_SA-selected/   <-- Raw data (large, not in git)
|-- per_well_processed/                   <-- Per-well preprocessed CSVs
|
|-- CLAUDE.md                    <-- Detailed project specification
|-- RESULTS.md                   <-- Full results writeup
'-- DEEP_RESEARCH_BRIEF.md       <-- Scientific background
```

---

## Setup

### 1. Python environment

All scripts use the HIPPIE virtual environment. If you don't have HIPPIE installed,
you can install the dependencies for the analysis scripts (04-09) with:

```bash
pip install -r requirements.txt
```

For the full pipeline including HIPPIE training (step 03), you need the
[HIPPIE repository](https://github.com/braingeneers/HIPPIE) set up separately.

### 2. Configure paths

All paths are centralized in **`config.py`**. The defaults work on the original machine.
To run elsewhere, either edit `config.py` or set environment variables:

```bash
export HIPPIE_INPUT_DIR=/path/to/FA_T4           # preprocessed HIPPIE input data
export HIPPIE_CODE_DIR=/path/to/hip-hip-hippie   # HIPPIE code repository
export HIPPIE_RESULTS_DIR=/path/to/results/FA_T4 # HIPPIE embedding outputs
```

---

## Pipeline: Step by Step

The pipeline has **10 scripts** numbered `00` through `09`. Run them in order.
Each script is self-contained — you can also run later scripts independently
if the earlier outputs already exist.

### Step 00: Preprocess raw data

**Script:** `scripts/00_preprocess.py`

**What it does:** Converts Kilosort spike-sorted outputs into clean CSV files that
HIPPIE can read. For each of the ~370 wells, it:
1. Extracts the mean waveform for each neuron (from `templates.npy`)
2. Computes interspike interval (ISI) histograms
3. Computes autocorrelograms (ACGs)
4. Applies quality filters: SNR > 3, presence > 50%, ISI violations < 10%

**Input:** `data/raw/` (Kilosort phy_output directories)
**Output:** Per-well CSVs in `data/processed/` + combined dataset for HIPPIE training

```bash
python scripts/00_preprocess.py
```

---

### Step 01: Patch metadata (if needed)

**Script:** `scripts/01_patch_metadata.py`

**What it does:** Updates the metadata CSV with correct condition labels (A-E),
sex (Male/Female), and DIV for each unit. Only needed if you re-run preprocessing
or if labels are missing.

```bash
python scripts/01_patch_metadata.py
```

---

### Step 02: Raw metric EDA

**Script:** `scripts/02_raw_metric_eda.py`

**What it does:** Explores the quality metrics (SNR, amplitude, half-width, etc.)
*before* any HIPPIE embedding. This is a sanity check: are there already measurable
differences between mice or conditions in the raw electrophysiology?

**Output:** `results/eda/` — distribution plots, correlation heatmaps, statistical tests

```bash
python scripts/02_raw_metric_eda.py
```

**What to look at:**
- `metric_distributions_by_mouse.png` — Do metrics vary across mice?
- `kruskal_wallis_by_mouse.csv` — Statistical significance of metric differences

---

### Step 03: Train HIPPIE & extract embeddings

**Script:** `scripts/03_train_hippie.sh`

**What it does:** Trains the HIPPIE cVAE model on all the preprocessed data plus
public neuroscience datasets. After training, it extracts a 32-dimensional embedding
for every neuron.

**Key settings:** `full_model` config, 32 latent dimensions, beta=4 (KL weight)

```bash
bash scripts/03_train_hippie.sh 10      # train for 10 epochs
bash scripts/03_train_hippie.sh --all   # sweep 1, 10, 20 epochs
```

**Output:** Embedding CSVs in the HIPPIE results directory

---

### Step 04: Embedding analysis

**Script:** `scripts/04_embedding_analysis.py`

**What it does:** The main analysis of HIPPIE embeddings:
- **UMAP visualization** — 2D projection colored by condition, mouse, DIV, sex
- **Well-level aggregation** — Average embedding per well (reduces noise)
- **Classifier probing** — Can a simple classifier predict FA condition from embeddings?
- **Feature attribution** — Which latent dimensions correlate with condition or metrics?

**Output:** `results/embeddings/epochs_N/` — UMAP plots, classifier accuracy, correlations

```bash
python scripts/04_embedding_analysis.py --config full_model --z-dim 32 --beta 4 --epochs 10
```

**What to look at:**
- `umap_by_condition.png` — Do FA conditions form separate clusters?
- `classifier_probing_lomo.csv` — Classification accuracy (chance = 0.2 for 5 conditions)
- `umap_well_level_by_condition.png` — Cleaner view with one dot per well

---

### Step 05: Network metrics

**Script:** `scripts/05_network_metrics.py`

**What it does:** Computes population-level activity metrics directly from spike trains
(no HIPPIE needed). These are standard MEA metrics:
- Mean firing rate
- Burst rate and inter-burst intervals
- Network synchrony (Fano factor)

**Output:** `results/network/network_metrics_all_wells.csv` + plots

```bash
python scripts/05_network_metrics.py
```

---

### Step 06: Condition trajectory analysis

**Script:** `scripts/06_condition_trajectory.py`

**What it does:** Tracks how metrics change over developmental time (DIV) for each
FA condition. Compares "early" (DIV 3-6) vs "late" (DIV 20-23) to see if FA
exposure produces divergent developmental trajectories.

**Output:** `results/condition_eda/` — trajectory plots, before/after violin plots, heatmaps

```bash
python scripts/06_condition_trajectory.py
```

---

### Step 07: HDBSCAN clustering

**Script:** `scripts/07_hdbscan_clustering.py`

**What it does:** Uses HDBSCAN (a density-based clustering algorithm) on the UMAP
coordinates to discover natural neuron types in the embedding space. Then profiles
each cluster by its waveform shape, ISI distribution, and autocorrelogram.

**Output:** `results/embeddings/epochs_N_hdbscan/` — cluster maps, profile grids

```bash
python scripts/07_hdbscan_clustering.py --epochs 100
```

**What to look at:**
- `cluster_profiles_wave_acg_isi.png` — What does each neuron type look like?
- `umap_categorical.png` — Cluster labels on the UMAP

---

### Step 08: E/I proportion analysis

**Script:** `scripts/08_ei_proportion.py`

**What it does:** Classifies HDBSCAN clusters as excitatory or inhibitory based on
waveform shape, then compares the excitatory fraction across FA conditions.

**Output:** `results/embeddings/epochs_100_hdbscan/ei_proportion/`

```bash
python scripts/08_ei_proportion.py
```

---

### Step 09: UMAP animation

**Script:** `scripts/09_umap_animation.py`

**What it does:** Creates animated GIFs showing how the neuron population evolves
across recording dates on a fixed UMAP coordinate space.

```bash
python scripts/09_umap_animation.py --color condition --fps 1
python scripts/09_umap_animation.py --color hdbscan   --fps 1
```

---

## Key Findings

1. **Raw metrics already differ between mice** — All quality metrics are highly significant
   across mouse IDs (Kruskal-Wallis p near 0), confirming biological variability.

2. **HIPPIE embeddings show weak condition separation** — At 10 epochs, leave-one-mouse-out
   classifier accuracy is 0.248 (chance = 0.2). The signal is subtle.

3. **Network activity increases dramatically over development** — Burst rate goes from
   2-10/min at DIV 3 to 25-35/min at DIV 23, as expected for maturing cultures.

4. **10 mg excess (condition C) is confounded** — It only appears in 2 late-addition mice
   with fewer time points, making it hard to separate condition from mouse effects.

5. **HDBSCAN identifies interpretable neuron clusters** — At 100 epochs, clusters
   correspond to recognizable waveform types (narrow-spiking, broad-spiking, etc.).

For the full writeup, see [RESULTS.md](RESULTS.md).

---

## Glossary

| Term | Meaning |
|------|---------|
| **MEA** | Multi-Electrode Array — a chip with hundreds of tiny electrodes that record electrical activity from neurons |
| **DIV** | Days In Vitro — how many days the neurons have been growing in the dish |
| **Spike sorting** | Algorithms (like Kilosort) that separate mixed electrical signals into individual neurons |
| **Waveform** | The characteristic shape of a neuron's electrical spike (~1-2 ms) |
| **ISI** | Interspike Interval — time between consecutive spikes from the same neuron |
| **ACG** | Autocorrelogram — histogram of all pairwise spike time differences for one neuron |
| **SNR** | Signal-to-Noise Ratio — how clean the spike signal is |
| **HIPPIE** | A deep learning model that learns compact representations of neuron activity |
| **cVAE** | Conditional Variational Autoencoder — the neural network architecture HIPPIE uses |
| **Embedding** | A compact numerical summary (here: 32 numbers) that captures a neuron's key properties |
| **UMAP** | Uniform Manifold Approximation and Projection — a method to visualize high-dimensional data in 2D |
| **HDBSCAN** | Hierarchical Density-Based Spatial Clustering — finds clusters of varying density without needing to specify the number of clusters |
| **Latent dimension** | One of the 32 numbers in the embedding (z0 through z31) |
| **Well** | A single chamber on the MEA plate containing one neuronal culture |
| **Fano factor** | Variance/mean ratio of spike counts — measures burstiness |

---

## Collaborators

- **Roy Ben-Shalom** and **Shruti Shah** (UC Davis) — experimental design, data collection
- Analysis pipeline built with [HIPPIE](https://github.com/braingeneers/HIPPIE)
# folic_acid_hippie
