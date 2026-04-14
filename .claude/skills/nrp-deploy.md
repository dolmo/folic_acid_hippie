# /nrp-deploy - NRP Job Deployment

Deploy Kubernetes jobs on Nautilus Research Platform with smoke-gate validation pattern.

## Usage

```
/nrp-deploy [--method <method>] [--config <config>] [--datasets <list>] [--folds <range>]
            [--script <type>] [--dry-run] [--smoke-test] [--run-tag <tag>]
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Method to deploy: `hippie`, `physmap`, `nemo`, or `all` | `all` |
| `--config` | HIPPIE config name (from `ExperimentConfigs`) | `full_model` |
| `--datasets` | Comma-separated dataset list | `hull_cell_type,hausser_cell_type,lissberger_labeled_cell_type` |
| `--folds` | CV fold range (e.g., `0-4` or `0`) | `0-4` |
| `--script` | Training script: `transductive`, `holdout`, `bimodal` | `transductive` |
| `--dry-run` | Print YAML without submitting | `false` |
| `--smoke-test` | Use smoke-test epochs (small runs) | `false` |
| `--run-tag` | Tag for this benchmark run | Auto-generated timestamp |

## Key Scripts

- **Smoke test deployment:** `scripts/smoke_test_nrp/deploy_smoke_test.sh`
- **Scaling benchmark:** `scripts/scaling_benchmark_nrp/deploy_scaling_benchmark.sh`
- **Job templates:**
  - `scripts/smoke_test_nrp/jobdefinition_hippie.yaml`
  - `scripts/smoke_test_nrp/jobdefinition_physmap.yaml`
  - `scripts/smoke_test_nrp/jobdefinition_nemo.yaml`
  - `scripts/scaling_benchmark_nrp/jobdefinition_scaling.yaml`

## Smoke-Gate Pattern

**Always validate with a single job before launching full benchmarks:**

```bash
# 1. First: run one fold on one dataset as smoke test
/nrp-deploy --method hippie --datasets hull_cell_type --folds 0 --smoke-test --dry-run

# 2. If dry-run looks good, submit the smoke test
/nrp-deploy --method hippie --datasets hull_cell_type --folds 0 --smoke-test

# 3. Monitor until complete
/nrp-monitor --status all

# 4. Only after smoke passes, launch full benchmark
/nrp-deploy --method all --folds 0-4 --run-tag rebuttal-final
```

## Environment Variables

These can be set in `secrets.sh` or passed via environment:

```bash
AWS_ACCESS_KEY_ID       # Required
AWS_SECRET_ACCESS_KEY   # Required
S3_ENDPOINT             # Default: http://rook-ceph-rgw-nautiluss3.rook
S3_BUCKET               # Default: braingeneersdev
DATASETS_PREFIX         # Default: jgf/unified_datasets
RESULTS_PREFIX          # Default: jgf/results/rebuttal_benchmark
WANDB_API_KEY           # Optional (for W&B logging)
```

## Output Paths

Results are uploaded to:
```
s3://braingeneersdev/jgf/results/rebuttal_benchmark/<run_tag>/<method>/<config>/<dataset>/fold_<i>/
```

Each fold directory contains:
- `timings.csv` - Compute parity timing data
- `predictions.csv` - Model predictions
- `embeddings.npy` - Learned embeddings (optional)
- `confusion_matrix.png` - Per-fold confusion matrix

## Examples

```bash
# Dry-run to see generated YAML
/nrp-deploy --dry-run --method hippie --datasets hull_cell_type

# Submit smoke test for all methods
/nrp-deploy --smoke-test --method all --folds 0

# Full benchmark with custom run tag
/nrp-deploy --method all --folds 0-4 --run-tag rebuttal-v2

# HIPPIE-only with specific config
/nrp-deploy --method hippie --config waveform_only --folds 0-4

# Bimodal ablation (waveform + ISI only)
/nrp-deploy --method hippie --script bimodal --folds 0-4
```

## Related Skills

- `/preflight` - Run before deploying
- `/nrp-monitor` - Monitor job status
- `/s3-data` - Verify data availability
