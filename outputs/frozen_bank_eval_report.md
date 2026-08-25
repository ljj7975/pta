# Frozen-Bank Evaluation Results (Experiments B + C)

See `outputs/offline_experiments_plan.md` for the full method and caveats. Raw per-dataset output: `outputs/frozen_bank_eval/*.json`.

## Experiment C: control (appearance weight) vs. TCR weight accuracy

| Dataset | n | Control acc. | TCR acc. | Delta |
|---|---|---|---|---|
| dtd | 1692 | 46.57% | 46.69% | +0.12pp |
| oxford_flowers | 2463 | 73.89% | 73.73% | -0.16pp |
| oxford_pets | 3669 | 90.79% | 90.43% | -0.35pp |

## Experiment B: does appearance weight track purity?

| Dataset | Clusters touched | Overall purity | Low-appearance tercile | Mid-appearance tercile | High-appearance tercile |
|---|---|---|---|---|---|
| dtd | 101 | 64.0% | 29.8% | 62.8% | 65.7% |
| oxford_flowers | 267 | 74.1% | 73.8% | 79.7% | 71.6% |
| oxford_pets | 250 | 84.9% | 83.5% | 87.8% | 83.3% |

