# Prototype Separability — Variance-Aware Re-check (Experiment A)

Same question as `outputs/prototype_separability_report.md`, but using the live scoring formula (center *and* variance, scaled Mahalanobis + exponential from `utils/kmeans.py`) instead of plain cosine similarity between centers. Reuses `outputs/patch_bank_dumps/` — no new GPU runs.

Match score is in the same [0, 1] range the live method actually computes (1.0 = perfect match). Separability margin = intra-class match score minus nearest cross-class match score; negative means a class's own prototypes aren't a better match for each other than for their nearest neighbor in another class.

## All prototypes

| Dataset | Prototypes | Classes | Intra-class match (mean) | Nearest cross-class match (mean) | Background cross-class match (mean) | Separability margin | v1 (cosine-only) margin |
|---|---|---|---|---|---|---|---|
| dtd | 151 | 39 | 0.557 | 0.752 | 0.497 | -0.194 | -0.178 |
| oxford_flowers | 379 | 93 | 0.556 | 0.867 | 0.559 | -0.311 | -0.233 |
| oxford_pets | 294 | 37 | 0.512 | 0.797 | 0.499 | -0.285 | -0.218 |

## High-appearance-only prototypes (top half per class)

| Dataset | Prototypes | Classes | Intra-class match (mean) | Nearest cross-class match (mean) | Background cross-class match (mean) | Separability margin | v1 (cosine-only) margin |
|---|---|---|---|---|---|---|---|
| dtd | 128 | 39 | 0.593 | 0.771 | 0.533 | -0.178 | -0.121 |
| oxford_flowers | 281 | 93 | 0.626 | 0.880 | 0.639 | -0.254 | -0.130 |
| oxford_pets | 194 | 37 | 0.590 | 0.852 | 0.604 | -0.262 | -0.114 |

