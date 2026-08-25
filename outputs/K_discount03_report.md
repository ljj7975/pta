# Experiment K — disagree_discount=0.3 — Statistical Results

Comparison baseline is plain PTA-CS (same convention as every prior experiment in this series). `patch_image.net` is the `flip_metrics_v2` mechanistic check (corrections minus regressions from adding the patch term on top of image-level PTA), pooled across the arm's 4-seed record sets per dataset -- must be > 0 for SUPPORT, same as Stage B's decision rule.

## topk20

| Dataset | PTA-CS (s1-4) | K-arm (s1-4) | Delta mean | p_sign | 95% CI | patch_image.net | Threshold | Verdict |
|---|---|---|---|---|---|---|---|---|
| dtd | 47.64 / 47.46 / 47.46 / 47.52 | 46.93 / 48.11 / 46.93 / 46.51 | -0.40pp | 0.8750 | [-0.89, +0.31] | -118 | 1.000pp | REFUTE |
| oxford_flowers | 74.46 / 74.79 / 74.26 / 74.99 | 72.27 / 72.47 / 71.74 / 72.68 | -2.33pp | 1.0000 | [-2.47, -2.22] | -1091 | 1.000pp | REFUTE |
| oxford_pets | 91.01 / 91.14 / 90.95 / 91.20 | 89.62 / 89.21 / 89.10 / 88.44 | -1.98pp | 1.0000 | [-2.53, -1.53] | -1466 | 1.000pp | REFUTE |

### topk20 — secondary comparison: vs. vanilla PatchModPTA-CS (no gate)

The more direct "did the gate help" question — same patch method, gate on vs. off — not gated by the (much harder, already-known) "beats plain PTA" bar above.

| Dataset | Vanilla PatchModPTA-CS (s1-4) | K-arm (s1-4) | Delta mean | p_sign | 95% CI |
|---|---|---|---|---|---|
| dtd | 46.75 / 48.23 / 47.22 / 46.63 | 46.93 / 48.11 / 46.93 / 46.51 | -0.09pp | 0.8125 | [-0.25, +0.10] |
| oxford_flowers | 71.66 / 72.39 / 72.39 / 72.43 | 72.27 / 72.47 / 71.74 / 72.68 | +0.07pp | 0.3750 | [-0.42, +0.48] |
| oxford_pets | 89.48 / 89.34 / 88.83 / 88.66 | 89.62 / 89.21 / 89.10 / 88.44 | +0.01pp | 0.4375 | [-0.17, +0.20] |

## otsu_mean

| Dataset | PTA-CS (s1-4) | K-arm (s1-4) | Delta mean | p_sign | 95% CI | patch_image.net | Threshold | Verdict |
|---|---|---|---|---|---|---|---|---|
| dtd | 47.64 / 47.46 / 47.46 / 47.52 | 46.99 / 48.05 / 46.93 / 46.51 | -0.40pp | 0.8750 | [-0.89, +0.28] | -115 | 1.000pp | REFUTE |
| oxford_flowers | 74.46 / 74.79 / 74.26 / 74.99 | 72.27 / 72.63 / 71.74 / 72.59 | -2.31pp | 1.0000 | [-2.46, -2.17] | -1075 | 1.000pp | REFUTE |
| oxford_pets | 91.01 / 91.14 / 90.95 / 91.20 | 89.59 / 89.32 / 89.29 / 88.47 | -1.91pp | 1.0000 | [-2.46, -1.52] | -1490 | 1.000pp | REFUTE |

### otsu_mean — secondary comparison: vs. vanilla PatchModPTA-CS (no gate)

The more direct "did the gate help" question — same patch method, gate on vs. off — not gated by the (much harder, already-known) "beats plain PTA" bar above.

| Dataset | Vanilla PatchModPTA-CS (s1-4) | K-arm (s1-4) | Delta mean | p_sign | 95% CI |
|---|---|---|---|---|---|
| dtd | 46.75 / 48.23 / 47.22 / 46.63 | 46.99 / 48.05 / 46.93 / 46.51 | -0.09pp | 0.8125 | [-0.25, +0.13] |
| oxford_flowers | 71.66 / 72.39 / 72.39 / 72.43 | 72.27 / 72.63 / 71.74 / 72.59 | +0.09pp | 0.3750 | [-0.42, +0.50] |
| oxford_pets | 89.48 / 89.34 / 88.83 / 88.66 | 89.59 / 89.32 / 89.29 / 88.47 | +0.09pp | 0.3750 | [-0.11, +0.34] |

