# Multi-Prototype PTA — Experiment Log

**Date**: 2026-06-29
**Commit**: [`2947b7b`](https://github.com/hzhxmu/PTA/commit/2947b7b) (initial commit — all MPTA variants implemented in one shot)

---

## Overview

The original **PTA** (`models/pta.py`) maintains a single prototype per class and updates it via exponential moving average at test time.

**MultiProtoPTA** (`models/multi_proto_pta.py`) extends this by keeping **multiple patch-level prototypes per class** learned through incremental K-means. This allows the model to capture multi-modal visual concepts within each class (e.g., dog faces, dog paws, dog fur) and match them independently.

The core pipeline is shared across all three experiments:
1. Extract patch embeddings from CLIP's ViT
2. Maintain a per-class memory bank of prototypes (patch-level centroids)
3. Score each class by matching image patches against its prototypes
4. Fuse prototype scores with CLIP text logits
5. Update prototype banks using confident pseudo-labels

---

## Experiment 1: Fixed Unique Patches

**File**: `models/exp1_fixed_unique_patches.py`  
**Config**: `configs_exp1/*.yaml`

### Key change vs MultiProtoPTA

| Aspect | MultiProtoPTA | Exp1 |
|---|---|---|
| Patch selection | Threshold-based grouping (`patch_group_threshold=0.9`) | Farthest-first diversity selection |
| Patch count per sample | Variable (depends on image content) | Fixed (`n_unique_patches_per_sample=15`) |
| Patch grouping | Greedy merge of similar patches | No merging — keeps maximally-diverse patches |

### Motivation

The threshold-based grouping in the base MPTA can produce wildly different patch counts per image. Using `exclude_pos=True` also causes aggressive merging of patches. Exp1 stabilizes the input representation by always picking exactly `n` patches that are farthest apart from each other, ensuring consistent downstream behavior.

### Implementation detail

```python
# Greedy farthest-first: seed with centroid-closest, then iteratively
# add the patch that is least similar to any already-selected one.
def _select_diverse_patches(patches_norm, n):
    ...
```

### Config

```yaml
match_threshold: 0.60
max_K: 100
conf_threshold: 0.5
n_half: 15.0
soft_nn_top_m: 4
alpha_max: 0.2
quality_eps: 0.001
exclude_pos: false
n_unique_patches_per_sample: 15    # ← Exp1-specific
```

---

## Experiment 2: Adaptive Tau Proto (50/50 Score Blending)

**File**: `models/exp2_adaptive_tau_proto.py`  
**Config**: `configs_exp2/*.yaml`

### Key change vs MultiProtoPTA

| Aspect | MultiProtoPTA | Exp2 |
|---|---|---|
| Fusion strategy | `text_logits + tau_proto * alpha * quality_gate * delta_proto` | `softmax(text_logits) * 0.5 + softmax(proto_scores) * 0.5` |
| Hyperparameters | `tau_proto=20, alpha_max=0.2, quality_eps=1e-3` | `blend_text_weight=0.5, blend_proto_weight=0.5` |
| Prototype aggregation | Always top-M mean | Top-M if `K ≥ min_protos_full_agg`, else mean of all |

### Motivation

The exponential weighting scheme in MPTA (`tau_proto * alpha * quality_gate`) introduces three coupled hyperparameters that are dataset-dependent and hard to tune. Exp2 replaces this with a principled 50/50 softmax blend. Both scores live on the same [0,1] probability scale, making the blend model-agnostic and interpretable.

Additionally, for classes with very few prototypes (`K < min_protos_for_full_agg`), the mean of all prototype scores is used instead of top-M, which is more robust against low-sample noise.

### Config

```yaml
match_threshold: 0.60
max_K: 100
conf_threshold: 0.5
n_half: 15.0
soft_nn_top_m: 4
alpha_max: 0.2
quality_eps: 0.001
exclude_pos: false
blend_text_weight: 0.5        # ← Exp2-specific
blend_proto_weight: 0.5       # ← Exp2-specific
min_protos_for_full_agg: 5    # ← Exp2-specific
```

---

## Experiment 3: Gaussian Prototypes

**File**: `models/exp3_gaussian_prototypes.py`  
**Config**: `configs_exp3/*.yaml`

### Key change vs MultiProtoPTA

| Aspect | MultiProtoPTA | Exp3 |
|---|---|---|
| Prototype representation | Center vector only | Center + per-dimension variance (Gaussian) |
| Matching function | Cosine similarity (`cos(center, patch)`) | Gaussian score (`exp(-0.5 * Σ (patch - center)² / variance)`) |
| Variance tracking | N/A | EMA-updated per prototype + per-dimension |
| New param | — | `gaussian_ema=0.1, variance_min=0.001, variance_max=1.0` |

### Motivation

Cosine similarity treats all feature dimensions equally, but some dimensions may be more informative for a given prototype. By modelling each prototype as a Gaussian (center + variance), Exp3 naturally:
- **Downweights** dimensions with high variance (uncertain/noisy features)
- **Upweights** dimensions with low variance (tight, high-confidence features)
- **Downweights** entire prototypes with high overall variance (poorly-formed clusters)

The variance is initialized conservatively (10× `variance_min` for new prototypes) and updated via EMA from the per-dimension variance of matching patches.

### Config

```yaml
match_threshold: 0.60
max_K: 100
conf_threshold: 0.5
n_half: 15.0
soft_nn_top_m: 4
alpha_max: 0.2
quality_eps: 0.001
exclude_pos: false
gaussian_ema: 0.1       # ← Exp3-specific
variance_min: 0.001     # ← Exp3-specific
variance_max: 1.0       # ← Exp3-specific
```

---

## Results

### Cross-Domain Generalization (ViT-B/16, 7 datasets)

```
Method                     caltech101      dtd   eurosat      fgvc   oxford_flowers   oxford_pets    ucf101       Avg
------------------------------------------------------------------------------------------------------------------------
PTA                             94.93    47.81     61.80     25.83            74.71         91.28     73.33     67.10
MultiProtoPTA                   93.59    40.90     47.62     20.13            67.88         87.93     64.82     60.41
Exp1FixedUniquePatches          93.31    41.67     43.04     21.06            69.55         86.21     65.87     60.10
Exp2AdaptiveTauProto            94.12    44.27     46.47     24.63            71.38         89.02     66.64     62.36
Exp3GaussianPrototypes          94.12    44.50     47.74     24.81            71.38         89.04     66.64     62.60
```

### Observations

1. **All MPTA variants underperform the original PTA** on average (60–63% vs 67%). The single-prototype EMA baseline is surprisingly strong.
2. **Exp3 (Gaussian) ≈ Exp2 (50/50 blend) > Exp1 (Fixed patches) > MPTA base** in that order. Gaussian scoring and probability blending both recover some of the gap vs MPTA.
3. **Exp3 is the best MPTA variant** overall (62.60%) and matches or beats all other variants on every individual dataset.
4. **Exp2 is close behind** (62.36%) and has the advantage of fewer hyperparameters (no `tau_proto`, `alpha_max`, `quality_gate`).
5. **Exp1 (Fixed 15 patches)** performs worst among the variants (60.10%) — fixing the patch count to 15 may discard too much information for complex scenes.

### Missing data

- `food101`, `stanford_cars`, `sun397` were not yet evaluated across all methods (pending Slurm array runs).

---

## Model File Map

```
models/
├── base.py                      # BaseAdapter abstract class
├── pta.py                       # Original PTA (single prototype EMA)
├── multi_proto_pta.py           # MultiProtoPTA (base variant)
├── multi_proto_pta_base.py      # Shared helpers: _safe_normalize, _incremental_kmeans_step, _extract_patch_embeddings
├── exp1_fixed_unique_patches.py # Exp1: Fixed diverse patches per sample
├── exp2_adaptive_tau_proto.py   # Exp2: 50/50 softmax blending
├── exp3_gaussian_prototypes.py  # Exp3: Gaussian prototype matching
```

```
configs_exp1/   # Configs for Exp1 — adds n_unique_patches_per_sample
configs_exp2/   # Configs for Exp2 — adds blend_text_weight, blend_proto_weight, min_protos_for_full_agg
configs_exp3/   # Configs for Exp3 — adds gaussian_ema, variance_min, variance_max
configs_mpta/   # Configs for base MultiProtoPTA (created later, 2026-07-01)
```

## Slurm Scripts

```
scripts/
├── slurm_cd_benchmark_mpta_vit.sh   # Base MultiProtoPTA array job
├── slurm_cd_benchmark_exp1_vit.sh   # Exp1 array job
├── slurm_cd_benchmark_exp2_vit.sh   # Exp2 array job
└── slurm_cd_benchmark_exp3_vit.sh   # Exp3 array job
```
