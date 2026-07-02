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

## Experiment 4: Full Fusion (Three-Branch, Fixed Weights)

**File**: `models/exp4_full_fusion.py`  
**Config**: `configs_exp4/*.yaml`

### Key change vs Exp3

| Aspect | Exp3 (Gaussian) | Exp4 (Full Fusion) |
|---|---|---|
| Logit branches | 2 (text + patch-level Gaussian) | 3 (text + image-level PTA + patch-level Gaussian) |
| Image-level prototype | None | PTA-style EMA per class, permissive update gate |
| Patch-level prototype | Gaussian (center + variance) | Same as Exp3, but with strict confidence + margin gate |
| Fusion weights | `text_logits + tau_proto * alpha * quality_gate * delta_proto` | `clip_logits + 100.0 * image_proto_logits + tau_proto * proto_alpha * quality_gate * patch_proto_logits` |
| Update gate (image-level) | N/A | Permissive: any class with softmax prob ≥ 0.1 |
| Update gate (patch-level) | Confidence threshold only | Strict: top-1 CLIP confidence ≥ `conf_threshold` **AND** margin ≥ `conf_margin_threshold` |

### Motivation

Exp3 showed that Gaussian prototypes alone don't close the gap to PTA. Exp4 hypothesizes that combining *both* image-level prototypes (PTA's strength: stable, per-sample EMA) *and* patch-level Gaussian prototypes (MPTA's strength: multi-modal visual concepts) may capture complementary signals. The image-level branch provides a strong baseline signal, while the patch-level branch adds fine-grained discrimination for complex classes.

The dual-gate strategy is intentional: the image-level branch updates frequently (permissive gate) to maintain responsive prototypes, while the patch-level branch updates conservatively (strict gate) to avoid corrupting Gaussian parameters with noisy samples.

### Config

```yaml
# Image-level prototype (PTA)
alpha: 0.01
T: 50.0

# Patch-level Gaussian prototype (Exp4)
match_threshold: 0.60
max_K: 100
conf_threshold: 0.5
conf_margin_threshold: 0.05    # ← Exp4-specific
n_half: 15.0
soft_nn_top_m: 4
proto_alpha_max: 0.2
quality_eps: 0.001
exclude_pos: false
patch_group_threshold: 0.9
tau_proto: 20.0
gaussian_ema: 0.1
variance_min: 0.001
variance_max: 1.0
```

---

## Experiment 5: Tunable Fusion (Three-Branch, Configurable Weights)

**File**: `models/exp5_tunable_fusion.py`  
**Config**: `configs_exp4/*.yaml` (shares config directory with Exp4; reads `tau_*` params from YAML)

### Key change vs Exp4

| Aspect | Exp4 (Full Fusion) | Exp5 (Tunable Fusion) |
|---|---|---|
| Fusion weights | Hardcoded: `clip_logits + 100.0 * image_proto + tau_proto * alpha * quality * patch_proto` | Configurable: `tau_text * clip + tau_image_proto * image_proto + tau_patch_proto * alpha * quality * patch_proto` |
| `tau_text` | Implicitly `1.0` (hardcoded) | Read from YAML (default: `1.0`) |
| `tau_image_proto` | Hardcoded `100.0` | Read from YAML (default: `100.0`) |
| `tau_patch_proto` | Uses `tau_proto` from config | Read from YAML as `tau_patch_proto` (default: `20.0`) |
| Prototype systems | Same dual system (image-level + patch-level) | Same dual system (identical to Exp4) |

### Motivation

Exp4's hardcoded `100.0` multiplier on image-level prototypes works well but is not principled — it was inherited from the original PTA implementation. Exp5 exposes all three branch weights as configurable hyperparameters, enabling systematic ablation of the relative contribution of each signal without code changes. This also makes it easy to test whether patch-level prototypes add value when image-level prototypes are already strong (by varying `tau_patch_proto` independently).

### Config

```yaml
# Same as Exp4, plus tunable tau weights:
tau_text: 1.0              # ← Exp5-specific (default)
tau_image_proto: 100.0     # ← Exp5-specific (default; replaces hardcoded 100.0)
tau_patch_proto: 20.0      # ← Exp5-specific (default; replaces tau_proto)
```

---

## Results

### Cross-Domain Generalization (ViT-B/16, 7 datasets)

```
Method                     caltech101      dtd   eurosat      fgvc   oxford_flowers   oxford_pets    ucf101       Avg
------------------------------------------------------------------------------------------------------------------------
PTA                             95.01    47.81     61.72     25.59            74.67         91.28     73.28     67.05
MultiProtoPTA                   93.59    40.90     47.62     20.13            67.88         87.93     64.82     60.41
Exp1FixedUniquePatches          93.31    41.67     43.04     21.06            69.55         86.21     65.87     60.10
Exp2AdaptiveTauProto            94.12    44.27     46.47     24.63            71.38         89.02     66.64     62.36
Exp3GaussianPrototypes          94.12    44.50     47.74     24.81            71.38         89.04     66.64     62.60
Exp5TunableFusion               94.97    47.64     61.93     25.68            74.58         91.17     72.91     66.98
Exp4FullFusion                  94.97    47.75     61.73     25.47            74.71         91.14     72.96     66.96
```

### Observations

1. **Exp4 and Exp5 match PTA** (66.96–66.98% vs 67.05%). The three-branch fusion strategy successfully closes the gap that earlier MPTA variants left open.
2. **Exp4 ≈ Exp5 ≈ PTA > Exp3 > Exp2 > Exp1 > MPTA base**. The ranking is clear: combining image-level + patch-level prototypes recovers all performance, while patch-level-only variants (Exp1–Exp3) lag behind.
3. **Exp5 (Tunable) slightly edges Exp4 (Fixed)** on average (66.98% vs 66.96%), but the difference is negligible — the default tunable weights (`tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=20.0`) produce nearly identical behavior to Exp4's hardcoded weights.
4. **The image-level prototype branch is the key signal**. Exp4/Exp5's permissive image-level updates (softmax ≥ 0.1) provide a strong, stable baseline that patch-level Gaussian prototypes refine. Exp3 (patch-level only) achieves only 62.60%, confirming that patch-level prototypes alone are insufficient.
5. **Exp1 (Fixed 15 patches)** remains the worst variant (60.10%) — fixing the patch count discards too much information for complex scenes.
6. **All methods agree on dataset difficulty**: `fgvc` is hardest (20–26%), `caltech101` and `oxford_pets` are easiest (91–95%).

### Missing data

- `food101`, `stanford_cars`, `sun397` were not yet evaluated across all methods (pending Slurm array runs).

---

## Fusion Weight Analysis

### Exact fusion formulas

| Method | Formula |
|---|---|
| **PTA** | `final = 1.0 × clip_logits + 100.0 × image_proto_logits` |
| **Exp4** | `final = 1.0 × clip_logits + 100.0 × image_proto_logits + 20.0 × proto_alpha × quality_gate × patch_proto_logits` |
| **Exp5** | `final = tau_text × clip_logits + tau_image_proto × image_proto_logits + tau_patch_proto × proto_alpha × quality_gate × patch_proto_logits` |

### Effective weights (Exp4 / Exp5 defaults)

| Branch | Nominal Weight | Effective Weight | Notes |
|---|---|---|---|
| `clip_logits` | 1.0 | ~1.0 | Raw CLIP zero-shot cosine similarities (range ≈ [-1, 1], scaled by temperature ~100) |
| `image_proto_logits` | **100.0** | **~100.0** | Refined prototype cosine similarities — the **dominant signal** |
| `patch_proto_logits` | 20.0 × 0.2 × ~0.8 | **~3.2** (max) | `tau_proto × proto_alpha_max × quality_gate`; typically much smaller |

The `proto_alpha` term is computed dynamically via `_alpha_from_evidence()` and caps at `proto_alpha_max = 0.2`. The `quality_gate` is `proto_var / (proto_var + 0.001)`, typically **0.5–1.0** depending on prototype variance. So the effective patch-level weight is at most **~3.2**, and often lower.

### How much does each prototype branch impact the decision?

The **image-level prototype dominates** (~99% of prototype influence):

- The `100.0` multiplier on `image_proto_logits` is inherited from PTA and exists because both `clip_logits` and `image_proto_logits` are cosine similarities, but the refined prototype signal needs amplification to compete with the raw CLIP temperature scaling.
- The patch-level Gaussian prototypes contribute a **small correction** (~1–3% effective weight), acting as a **tiebreaker** for ambiguous samples rather than a primary signal.
- This explains why Exp4/Exp5 match PTA so closely (66.96–66.98% vs 67.05%) — they are essentially running PTA with a small patch-level bonus on top.

### Why does the `100.0` multiplier exist?

Both `clip_logits` and `image_proto_logits` are computed as `image_features @ text_features.T` (cosine similarity). The CLIP model temperature (~100) already scales raw logits, but the refined prototype features need the same amplification to be competitive. Without `100.0`, the prototype branch would be swamped by the raw CLIP signal.

### Implications for future experiments

- **Ablating `tau_image_proto`** (via Exp5) will reveal how much the image-level prototype contributes vs. raw CLIP. Setting it to `0.0` should reproduce patch-only behavior (similar to Exp3).
- **Ablating `tau_patch_proto`** will confirm whether patch-level prototypes add measurable value. Setting it to `0.0` should reproduce PTA exactly.
- **Increasing `tau_patch_proto`** (e.g., to 50–100) may overfit to patch-level noise — the Gaussian prototypes are less stable than image-level EMA prototypes.

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
├── exp4_full_fusion.py          # Exp4: Three-branch fusion (fixed weights)
├── exp5_tunable_fusion.py       # Exp5: Three-branch fusion (configurable tau weights)
```

```
configs_exp1/   # Configs for Exp1 — adds n_unique_patches_per_sample
configs_exp2/   # Configs for Exp2 — adds blend_text_weight, blend_proto_weight, min_protos_for_full_agg
configs_exp3/   # Configs for Exp3 — adds gaussian_ema, variance_min, variance_max
configs_exp4/   # Configs for Exp4/Exp5 — adds conf_margin_threshold, tau_text, tau_image_proto, tau_patch_proto
configs_mpta/   # Configs for base MultiProtoPTA (created later, 2026-07-01)
```

## Slurm Scripts

```
scripts/
├── slurm_cd_benchmark_mpta_vit.sh   # Base MultiProtoPTA array job
├── slurm_cd_benchmark_exp1_vit.sh   # Exp1 array job
├── slurm_cd_benchmark_exp2_vit.sh   # Exp2 array job
├── slurm_cd_benchmark_exp3_vit.sh   # Exp3 array job
├── slurm_cd_benchmark_exp4_vit.sh   # Exp4/Exp5 array job (runs pta, exp4, exp5)
├── run_cd_benchmark_exp4_vit.sh     # Non-Slurm runner for Exp4/Exp5
└── run_cd_benchmark_exps_vit.sh     # Combined runner for all experiments
```
