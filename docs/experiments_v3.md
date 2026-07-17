# Experiment Set 3 — Three-Branch Fusion Family (Exp4–Exp12)

Results from the three-branch fusion experiments: CLIP text + image-level EMA prototype + patch-level Gaussian prototype, tested on 6 CD datasets (ViT-B/16).

**Source**: `old_outputs/exp_results.txt`

---

## Overview

All methods in this set extend the two-branch architecture (CLIP text + image-level EMA prototype) with a third branch: **patch-level Gaussian prototypes**. The image-level and patch-level prototypes are independent — both extract information from the same image but at different granularities.

### Adapter Comparison

| Adapter | Method Name | Fusion | Update Gate | Key Change |
|---------|-------------|--------|-------------|------------|
| `pta` | Baseline PTA | Two-branch (text + image proto) | Binary top-1 | Original PTA — no patch-level prototypes |
| `exp4_full_fusion` | Exp4FullFusion | Three-branch (fixed weights) | Dual gate (permissive image, strict patch) | Hardcoded `tau_image_proto=100`, `tau_patch_proto=20` |
| `exp5_tunable_fusion` | Exp5TunableFusion | Three-branch (configurable) | Binary top-1 | Exposes `tau_text`, `tau_image_proto`, `tau_patch_proto` |
| `exp7_inverted_image_weight` | Exp7InvertedImageWeight | Three-branch + adaptive per-class | Binary top-1 | `tau_image[c]` proportional to `proto_alpha[c]` (inverted) |
| `exp8_boosted_patch` | Exp8BoostedPatch | Three-branch (boosted patch) | Binary top-1 | `tau_patch_proto=500` (25× default) |
| `exp10_soft_patch_gate` | Exp10SoftPatchGate | Three-branch (configurable) | Soft per-class proportional | Every class with `softmax_prob > threshold` updates |
| `exp11_tunable_fusion_new` | Exp11TunableFusionNew | Three-branch (configurable) | Binary top-1 | Control/repeat of Exp5 with new config naming |
| `exp12_patch_quality_modulation` | Exp12PatchQualityModulation | Three-branch (configurable) | Binary top-1 | Patch quality modulates image-level EMA update rate |
| `exp6_adaptive_image_weight` | Exp6AdaptiveImageWeight | Three-branch + adaptive per-class | Binary top-1 | `tau_image[c]` inversely proportional to `proto_alpha[c]` |

---

## Results

### Cross-Domain Generalization (ViT-B/16, 6 datasets)

| Method | caltech101 | dtd | eurosat | fgvc | oxford_flowers | oxford_pets | **Avg** | Δ vs PTA |
|--------|:----------:|:---:|:-------:|:----:|:--------------:|:-----------:|:-------:|:--------:|
| PTA (baseline) | 94.97 | 47.81 | 61.78 | 25.86 | 74.71 | 91.41 | **66.09** | — |
| Exp4FullFusion | 94.97 | 47.75 | 61.88 | 25.50 | 74.71 | 91.09 | **65.98** | −0.11 |
| Exp5TunableFusion | 94.93 | 47.64 | 61.93 | 25.59 | 74.58 | 91.11 | **65.96** | −0.13 |
| Exp7InvertedImageWeight | 94.73 | 41.02 | 57.52 | 20.10 | 73.69 | 90.76 | **62.97** | −3.12 |
| Exp8BoostedPatch | 94.97 | 47.64 | 61.84 | 25.74 | 74.34 | 91.17 | **65.95** | −0.14 |
| Exp10SoftPatchGate | 94.93 | 47.70 | 61.68 | 25.80 | 74.71 | 91.33 | **66.02** | −0.07 |
| Exp11TunableFusionNew | 94.93 | 47.70 | 61.81 | 25.68 | 74.75 | 91.39 | **66.04** | −0.05 |
| Exp12PatchQualityModulation | 94.89 | 47.99 | 61.84 | 25.83 | 74.58 | 91.39 | **66.09** | 0.00 |
| Exp6AdaptiveImageWeight | 94.48 | 36.70 | 36.69 | 22.05 | 55.87 | 90.95 | **56.12** | −9.97 |

### Delta Ranking (Δ Avg vs PTA)

| Method | Δ Avg vs PTA | Verdict |
|--------|:-----------:|---------|
| Exp12PatchQualityModulation | **0.00** | Tied with baseline — best of the variants |
| Exp11TunableFusionNew | **−0.05** | Negligible — within noise |
| Exp10SoftPatchGate | **−0.07** | Negligible — within noise |
| Exp4FullFusion | **−0.11** | Marginal degradation |
| Exp5TunableFusion | **−0.13** | Marginal degradation |
| Exp8BoostedPatch | **−0.14** | Marginal degradation — 25× boost didn't help |
| Exp7InvertedImageWeight | **−3.12** | Significant degradation |
| Exp6AdaptiveImageWeight | **−9.97** | Catastrophic degradation |

---

## Headline Findings

1. **No experiment beats PTA.** The best result is Exp12PatchQualityModulation (66.09), exactly tied with PTA (66.09). The three-branch fusion architecture provides no measurable advantage over the two-branch baseline on 6-dataset average.

2. **Exp12 (Patch Quality Modulation) is the best variant** — it matches PTA exactly (66.09 vs 66.09). Using patch-level quality to modulate the image-level EMA update rate is the only approach that doesn't degrade performance.

3. **Exp10 (Soft Patch Gate) is nearly neutral** (66.02, −0.07). Replacing binary top-1 update with soft per-class proportional gate is harmless but not beneficial.

4. **Exp4/Exp5/Exp8 cluster tightly** (65.95–65.98). All three-branch variants with standard fusion perform nearly identically, with marginal degradation from the added patch branch.

5. **Exp11 (Exp5 repeat) confirms consistency** (66.04 vs 65.96 for original Exp5). Small difference is within expected variance from random seeds.

6. **Exp7 (Inverted Image Weight) degrades significantly** (−3.12). Amplifying image-level weight when patch evidence is abundant hurts — the inverse hypothesis is wrong.

7. **Exp6 (Adaptive Image Weight) fails catastrophically** (−9.97). Downweighting the image-level branch for classes with many prototypes is destructive. Worst hit: eurosat (−25.09pp), oxford_flowers (−18.84pp), dtd (−11.11pp).

8. **25× patch boost (Exp8) doesn't help** (−0.14). Despite making patch-level weight competitive with image-level, performance barely changes. Patch-level Gaussian prototypes don't carry enough discriminative signal.

9. **All methods agree on dataset difficulty**: fgvc is hardest (20–26%), caltech101 and oxford_pets are easiest (90–95%).

---

## Individual Experiment Analyses

### Exp6: Adaptive Image Weight (Catastrophic Failure)

**File**: `models/exp6_adaptive_image_weight.py`  
**Config**: `configs_exp6/*.yaml`

#### Hypothesis

When a class has many high-quality patch-level prototypes (high `proto_alpha[c]`), the image-level signal is redundant. When a class has few prototypes (low `proto_alpha[c]`), the image-level EMA should be weighted heavily.

#### Key Change

Per-class adaptive image-level weight:
```
tau_image[c] = tau_image_max - (tau_image_max - tau_image_min) * proto_alpha[c]
```

- `proto_alpha[c] ≈ 0` → `tau_image[c] ≈ tau_image_max` (100)
- `proto_alpha[c] ≈ alpha_max` → `tau_image[c] ≈ tau_image_min` (50)

#### Per-Dataset Breakdown

| Dataset | PTA | Exp6 | Δ |
|---------|:---:|:----:|:-:|
| caltech101 | 94.97 | 94.48 | −0.49 |
| dtd | 47.81 | 36.70 | **−11.11** |
| eurosat | 61.78 | 36.69 | **−25.09** |
| fgvc | 25.86 | 22.05 | −3.81 |
| oxford_flowers | 74.71 | 55.87 | **−18.84** |
| oxford_pets | 91.41 | 90.95 | −0.46 |

#### Analysis

The adaptive weighting scheme is **uniformly harmful** across all 6 datasets, with catastrophic degradation on eurosat, oxford_flowers, and dtd. Only caltech101 and oxford_pets (high-base datasets) are minimally affected.

- **Over-squashing**: Reducing `tau_image` from 100→50 for high-α classes cuts the image-level signal in half. Patch-level prototypes cannot compensate.
- **Wrong proxy**: `proto_alpha[c]` (sample count) may not correlate with "how much to trust image-level vs. patch-level."
- **The image-level prototype is the dominant signal** (~99% of prototype influence). Reducing it below 100 causes severe degradation.

#### Recommendation

**Do not use.** The approach is fundamentally flawed. See `NEW_EXPERIMENTS.md` for detailed ablation.

---

### Exp7: Inverted Image Weight (Significant Degradation)

**File**: `models/exp7_inverted_image_weight.py`  
**Config**: `configs_exp7/*.yaml`

#### Hypothesis

When a class has abundant prototype evidence (high `proto_alpha[c]`), the image-level prototype should be **amplified** because it has been refined by many samples and is more trustworthy.

#### Key Change

Inverted adaptive image-level weight:
```
tau_image[c] = tau_image_min + (tau_image_max - tau_image_min) * proto_alpha[c]
```

- `proto_alpha[c] ≈ 0` → `tau_image[c] ≈ tau_image_min` (100)
- `proto_alpha[c] ≈ alpha_max` → `tau_image[c] ≈ tau_image_max` (150)

#### Per-Dataset Breakdown

| Dataset | PTA | Exp7 | Δ |
|---------|:---:|:----:|:-:|
| caltech101 | 94.97 | 94.73 | −0.24 |
| dtd | 47.81 | 41.02 | **−6.79** |
| eurosat | 61.78 | 57.52 | −4.26 |
| fgvc | 25.86 | 20.10 | −5.76 |
| oxford_flowers | 74.71 | 73.69 | −1.02 |
| oxford_pets | 91.41 | 90.76 | −0.65 |

#### Analysis

Amplifying image-level weight when patch evidence is abundant hurts performance, particularly on dtd (−6.79pp), fgvc (−5.76pp), and eurosat (−4.26pp). The inverse hypothesis was wrong: more prototypes do not mean a more trustworthy image-level signal.

- The higher `tau_image_max=150` pushes the image-level weight above the baseline `100`, which overshoots for already-well-represented classes.
- The degradation pattern mirrors Exp6 but less severe — both per-class adaptive schemes harm performance.

---

### Exp12: Patch Quality Modulation (Best Variant)

**File**: `models/exp12_patch_quality_modulation.py`  
**Config**: `configs_exp12/*.yaml`

#### Hypothesis

Patch-level quality can modulate the image-level EMA update rate: high-quality patch evidence → faster prototype refinement; noisy patch evidence → default update speed.

#### Key Change

Quality-gated EMA update:
```
w_new *= (1 + quality_modulation * quality_gate)
```

Pipeline reordered: patch extraction → quality gate → image-level update (modulated) → fusion.

#### Per-Dataset Breakdown

| Dataset | PTA | Exp12 | Δ |
|---------|:---:|:-----:|:-:|
| caltech101 | 94.97 | 94.89 | −0.08 |
| dtd | 47.81 | 47.99 | +0.18 |
| eurosat | 61.78 | 61.84 | +0.06 |
| fgvc | 25.86 | 25.83 | −0.03 |
| oxford_flowers | 74.71 | 74.58 | −0.13 |
| oxford_pets | 91.41 | 91.39 | −0.02 |

#### Analysis

Exp12 matches PTA exactly (66.09 vs 66.09) — the only variant that doesn't degrade. The quality-gated EMA modulation provides a small positive on dtd (+0.18) and eurosat (+0.06), offset by small negatives elsewhere. The net effect is zero — the modulation is neutral.

This is the same finding as the post-refactoring `QualityGateOnly` experiment: the quality gate's contribution to image-level EMA modulation is the entire benefit of patch-level analysis, not the patch logits in fusion.

---

### Exp8: Boosted Patch (25× tau_patch_proto)

**File**: `models/exp8_boosted_patch.py`  
**Config**: `configs_exp8/*.yaml`

#### Hypothesis

The default `tau_patch_proto=20` produces an effective patch weight of ~3.2, which is dwarfed by the image-level weight (~100). Boosting `tau_patch_proto` to 500 makes the patch branch competitive.

#### Key Change

`tau_patch_proto = 500.0` (25× default) — all other logic identical to Exp5.

#### Per-Dataset Breakdown

| Dataset | PTA | Exp8 | Δ |
|---------|:---:|:----:|:-:|
| caltech101 | 94.97 | 94.97 | 0.00 |
| dtd | 47.81 | 47.64 | −0.17 |
| eurosat | 61.78 | 61.84 | +0.06 |
| fgvc | 25.86 | 25.74 | −0.12 |
| oxford_flowers | 74.71 | 74.34 | −0.37 |
| oxford_pets | 91.41 | 91.17 | −0.24 |

#### Analysis

Despite a 25× increase, performance barely changes (−0.14). The `proto_alpha` and `quality_gate` terms still cap the effective weight, and the Gaussian prototypes don't add enough discriminative signal to move the needle — or they introduce noise that cancels out.

---

### Exp10: Soft Patch Gate

**File**: `models/exp10_soft_patch_gate.py`  
**Config**: `configs_exp10/*.yaml`

#### Hypothesis

The binary top-1 update gate discards signal from plausible-but-not-top classes. A soft per-class proportional gate should capture more useful update signal.

#### Key Change

Soft per-class proportional update gate:
```python
for c in range(C):
    weight = softmax_probs[c]
    if weight > soft_gate_threshold:
        relaxed_thresh = match_threshold - weight * 0.1
        update_prototype(class=c, threshold=relaxed_thresh)
```

#### Per-Dataset Breakdown

| Dataset | PTA | Exp10 | Δ |
|---------|:---:|:-----:|:-:|
| caltech101 | 94.97 | 94.93 | −0.04 |
| dtd | 47.81 | 47.70 | −0.11 |
| eurosat | 61.78 | 61.68 | −0.10 |
| fgvc | 25.86 | 25.80 | −0.06 |
| oxford_flowers | 74.71 | 74.71 | 0.00 |
| oxford_pets | 91.41 | 91.33 | −0.08 |

#### Analysis

SoftPatchGate is **effectively indistinguishable from baseline** (66.02 vs 66.09). The soft gate introduces no degradation but also no improvement. The binary top-1 gate is not the bottleneck.

---

## Effective Weight Analysis

### Fusion Formulas

| Method | Formula |
|---|---|
| **PTA** | `final = 1.0 × clip + 100.0 × image_proto` |
| **Exp4** | `final = 1.0 × clip + 100.0 × image_proto + 20.0 × proto_alpha × quality_gate × patch_proto` |
| **Exp5** | `final = tau_text × clip + tau_image_proto × image_proto + tau_patch_proto × proto_alpha × quality_gate × patch_proto` |
| **Exp6** | Same as Exp5 but `tau_image[c] = tau_image_max - (tau_image_max - tau_image_min) × proto_alpha[c]` |
| **Exp7** | Same as Exp5 but `tau_image[c] = tau_image_min + (tau_image_max - tau_image_min) × proto_alpha[c]` |
| **Exp8** | Same as Exp5 but `tau_patch_proto = 500.0` |
| **Exp10** | Same as Exp5 fusion, soft per-class update gate |
| **Exp11** | Identical to Exp5 (control run) |
| **Exp12** | Same as Exp5 fusion, image-level update modulated: `w_new *= (1 + quality_modulation × quality_gate)` |

### Effective Weights (Exp4/Exp5 defaults)

| Branch | Nominal Weight | Effective Weight | Notes |
|---|---|---|---|
| `clip_logits` | 1.0 | ~1.0 | Raw CLIP zero-shot cosine similarities |
| `image_proto_logits` | **100.0** | **~100.0** | Refined prototype — **dominant signal** |
| `patch_proto_logits` | 20.0 × 0.2 × ~0.8 | **~3.2** (max) | `tau_proto × proto_alpha_max × quality_gate` |

The image-level prototype dominates (~99% of prototype influence). The patch-level branch contributes a small correction (~1–3%), acting as a tiebreaker for ambiguous samples.

---

## Cross-Reference

This experiment set is the pre-refactoring version of the three-branch fusion family. For the post-refactoring equivalents (using the pluggable `PatchModulatedPTA` adapter), see `NEW_EXPERIMENTS.md`.

For the full experiment documentation with all configurations, see `experiments.md`.

---

## Caveats

- **ucf101 missing**: This run covers 6 CD datasets only. ucf101 was evaluated separately in `old_outputs/exp_results_2.txt`.
- **No entropy fusion (Exp9)**: Exp9 was excluded from this run due to catastrophic failure in the full 7-dataset benchmark (30.05 avg).
- **Single run**: Each method was run once. No confidence intervals or multi-seed averaging.
- Results are from the pre-refactoring codebase — adapter code has since been reorganized into the pluggable architecture.
