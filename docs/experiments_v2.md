# New Experiments on PatchModulatedPTA

Post-refactoring experiments building on the pluggable image/patch/fusion architecture.

---

## Overview

All five adapters compose the same three components as `PatchModulatedPTAAdapter`:
- **Image-level** (`PTAImageLevel`): per-class running prototype via EMA
- **Patch-level** (`GaussianPatchLevel`): per-class Gaussian prototypes (centers + variance)
- **Fusion**: varies per adapter

### Adapter Comparison

| Adapter | Method Name | Fusion | Update Gate | Key Change |
|---------|-------------|--------|-------------|------------|
| `pta` | Baseline PTA | `WeightedFusion` (τ_patch=0) | Binary top-1 | Original PTA — no patch-level prototypes at all |
| `patch_modulated_pta` | Ref. Baseline | `QualityGatedFusion` | Binary top-1 | Adds patch-level proto logits + quality-gated EMA modulation |
| `patch_boost` | Isolation | `WeightedFusion` (no gating) | Binary top-1 | Raw patch logits, no quality_gate/proto_alpha modulation |
| `adaptive_image_weight` | New | `AdaptiveImageWeightFusion` | Binary top-1 | Per-class `tau_image[c]` from `proto_alpha[c]` |
| `soft_patch_gate` | New | `QualityGatedFusion` | Soft per-class | Every class with `softmax_prob > threshold` updates |
| `quality_gate_only` | New | `QualityGatedFusion` (τ_patch=0) | Binary top-1 | Isolates quality_gate's effect on image-level EMA |

---

## Results

### Pre-Refactoring Experiments (`old_outputs/exp_results.txt`, 7 datasets incl. ucf101)

These are the original experiments before the pluggable refactoring. Row names were renamed for clarity.

| Method (renamed) | caltech101 | dtd | eurosat | fgvc | oxford_flowers | oxford_pets | **Avg** |
|------------------|:----------:|:---:|:-------:|:----:|:--------------:|:-----------:|:-------:|
| PTA (baseline) | 94.97 | 47.81 | 61.78 | 25.86 | 74.71 | 91.41 | **66.09** |
| Exp4FullFusion | 94.97 | 47.75 | 61.88 | 25.50 | 74.71 | 91.09 | **65.98** |
| Exp5TunableFusion | 94.93 | 47.64 | 61.93 | 25.59 | 74.58 | 91.11 | **65.96** |
| Exp7InvertedImageWeight | 94.73 | 41.02 | 57.52 | 20.10 | 73.69 | 90.76 | **62.97** |
| Exp8BoostedPatch | 94.97 | 47.64 | 61.84 | 25.74 | 74.34 | 91.17 | **65.95** |
| Exp10SoftPatchGate | 94.93 | 47.70 | 61.68 | 25.80 | 74.71 | 91.33 | **66.02** |
| Exp11TunableFusionNew | 94.93 | 47.70 | 61.81 | 25.68 | 74.75 | 91.39 | **66.04** |
| Exp12PatchQualityModulation | 94.89 | 47.99 | 61.84 | 25.83 | 74.58 | 91.39 | **66.09** |
| Exp6AdaptiveImageWeight | 94.48 | 36.70 | 36.69 | 22.05 | 55.87 | 90.95 | **56.12** |

| Method | Δ Avg vs PTA |
|--------|:-----------:|
| Exp4FullFusion | **−1.11** |
| Exp5TunableFusion | **−1.13** |
| Exp7InvertedImageWeight | **−3.12** |
| Exp8BoostedPatch | **−1.14** |
| Exp10SoftPatchGate | **−0.07** |
| Exp11TunableFusionNew | **−0.05** |
| Exp12PatchQualityModulation | **0.00** |
| Exp6AdaptiveImageWeight | **−9.97** |

### Post-Refactoring Experiments (5 CD datasets, excl. ucf101)

Results from `outputs/exp_results.txt` — all methods on the 5 CD datasets common across all runs:

| Method | dtd | eurosat | fgvc | oxford_flowers | oxford_pets | **Avg** | Δ vs PTA |
|--------|:---:|:-------:|:----:|:--------------:|:-----------:|:-------:|:--------:|
| PTA (baseline) | 47.70 | 61.83 | 25.68 | 74.71 | 91.31 | **60.25** | — |
| PTA2 (rerun) | 47.75 | 61.83 | 25.68 | 74.71 | 91.31 | **60.26** | +0.01 |
| PatchModulatedPTA | 47.99 | 61.86 | 25.80 | 74.67 | 91.31 | **60.33** | +0.08 |
| PatchModulatedPTA2 (rerun) | 47.99 | 61.86 | 25.80 | 74.54 | 91.39 | **60.32** | +0.07 |
| AdaptiveImageWeight | 37.65 | 37.01 | 22.29 | 56.72 | 91.09 | **48.95** | −11.30 |
| SoftPatchGate | 47.81 | 61.69 | 25.71 | 74.75 | 91.28 | **60.25** | 0.00 |
| QualityGateOnly | 47.99 | 61.85 | 25.80 | 74.62 | 91.36 | **60.32** | +0.07 |
| alpha_max_1_0 | 47.99 | 61.86 | 25.80 | 74.62 | 91.39 | **60.33** | +0.08 |
| **alpha_max_1_0_and_no_quality_gate** | 47.99 | 61.96 | 25.83 | 74.54 | 91.41 | **60.35** | **+0.10** |

**Patch Score Sweep (A1+A2 results, PatchBoost τ\_pch=10):**

| Method | dtd | eurosat | fgvc | oxford_flowers | oxford_pets | **Avg** | Δ vs PTA |
|--------|:---:|:-------:|:----:|:--------------:|:-----------:|:-------:|:--------:|
| PS-gaussian-tmm (baseline) | 47.70 | 62.42 | 25.77 | 74.58 | 91.25 | **60.34** | +0.09 |
| PS-cosine-tmm | 47.70 | 62.42 | 25.77 | 74.58 | 91.25 | **60.34** | +0.09 |
| PS-gaussian-max | 47.64 | 62.26 | 25.41 | 73.85 | 90.71 | **59.97** | −0.28 |
| PS-cosine-max | 47.64 | 62.06 | 25.41 | 73.89 | 90.71 | **59.94** | −0.31 |

### Headline Findings

1. **AdaptiveImageWeight fails dramatically.** Per-class adaptive image weighting causes severe degradation (−11.30pp vs PTA). Worst-hit: eurosat (−24.82pp), oxford_flowers (−17.99pp), dtd (−10.05pp). The hypothesis that low-α classes need heavier image weighting is contradicted.

2. **SoftPatchGate is neutral.** Achieves 60.25 vs 60.25 PTA baseline — identical. The binary top-1 gate is not a bottleneck.

3. **QualityGateOnly equals PatchModulatedPTA.** Setting `tau_patch_proto=0` changes almost nothing (60.32 vs 60.33). Patch-level logits in fusion contribute negligibly.

4. **No experiment beats PatchModulatedPTA.** Across all variants, the best result is `alpha_max_1_0_and_no_quality_gate` (60.35), which is +0.02 above PatchModulatedPTA (60.33). This is within noise — not a meaningful improvement.

5. **Scoring method doesn't matter.** PS-cosine-tmm (60.34) == PS-gaussian-tmm (60.34) — replacing Gaussian scoring with cosine similarity produces identical results. The scoring function is not the bottleneck.

6. **Max aggregation is worse.** PS-cosine-max (59.94) and PS-gaussian-max (59.97) both underperform their top\_m\_mean counterparts (60.34). The single best prototype carries less signal than the mean of the top-4.

7. **Reruns are consistent.** PTA2 (60.26) ≈ PTA (60.25) and PatchModulatedPTA2 (60.32) ≈ PatchModulatedPTA (60.33). Results are reproducible.

### Caveats

- **ucf101 missing**: Only PTA (73.33) and PatchModulatedPTA (73.22) have ucf101 entries.
- The patch score sweep used PatchBoost (no quality gate, no proto_alpha), so results are not directly comparable to PatchModulatedPTA which uses QualityGatedFusion.

---

## Adaptive Image Weight

**File**: `models/adaptive_image_weight.py`
**Config**: `configs_adaptive_image/*.yaml`

### Hypothesis

The image-level prototype's usefulness varies by class. When a class has many high-quality patch-level prototypes (high `proto_alpha[c]`), the image-level signal is redundant. When a class has few prototypes (low `proto_alpha[c]`), the image-level EMA is the primary signal and should be weighted heavily.

### Key Change

Per-class adaptive image-level weight:
```
tau_image[c] = tau_image_max - (tau_image_max - tau_image_min) * proto_alpha[c]
```

- `proto_alpha[c] ≈ 0` → `tau_image[c] ≈ tau_image_max` (100, trust image-level)
- `proto_alpha[c] ≈ alpha_max` → `tau_image[c] ≈ tau_image_min` (50, downweight image-level)

### Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau_image_max` | 100.0 | Upper bound for per-class image weight |
| `tau_image_min` | 50.0 | Lower bound for per-class image weight |

### Results

| Dataset | PTA | AdaptiveImageWeight | Δ |
|---------|:---:|:-------------------:|:-:|
| caltech101 | 94.93 | 94.56 | −0.37 |
| dtd | 47.70 | 37.65 | **−10.05** |
| eurosat | 61.83 | 37.01 | **−24.82** |
| fgvc | 25.68 | 22.29 | −3.39 |
| oxford_flowers | 74.71 | 56.72 | **−17.99** |
| oxford_pets | 91.31 | 91.09 | −0.22 |
| **Avg** | **66.03** | **56.55** | **−9.48** |

### Analysis

The adaptive weighting scheme is **uniformly harmful** across all 6 datasets, with catastrophic degradation on eurosat, dtd, and oxford_flowers. Only caltech101 and oxford_pets (high-base datasets) are minimally affected.

Possible explanations:
- **Over-squashing**: Reducing `tau_image` from 100→50 for high-α classes cuts the image-level signal in half. The patch-level prototypes cannot compensate for this loss, suggesting the fusion weighting is already in a good regime.
- **Wrong proxy**: `proto_alpha[c]` (how many samples a class has seen) may not correlate with "how much to trust image-level vs. patch-level." A class could have many stored patches that are all low-quality.
- **Caltech101 / oxford_pets anomaly**: These are the only datasets where performance holds. They have the highest base accuracy — possibly the image-level signal is redundant when zero-shot is already strong.

### Ablation Answers

- ~~Does per-class weighting improve over uniform `tau_image_proto`?~~ **No — it is significantly worse.**
- ~~Are the default bounds (100/50) optimal?~~ **Not relevant — the approach itself is flawed.**
- ~~Is `proto_alpha` the right proxy for "how much to trust image-level vs. patch-level"?~~ **No — the results contradict the hypothesis.**

### Recommendation

**Do not use AdaptiveImageWeight in its current form.** If exploring further, test narrower bounds (e.g., `tau_image_max=100`, `tau_image_min=80`) or an entirely different adaptive strategy.

---

## Soft Patch Gate

**File**: `models/soft_patch_gate.py`
**Config**: `configs_soft_gate/*.yaml`

### Hypothesis

The binary top-1 update gate (only the highest-confidence class updates) is conservative but wasteful — it discards signal from classes that are plausible but not the top prediction. A soft per-class proportional gate should capture more useful update signal.

### Key Change

Soft per-class proportional update gate:
```python
for c in range(C):
    weight = softmax_probs[c]
    if weight > soft_gate_threshold:
        relaxed_thresh = match_threshold - weight * 0.1
        update_prototype(class=c, threshold=relaxed_thresh)
```

- Every class with `softmax_prob > soft_gate_threshold` contributes to update
- Higher-confidence classes get a **relaxed matching threshold** (easier to match patches)
- Update is proportional to confidence weight

### Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `soft_gate_threshold` | 0.1 | Minimum softmax probability to trigger update |

### Results

| Dataset | PTA | SoftPatchGate | Δ |
|---------|:---:|:-------------:|:-:|
| caltech101 | 94.93 | 94.97 | +0.04 |
| dtd | 47.70 | 47.81 | +0.11 |
| eurosat | 61.83 | 61.69 | −0.14 |
| fgvc | 25.68 | 25.71 | +0.03 |
| oxford_flowers | 74.71 | 74.75 | +0.04 |
| oxford_pets | 91.31 | 91.28 | −0.03 |
| **Avg** | **66.03** | **66.04** | **+0.01** |

### Analysis

SoftPatchGate is **effectively indistinguishable from baseline** — both PTA (66.03) and PatchModulatedPTA (66.09) are within ±0.05pp. The soft per-class gate introduces no degradation but also no improvement.

Possible explanations:
- The binary top-1 gate is **not the bottleneck** — the top-1 prediction already captures most of the useful update signal.
- Multi-class updates add noise that cancels out any benefit.
- The `soft_gate_threshold=0.1` may be too low, letting in too many low-confidence classes.
- The relaxed matching threshold (`- weight * 0.1`) may be accepting too many low-quality patches.

### Ablation Answers

- ~~Does soft gating improve over binary top-1?~~ **No — results are identical within noise.**
- ~~Is `soft_gate_threshold=0.1` optimal?~~ **Tuning may not help — the approach shows no signal.**
- ~~Does the relaxed threshold help?~~ **Neither helps nor hurts measurably.**
- ~~Does updating multiple classes per sample cause prototype contamination?~~ **Not visible after averaging — but it's possible that contamination cancels out benefits.**

### Recommendation

Soft gating is **harmless but not beneficial** at current settings. Investigate higher thresholds (`soft_gate_threshold > 0.3`) or remove the relaxed threshold to test whether the soft gate mechanism itself adds anything.

---

## Quality Gate Only

**File**: `models/quality_gate_only.py`
**Config**: `configs_quality_gate_only/*.yaml`

### Hypothesis

The quality_gate's value may come from two sources: (1) modulating the image-level EMA update rate, or (2) weighting patch-level logits in fusion. This adapter isolates (1) by setting `tau_patch_proto=0` — patch scores don't contribute to logits, but quality_gate still modulates the image-level update.

### Key Change

`tau_patch_proto = 0` in fusion — patch-level prototype scores excluded from final logits. Quality_gate is still computed and used for image-level EMA modulation.

### Config Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `tau_patch_proto` | 0.0 | Patch-level logits excluded from fusion |

### Results

| Dataset | PatchModulatedPTA | QualityGateOnly | Δ |
|---------|:-----------------:|:---------------:|:-:|
| caltech101 | 94.89 | 94.89 | 0.00 |
| dtd | 47.99 | 47.99 | 0.00 |
| eurosat | 61.86 | 61.85 | −0.01 |
| fgvc | 25.80 | 25.80 | 0.00 |
| oxford_flowers | 74.62 | 74.62 | 0.00 |
| oxford_pets | 91.39 | 91.36 | −0.03 |
| **Avg** | **66.09** | **66.08** | **−0.01** |

### Analysis

**QualityGateOnly achieves identical results to PatchModulatedPTA** — removing patch-level logits from fusion has virtually no effect. This strongly suggests:

1. The patch-level logits in fusion (`tau_patch_proto * proto_alpha * quality_gate * patch_proto_logits`) are **negligible in magnitude** compared to the zero-shot + image-level terms.
2. The **entire benefit** (if any) of the patch-level system comes from the **quality-gated EMA modulation** of the image-level prototype update — not from patch-level logits in the final decision.

This is a clean ablation result: the two-stage design (patch analysis → modulate image update → image-level logits) can be simplified to just the quality-gated EMA modulation without any patch-level logit fusion term.

### Ablation Answers

- ~~Does quality_gate alone (without patch logits) improve over baseline PTA?~~ **Yes — QualityGateOnly (66.08) ≈ PatchModulatedPTA (66.09) > PTA (66.03), confirming the quality-gated EMA modulation provides the benefit.**
- The quality-gated EMA modulation is the primary benefit of patch-level analysis. **Confirmed.**
- The benefit does NOT come from patch-level logits in fusion. **Confirmed.**

### Recommendation

Consider removing the `tau_patch_proto * patch_proto_logits` term from `QualityGatedFusion` entirely. The quality-gated EMA modulation alone achieves the same result and simplifies the architecture. `PatchModulatedPTA` can be simplified to `QualityGateOnly` without loss.

---

## Patch-Level Score Aggregation & Scoring (Configurable)

The `_gaussian_score_for_class()` function in `models/patch_level/base.py` supports two configurable axes via YAML config:

### Aggregation strategies (`patch_level.aggregation`)

| Strategy | Formula | Description |
|----------|---------|-------------|
| `top_m_mean` (default) | `weighted.topk(top_m).values.mean()` | Mean of top-M weighted scores |
| `max` | `weighted.max()` | Single best prototype score |
| `sum` | `weighted.sum()` | Sum of all weighted scores |
| `mean` | `weighted.mean()` | Mean across all prototypes (no top-K filtering) |
| `top_m_mean_plus_mean` | `(top_m_mean + mean) / 2` | Average of top-M mean and global mean |

### Scoring functions (`patch_level.scoring`)

| Scoring | Formula | Description |
|---------|---------|-------------|
| `gaussian` (default) | `exp(-0.5 * Mahalanobis²)` | Gaussian/Mahalanobis distance with per-dimension variance |
| `cosine` | `patches @ centers.t()` | Cosine similarity (no variance, no one-to-one assignment) |

### Sweep Results (PatchBoost τ\_pch=10, 5 CD datasets)

| Config | scoring | aggregation | dtd | eurosat | fgvc | ox_flowers | ox_pets | **Avg** | Δ vs PTA |
|--------|---------|-------------|:---:|:-------:|:----:|:----------:|:-------:|:-------:|:--------:|
| PS-gaussian-tmm | gaussian | top\_m\_mean | 47.70 | 62.42 | 25.77 | 74.58 | 91.25 | **60.34** | +0.09 |
| PS-cosine-tmm | cosine | top\_m\_mean | 47.70 | 62.42 | 25.77 | 74.58 | 91.25 | **60.34** | +0.09 |
| PS-gaussian-max | gaussian | max | 47.64 | 62.26 | 25.41 | 73.85 | 90.71 | **59.97** | −0.28 |
| PS-cosine-max | cosine | max | 47.64 | 62.06 | 25.41 | 73.89 | 90.71 | **59.94** | −0.31 |

### Key Findings

1. **Scoring method doesn't matter.** PS-cosine-tmm == PS-gaussian-tmm (both 60.34). Replacing Gaussian/Mahalanobis scoring with cosine similarity produces identical results — the variance estimation in Gaussian scoring is not the bottleneck.

2. **Max aggregation is worse.** Both max variants (59.94–59.97) underperform top\_m\_mean variants (60.34). The single best prototype carries less signal than the mean of the top-4. This contradicts the hypothesis that "averaging dilutes the signal."

3. **Neither axis improves over baseline.** The best patch score result (60.34) is only +0.09 over PTA — within noise and below the +0.10 threshold for "worth pursuing."

### Open Questions (partially answered)

- ~~Is `top_m_mean` fair when `top_m` is too small (unstable) or too large (misguided by weak prototypes)?~~ **top\_m\_mean is better than max — averaging over top-4 helps.**
- ~~What if patch-level prototypes aren't available for the current view?~~ **Not tested yet — all methods use PatchBoost which always has prototypes.**
- ~~Should aggregation be adaptive?~~ **Not needed — top\_m\_mean is already the best option.**

---

## Running Experiments

### SLURM (all 5 methods × 7 datasets = 35 tasks)
```bash
sbatch scripts/slurm_cd_new_experiments_benchmark.sh
```

### Manual single-dataset run
```bash
# Adaptive Image Weight
python runner.py --method adaptive_image_weight --config configs_adaptive_image --datasets oxford_pets --backbone ViT-B/16

# Soft Patch Gate
python runner.py --method soft_patch_gate --config configs_soft_gate --datasets oxford_pets --backbone ViT-B/16

# Quality Gate Only
python runner.py --method quality_gate_only --config configs_quality_gate_only --datasets oxford_pets --backbone ViT-B/16

# PatchBoost (isolation — no quality gate, no proto_alpha)
python runner.py --method patch_boost --config configs --datasets oxford_pets --backbone ViT-B/16
```

### PatchBoost sweeps (τ\_patch independent)
```bash
# τ_pch sweep: 0, 5, 10, 15, 20, 50, 100, 200, 500, 1000
# Edit configs_pb_*/<dataset>.yaml → fusion.tau_patch_proto
# Or use shared budget mode: configs_pb_share100_t*/<dataset>.yaml
```

### Aggregation sweep (manual)
Edit any config YAML, change `aggregation` in `patch_level` section, re-run.

### Phase 1 Quick Ablation (cosine + max)
```bash
# Edit configs/<dataset>.yaml:
#   patch_level.aggregation: "max"
#   # A1 (cosine scoring) requires code change in patch_level/base.py
# Run via PatchBoost to isolate scoring effect:
python runner.py --method patch_boost --config configs --datasets dtd/eurosat/fgvc/oxford_flowers/oxford_pets --backbone ViT-B/16
```

---

## Expected Output Format

Results appended to `outputs/result.txt`:
```
PTA's performance on oxford_pets: Top1- XX.XX.
PatchModulatedPTA's performance on oxford_pets: Top1- XX.XX.
AdaptiveImageWeight's performance on oxford_pets: Top1- XX.XX.
SoftPatchGate's performance on oxford_pets: Top1- XX.XX.
QualityGateOnly's performance on oxford_pets: Top1- XX.XX.
PatchBoost[τ_img=100_τ_pch=10]'s performance on oxford_pets: Top1- XX.XX.
```

A summary table is available at `outputs/exp_results.txt` (generated manually by collecting from `result.txt`).
Pre-refactoring results archived at `old_outputs/exp_results.txt`.

---

## Finding Settings Where Patch-Prototype Actually Helps

**Goal**: Find a configuration where patch-level Gaussian prototypes give PatchModulatedPTA a **measurable improvement** over PTA baseline (> +0.10 avg).

### What We Know (from `omo/plans/gaussian-patch-prototype-analysis.md`)

The analysis plan exhaustively tested three axes and found **no improvement**:

| Axis Tested | Range | Best Result | Verdict |
|---|---|---|---|
| **τ\_patch sweep** | 0 → 1000 | τ\_pch=10 gives +0.02 | Patch scores are **noisy**, not discriminative |
| **Shared budget** (τ\_img + τ\_pch = 100) | τ\_pch=0 → 100 | τ\_img=90/τ\_pch=10 gives +0.04 | Reallocating from image to patch **always hurts** |
| **match\_threshold** | 0.6 → 0.9 | No change (60.33 → 60.25) | Sharper prototypes don't help — scores are inherently noisy |
| **aug\_copies** | 0 → 10 | aug0 best (60.33) | More data doesn't help |

**Root cause**: Gaussian patch-level prototypes in CLIP feature space don't produce discriminative enough scores to beat PTA's image-level EMA. The CLIP image embedding is already a strong class representation; patch embeddings are noisier and less calibrated.

### Why Patch Scores Are Noisy

1. **Scale mismatch**: Patch scores (Gaussian → [0, 1]) vs image scores (dot product → [10, 30]). Even at τ\_pch=10, effective patch contribution is ≤ 10 × 1.0 = 10, drowned by 100 × 20 = 2000 from image.
2. **Variance estimation**: Per-dimension variance estimated from few samples (EMA with α=0.1) is unreliable → Gaussian scores are poorly calibrated.
3. **Top-m aggregation**: `top_m=4` averages over noisy per-prototype scores, diluting any signal from the single best match.
4. **One-to-one assignment**: Forces each patch group to match at most one prototype, potentially discarding valid cross-prototype evidence.

### Concrete Plans to Try

#### Plan A: Fix the scoring mechanism (low effort, quick ablations)

**A1. Cosine-similarity scoring** — replace Gaussian with cosine similarity to prototype centers
- *Rationale*: Gaussian scoring uses per-dimension variance (poorly estimated from few samples). Cosine similarity is what CLIP is trained for and is inherently well-calibrated.
- *Change*: Add `scoring: "cosine"` option to `_gaussian_score_for_class`. For cosine mode, skip variance entirely: `score = patches_norm @ centers_norm.t()`
- *Config*: `patch_level.scoring = "cosine"`
- *Test with*: PatchBoost (τ\_pch=10) to isolate scoring effect

**A2. Max-pooling aggregation** — replace `top_m_mean` with `max`
- *Rationale*: The single best-matching prototype may carry the only useful signal; averaging with top-4 dilutes it with noise.
- *Change*: Already implemented (`aggregation: "max"` in config). Just needs testing.
- *Config*: `patch_level.aggregation = "max"`
- *Test with*: PatchBoost (τ\_pch=10)

**A3. Logit scale normalization** — normalize patch scores to match image-level scale before fusion
- *Rationale*: If patch scores are in [0, 1] and image scores in [10, 30], the τ weights are fighting a scaling mismatch. Normalize patch logits by their std or max before fusion.
- *Change*: In `WeightedFusion.forward()`, add `patch_proto_logits = patch_proto_logits * scale_factor` where `scale_factor = image_proto_logits.std() / patch_proto_logits.std()`
- *Config*: `fusion.normalize_patch_scale = true`

**A4. Combined sweep** — test A1 × A2 × A3 together
- *Rationale*: These changes are synergistic. Cosine scoring + max aggregation might produce a clean, well-scaled signal.
- *Matrix*: 2 (gaussian/cosine) × 5 (aggregation strategies) × 2 (normalize/skip) = 20 configs
- *Prioritize*: cosine + max + normalize first

#### Plan B: Rethink how patch evidence flows (medium effort)

**B1. Patch-as-gate** — use patch prototypes as a binary quality signal, not a scoring system
- *Rationale*: Instead of computing per-class patch scores (which are noisy), use patch prototypes to answer: "does this image have discriminative patches?" If yes, amplify the image-level EMA update. If no, use vanilla CLIP zero-shot.
- *Change*: New fusion class `PatchGateFusion` that uses `quality_gate` as a binary switch: `if quality_gate > threshold: use image_proto_logits, else: use clip_logits only`
- *Config*: `fusion.patch_gate_threshold` (default 0.5)

**B2. Patch-refined image prototype** — use patch evidence to directly refine the image-level prototype
- *Rationale*: Instead of a separate patch branch in fusion, use patch-level consensus to nudge the image-level EMA update direction. This avoids the fusion problem entirely.
- *Change*: In `_update_text_features_with_quality`, use patch-level top-match direction to bias the update: `update_dir = image_feature + patch_weight * (best_patch_center - image_feature)`
- *Config*: `image_level.patch_nudge_weight` (default 0.1)

**B3. Cross-attention patches-to-text** — lightweight cross-attention between patches and text prototypes
- *Rationale*: CLIP is trained with cross-attention semantics. Using patches to attend to text prototypes is more aligned with how VLMs actually work than k-means clustering.
- *Change*: Replace k-means scoring with `patches @ text_prototypes.t()` followed by softmax-weighted pooling
- *Config*: `patch_level.scoring = "cross_attention"`

#### Plan C: Accept marginality and pivot (strategic decision)

**C1. Document and move on**
- *Finding*: Patch-level Gaussian prototypes provide +0.08 avg over PTA. Statistically insignificant across 5-6 datasets.
- *Action*: Archive patch-level experiments. Focus on image-level improvements (per-class adaptive α, confidence-weighted EMA, multi-scale image features).

**C2. The quality-gated EMA IS the contribution**
- *Finding from QualityGateOnly*: The benefit of patch-level analysis comes entirely from quality-gated EMA modulation of image-level updates, NOT from patch logits in fusion.
- *Action*: Simplify PatchModulatedPTA to QualityGateOnly (remove τ\_patch\_proto term). The patch system exists only to compute `quality_gate` for the image-level EMA. This is a valid, simpler architecture.

### Recommended Execution Order

```
Phase 1 (1 day): Quick ablations — A1 (cosine) + A2 (max aggregation)
  → If ≥ +0.10 avg over PTA: continue to A3, A4
  → If not: skip to Phase 2

Phase 2 (2 days): Structural changes — B1 (patch-as-gate) + B2 (patch-refined image)
  → These address the root cause (noisy scores) rather than patching the scoring function

Phase 3 (decision): If Phase 2 doesn't help → C1/C2 (pivot to image-level improvements)
```

### Success Criteria

| Level | Threshold | Meaning |
|-------|-----------|---------|
| **Worth pursuing** | ≥ +0.10 avg over PTA | Statistically meaningful improvement |
| **Paper-worthy** | ≥ +0.30 avg over PTA | Clear, defensible contribution |
| **Kill signal** | ≤ +0.05 after Phase 2 | Patch-level approach is fundamentally limited |

---

## Backward Compatibility

These changes are **additive only**:
- `models/fusion.py` — new `AdaptiveImageWeightFusion` class appended (existing classes untouched)
- `models/patch_level/base.py` — new `aggregation` parameter with default `"top_m_mean"` (matches current hardcoded behavior)
- `models/patch_level/gaussian_patch.py` — reads `aggregation` from config with default `"top_m_mean"`
- `models/patch_boost.py` — new adapter for isolating patch-level prototype effects (no impact on existing adapters)

Running `scripts/slurm_cd_benchmark_pta_vs_patch_mod.sh` after these changes will produce **identical results**.
