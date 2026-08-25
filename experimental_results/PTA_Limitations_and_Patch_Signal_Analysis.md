# PTA Limitations and Patch-Level Signal Analysis

**Scope**: ViT-B/16 backbone, CLIP Surgery, 3 primary datasets (dtd, oxford_flowers, oxford_pets), 4 seeds, 7 fusion/write-rule configurations.

---

## Background: How PTA Works

PTA (Prototype-Based Test-Time Adaptation) improves CLIP's zero-shot predictions by maintaining running "prototypes" — reference representations that get updated as the model processes test images.

### Two Types of Prototypes

**Image-level prototype** (standard PTA):
- **What**: One running average per class, stored as a single feature vector
- **Write time**: When CLIP predicts class *c* with high confidence for an image, the image's CLS-pooled embedding (the "[CLS]" token's output, representing the whole image) is blended into class *c*'s running average via exponential moving average (EMA)
- **Inference time**: The stored prototype is compared to a new image's CLS embedding via cosine similarity; this produces a "prototype logit" that competes with CLIP's own zero-shot logit
- **Failure mode**: If CLIP is confidently wrong, the wrong prediction gets blended in. Over time, the prototype can drift toward incorrect representations — this is "prototype drift"

**Patch-level prototype** (PatchModPTA, the variant under investigation):
- **What**: Multiple cluster centroids per class, stored as a small bank of Gaussian components (each with a center, variance, and appearance count)
- **Write time**: When CLIP predicts class *c* with high confidence, the image's *patch embeddings* (196 spatial patches from the ViT, not the CLS token) are clustered and matched against class *c*'s existing bank. Matching clusters get their centers updated; non-matching patches may form new clusters
- **Inference time**: Each patch in a new image is scored against all clusters in each class bank using a Gaussian scoring formula (center similarity + variance weighting). Scores are aggregated (e.g., mean, top-k mean) to produce a "patch logit" per class
- **Key difference from image-level**: Instead of one average per class, the bank captures multiple visual patterns (e.g., different poses, backgrounds, textures). This could in theory capture intra-class diversity that a single average cannot

### Fusion: Combining the Signals

At inference, the final prediction combines three sources:
1. **CLIP zero-shot**: cosine similarity between CLS embedding and text class embeddings
2. **Image-level prototype**: cosine similarity between CLS embedding and stored prototype
3. **Patch-level prototype** (when enabled): aggregated patch-to-bank similarity

Different fusion mechanisms control how these combine:
- **ProtoAlpha**: Patch term always contributes (weighted by a learned alpha)
- **QualityGated**: Same as ProtoAlpha, but gated by prediction quality
- **MajorityVote**: 2-of-3 vote — if any two sources agree, that class wins
- **AgreementGate**: Patch term only contributes when it agrees with CLIP

---

## Part 1: PTA's Prototype Drift Problem

### 1.1 The Tie-Breaking Failure

When CLIP and the image-level prototype disagree, which one should we trust? We analyzed such "tie" cases.

**Dataset**: dtd, seed 1 (1692 samples)

| Metric | Value |
|--------|-------|
| Total samples | 1692 |
| Ties (CLIP disagrees with prototype) | 527 (31.1%) |
| Of ties, prototype is correct | 130 (24.7%) |
| Of ties, prototype is wrong | **397 (75.3%)** |

**Finding**: When the two sources disagree, the image-level prototype is wrong **75.3%** of the time. The prototype has drifted from CLIP's prediction, and in most cases, CLIP was right.

> **Note**: This analysis was performed on dtd only (seed 1). The pattern — ties occurring in ~30% of samples, with the prototype wrong ~75% of the time — is consistent with the broader finding that PTA's adaptation can hurt when it diverges from CLIP's prediction.

### 1.2 Why This Happens: The Write-Time Problem

PTA's prototype update is gated only by CLIP's own confidence score. There is no ground truth available at test time. This creates a fundamental problem:

- **Confident-right predictions** correctly reinforce the prototype
- **Confident-wrong predictions** incorrectly contaminate the prototype
- **From inside the system, these look identical** — both have high confidence scores

PTA's EMA update partially mitigates this: wrong predictions get diluted over time as they compete with correct ones in the running average. But the contamination still accumulates, especially for classes that CLIP consistently confuses with a similar neighbor.

---

## Part 2: Why Patch-Level Prototypes Don't Fix PTA

The hypothesis: patch-level prototypes might provide additional signal that either (a) corrects image-level prototype drift, or (b) provides an independent quality check on predictions.

### 2.1 Bank Quality: Purity

The patch-level bank is built from CLIP's own guesses (no ground truth). **Purity** measures what fraction of written patches actually belong to the claimed class.

**Write-time settings**:
- **Confidence source**: text embeddings (`conf_source="text"`)
- **Confidence threshold**: 0.3 (`conf_threshold=0.3`) — only writes when CLIP's softmax confidence exceeds this value
- **Multi-gate**: allows writes to multiple class banks when confidence exceeds threshold for several classes simultaneously

**Method**: Replay the `multi_gate` write mask from stored zero-shot CLIP scores and true labels. No new GPU runs.

| Dataset | Overall purity | Primary-write purity | Collateral-write purity | Collateral share |
|---------|----------------|---------------------|------------------------|------------------|
| dtd | 62.6% | 63.4% | 46.7% | 4.5% |
| oxford_flowers | 74.1% | 77.2% | 38.1% | 7.9% |
| oxford_pets | 85.7% | 89.9% | 40.1% | 8.4% |

**Terminology**:
- **Primary-write purity**: When an image is written to its *predicted* class's bank, what fraction of those writes are correct? (e.g., image predicted as "bengal" → written to bengal bank → was it actually a bengal?)
- **Collateral-write purity**: When an image triggers writes to *multiple* class banks (multi-write events), what fraction of those *additional* writes are correct? (e.g., image predicted as "bengal" also triggers writes to "abyssinian" bank → was it actually an abyssinian?)
- **Collateral share**: What fraction of all writes are collateral (multi-class) writes?

**Key insight**: Collateral writes have much lower purity (38-47%) than primary writes (63-90%), meaning when the system writes to multiple classes, those extra writes are often wrong.

**Interpretation**: The result appears "backwards" — dtd has the dirtiest bank (62.6% purity) yet patch fusion helps there, while oxford_pets has the cleanest bank (85.7%) yet patch fusion hurts most. However, this doesn't necessarily indicate a problem with the bank itself. When CLIP is incorrect (which happens more often on dtd due to its lower baseline accuracy), the system *needs* to write to alternative classes to potentially correct the error. The question is not whether the bank is pure, but **what weight should patch-level prototypes receive** such that incorrect-class writes don't dominate the final prediction.

### 2.2 Bank Quality: Separability

Even if a bank is 100% pure, do those patches actually *look different* from a neighboring class's patches? Two visually-similar breeds of cat can have fur patches that look more alike to each other than to other photos of the same breed.

**Method**: Load final per-class Gaussian bank, measure intra-class coherence vs. nearest cross-class confusability.

**Cosine similarity between cluster centers**:

| Dataset | Intra-class similarity | Nearest cross-class similarity | Margin |
|---------|----------------------|-------------------------------|--------|
| dtd | 0.738 | 0.859 | **-0.121** |
| oxford_flowers | 0.799 | 0.929 | **-0.130** |
| oxford_pets | 0.795 | 0.908 | **-0.114** |

**Variance-aware scoring** (uses both center and spread):

| Dataset | Intra-class match | Nearest cross-class match | Margin |
|---------|-------------------|--------------------------|--------|
| dtd | 0.593 | 0.771 | **-0.178** |
| oxford_flowers | 0.626 | 0.880 | **-0.254** |
| oxford_pets | 0.590 | 0.852 | **-0.262** |

**Result**: Every dataset has **negative margins** — a class's own patches are consistently *less similar to each other* than to their nearest neighbor in another class. This is a fundamental problem: even perfect purity cannot make patch banks discriminative.

### 2.3 The Structural Asymmetry

Neither purity nor separability explains why patch fusion helps on dtd but hurts on oxford_pets. The key difference is **baseline accuracy**:

| Dataset | PTA Accuracy | Room to Improve |
|---------|--------------|-----------------|
| dtd | 47.5% | Many wrong answers to fix |
| oxford_flowers | 74.6% | Moderate |
| oxford_pets | 91.1% | Very few wrong answers left |

Adding a noisy, mediocre signal to a high-accuracy baseline (oxford_pets, 91%) breaks more correct answers than it fixes. Adding the same signal to a low-accuracy baseline (dtd, 47.5%) can help slightly, but the effect is sub-threshold.

---

## Part 3: Patch Agreement as a Quality Signal

Despite patch content not helping predictions, we discovered a genuinely useful **meta-signal**: whether CLIP's CLS-level guess and the patch-level vote agree.

### 3.1 The Agreement Signal

The patch embeddings can be compared directly to text embeddings (patch-to-text zero-shot), independent of any bank state. This produces a second vote from the *same model, same image* — global pooling (CLS) vs. local pooling (patches).

**When CLIP and patch vote agree, CLIP is much more likely to be correct**:

| Dataset | CLIP accuracy when agree | CLIP accuracy when disagree | Gap |
|---------|-------------------------|----------------------------|-----|
| dtd | 58.4% | 23.0% | **+35.5pp** |
| oxford_flowers | 94.7% | 64.5% | **+30.1pp** |
| oxford_pets | 95.3% | 86.4% | **+8.9pp** |

This is a **real, validated signal**: it works across multiple seeds, and the agreement rate varies per-image within a class (not just per-class difficulty).

### 3.2 Aggregation Methods

Mean-pooling across all patches can be diluted by background. We tested 13 pooling variants. "Standalone accuracy" here means: if we **only** use the patch-to-text similarity for classification (ignoring CLIP's CLS embedding and any prototypes), what's the accuracy?

| Dataset | Method | Standalone accuracy | Agreement rate |
|---------|--------|--------------------|----|
| dtd | mean | 40.8% | 59.6% |
| dtd | topk20 | 41.3% | 60.0% |
| oxford_flowers | mean | 20.7% | 20.5% |
| oxford_flowers | topk20 | 29.0% | 31.6% |
| oxford_pets | mean | 30.2% | 30.0% |
| oxford_pets | topk20 | **65.4%** | **64.8%** |

**Key finding**: `topk20` (average the top-20 most similar patches) roughly **doubles** standalone accuracy on oxford_pets (30% → 65%) by focusing on the most relevant patches. An Otsu-threshold variant (`otsu_mean`) achieves similar results without tuning k.

### 3.3 Validation Across Seeds

The agreement signal holds up across 4 seeds:

| Dataset | Variant | Purity gap range (s1-s4) |
|---------|---------|-------------------------|
| dtd | topk20 | +30.8 to +37.1pp |
| oxford_flowers | topk20 | +16.1 to +18.4pp |
| oxford_pets | topk20 | +20.9 to +23.4pp |

The signal is **genuinely new information** — it predicts correctness *within* both confident and ambiguous predictions, and is independent of which prototypes are geometrically separable.

### 3.4 Why This Doesn't Help Predictions

The agreement signal can flag "this prediction is uncertain" but cannot tell you what the *right* answer is. When CLIP is wrong, the patch vote only helps 10-15% of the time — not enough to serve as an independent classifier.

Multiple attempts to convert this signal into accuracy wins (gating write confidence, down-weighting disagreement) produced null results:

| Mechanism | Result |
|-----------|--------|
| Quality-gated writes | No improvement |
| Agreement-gated writes | No improvement |
| Soft down-weighting at 0.3 discount | No improvement (single-seed pilot signal did not replicate) |

---

## Part 4: Fusion Mechanisms

This section summarizes how different fusion mechanisms combine the three signals (CLIP, image-level prototype, patch-level prototype) and their empirical performance.

### 4.1 Mechanism Comparison

| Method | Formula | Behavior |
|--------|---------|----------|
| **ProtoAlpha** | `tau_patch × proto_alpha × squash(patch)` | Patch always contributes, scaled by `proto_alpha` |
| **QualityGated** | `tau_patch × proto_alpha × quality_gate × squash(patch)` | Same as ProtoAlpha, gated by sample quality |
| **MajorityVote** | 2-of-3 vote on argmax | Each source casts a vote; ≥2 agree = win |
| **AgreementGate** | `tau_patch × (1 if patch.argmax==clip.argmax else 0) × squash(patch)` | Patch only contributes when it agrees with CLIP |

### 4.2 Results (4 seeds, 3 datasets)

| Method | dtd | oxford_flowers | oxford_pets | Mean |
|--------|-----|----------------|-------------|------|
| PTA (baseline) | 47.52 | 74.62 | 91.07 | 71.07 |
| PatchModPTA (default) | 47.21 | 72.22 | 89.08 | 69.50 |
| PatchModPTA-QGated | 47.21 | 72.25 | 89.10 | 69.52 |
| PatchModPTA-MVote | 48.06 | 74.17 | 90.67 | 70.97 |
| PatchModPTA-AGate | 48.12 | 74.14 | 90.83 | 71.03 |

### 4.3 Why Always-On Fusion Collapses

**ProtoAlpha/QGated** always contribute the patch term (just scaled). On fine-grained datasets, patch is noisy and independent ~59% of the time. This adds noise, dragging performance down by -2.0 to -2.8pp.

**MVote/AGate** prevent collapse by silencing bad patch predictions:
- MVote: 2-of-3 vote — patch only wins if at least one other source agrees
- AGate: patch only contributes when it agrees with CLIP

### 4.4 Delta vs PTA (5 datasets, 5 seeds)

| Dataset | ProtoAlpha | QGated | MVote | AGate |
|---------|------------|--------|-------|-------|
| dtd | -0.08% | -0.09% | +0.65% | +0.60% |
| eurosat | +1.04% | +1.04% | +0.59% | -0.13% |
| fgvc | -2.75% | -2.72% | -0.04% | -0.10% |
| oxford_flowers | -2.29% | -2.24% | -0.13% | -0.24% |
| oxford_pets | -2.01% | -2.00% | -0.39% | -0.31% |
| **AVG** | **-1.22%** | **-1.20%** | **+0.14%** | **-0.04%** |

**Pattern**: Gains on texture/satellite (dtd, eurosat where local patterns matter); collapse on fine-grained (FGVC, flowers, pets where always-on fusion hurts).

---

## Summary

| Finding | Implication |
|---------|-------------|
| Image-level prototypes drift when CLIP is confidently wrong | PTA's adaptation can hurt on 31% of samples |
| Patch fusion never reliably beats PTA across 30+ experiments | Patch content is too noisy and non-separable |
| Patch agreement with CLIP is a strong quality signal (30+ pp gap) | Genuine independent information exists |
| This signal doesn't convert to accuracy wins | Can flag uncertainty but not correct errors |

The fundamental limitation is structural: patch banks are built from CLIP's own guesses (no ground truth), class clusters overlap at the patch level (negative separability margins), and the signal quality is fixed by the underlying CLIP features.

---

## Appendix: Supplementary Data

### A.1 Extended Results (5 datasets, 5 seeds)

| Method | dtd | eurosat | fgvc | oxford_flowers | oxford_pets | Mean |
|--------|-----|---------|------|----------------|-------------|------|
| PTA | 47.43 | 61.34 | 25.65 | 74.53 | 91.15 | **60.02** |
| PatchModPTA | 47.35 | 62.38 | 22.90 | 72.24 | 89.14 | **58.80** |
| PatchModPTA-MVote | 48.09 | 61.93 | 25.61 | 74.40 | 90.76 | **60.16** |
| PatchModPTA-AGate | 48.04 | 61.21 | 25.55 | 74.28 | 90.85 | **59.99** |

### A.2 Correction/Regression Counts

| Arm | Dataset | patch_alone corrections | patch_alone regressions | net |
|-----|---------|------------------------|------------------------|-----|
| PatchModPTA-CS | dtd | 422 | 768 | -346 |
| PatchModPTA-CS | oxford_flowers | 296 | 3218 | -2922 |
| PatchModPTA-CS | oxford_pets | 404 | 6110 | -5706 |

Patch predictions are **net-negative everywhere** — more regressions than corrections.

### A.3 Qualitative Examples (oxford_pets, topk20)

**Rescue** (patch correct, CLIP wrong):
- target=`bengal`, CLIP=`abyssinian`, patch=`bengal`
- target=`birman`, CLIP=`ragdoll`, patch=`birman`
- target=`beagle`, CLIP=`basset_hound`, patch=`beagle`

**Corrupt** (CLIP correct, patch wrong):
- target=`wheaten_terrier`, CLIP=`wheaten_terrier`, patch=`pug`
- target=`ragdoll`, CLIP=`ragdoll`, patch=`siamese`

Both dominated by visually-similar breed confusions — the signal does real discrimination, just not reliably enough.

---

*This document is self-contained. For the full investigation narrative including all experimental rounds, see `patch_level_fusion_summary.md` in the original outputs directory.*
