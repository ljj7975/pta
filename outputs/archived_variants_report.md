# Archived Variants — Ideas & Motivation
**Date:** 2026-06-19  
**Status:** Both variants archived (files removed). No benchmark results were collected.

---

## 1. MultiProtoPTA-Gaussian (`multi_proto_pta_gaussian.py`)

### Core idea
Replace cosine similarity in the prototype scoring step with a **Mahalanobis-like Gaussian score**.

In the base MultiProtoPTA, a prototype is a single point in feature space and is scored by how similar the best matching patch is to that point. The Gaussian variant augments each prototype with a per-dimension **variance** tensor (a "spread"), turning the prototype from a point into a fuzzy cloud:

```
score(patch, prototype) = exp(-0.5 * Σ_d (patch_d - center_d)² / variance_d)
```

This has two desirable properties:
- A **tight** prototype (low variance) penalises patches that aren't very close — it acts as a strict discriminator.
- A **loose** prototype (high variance) is more forgiving — it signals the cluster hasn't converged yet and shouldn't be trusted as strongly.

Variance is maintained per prototype via **EMA** (exponential moving average) over the residuals of patches that match that prototype each update. New prototypes initialise with a default spread (`gaussian_new_var`) until enough evidence accumulates.

### Relationship to Exp3
Exp3 (`exp3_gaussian_prototypes.py`) explores the same Gaussian scoring idea but within the cleaner Exp-series architecture. The Gaussian variant here is an earlier, less integrated attempt built directly on top of `MultiProtoPTABase` via subclassing.

### Config
`configs_multi_proto_gaussian/` (full dataset coverage including ImageNet variants).

---

## 2. MultiProtoPTA-TCR (`multi_proto_pta_tcr.py`)

### Core idea
Add **cross-class specificity weighting** to each prototype via a **Term-Class Relevance (TCR)** mechanism — borrowed conceptually from TF-IDF in information retrieval.

The problem it targets: some prototypes that accumulate for a given class are actually generic visual patterns that appear in many classes (e.g., backgrounds, textures). These prototypes should be down-weighted at scoring time because they don't discriminate well.

### Mechanism
Each prototype tracks two counters:
- `appearance[k]` — how many images from this class activated prototype k (intra-class frequency, same as base)
- `cc_hits[k]` — how many times prototype k was geometrically close (cosine sim ≥ 0.92) to a prototype from another class after that other class received a visual update

A **Beta-posterior specificity** estimate is computed per prototype:
```
spec_k = (appearance[k] + m) / (appearance[k] + cc_hits[k] + 2m)
```
(Prior `Beta(m, m)` with mean 0.5, so a new prototype starts neutral.)

This is then gated by a **confidence term** that starts at zero and grows with evidence:
```
confidence_k = 1 - exp(-N_k / n_half)
weight_k     = (1 - confidence) * 1.0  +  confidence * (2 * spec_k)
```

Result: weights live in (0, 2). Class-specific prototypes (high appearance, low cc_hits) get amplified toward 2.0; cross-class prototypes (high cc_hits) get suppressed toward 0.0. New prototypes start at weight 1.0 (no distortion until evidence accumulates).

### Key design choices
- **Skip text-initialized classes in the cc scan**: CLIP text embeddings cluster tightly (cosine sim > 0.85 between related classes), so scanning against uninitialised classes would spike cc_hits immediately. Only classes that have seen at least one real image update are included.
- **cc_hits decay on prototype movement**: when a prototype's center moves via EMA, its old cross-class evidence is decayed by `cc_decay_rate` (default 0.05). A prototype that drifts away from a cross-class region recovers its specificity score within ~20 updates.
- **Replaces `penalize_common`**: the base class has a simpler global penalty for common prototypes; TCR supersedes this with a per-prototype, evidence-accumulating version.

### Config
No dedicated config directory — uses `configs_multi_proto/`.

---

## Why archived

Neither variant was benchmarked before the Exp1/2/3 series was prioritised. Exp3 (Gaussian Prototypes) subsumes the Gaussian variant's core idea in a cleaner form and did produce results. TCR remains an untested but theoretically motivated idea for future reference.
