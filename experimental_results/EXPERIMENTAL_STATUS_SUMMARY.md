# Experimental Status Summary — PTA Improvement Investigation

**Date**: Sep 2026  
**Purpose**: Concise reference for revisiting research direction with teammates.  
**Scope**: ViT-B/16, CLIP Surgery, dev set = {dtd, oxford_flowers, oxford_pets}, 4 seeds.  
**Baseline**: PTA fused = 71.07% avg (dtd 47.47 / flowers 74.55 / pets 91.18). CLIP zero-shot = 68.28%.

---

## Core Finding

**PTA's single-mean EMA prototype is a local optimum. Every proposed improvement makes it worse. The write-stream bias — CLIP's own predictions drive all prototype updates, with no ground truth — is the binding constraint that cannot be worked around from inside the system.**

---

## What Was Tried (Summary)

### A. Patch-Level Fusion (earliest work)

**Hypothesis**: Patch-level features provide independent spatial evidence that can correct image-level prototype drift.

**Result**: Null across 30+ experiments. Patch banks have negative separability margins everywhere (−0.11 to −0.24); class clusters overlap in patch space regardless of purity. Patch fusion helps slightly on the hardest dataset (dtd) but collapses on fine-grained sets (flowers, pets).

**Key diagnostic**: The agreement signal (patch vote vs. CLIP CLS-guess) is genuinely strong (+9 to +35pp purity gap), but this is a *diagnostic*, not a *lever*. It has been tested in every configuration (read-time reweighting, write-gating, write-reweighting) and fails everywhere.

**Status**: Closed.

---

### B. Confusability/Repulsion Line (Phases 1–10)

**Hypothesis**: Trust signals can identify bad writes and either block them or correct the prototype geometry.

| Phase | Approach | Result |
|-------|----------|--------|
| 1–3 | Read-time trust fusion (120 runs) | Null. Trust regime overlaps tie regime; reweighting has nowhere to act. |
| 1–3 | Foreground-weighted embedding | Killed offline. −40 to −44pp vs CLS on fine-grained. |
| 1–3 | Image-level multi-prototype | Killed offline. Clustering margins near zero. |
| 4–6 | Class-prediction calibration | Null. Peak +0.15pp, reverses on higher-class-count datasets. |
| 4–6 | Prototype confusability freeze | Strong diagnostic (+9 to +16.5pp gap). Monotonic regression as adapter (−2.2 to −10.0pp). |
| 4–6 | Multi-view consistency | Strongest diagnostic in campaign (+16.6 to +27.4pp). Zero improvement as adapter. |
| 7 | Write-gate/reweight on new signals (18 settings) | All null. Diagnostic strength *inversely* related to write-time usefulness. |
| 8 | Prototype repulsion (undirected) | Mechanism works (confusability drops). Net negative (oxford_pets −0.86 to −8.0pp). Cannot distinguish drift from genuine similarity. |
| 9 | Drift-gated repulsion | Mildest result in campaign (lr=0.02: zero regression anywhere, +0.03 to +0.07pp avg). Below promotion bar. |
| 10 | Floor ablation / text-anchored EMA / prob-weighted EMA / bilateral drift | All 16 settings null. Drift gate is the active ingredient (Phase 9); floor is safety net only. |

**Mechanistic conclusion**: Diagnostic quality does not predict intervention usefulness. The strongest diagnostic (multi-view consistency) produces the *most destructive* write gate. Combining signals never beats the stronger individual one. Every relaxation of the intervention (permanent → per-sample, hard gate → soft reweight) narrows the loss but never flips the sign.

**Status**: Exhaustively explored. Recommendation: close this line entirely.

---

### C. Representation Upgrade Study (Waves 1–3)

**Hypothesis**: Improving the per-class prototype representation (variance-aware scoring, K-prototype banks, compactness gates, augmentation ensembling, text-anchor damping) beats base PTA.

**Pre-validation gates**: Every mechanism passed cheap offline diagnostics before any cluster compute was spent.

**Online results**: Every mechanism failed.

| Method | Avg Δ vs PTA | Failure mode |
|--------|-------------|--------------|
| GaussPTA (Mahalanobis scoring) | −3.4 to −32.8pp | Variance estimated from biased stream amplifies error |
| BankPTA (K=3, K=5) | −0.6pp | Rich-get-richer collapse (K=3 ≡ K=5) |
| BankV2 (anti-collapse) | −4.4 to −5.5pp | Growth rule plants CLIP error modes |
| CompactPTA (distance gate) | −0.2 to −6.0pp | Gate removes correct updates along with wrong ones |
| BankCompact / GaussCompact | −5.9 to −21.7pp | Compounds individual errors |
| AugPTA (augmentation ensembling) | −0.2 to −2.0pp | Ensembled logits *less* accurate than single view |
| AnchorPTA (text-anchor damping) | −0.01/−0.02pp | Damping degenerates to uniform scaling after warm-up |

**The pattern**: Offline gates pass because they use favorable proxies (true-label statistics, offline k-means). Online, every mechanism must estimate from CLIP's biased write stream, and each fails for a distinct, identifiable reason — but all share the same root cause.

**Hyperparameter sweep**: Default α=0.01 / T=20 sits at the top of its local grid. Not a strawman.

**Status**: Closed. No mechanism whose advantage depends on statistics estimated from CLIP's own guesses has beaten base PTA.

---

### D. DEC Certainty Regularizer (Phase 1 only)

**Hypothesis**: Entropy + logit-norm temperature modulates write weight to reduce impact of noisy high-confidence samples.

**Status**: Implemented, committed, **not GPU-validated**. Offline pre-validation found the committed temperature mapping runs 0.72–0.83 (near-constant, not an adaptive filter) with direction *inverted* relative to the strongest purity signal (entropy tertiles on dtd: low-H 63.2% purity / high-H 16.4%). Would need reparameterization before cluster runs.

**Recommendation**: Reparameterize as entropy-gated/entropy-weighted write rule, or close.

---

## Five Facts That Constrain All Future Work

1. **The prototype is the right lever.** On CLIP↔prototype ties, the prototype is the more reliable predictor on every dataset. The method works; improving it is the goal.

2. **The write-stream bias is the binding constraint.** All writes are CLIP-driven with no ground truth. Every mechanism that must estimate statistics from this stream (variance, cluster membership, trust scores) fails because CLIP's biased guesses corrupt the estimates.

3. **Diagnostics are strong; interventions are not.** Patch agreement (+30pp), prototype confusability (+16pp), multi-view consistency (+27pp) — all real, reproducible signals. None converts to an accuracy lever via any scalar mechanism tested (gate, up-weight, down-weight, repulsion, damping, ensembling).

4. **The space of scalar trust levers is exhausted.** Write × {gate, up, down, two-sided}, read × {up, down, two-sided}, 3 signal sources, all combinations — consistent null or negative results. The bottleneck is not signal quality but the structural overlap between "tie" samples and "untrusted" samples (~88% overlap regardless of signal source).

5. **PTA's default EMA is already optimal within this class.** The α/T sweep confirms the default is not a strawman. Every modification to the write rule or fusion rule degrades performance.

---

## What Would Need to Be True for a Pivot to Succeed

A fundamentally different mechanism is needed — one that does *not* rely on:
- Scalar trust scores applied per-sample or per-class
- Statistics estimated from CLIP's own biased write stream
- Gating or reweighting the EMA write

Possible directions that have NOT been tested (and their risk):
- **External grounding**: Using a second model, retrieval database, or self-training loop to provide pseudo-labels independent of CLIP's own zero-shot. High effort, untested.
- **Batch-aware methods**: Processing groups of samples jointly (memory buffer, contrastive objectives). Requires architectural changes to PTA's sequential design.
- **Representation-level change**: Moving beyond ViT-B/16 CLIP Surgery features entirely (stronger backbone, different pre-training). Different research question.
- **Abandoning prototype improvement**: Accepting PTA's +2–3pp as the ceiling for this specific adaptation mechanism and pivoting to a different TTA paradigm entirely.

---

*For detailed methodology, see [METHODOLOGY.md](./METHODOLOGY.md). For per-phase analysis, see individual phase reports in this directory.*
