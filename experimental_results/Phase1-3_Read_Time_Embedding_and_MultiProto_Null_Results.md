# Read-Time Fusion, Foreground-Weighted Embeddings, and Image-Level Multi-Prototype: Three Null Results

**Scope**: ViT-B/16 backbone, CLIP Surgery, 3 primary datasets (dtd, oxford_flowers, oxford_pets), 4 seeds
(1-4) for the GPU sweeps; single-pass diagnostics use seed 1. Follow-up to
`PTA_Limitations_and_Patch_Signal_Analysis.md` and `PatchModPTA_Purity_Separability_Trust_Analysis.md`
(hereafter "the prior reports"), which established that the confidence x patch-agreement trust signal
is a strong *diagnostic* of correctness but that every **write-time** lever built on it (hard gate,
up-weight, down-weight, two-sided) fails to beat base PTA (Parts 4a/4b).

## Background: What This Report Adds

The prior reports left three levers untested, each targeting a different part of the PTA pipeline:

1. **Read-time (fusion) trust weighting** — leave the prototype write untouched; scale how much the
   *existing* prototype is trusted at prediction time, per sample.
2. **Foreground-weighted embedding** — replace the CLS token feeding the prototype with a CLIP-Surgery
   relevance-weighted pool of patch tokens, attacking prototype drift at the representation level instead
   of gating a separate signal.
3. **Image-level multi-prototype** — cluster whole-image (CLS) embeddings per class (K centers) instead of
   maintaining one running mean, since patch-level clustering was proven non-separable but image-level
   embeddings were never checked.

All three were pre-registered with numeric go/no-go rules before any adapter was built (see
`/home/brandon/.claude/plans/toasty-pondering-eagle.md`). **All three are negative.** Two were killed at
the cheap-diagnostic stage before any full adapter sweep was run, exactly as designed; the third
(read-time fusion) was run to completion (120 GPU runs total) and understood mechanistically before being
closed out.

---

## Part 1: Trust-Adaptive Read-Time Fusion

### 1.0 Mechanism

New adapter `models/trust_fusion_pta.py` + `models/fusion.py::TrustAdaptiveFusion`. The write rule is
byte-identical to base PTA (`PTAImageLevel.update_prototypes`, unmodified). Only the fusion weight on the
image prototype is scaled per sample:

```
trusted  = (clip_margin >= 0.2) and (patch_vote_pred(topk20) == argmax(clip))
tau_eff  = tau_image_proto * (tau_scale_trusted if trusted else tau_scale_untrusted)
final    = clip + tau_eff * image_proto
```

`(tau_scale_trusted, tau_scale_untrusted) = (1.0, 1.0)` reproduces base PTA exactly (verified: dtd seed 1
control run scored 47.64% vs. the recorded base-PTA 47.70%, a 1-sample discrepancy from the extra
read-only patch-vote forward pass, well within noise).

### 1.1 Sweep v1 — bidirectional trust scaling

7 settings (control + 6 variants spanning `tau_scale_trusted in {0.3, 0.5, 1.0}` x
`tau_scale_untrusted in {1.0, 1.5, 2.0}`) x 3 datasets x 4 seeds = 84 runs
(`scripts/run_trust_fusion_sweep.sh`, `outputs/result_trust_fusion.txt`).

| Setting | dtd | flowers | pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control (1.0, 1.0) | 47.52 | 74.62 | 91.08 | 71.073 | +0.000 |
| downT-0.5 (0.5, 1.0) | 47.74 | 74.41 | 91.05 | 71.062 | -0.011 |
| downT-0.3 (0.3, 1.0) | 47.71 | 74.33 | 91.01 | 71.017 | -0.057 |
| upU-1.5 (1.0, 1.5) | 47.46 | 74.75 | 91.11 | 71.104 | +0.031 |
| upU-2.0 (1.0, 2.0) | 47.33 | 74.77 | 91.12 | 71.072 | -0.002 |
| both-mod (0.5, 1.5) | 47.68 | 74.53 | 91.08 | 71.095 | +0.022 |
| both-agg (0.3, 2.0) | 47.52 | 74.47 | 91.06 | 71.018 | -0.055 |

Every setting lands within ±0.06pp of control on the 3-dataset average — all inside the ~0.5pp per-seed
noise band established in the prior reports. **All 6 variants: no-go.**

### 1.2 Why the sweep is flat: regime/tie overlap analysis

A linear reweighting of `tau_image_proto` can only change the fused argmax on samples where the two
components already disagree — i.e., **tie** samples (`clip.argmax != image_proto.argmax`, same definition
as `PTA_Limitations...md` Part 1.1). Cross-tabulating the causal `trust_regime` against tie membership on
the pooled control records (31,296 samples, 3 datasets x 4 seeds):

| | Tie | Non-tie |
|---|---|---|
| Trusted (n=14,688) | 646 (4.4% of trusted) | 14,042 |
| Untrusted (n=16,608) | 4,581 (27.6% of untrusted) | 12,027 |

Only 4.4% of "trusted" (confident + patch-agreeing) samples are ties — the `tau_scale_trusted` sweep
therefore has almost no samples where it *could* flip a decision, explaining why `downT-0.3`/`downT-0.5`
are flat. Conversely, 87.6% of all 5,227 ties (4,581/5,227) fall in the "untrusted" bucket — but
upweighting that bucket (`upU-1.5`/`upU-2.0`) just reinforces whichever term already wins the tie at
`tau=100` (image_proto already tends to win ties at the base weight), so pushing the weight higher changes
almost nothing.

### 1.3 Sweep v2 — downweighting the untrusted (tie-concentrated) regime

The one direction the v1 sweep never tested: pulling `tau_image_proto` **down** specifically on untrusted
samples, to let CLIP's own vote compete more where the disagreement actually lives. 3 settings x 3
datasets x 4 seeds = 36 more runs (`scripts/run_trust_fusion_sweep_v2.sh`), appended to the same result
file.

| Setting | dtd | flowers | pets | Avg | Δ vs control | Tie Acc |
|---|---|---|---|---|---|---|
| downU-0.7 (1.0, 0.7) | 47.62 | 74.49 | 91.07 | 71.060 | -0.013 | 38.1% |
| downU-0.5 (1.0, 0.5) | 47.70 | 74.13 | 90.90 | 70.907 | -0.167 | 37.0% |
| downU-0.3 (1.0, 0.3) | 47.44 | 73.59 | 90.52 | 70.516 | **-0.558** | 34.6% |

Accuracy degrades **monotonically** as `tau_scale_untrusted` shrinks — the opposite of a hoped-for gain.
On ties specifically (control tie acc = 38.2%), `downU-0.3` drops tie accuracy to 34.6%: a direct
helps/hurts breakdown on ties (`downU-0.3` vs. control, same methodology as
`PTA_Limitations...md` Part 1.3) gives **helps=236, hurts=425, both_right=1571, both_wrong=2995 — help
rate 35.7%** (below the 50% break-even line). **All 3 variants: no-go.**

### 1.4 Mechanism & conclusion

This closes the loop on the read-time lever with a clean, opposite-of-hypothesis finding: even in the
disagreement zone (where CLIP's confidence x agreement signal says CLIP itself is *less* reliable),
image_proto's opinion is **still, on net, more often right than CLIP's own** for tie-breaking — reducing
image_proto's say there breaks more ties than it fixes, monotonically with strength. The confidence x
agreement signal is a good predictor of **absolute correctness** (Part 3.1 of the prior report), but this
result shows it is a poor predictor of **which of two disagreeing sources to trust** — a materially
different question that the signal was never validated against. Combined with 1.2's overlap analysis, no
scalar reweighting of `tau_image_proto` — up, down, or split by trust regime — has room to help: the
trusted regime rarely contains ties (nothing to fix there), and the untrusted regime's ties are already
better resolved by the existing tau=100 default than by any adjustment tested.

**Conclusion (negative):** trust-adaptive read-time fusion, across 10 settings (120 GPU runs) and both
directions of the hypothesis space, does not beat base PTA on the 3-dataset dev set. Unlike the write-side
levers (Part 4a/4b), which actively degraded accuracy, the read-time lever is largely *inert* except when
pushed toward down-weighting image_proto on ties, where it actively hurts — a cleaner, fully
mechanistically-explained null result.

---

## Part 2: Foreground-Weighted Embedding (Phase 2a diagnostic — killed before adapter build)

### 2.0 Mechanism tested

Cheap diagnostic (`scripts/check_surgery_embedding.py`, no adapter, no TTA loop, single forward pass per
image): for each image, build `fg_embedding = normalize(sum_i relu(relevance_i[top1]) * patch_i /
sum_i relu(relevance_i[top1]))` using CLIP-Surgery's per-patch, per-class relevance
(`utils/clip_inference.compute_surgery_scores`, already implemented, previously only used to filter the
Gaussian patch bank) restricted to the CLIP-predicted top-1 class. Standalone zero-shot accuracy of
`fg_embedding` alone is compared against CLS zero-shot accuracy — mirroring the "standalone accuracy"
methodology already used for the patch vote (prior report Part 3.2).

### 2.1 Result

| Dataset | N | CLS zero-shot acc | fg_embedding standalone acc | Δ (fg - cls) | mean cos(fg, cls) |
|---|---|---|---|---|---|
| dtd | 1692 | 44.09% | 42.26% | -1.83pp | 0.652 |
| oxford_flowers | 2463 | 70.73% | 26.63% | **-44.09pp** | 0.639 |
| oxford_pets | 3669 | 89.04% | 48.68% | **-40.37pp** | 0.648 |

Pre-registered go/no-go bar: proceed to the full adapter sweep only if `fg_embedding` standalone accuracy
is within ~10pp of CLS zero-shot on >=2/3 datasets. Only dtd clears the bar (1/3). **Verdict: STOP — Phase
2b adapter not built.**

### 2.2 Interpretation

This is the same failure shape the prior report found for the raw mean-pooled patch vote (Part 3.2: 20.9%
vs. 71.4% CLIP zero-shot on flowers) — a naive (here, relevance-weighted rather than uniformly-pooled)
patch aggregate is nowhere near as discriminative as the CLS token, which is specifically the token CLIP's
projection head was trained to align with text. Restricting relevance to the top-1 class (rather than
mean-pooling all patches) does noticeably narrow the gap on the hardest dataset (dtd: -1.8pp vs. patch
vote's much larger gaps elsewhere) but the fine-grained collapse (-40 to -44pp) is decisive: this candidate
representation is not close enough to CLS to be a safe blend target for the adaptive prototype, and the
full adapter sweep (Phase 2b) was correctly never run.

---

## Part 3: Image-Level Multi-Prototype (Phase 3a diagnostic — killed before adapter build)

### 3.0 Mechanism tested

Cheap diagnostic (`scripts/check_image_level_separability.py`, no adapter, single feature-dump pass per
dataset): dump CLS embeddings + ground-truth labels for the whole test set (same retrospective use of
labels as the prior report's Part 2.2 bank-separability analysis), then per class run a deterministic
batch k-means (k=1,2,3) on that class's embeddings and compute the identical margin metric as Part 2.2:
intra-class coherence (mean pairwise similarity among a class's own k centers) minus nearest cross-class
confusability (max similarity to the single most similar other class's centers).

### 3.1 Result

| Dataset | k=1 nearest-cross (ref) | k=2 margin | k=3 margin |
|---|---|---|---|
| dtd | 0.943 | -0.045 | -0.066 |
| oxford_flowers | 0.945 | **+0.005** | -0.010 |
| oxford_pets | 0.968 | -0.020 | -0.030 |

Pre-registered go/no-go bar: proceed to Phase 3b only if k>=2 margins are positive on >=2/3 datasets. Only
oxford_flowers at k=2 is (barely) positive; 1/3 datasets clears the bar. **Verdict: STOP — Phase 3b
adapter not built.**

### 3.2 Interpretation

The margins here (-0.02 to -0.07) are an order of magnitude less negative than the patch-level margins
from the prior report (-0.11 to -0.24) — confirming the intuition that image-level CLIP embeddings really
are far more separable than patch tokens. But "far less bad" is not "positive": base cosine similarities
between *any* two natural-image CLIP embeddings are already very high (0.90-0.97), so splitting a class's
already-tight cluster into k>=2 sub-clusters mostly just measures k-means noise at this similarity scale,
not genuine sub-class structure the single running mean is failing to capture. The pre-registered stopping
rule correctly avoided sinking engineering effort into a multi-prototype adapter whose representational
substrate doesn't clear its own separability bar.

---

## Summary

| Direction | Stage reached | Verdict | Key number |
|---|---|---|---|
| Read-time trust-adaptive fusion | Full sweep (120 runs) | **No-go** | best avg delta +0.031pp (upU-1.5), worst -0.558pp (downU-0.3); all within/below noise |
| Foreground-weighted embedding | Cheap diagnostic only | **No-go** | fg_embedding standalone acc -40 to -44pp vs. CLS on fine-grained sets |
| Image-level multi-prototype | Cheap diagnostic only | **No-go** | k>=2 margins negative on 2/3 datasets (though far less negative than patch-level) |

All three previously-untested levers on the confidence x patch-agreement trust signal — write-time (prior
report Parts 4a/4b), and now read-time (this report Part 1) — are negative. The signal remains validated
as a strong *diagnostic* of prediction correctness in the aggregate, but no mechanism tested so far (six
in total, across both reports) converts it into an accuracy lever anywhere in the PTA pipeline: not the
write, not the read, and not the underlying embedding. The image-level multi-prototype idea is
independently ruled out by a representational ceiling (near-zero clustering margins) unrelated to the
trust signal.

**Implication for future work**: further scalar-reweighting variations of the confidence x agreement
signal are unlikely to be productive — the space (write x {gate, up, down, two-sided}, read x {up, down,
two-sided}) is now exhausted in both directions with consistent null-or-negative results. Any future
attempt to beat base PTA should look for a signal or mechanism structurally different from "patch-vote
agreement as a trust score," since that specific idea has now been tested exhaustively and the ceiling
appears to be the confidence-agreement signal's inherent limitation as a *relative* (this-source-vs-that-
source) rather than *absolute* correctness predictor (Part 1.4).

---

## Appendix: Reproducing this report

- Part 1: `bash scripts/run_trust_fusion_sweep.sh --run` then `bash scripts/run_trust_fusion_sweep_v2.sh --run`,
  then `python scripts/analyze_trust_fusion.py`. Raw results: `outputs/result_trust_fusion.txt`,
  `outputs/records_trust_fusion/`, `outputs/trust_fusion_report.md`.
- Part 2: `python scripts/check_surgery_embedding.py --datasets dtd/oxford_flowers/oxford_pets --seed 1`.
  Output: `outputs/surgery_embedding_diagnostic.md`.
- Part 3: `python scripts/check_image_level_separability.py --datasets dtd/oxford_flowers/oxford_pets --seed 1`.
  Output: `outputs/image_level_separability_report.md`.
- Full experiment plan and pre-registered go/no-go rules: `/home/brandon/.claude/plans/toasty-pondering-eagle.md`.
