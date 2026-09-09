# Phase 4-6: Class-Prediction Calibration, Prototype Confusability, and Multi-View Consistency — Null Results

**Scope**: ViT-B/16 backbone, CLIP Surgery, 3 primary datasets (dtd, oxford_flowers, oxford_pets), 4 seeds
(1-4). Follow-up to `Phase1-3_Read_Time_Embedding_and_MultiProto_Null_Results.md`, which closed out every
lever built on the confidence x patch-agreement signal (write-gating, write-reweighting, read-time
reweighting) and two patch/embedding-derived directions (foreground-weighted embedding, image-level
multi-prototype). All of those shared one root cause: patch-level content is structurally non-separable,
and CLS-level embeddings don't cluster either.

This report covers three directions chosen specifically to share **nothing** with patch content or
patch-vote signals: a purely statistical prediction-frequency calibration, a prototype-bank-native
confusability monitor (class-level, not sample-level), and a multi-view perturbation-consistency signal.
Each was tested to a pre-registered go/no-go rule (see
`/home/brandon/.claude/plans/toasty-pondering-eagle.md`).

**All three are negative** — but Part 3 in particular provides the strongest positive *diagnostic* signal
found across this entire investigation, and the way it still fails to convert into an accuracy win
substantially sharpens the mechanistic explanation for why read-time reweighting doesn't work, regardless
of what feeds it.

---

## Part 1: Class-Prediction-Frequency Calibration

### 1.0 Mechanism

PTA processes test images sequentially; if the running prototype drifts toward one class, subsequent
similar images can be pulled toward the same wrong class. This direction checks a purely statistical
correction: track the running frequency of the adapter's own predictions per class, and subtract a
log-prior-ratio term from over-predicted classes' logits (classic long-tail/prior-shift logit adjustment):

```
running_freq[c] = (samples predicted as c so far) / (samples seen so far)
adjusted_logits[c] = final_logits[c] - lam * log(running_freq[c] / expected_freq[c] + eps)
```

`expected_freq = 1/C` (uniform). `lam=0` is the control.

### 1.1 Offline diagnostic (zero new GPU runs)

Every completed run in this project logs per-sample `final_logits` and `target` to `records.jsonl`. This
means the idea can be replayed **entirely offline** against the already-existing `TFP-control-*` records
(control ≡ base PTA) from the prior round — no adaptation runs needed for a first read.
`scripts/check_prior_calibration_signal.py` replays the sequence causally (running_freq derived only from
the replay's own emitted predictions) and recomputes accuracy under a swept `lam`.

**Initial grid** (`lam ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0}`):

| lam | dtd | oxford_flowers | oxford_pets | Avg | Δ vs lam=0 |
|---|---|---|---|---|---|
| 0.0 | 47.52 | 74.62 | 91.07 | 71.072 | +0.000 |
| 0.05 | 47.62 | 74.63 | 91.09 | 71.114 | +0.042 |
| 0.1 | 47.56 | 74.72 | 91.13 | 71.135 | +0.063 |
| 0.2 | 47.68 | 74.72 | 91.18 | 71.191 | +0.119 |
| 0.5 | 47.72 | 74.64 | 91.13 | 71.166 | +0.094 |
| 1.0 | 47.84 | 74.57 | 91.24 | 71.220 | +0.148 |

**Widened grid** (`lam` up to 12, since the trend was still rising at `lam=1.0`):

| lam | dtd | oxford_flowers | oxford_pets | Avg | Δ vs lam=0 |
|---|---|---|---|---|---|
| 1.0 | 47.84 | 74.57 | 91.24 | 71.220 | +0.148 |
| 2.0 | 47.90 | 73.95 | 91.20 | 71.018 | -0.054 |
| 3.0 | 47.86 | 73.41 | 91.29 | 70.852 | -0.220 |
| 5.0 | 47.74 | 72.11 | 91.20 | 70.350 | -0.722 |
| 8.0 | 47.25 | 69.59 | 90.80 | 69.214 | -1.858 |
| 12.0 | 46.29 | 67.01 | 90.48 | 67.928 | -3.144 |

The effect peaks near `lam≈1-3` at a **maximum average gain of +0.15pp** (never clearing the 0.3pp bar on
more than 1/3 datasets at any `lam`), then reverses sharply. `oxford_flowers` (102 classes) degrades
monotonically past `lam=1.0`, while `dtd` (47 classes) and `oxford_pets` (37 classes) hold up slightly
better — consistent with the mechanism being noisier for datasets with more classes, since each class's
`running_freq` estimate is based on fewer expected samples per class as `C` grows.

**Go/no-go**: requires some `lam>0` to beat `lam=0` by >0.3pp on ≥2/3 datasets. **Never cleared on any
`lam`. No-go — Phase 1b (adapter build) was not started.**

### 1.2 Conclusion

The self-reinforcing-drift hypothesis behind this mechanism is directionally correct (the effect is never
negative for small `lam`), but the magnitude is negligible (~0.15pp peak) and it becomes actively harmful
once `lam` is large enough to matter, especially on higher-class-count datasets. Prediction-frequency
skew is real but too small and too dataset-size-sensitive a signal to be useful here.

---

## Part 2: Prototype Confusability Monitor

### 2.0 Mechanism

Unlike every other lever in this line of work, this one uses no CLIP confidence and no patch content —
only the prototype bank's own geometry. After each write, compute each class's cosine similarity to its
nearest *other* class's prototype ("confusability"):

```
sims = normalize(prototype_state) @ normalize(prototype_state).T   # [C, C]
sims.fill_diagonal_(-1)
nearest_other[c] = sims[c].max()
```

The hypothesis: classes whose prototype has drifted close to a different class's prototype are the ones
losing accuracy, and freezing further writes to a class once it crosses a confusability threshold should
protect it from further contamination.

### 2.1 Phase 2a — cheap diagnostic (single extra matmul per step, no extra CLIP calls)

`scripts/check_prototype_confusability.py` runs the **unmodified** base-PTA loop and, at the end, checks
whether classes in the bottom confusability tercile (least confusable) have higher accuracy than the top
tercile (most confusable):

| Dataset | Confusability range (min/p25/median/p75/max) | Low-tercile acc | High-tercile acc | Gap |
|---|---|---|---|---|
| dtd (C=47) | 0.853 / 0.929 / 0.946 / 0.965 / 0.978 | 57.59 | 41.11 | **+16.48pp** |
| oxford_flowers (C=102) | 0.000 / 0.940 / 0.960 / 0.978 / 0.999 | 73.27 | 56.93 | **+16.33pp** |
| oxford_pets (C=37) | 0.926 / 0.965 / 0.973 / 0.977 / 0.997 | 93.49 | 84.31 | **+9.17pp** |

**Go/no-go**: ≥5pp gap on ≥2/3 datasets. **3/3 cleared — the first positive diagnostic in the entire
campaign** (stronger than nearly all prior signal checks). Note the raw cosine values sit in a narrow,
dataset-dependent band (e.g. dtd tops out at 0.978, oxford_pets bottoms out at 0.926) — a single fixed
absolute cutoff would not transfer across datasets, so Phase 2b uses a **running percentile** of the
current confusability distribution instead of an absolute threshold.

### 2.2 Phase 2b — adapter (write-freeze gate)

`models/confusability_gated_pta.py`: identical write math to base PTA except a class is **permanently
frozen** (excluded from all further EMA writes) once its confusability exceeds the `freeze_percentile`-th
percentile of all *already-written* classes' current confusability (a minimum of 5 written classes is
required before any freeze fires, to avoid spurious early freezing when the bank is mostly still empty —
an earlier version without this guard froze 26/47 dtd classes within the first 300 samples).
`freeze_percentile=100` is the control (threshold == current max, nothing can exceed it, exactly
reproduces base PTA — verified: 47.64% on dtd seed 1, matching the known baseline to 2 decimal places).

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control | Mean final n_frozen (dtd/flowers/pets) |
|---|---|---|---|---|---|---|
| control (p100) | 47.52 | 74.62 | 91.08 | 71.073 | +0.000 | 0.0 / 0.0 / 0.0 |
| p95 | 45.57 | 73.98 | 86.97 | 68.838 | -2.235 | 3.0 / 4.0 / 3.0 |
| p90 | 44.36 | 72.40 | 85.39 | 67.382 | -3.691 | 5.8 / 11.0 / 8.5 |
| p80 | 41.81 | 70.46 | 80.86 | 64.377 | -6.697 | 12.5 / 21.2 / 14.2 |
| p66 | 39.89 | 66.07 | 77.12 | 61.027 | -10.046 | 19.5 / 39.2 / 20.2 |

The regression is **monotonic and severe**, appearing even at the gentlest setting tested (p95, freezing
only ~3-4 classes on average): -2.2pp average, with oxford_pets alone dropping 4.1pp. All four settings:
**no-go.**

### 2.3 Conclusion

This is the same failure shape as the prior round's write-gate study (Part 4a there): a signal that
correctly *identifies* troubled classes does not make a safe *intervention* target. Freezing a class stops
it from absorbing more contaminating writes, but it equally stops it from absorbing the correct writes that
could have fixed the confusion — and a class only becomes "confusable" in the first place because it's
still ambiguous, i.e. exactly the class that most needs continued, not less, refinement. The confusability
signal is a genuine diagnostic of where accuracy is being lost; it is not, on its own, a usable trigger for
a hard write intervention.

---

## Part 3: Multi-View Consistency Trust Signal

### 3.0 Mechanism

Reuses `utils/augmentation.py:_augment_image` (rotation, affine, brightness/contrast, occasional
grayscale/edge-blend — already validated, already used to build the patch-level Gaussian bank) to generate
`N-1` perturbed copies of each test image. Concretely, for every single test image, before the final
prediction is made (`models/view_consistency_pta.py`):

```python
top1 = int(clip_logits.argmax(dim=-1).item())            # CLIP's own zero-shot guess on the real image

aug_views = [_augment_image(images) for _ in range(n_views - 1)]   # 3 perturbed copies (n_views=4 default)
batched = torch.cat(aug_views, dim=0).cuda()
aug_feats = encoder.encode_image(batched)                 # run CLIP on the 3 perturbed copies from scratch
aug_logits = 100.0 * aug_feats @ text_embeddings
aug_preds = aug_logits.argmax(dim=-1)                      # [3] -- each perturbed copy's own top-1 guess

agreement = float((aug_preds == top1).float().mean())      # fraction of the 3 copies that agree with the original
trusted   = agreement >= agreement_thresh                   # default agreement_thresh = 1.0 -> ALL 3 must agree
```

Each of the 3 extra views is the *same* image, slightly distorted, re-run through CLIP independently. If
CLIP still gives the same top-1 class on all 3 distorted copies, that is evidence the original prediction
is robust; if even one distorted copy flips to a different class, that is a red flag. This computation is
purely diagnostic bookkeeping — it does **not** touch the prototype write step, which stays byte-identical
to base PTA.

This is a fundamentally different "second opinion" than spatial patch decomposition — it tests robustness
under stochastic perturbation of the whole image, with precedent in the TTA literature (TPT/MEMO-style
confidence-filtered augmentation ensembles) — and can plug directly into the **already-built**
`TrustAdaptiveFusion` (`models/fusion.py`) from the prior round's Phase 1, swapping only the trust-signal
source:

```python
scale     = tau_scale_trusted if trusted else tau_scale_untrusted
tau_eff   = tau_image_proto * scale                        # tau_image_proto = 100.0, the base-PTA weight
final     = tau_text * clip_logits + tau_eff * image_proto_logits
```

The only thing that varies across the Phase 3b sweep below is `(tau_scale_trusted, tau_scale_untrusted)`
-- how much weight the prototype's opinion gets, conditioned on whether all 3 perturbed views agreed with
CLIP's original guess.

### 3.1 Phase 3a — cheap diagnostic (one pass per image, N=4 views, no TTA state)

`scripts/check_view_consistency_signal.py` measures the purity gap: accuracy of the original image's own
zero-shot prediction when all augmented views agree vs. when at least one disagrees.

| Dataset | N agree | Agree Acc | N disagree | Disagree Acc | Gap | Samples/sec |
|---|---|---|---|---|---|---|
| dtd | 395 | 65.06 | 1297 | 37.70 | **+27.36** | 60.4 |
| oxford_flowers | 637 | 90.27 | 1826 | 63.91 | **+26.36** | 59.3 |
| oxford_pets | 2062 | 96.36 | 1607 | 79.78 | **+16.59** | 60.8 |

**Go/no-go**: purity gap ≥9pp (low end of the already-validated patch-vote gap) on ≥2/3 datasets. **3/3
cleared, comfortably — this is the strongest diagnostic signal found across the entire campaign** (dtd and
oxford_flowers gaps exceed the patch-vote signal's typical range). Throughput (~60 samples/sec even with 4
forward passes per image) confirmed a full sequential-TTA sweep was feasible on the 16GB/no-slurm machine.

### 3.2 Phase 3b — adapter sweep

`models/view_consistency_pta.py`: write side byte-identical to base PTA; only the `TrustAdaptiveFusion`
read weight is scaled by the multi-view `agreement` signal instead of the patch-vote signal. Re-swept the
**exact same** `(tau_scale_trusted, tau_scale_untrusted)` grid as the prior round's patch-vote-based
`trust_fusion_pta` study, for direct comparability (control verified: 40.67% on a 300-sample dtd slice,
exactly matching base PTA).

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control | 47.52 | 74.62 | 91.08 | 71.073 | +0.000 |
| downT-0.3 | 47.53 | 74.61 | 91.03 | 71.059 | -0.014 |
| downT-0.5 | 47.62 | 74.63 | 91.09 | 71.115 | +0.042 |
| upU-1.5 | 47.48 | 74.92 | 91.12 | 71.169 | +0.096 |
| upU-2.0 | 47.27 | 75.02 | 91.10 | 71.130 | +0.057 |
| both-mod | 47.58 | 74.93 | 91.12 | 71.210 | +0.137 |
| both-agg | 47.28 | 75.01 | 91.06 | 71.117 | +0.044 |

Go/no-go bar: >0.3pp avg gain, no regression >0.5pp. **All 7 settings: no-go** — but every setting is now
non-negative (unlike the patch-vote sweep, where `downU` settings actively regressed accuracy); the best
setting (`both-mod`) reaches only +0.14pp, still 2x short of the bar.

What each setting name means — every row is a `(tau_scale_trusted, tau_scale_untrusted)` pair, reusing the
exact grid from the original patch-vote study for apples-to-apples comparison:

| Setting | tau_scale_trusted | tau_scale_untrusted | What it does |
|---|---|---|---|
| control | 1.0 | 1.0 | No change — reproduces base PTA exactly |
| downT-0.5 | 0.5 | 1.0 | When robust (all views agree), trust the prototype *half* as much. Shaky case untouched. |
| downT-0.3 | 0.3 | 1.0 | Same idea, stronger: prototype counts for only 30% as much when robust. |
| upU-1.5 | 1.0 | 1.5 | When shaky (some view disagreed), trust the prototype 1.5x more. Robust case untouched. |
| upU-2.0 | 1.0 | 2.0 | Same idea, stronger: 2x more trust in the shaky case. |
| both-mod | 0.5 | 1.5 | **Both** levers pulled together, at moderate strength: down-weight the prototype when robust *and* up-weight it when shaky, simultaneously. |
| both-agg | 0.3 | 2.0 | Same combination, at aggressive strength (0.3 / 2.0 instead of 0.5 / 1.5). |

`downT`/`upU` isolate one side of the lever each; `both-mod`/`both-agg` are not a different mechanism, just
the combined change (both sides pulled at once) at two strengths — "mod" (moderate) and "agg" (aggressive).

**Mechanism check (why, despite a much stronger diagnostic signal, the outcome doesn't change):** cross-
tabulating the control run's `trust_regime` against tie membership (`clip.argmax != image_proto.argmax`,
the only samples any reweighting can flip):

| | This report (multi-view) | Prior report (patch-vote) |
|---|---|---|
| Fraction of ties that are "untrusted" | 88.1% | 87.6% |
| Fraction of "trusted" samples that are ties | 5.0% | 4.4% |

These numbers are nearly identical despite the two trust signals being computed from completely
independent evidence (image perturbation vs. spatial patch content). This generalizes the prior round's
finding: it isn't that the patch-vote signal specifically was too weak — **any** reasonable per-sample
uncertainty proxy will show this same structure, because a sample becomes a "tie" (CLIP and the prototype
disagree) precisely when the underlying prediction is genuinely ambiguous, and any independent proxy for
ambiguity (consistency under perturbation, patch agreement, confidence margin) will *also* flag that same
sample as untrusted. There is essentially no room for a scalar reweighting rule to selectively act on ties
without acting almost identically on the (much larger) set of already-correctly-classified confident
samples too.

### 3.3 Conclusion

The multi-view consistency signal is a **better-validated diagnostic than the patch-vote signal** (larger
purity gap on 2/3 datasets) and is structurally unrelated to it, yet produces the **same null result** when
plugged into the same read-time reweighting lever. This is strong evidence that read-time trust-adaptive
fusion is a structurally limited lever *regardless of the trust signal's source or quality* — the
bottleneck is the near-total overlap between "tie" and "untrusted," not the quality of any one signal.

---

## Summary

| Direction | Diagnostic result | Adapter result | Verdict |
|---|---|---|---|
| Class-prediction-frequency calibration | N/A (offline replay only) | Peak +0.15pp, reverses beyond `lam≈3`, worse on high-class-count datasets | **No-go** |
| Prototype confusability monitor | Strong (+9 to +16.5pp tercile gap, 3/3 datasets) | Monotonic regression at every freeze strength (-2.2 to -10.0pp) | **No-go** |
| Multi-view consistency trust signal | Strongest in the campaign (+16.6 to +27.4pp purity gap, 3/3 datasets) | Best setting +0.14pp, all non-negative but none clear the bar | **No-go** |

No setting from any of the three directions cleared its dev-set go/no-go bar, so none were promoted to
held-out validation (`caltech101`/`eurosat`/`ucf101`).

Combined with the prior round, this closes out: patch-content fusion, write-gating, write-reweighting,
read-time reweighting (two independent trust signals), foreground-weighted embeddings, image-level
multi-prototype clustering, prediction-frequency calibration, and prototype-confusability write-freezing —
**eight structurally distinct mechanisms**, several of them (confusability, multi-view consistency) backed
by the strongest diagnostic signals found in the whole investigation. The consistent pattern: PTA's
image-level prototype is already a strong, near-optimal use of the available per-sample evidence, and
every scalar lever tested — on the write side or the read side, fed by CLIP confidence, patch content,
prototype geometry, prediction history, or perturbation robustness — either does nothing or actively hurts
once pushed hard enough to matter. Further improvement attempts on this specific class of scalar
gating/reweighting mechanisms are unlikely to be productive; a fundamentally different mechanism (e.g. one
that doesn't rely on a single scalar "trust" applied uniformly per sample or per class) would be needed.

---

## Appendix: Reproduction

```bash
# Part 1 — offline calibration diagnostic (no GPU)
python scripts/check_prior_calibration_signal.py \
    --lams 0,0.05,0.1,0.2,0.5,1.0 --out outputs/prior_calibration_diagnostic.md
python scripts/check_prior_calibration_signal.py \
    --lams 0,1.0,2.0,3.0,5.0,8.0,12.0 --out outputs/prior_calibration_diagnostic_wide.md

# Part 2 — confusability diagnostic + sweep
python scripts/check_prototype_confusability.py \
    --datasets dtd/oxford_flowers/oxford_pets --seed 1 \
    --out outputs/prototype_confusability_report.md
bash scripts/run_confusability_sweep.sh --run
python scripts/analyze_confusability.py

# Part 3 — view-consistency diagnostic + sweep
python scripts/check_view_consistency_signal.py \
    --datasets dtd/oxford_flowers/oxford_pets --seed 1 --n-views 4 \
    --out outputs/view_consistency_diagnostic.md
MAX_PARALLEL=2 bash scripts/run_view_consistency_sweep.sh --run
python scripts/analyze_view_consistency.py
```
