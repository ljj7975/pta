# Patch-Level Fix Experiments — Plan (Round 2: E-H)

`outputs/offline_experiments_plan.md` (Experiments A-D) diagnosed *why*
patch-level prototypes don't help: writes are gated only by CLIP's own
confidence, with no ground truth, so a confidently-wrong CLIP call
contaminates the bank in a way nothing derived from CLIP's own signals can
detect. Baseline PTA has the same root exposure (`models/image_level/pta_image.py`
also gates purely on `softmax(clip_logits)`) but survives it because
mistakes get *diluted* — one running mean per class, continuously
EMA-blended, so a minority of wrong writes gets averaged away. Patch-level
instead *segregates* patches into many separate clusters and lets the most-
repeated one win, so a systematic, repeated mistake can form its own
cluster and get reinforced instead of diluted.

This round tests four concrete fixes, all discussed and agreed with the
user. Each aims at either giving patch-level the same "dilution" property
baseline PTA has, or reducing how much a single (possibly-wrong) CLIP
opinion controls the outcome.

Scope constraint agreed with the user: any TTA step may run **one model**
(CLIP or CLIP Surgery — not both) on the **original, unaugmented** image,
any number of times. No image augmentation, no second model (e.g. DINO is
out).

Status: **Phase 1 complete for all four (E, F, G, H).** See "Results"
below. Bottom line: E, F, and G are dead ends; H is a real, strong signal
and the clear priority for Phase 2.

**Round 3 (I, J, K)**: before building a live implementation on top of H,
the user raised a concrete objection to H's pooling choice (mean cosine
similarity can be diluted by background patches) and asked for a proper
comparison of patch-level aggregation signals first. I = compare pooling
variants offline; J = deep-validate the shortlist (multi-seed, margin-graded,
per-class, cross-signal correlation); K = live soft-down-weighted bank
writes built on J's recommendation, extended to all 5 dev datasets if it
clears SUPPORT. Status: **I and J complete.** I: the objection was right,
especially on oxford_pets (standalone accuracy and agreement rate both ~2x
under `topk20` vs `mean`). J: multi-seed-stable, margin carries graded
information (not just binary), correlates with but isn't explained by
CLIP's per-class difficulty, and is confirmed non-redundant with both the
write-confidence split and Experiment F's confusability margin. Added
`otsu_mean` (a hyperparameter-free alternative to `topk20`'s fixed k, via
Otsu's threshold method) at the user's request; it passed every I and J
check exactly as cleanly as `topk20`, with neither dominating the other
across all three datasets (otsu stronger on dtd/flowers, topk20 stronger
on pets). Final recommendation: **both `topk20` and `otsu_mean`**, each
with a continuous margin-derived weight, carried forward as parallel
candidates for Experiment K.

**K: implemented, piloted, and statistically validated at
`disagree_discount=0.3` (both aggregations, full 3-dataset × 4-seed sweep)
— null result.** No real improvement over vanilla `PatchModPTA-CS` on any
dataset (all deltas within noise, `p_sign` far from significant); the
promising single-seed pilot signal on oxford_flowers (+0.61pp) did not
replicate across seeds. Still loses to plain PTA-CS on oxford_flowers/
oxford_pets, same as before this round. See "K" section below for the full
write-up, two real bugs caught and fixed along the way, and what this
does/doesn't say about the corroboration signal itself.

---

## E — Appearance decay (EMA instead of a running count)

**Idea**: `models/patch_level/gaussian_patch.py` currently does
`appearance = appearance + (1.0 if occurred else 0.0)` — a flat, permanent
counter. A wrong cluster that gets reinforced early stays "trusted" forever.
Baseline PTA never does this — its EMA (`w_new = 1 - exp(-w/T)`) always lets
recent evidence outweigh old evidence. Give patch-level appearance the same
property: `appearance = decay*appearance + (1-decay)*occurred`.

**Why this needs a live rerun, not offline replay**: appearance directly
controls how much each cluster is trusted *during* the run (it's read back
into the `weighted_mean` aggregation immediately, not just at the end), so
changing its update rule changes the whole trajectory of what the bank
looks like at every subsequent step — not something recoverable from the
already-frozen final bank dumps.

**Method**:
1. Add a `appearance_decay` config knob to `GaussianPatchLevel` (default
   `1.0` = current behavior, so this is opt-in and doesn't touch the
   existing arms).
2. Pilot first: 1 dataset (dtd — cheapest, and the one where patch fusion
   is least bad today) x 1 seed x a small decay grid (e.g. `0.9, 0.95,
   0.99`) — quick GPU check for any signal before committing more compute.
3. If the pilot shows a real, non-noise move in the right direction, extend
   to all 3 datasets x 4 seeds x the most promising decay value(s), scored
   against the same pre-registered noise floor used throughout this study
   (`outputs/patch_benefit_prereg.md`).

**Cost**: GPU, live TTA reruns (cheap pilot first, full sweep only if
justified).

---

## F — Confusability-weighted appearance

**Idea**: Experiment C already tested a specificity-style reweighting
(TCR — density x cross-class frequency correction) and found no real gain.
But TCR's cross-class check was a coarse frequency proxy, not the actual
geometric confusability metric this study already built and validated in
Experiment A (variance-aware, Mahalanobis-style intra- vs. nearest-cross-
class match score, `scripts/analyze_prototype_separability_v2.py`). This
tests the sharper version: `trust = appearance * f(separation_margin)`,
where a cluster loses trust specifically for looking geometrically like a
*neighboring class's* cluster, not just for being common elsewhere.

**Method (Phase 1, offline, cheap)**: Extend `scripts/frozen_bank_eval.py`'s
existing control-vs-TCR harness (same frozen bank dumps, same simplified
`weighted_mean` aggregation, same disclosed caveat as Experiment C) with a
third condition: appearance weight multiplied by each cluster's
Experiment-A-style separation margin (computed once per bank, cheap).
Compare control vs. TCR vs. this new condition on accuracy, same stats
machinery as before.

**Method (Phase 2, only if Phase 1 shows signal)**: Move the reweighting
from a post-hoc read-time adjustment into the actual online write/aggregation
step (a live rerun), since a post-hoc reweighting can only change how
already-accumulated data is *read*, not what accumulates in the first
place.

**Cost**: Phase 1 is CPU-only, reuses existing bank dumps — minutes. Phase 2
is a live GPU rerun, contingent on Phase 1.

---

## G — Candidate-set-restricted patch voting (generalized, not fixed top-2)

**Idea**: patches currently can move any class's final logit, including
classes CLIP considered implausible for this image. Restrict patch
influence to a **candidate set** of classes CLIP already considers
plausible for that specific image, so patches can re-rank a shortlist but
never promote a class CLIP ruled out. Two interchangeable ways to define
the set, both worth testing:
- **top-K by rank** (K swept: 2, 3, 5, 10, ...)
- **threshold/margin-based** (all classes within some margin of CLIP's top
  score, or above some confidence cutoff — adapts automatically: shrinks to
  size 1 on easy images, grows only when the image is genuinely ambiguous)

**Method**: This is testable entirely offline, extending
`scripts/analyze_uncertainty_gated_fusion.py`'s existing per-sample
reconstruction (`baseline_logits`, `patch_term`, already computed from
stored records, all 3 datasets x 4 seeds, no bank access needed). Add a
per-sample candidate-set mask; zero out `patch_term` for any class outside
the set before summing into `gated_final`. Sweep K and the margin/threshold
version the same way `tau` was swept in Experiment D, and (naturally) also
combine with Experiment D's confidence gating, since they're compatible
mechanisms (one restricts *which classes* patches can affect, the other
restricts *how much* they can affect them).

**Cost**: CPU only, no new GPU runs — seconds, same as Experiment D.

---

## H — CLS-vote vs. patch-vote corroboration

**Idea**: Right now every write and every fusion decision is gated on one
number — CLIP's CLS-pooled confidence. But the patch embeddings already
extracted for the patch-level branch can *also* be compared directly to the
text embeddings (patch-to-text zero-shot, mean/max-pooled across patches),
completely independent of any accumulated bank state. That's a second,
structurally different vote from the *same single model, same one
unaugmented image* — global pooling vs. local/patch pooling. Where the two
agree, trust the write/prediction more; where they disagree, treat it as a
flagged, lower-trust case.

**Method (Phase 1, diagnostic, semi-offline)**: This specific quantity
(patch-to-text zero-shot logits) isn't in the stored records — it's not the
same as `patch_proto` (patch-to-*prototype*) already stored. But
`scripts/frozen_bank_eval.py` already does one fresh encoder forward pass
per image (for Experiments B/C); extend it to also compute patch-to-text
logits from that same pass (cheap — reuses patch embeddings already being
extracted, just an extra matmul against text embeddings already loaded for
zero-shot init). Then, purely as a diagnostic (no fusion changes yet): for
every write event, check whether CLS-vote and patch-vote agree on the
written class, and compare purity of agreeing vs. disagreeing writes. This
directly tests whether disagreement is actually informative before building
anything on top of it.

**Method (Phase 2, only if Phase 1 shows the purity gap is real)**: Wire the
agreement check into the live write gate (`multi_gate`) and/or the fusion
formula as an actual gating condition, and rerun live.

**Cost**: Phase 1 reuses `frozen_bank_eval.py`'s existing forward pass — no
new GPU cost beyond what Experiments B/C already paid. Phase 2 is a live
GPU rerun, contingent on Phase 1.

---

## Sequencing summary

| Phase | Experiments | Needs new GPU work? |
|---|---|---|
| 1 | F (offline reweighting check), G (offline candidate-set sweep), H (offline agreement-vs-purity diagnostic) | No — CPU/reused forward pass only |
| 1 (pilot) | E (small decay grid, 1 dataset x 1 seed) | Small, cheap live rerun |
| 2 | Whichever of E/F/H show real signal in Phase 1, implemented live and fully swept | Yes, live reruns |

G doesn't need a Phase 2 — a candidate-set restriction is naturally a
read-time/fusion-time mechanism, not something that needs to change how the
bank accumulates, so an offline finding for G is already the final answer
(modulo confirming it live before calling it done).

## Results

### G — Candidate-set-restricted patch voting: no hidden win

Same null pattern as Experiment D. Across the full sweep (top-K in
{1,2,3,5,10,20,unrestricted} x margin in {0.0..1.0}), no cell reaches
SUPPORT on any dataset. Best deltas over the baseline-only reconstruction:
dtd +0.52pp (margin=0.3), oxford_flowers +0.02pp (margin=0.02), oxford_pets
+0.01pp (margin=0.02) — all sub-noise-floor, same shape as every other
patch intervention tested so far. Full table:
`outputs/candidate_restricted_fusion_report.md`.

### F — Confusability-weighted appearance: mixed, and net negative on 2/3

Reweighting appearance by each prototype's own geometric confusability
(relative to its classmates, using the same variance-aware match score as
Experiment A) instead of raw frequency:

| Dataset | Control (appearance) | F-weighted | Delta |
|---|---|---|---|
| dtd | 46.57% | 45.80% | -0.77pp |
| oxford_flowers | 73.89% | 74.34% | +0.45pp |
| oxford_pets | 90.79% | 90.00% | -0.79pp |

Unlike every other patch intervention in this study (which consistently
helps dtd a little and hurts flowers/pets), this one flips the pattern —
small gain on flowers, losses on dtd and pets. All three deltas are still
under the noise floor, so none of this is a confirmed effect either way,
but there's no case for adopting it: it doesn't recover the flowers/pets
loss and it makes dtd (the one dataset patch fusion doesn't already hurt)
slightly worse. Raw output: `outputs/frozen_bank_eval_v2/*.json`.

### H — CLS-vote vs. patch-vote corroboration: the strongest signal in this whole investigation

This is a real, large, and consistent effect — bigger than anything else
tested across both rounds of experiments.

| Dataset | Purity when votes agree | Purity when votes disagree | Gap | Share of writes that agree |
|---|---|---|---|---|
| dtd | 72.2% | 40.8% | 31.4pp | 59.6% |
| oxford_flowers | 95.9% | 69.0% | 26.9pp | 20.5% |
| oxford_pets | 93.4% | 81.4% | 12.0pp | 30.0% |

And it's not just about bank writes — it tracks whether **CLIP's own
top-1 guess** is right:

| Dataset | CLIP accuracy when votes agree | CLIP accuracy when votes disagree |
|---|---|---|
| dtd | 58.4% | 23.0% |
| oxford_flowers | 94.7% | 64.5% |
| oxford_pets | 95.3% | 86.4% |

In plain terms: whether CLIP's whole-image guess and the patch-only guess
(computed from the same single forward pass, no bank, no training) agree
with each other is a strong, independent hint about whether CLIP is right
on this particular image — something raw confidence/margin (tested earlier)
provably cannot give us, since a confidently-wrong prediction looks
identical to a confidently-right one from inside CLIP's own score alone.
This is a genuinely different signal, and it works.

Two important caveats before calling this a fix:

1. **The patch-only vote is not a good classifier by itself.** When CLIP is
   wrong, the patch vote only happens to be right 10.7% / 4.3% / 14.7% of
   the time (dtd/flowers/pets) — so this can flag untrustworthy cases, it
   can't independently solve them.
2. **Agreement is rare on flowers/pets** (20.5% / 30.0% of images) — these
   are fine-grained datasets where a single patch rarely carries enough
   context to name the exact species/breed the way it can for dtd's
   textures. A hard "only write/trust on agreement" gate would throw away
   70-80% of all writes on those two datasets, which risks starving the
   bank more than it helps. A live implementation should probably
   *down-weight* disagreement rather than exclude it outright, at least on
   flowers/pets.

**This is the strongest candidate for a Phase 2 live implementation** of
anything tested in this study so far — recommend prioritizing it.

### E — Appearance decay: pilot says no, don't extend to a full sweep

| `appearance_decay` | dtd accuracy (seed 1) | Delta vs. decay=1.0 |
|---|---|---|
| 1.0 (sanity check — old behavior) | 46.75% | — (matches known value exactly) |
| 0.99 | 46.69% | -0.06pp |
| 0.95 | 46.63% | -0.12pp |
| 0.90 | 45.57% | -1.18pp |

The sanity check confirms the code change is behavior-neutral at the
default (`decay=1.0` reproduces the exact known accuracy). But every decay
value tested makes things *worse*, monotonically -- the more aggressive the
forgetting, the bigger the loss. Per the plan's own pilot-gating rule (only
extend to the full 3-dataset x 4-seed sweep if the pilot shows a real signal
in the *right* direction), this doesn't justify a full sweep. Stopping E
here.

**Why this probably doesn't transfer from baseline PTA, in hindsight**:
baseline PTA's EMA blends *content* -- old and new evidence merge into one
running feature vector, so the representation itself self-corrects toward
the true average. Appearance decay only changes how much a cluster's
*already-fixed* geometric identity gets to count in scoring -- it never
touches the cluster's center. It also shrinks the long-run trust ceiling
uniformly for every cluster, correct or not (geometric decay caps the
steady-state weight at `1/(1-decay)` instead of growing without bound), so
it doesn't selectively suppress wrong clusters any more than it suppresses
good, well-established ones -- it just makes the whole bank noisier. This is
a real disanalogy from what makes PTA's EMA work, not just an unlucky
hyperparameter choice, so it's not worth re-testing at different decay
values.

## I — Comparing patch-level aggregation signals

**Idea**: H's "patch vote" was built with a single pooling choice — mean
cosine similarity across all patches, dotted against the text class
embeddings. Mean pooling can be diluted by background patches, especially
when the object of interest is small relative to the image; a max-style
pooling ("does *any* patch look like class c") is a conceptually different
question and may behave very differently on exactly that kind of image.
`scripts/frozen_bank_eval_v2.py::patch_text_vote()` already computed a
`max_vote` alongside `mean_vote`, but only `mean_vote` was ever evaluated.

New script `scripts/analyze_patch_vote_aggregation.py` runs 12 pooling
variants — all from the same single CLIP Surgery forward pass per image (no
augmentation, no second model): mean, max, top-k mean (k ∈ {3,5,10,20}),
generalized power-mean (p ∈ {2,4,8,16}, a continuous mean↔max knob), and two
CLIP-Surgery-relevance-based variants (masking raw cosine similarity by the
surgery relevance map before mean-pooling; and mean-pooling the surgery
relevance score itself). Each variant is scored on the same diagnostics H
used (agreement rate with CLIP's own CLS-pooled prediction, CLS-accuracy-
by-agreement, write-purity-by-agreement, rescue/corrupt rates), plus a new
standalone-accuracy check and each variant's own top1-top2 margin (recorded
for Experiment J, not used for gating here).

Purely offline/diagnostic — no bank mutation, no live prediction change.
`scripts/frozen_bank_eval.py`'s `-s1` bank dump and write-purity replay
logic (`MATCH_THRESHOLD`, `WRITE_CONF_THRESHOLD`) are reused unchanged.

**Verification**: the `mean` variant reproduces Experiment H's exact stored
dtd numbers (`outputs/frozen_bank_eval_v2/dtd.json`) — agreement rate,
both accuracy-by-agreement numbers, both write-purity-by-agreement numbers
(plus their `n`), and rescue/corrupt rates all matched to float precision.
`mean`/`max` also spot-checked directly against `patch_text_vote()`'s own
output on 5 samples (`torch.allclose`, exact match).

**Results** (full table: `outputs/patch_vote_aggregation_report.md`):

The objection was right, and the effect is large on the dataset where it
should matter most. **oxford_pets** — cat/dog head-and-shoulders photos,
where the animal often fills only part of the frame — shows a dramatic
jump: mean-pooling's standalone accuracy (30.20%) and agreement rate
(30.0%) both roughly **double** under `topk20` (65.41% / 64.8%), with a
bigger agreement/disagreement accuracy gap too (18.4pp vs 8.9pp). `max`
alone gets most of the way there (62.55% / 62.2%); `topk20` beats plain
`max` on every metric in every dataset, so it's the better representative
of "does *some* patch (not just the single best one) look like class c."

On **dtd**, all variants are close — differences of 1-2pp either way, no
clear winner, consistent with dtd being about a whole-image texture rather
than a small foreground object.

On **oxford_flowers** the picture is more mixed: `mean` actually has the
*largest* agreement/disagreement gap (30.1pp) and the highest write purity
when it agrees (95.9%), it just agrees rarely (20.5% of images). `topk20`
trades some of that per-agreement purity for meaningfully more coverage
(31.6% agreement, purity 85.6%). Neither dominates the other here — this
tradeoff (rare-but-very-reliable vs. common-but-less-reliable) is exactly
the kind of thing Experiment J's margin-graded and per-class analysis needs
to resolve before picking one.

One negative result worth flagging: `surgery_score_mean` (mean-pooling
CLIP Surgery's own `surgery_no_labels` relevance score directly, instead of
raw cosine similarity) turned out to be **mathematically identical to
`mean`** on every single metric — not a near-tie, bit-for-bit identical.
Traced to `clip_feature_surgery`'s `redundant_feats` branch being a linear
shift by a class-independent constant, which cancels out under
mean-pooling + argmax/softmax. Dropped from further consideration; CLIP
Surgery's relevance map is still doing something real in
`surgery_masked_mean` (used as a per-class patch *mask*, not as the score
itself, which does NOT collapse the same way and shows genuinely different
numbers from `mean`).

**Shortlist for Experiment J** (best overall + one representative per
structurally distinct family, per this doc's decision criteria):
- `mean` — required as the reference/control.
- `topk20` — best or tied-best on nearly every metric, every dataset;
  dominates plain `max`.
- `pmean16` — the power-mean family's strongest setting, tracks `topk20`
  closely but via a different (continuously-tunable) mechanism.
- `surgery_masked_mean` — the one variant genuinely using CLIP Surgery's
  own relevance machinery; behaves differently enough from the cosine-only
  family to be worth validating separately (e.g. oxford_pets corrupt rate
  35.87% vs. `topk20`'s 30.46% — worse there, despite similar standalone
  accuracy).

**Addendum: `otsu_mean`, a hyperparameter-free alternative to `topk20`.**
The user objected to `topk20`'s fixed `k=20` needing its own tuning sweep
and asked for a data-driven alternative. Added `otsu_mean`: per class, find
the Otsu threshold (the classic zero-hyperparameter algorithm for splitting
a 1D distribution into a high/low group by maximizing between-group
variance — no k, no z-score cutoff to pick, a unique optimum given the
data) on that class's patch-similarity values, mean-pool the patches at or
above it. Verified it's genuinely adaptive, not a fixed count in disguise:
median patches kept is ~93-122 (of 196) on dtd's whole-image textures vs. a
much more selective, wider-ranging 11-149 on oxford_pets, varying per image
and per class.

Result: `otsu_mean` **wins outright on dtd** (best standalone accuracy,
agreement rate, gap, and corrupt rate of all 13 variants tested), is
essentially tied with the leaders on oxford_flowers (best write purity and
best rescue rate of any variant), but trails `topk20`/`pmean16` on
oxford_pets (61.1% vs. 65.4% standalone accuracy). Added to the shortlist
above — carried into Experiment J alongside `topk20`, per the user's
request to keep both as candidates for Experiment K rather than pick one
now.

## J — Deep validation of the shortlisted aggregation signal(s)

**Idea**: Experiment I's metrics are coarse, single-seed, whole-dataset
averages. Before building Experiment K's live write-time change on one of
these variants, validate the shortlist (`mean`, `topk20`, `pmean16`,
`surgery_masked_mean`) more deeply: does the write-purity gap hold up
against different final bank states (not just seed 1), does the variant's
own margin carry graded information beyond binary agree/disagree, does the
benefit concentrate in particular classes, and is agreement genuinely new
information or a repackaging of signals this investigation already has?

New script `scripts/validate_patch_vote_signal.py`.

**Bug caught and fixed during this experiment**: the multi-seed pass
initially reused `frozen_bank_eval.py`'s `load_stored_records()`, which is
hardcoded to seed-1's records file. Pairing seed-1 records (keyed by
*position* in seed-1's shuffled test loader) against a *different* seed's
shuffled loader silently scrambled every image/target pairing — standalone
accuracy collapsed to ~1-3% (chance level for each dataset's class count),
which is what exposed it. Fixed by loading each seed's own
`PatchModPTA-CS-<dataset>-s<seed>/records.jsonl` (these already existed on
disk for all 4 seeds). Re-verified: standalone accuracy and agreement rate
are then bit-for-bit identical across all 4 seeds, as expected (both are
pure per-image functions of zero-shot CLIP logits + patch embeddings,
neither of which depends on test-set processing order).

**Also required regenerating 9 missing bank dumps**
(`outputs/patch_bank_dumps/{dtd,oxford_flowers,oxford_pets}-s{2,3,4}.pt}`) —
only the seed-1 dumps existed on disk (Experiments B/C/H/I were all
implicitly seed-1-only). Regenerated via the existing `DUMP_PATCH_BANK` env
var hook in `models/patch_modulated_pta.py`, using the exact same
`patch_modulated_pta` / `configs/patch_modulated_pta` invocation as the
original study (`scripts/run_patch_benefit_study.sh`). Verified: resulting
accuracies matched the already-known `PatchModPTA-CS` seed 2-4 numbers
exactly (dtd 48.23/47.22/46.63, flowers 72.39/72.39/72.43, pets
88.83/89.34/88.66).

**Results** (full tables: `outputs/patch_vote_validation_report.md`):

1. **Multi-seed replication — holds up.** Write-purity-agree, -disagree, and
   the gap between them stay in a consistent band across all 4 seeds for
   every shortlisted variant, every dataset (e.g. `topk20` on oxford_pets:
   gap 20.9-23.4pp across seeds 1-4; on dtd: 30.8-37.1pp). Not a seed-1
   fluke.

2. **Margin-graded analysis — margin carries real, mostly-monotonic
   information.** Binning `topk20`'s own top1-top2 margin into deciles:
   CLS accuracy rises from 23.7%→83.6% (dtd) and 83.6%→99.7% (oxford_pets)
   from the lowest to the highest margin decile, with write purity
   following the same trend. oxford_flowers is noisier but still trends
   up overall. **This means a binary agree/disagree discount throws away
   information — a continuous, margin-derived weight is better justified.**

3. **Per-class breakdown — real correlation with CLIP's own difficulty, but
   not a full explanation.** Agreement rate positively correlates with
   CLIP's per-class baseline accuracy (Pearson r = 0.65 dtd, 0.25 flowers,
   0.46 pets) — the signal is somewhat concentrated in classes CLIP is
   already decent at, but the correlation is well short of 1.0, meaning it
   also captures real per-image variation *within* a class, not just "this
   is a hard class." Individual per-class gaps are noisy (13-100
   samples/class) and a few classes show reversed gaps by chance — judge by
   the correlation and Experiment I's pooled numbers, not single-class
   outliers.

4. **Cross-signal correlation — genuinely new information on both counts.**
   - vs. the earlier confident-vs-ambiguous CLIP write-margin split:
     agreement predicts purity *within both* the confident and ambiguous
     write buckets (e.g. oxford_pets: confident+agree 96.5% vs.
     confident+disagree 81.0%; ambiguous+agree 58.9% vs.
     ambiguous+disagree 38.1%) — not just acting as a proxy for write
     confidence.
   - vs. Experiment F's per-prototype confusability margin: essentially
     **no difference** between agree/disagree groups (e.g. oxford_pets:
     -0.323 vs. -0.328) — agreement is not just re-deriving which
     prototypes are geometrically well-separated; it's about whether *this
     image's* patches match the claimed class, a different axis entirely.

5. **Qualitative spot-check.** oxford_pets rescue/corrupt cases are
   dominated by visually-similar breed confusions (birman vs. ragdoll,
   bengal vs. abyssinian, pit-bull vs. staffordshire) — `topk20` is doing
   real fine-grained visual discrimination, just not reliably enough to
   trust alone (consistent with Experiment I's low rescue rates).

**Addendum: `otsu_mean` added and validated the same way.** The user
objected to `topk20`'s fixed `k=20` needing its own tuning sweep and asked
for a hyperparameter-free alternative (see the addendum to Experiment I's
results above for the method). Re-ran all of J's checks — multi-seed
replication, margin-decile, per-class correlation, cross-signal — for
`otsu_mean`; it passes every one exactly as cleanly as `topk20` did
(multi-seed stable, margin trends up ~20-40pp low-to-high decile, per-class
correlation 0.24-0.66 without reducing to it, confirmed non-redundant with
both the write-confidence split and Experiment F's confusability margin).
Full tables: `outputs/patch_vote_validation_report.md`.

**Final recommendation for Experiment K**: carry **both `topk20` and
`otsu_mean` forward as parallel candidates**, per the user's explicit
request, rather than picking one now. Practical difference (from
Experiment I): `otsu_mean` is stronger/tied on dtd and oxford_flowers,
`topk20`/`pmean16` are stronger on oxford_pets — neither dominates the
other across all three datasets, so Experiment K should implement both and
let the live results (against the locked noise-floor bar) decide. Both use
a **continuous, margin-derived trust weight** rather than a purely binary
agree/disagree discount, per finding 2 in the original J results above.
Both open questions from Experiment I's shortlist section are resolved for
both variants: multi-seed stability holds, and the signal is confirmed
non-redundant with everything else this investigation has tried.

## K — Soft down-weighted bank writes: implementation and results

### Implementation

Wired the CLS-vote-vs-patch-vote corroboration signal into the live
write-time path (see `utils/patch_vote.py`, `models/patch_level/gaussian_patch.py::update_state`'s
new `trust_weight` parameter, `models/patch_modulated_pta.py`'s new
`patch_vote_gate` config block). Pooling functions (`topk_pool`,
`otsu_mean`, Otsu threshold) were moved to a shared `utils/patch_vote.py`
module so the live gate can never silently drift from what Experiments I/J
validated — same implementation, not a re-derivation.

Two real bugs caught and fixed during pilot testing, before any full sweep:

1. **`torch.histc` has no deterministic CUDA kernel**, and the live TTA
   loop runs under `torch.use_deterministic_algorithms(True)` — crashed
   immediately on `otsu_mean`. Fixed by offloading the histogram to CPU
   (same fix already used elsewhere in this codebase for the `cpm`
   write-rule's `torch.cumsum`). Offline analysis scripts never hit this
   since they don't set that mode.
2. **Trust-weight formula design flaw**: the first version
   (`trust_weight = margin if agree else disagree_discount`) let a
   confidently-*disagreeing* write outweigh a barely-*agreeing* one at
   `disagree_discount=1.0`, inverting the intended relationship — the
   pilot showed every discount value strictly worse than baseline,
   including 1.0, which should have been a no-op. Fixed to
   `trust_weight = disagree_discount + (1 - disagree_discount) * margin`
   when agreeing (interpolates from the floor up to 1.0 by confidence),
   flat `disagree_discount` when disagreeing — guarantees agree >=
   disagree always, and `disagree_discount=1.0` collapses both branches to
   an exact 1.0 no-op.

**Verification**: `patch_vote_gate.enabled` unset (default) reproduces the
known `PatchModPTA-CS` accuracy exactly on both dtd (46.75%) and
oxford_pets (89.48%) seed 1. After the formula fix, `disagree_discount=1.0`
(gate enabled, but both branches collapse to weight 1.0) also reproduces
46.75%/89.48% exactly on both aggregations — confirms the corrected
formula is behavior-neutral at its boundary, the same free correctness
check every prior experiment in this series has used.

### Pilot sweeps (before committing to the full statistical run)

A wide pilot (dtd, seed 1, `disagree_discount` ∈ {1.0, 0.5, 0.1}) showed a
flat-to-negative trend on dtd and, on oxford_pets, a small positive nudge
at 0.5 (+0.05/+0.11pp) but a catastrophic collapse at 0.1 (−17 to −18pp —
almost certainly bank starvation: ~35-40% of writes get crushed toward
near-zero trust, especially early in adaptation before much evidence has
accumulated). A narrower pilot (all 3 datasets, seed 1,
`disagree_discount` ∈ {0.3, 0.5, 0.7, 0.9}) found a consistent, monotonic
positive trend on oxford_flowers/oxford_pets as discount decreased toward
0.3 (best single point: **+0.61pp on oxford_flowers at discount=0.3**, both
aggregations), with dtd staying flat/noisy throughout.

### Full statistical validation at disagree_discount=0.3

Per the user's request, ran the full 3-dataset × 4-seed × 2-aggregation
grid at the pilot's best setting (`disagree_discount=0.3`), with full
per-sample record capture so the same statistical bar used throughout this
investigation could be applied: locked noise-floor thresholds
(`outputs/patch_benefit_prereg.md`), paired sign-permutation p-value,
seed-level bootstrap CI, and the `flip_metrics_v2` mechanistic check.
Full tables: `outputs/K_discount03_report.md`.

**vs. plain PTA-CS** (the standing bar every patch-fusion variant in this
whole investigation has been judged against): REFUTE on all 3 datasets,
both aggregations (deltas −0.40pp on dtd, −2.3pp on oxford_flowers, −1.9 to
−2.0pp on oxford_pets — all with `p_sign` at or near 1.0 and negative
`flip_metrics_v2` `patch_image.net`). This isn't new information by
itself — vanilla, ungated patch fusion already loses to plain PTA on
oxford_flowers/oxford_pets, which is the premise this entire round of
experiments (E-K) was trying to fix.

**vs. vanilla `PatchModPTA-CS`** (same patch method, gate off — the more
direct "did the gate actually help" question): the effect **evaporates**.

| Dataset | topk20 delta mean | otsu_mean delta mean |
|---|---|---|
| dtd | −0.09pp (p=0.81) | −0.09pp (p=0.81) |
| oxford_flowers | +0.07pp (p=0.38) | +0.09pp (p=0.38) |
| oxford_pets | +0.01pp (p=0.44) | +0.09pp (p=0.38) |

Every delta is within noise, every `p_sign` is far above the 0.0625
significance threshold, every 95% CI straddles zero. **The +0.61pp
single-seed pilot signal on oxford_flowers does not replicate** — seed 3
alone showed −0.65pp at the same setting, and the 4-seed mean collapses to
+0.07pp. In hindsight this is a real lesson about the narrower pilot's own
methodology: sweeping `disagree_discount` at a single fixed seed produces
internally-consistent-*looking* trends across that axis, because every
point in that sweep shares the same seed's idiosyncratic noise — it says
nothing about whether the trend holds for other seeds.

### Bottom line

`disagree_discount=0.3` (both `topk20` and `otsu_mean`) is a **null
result** once properly validated: no real improvement over vanilla patch
fusion on any of the 3 datasets, and (unsurprisingly, unchanged from
before this round) still loses to plain PTA-CS on oxford_flowers/
oxford_pets. This doesn't necessarily mean the corroboration signal itself
is useless — Experiments H/I/J established real, robust, multi-seed
information content in it — but this particular mechanism (a soft
down-weight on write-time trust) isn't converting that information into an
accuracy win at this setting. Whether a different point in the
discount/margin-formula space would do better is open; per the standing
plan, any further sweep should be evaluated with multi-seed data from the
start, not single-seed pilots, given what happened here.

## Standing constraint

Per earlier instruction: **do not `git add`/`git commit`** anything produced
by this work.
