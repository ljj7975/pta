# Offline Follow-Up Experiments — Plan

Follow-up to `outputs/patch_benefit_report.md` (Sections 8-9). Three
questions came out of that discussion; this document plans how each is
tested, then (once run) reports what was found. Written to be readable on
its own — see `patch_benefit_report.md` Sections 8-9 for full background and
terminology (bank, purity, appearance weight, separability margin, TCR).

Status: **all four experiments are complete.** See "Results" at the bottom
for what was found, explained in plain language. (Experiment D was added
after a later discussion about what to do given A-C's findings — see its
own Method section below for that context.)

---

## Experiment A — Variance-aware separability re-check

**Question**: Section 8.2's confusability finding compared prototypes using
plain cosine similarity between their centers only. But each prototype also
has a "variance" (how tightly its member patches cluster around that
center), and the live scoring code uses *both* center and variance (a
Gaussian-likelihood style comparison, not raw cosine). Does the "every
dataset is confusable" finding survive using the same math the live method
actually uses?

**Method**: Reuse the frozen bank snapshots already saved in
`outputs/patch_bank_dumps/` (one per dataset, no new GPU run needed).
Replace the cosine-similarity confusability check with the same
scaled-Mahalanobis-plus-exponential scoring formula the production code uses
(`utils/kmeans.py`), scoring each class's own prototypes against every other
class's prototypes (using the *other* class's variance, since that's what
determines how forgiving that class's decision boundary actually is).

**Cost**: CPU only, no new runs — minutes.

**Output**: `outputs/prototype_separability_v2_report.md` — corrected
confusability numbers, compared against the original cosine-based ones.

---

## Experiment B — Per-cluster purity (does appearance weight track correctness?)

**Question**: Section 8.1's purity number was measured per *class* (did an
image's patches get written into the right class's bank at all?). It can't
say whether the specific prototypes that end up trusted most — the
high-appearance ones — are actually the correct ones, or just the most
*consistently repeated* ones (which, per Section 9.3, could also happen from
a systematic, repeated misclassification).

**Method — revised from the original plan.** The original plan for this
called for adding live logging of which cluster each write lands in and
re-running the adaptation loop. That turned out to be avoidable: matching an
image's patches against the *already-frozen, already-dumped* final bank
(same encoder forward pass Experiment C needs anyway) gives the same answer
without re-running the online adaptation loop or touching its code. For
every test image, using the write rule already validated in
`outputs/prototype_purity_report.md` (which classes an image's patches get
written to, replayed from stored zero-shot CLIP confidence), match its
patches against the nearest prototype *in that class's final bank* and
record whether the image's true label matches that class. Pooling this over
all images gives a purity score **per prototype**, which can then be
compared against that prototype's appearance weight and its confusability
score from Experiment A.

**Cost**: Shares the one encoder forward pass per dataset that Experiment C
also needs (see below) — no separate GPU cost.

**Output**: Per-prototype table: appearance weight, purity, confusability.
Directly answers whether high-appearance prototypes are reliably purer
(supporting the "appearance ≈ trustworthy" assumption both the existing code
and TCR depend on) or whether purity is flat/unrelated to appearance once
looked at per-prototype rather than per-dataset.

---

## Experiment C — Does TCR-style reweighting recover any accuracy?

**Question**: Section 9.4 proposed replacing the current flat appearance
weight with a specificity-aware (TCR-style) weight — one that additionally
downweights a prototype if a similar, frequently-seen prototype also exists
in *other* classes. Does swapping in that weight actually change accuracy
when evaluated against the same frozen bank?

**Method**: Using the same frozen bank dumps and the same per-image encoder
forward pass as Experiment B:

1. Compute a TCR-style weight per prototype from the frozen bank alone (no
   labels used) — its own frequency, corrected by how much a similar
   prototype also shows up with real weight in other classes, corrected
   further by how spread out that cross-class occurrence is.
2. Re-score every test image against the frozen bank twice — once with the
   bank's original appearance weight (**control**), once with the TCR weight
   (**treatment**) — using the *same* simplified aggregation for both, so
   the only thing that differs between the two runs is the weighting scheme.
3. Fuse each with the already-stored, already-validated zero-shot CLIP and
   image-level prototype logits from `outputs/records_patch_benefit/`
   (`PatchModPTA-CS`, the arm these bank dumps came from) using the exact
   same fusion formula the live method uses, and recompute accuracy for
   both.

**Important simplification, stated plainly**: the live method's real scoring
uses a z-score-normalized aggregation, which needs a running reference
distribution per prototype that wasn't saved in the bank dump (only the
final centers/variance/appearance were). Reproducing that exactly would need
another rerun with extra logging. Instead, both the control and the TCR
condition here use a simpler appearance-weighted average (still the same
core per-patch Gaussian similarity computation the live code uses — only the
top-level combination step is simplified). This means the control condition
is **not** expected to reproduce the exact live accuracy number, and that's
fine — the comparison is between control and TCR *under the same simplified
formula*, which is what isolates the effect of the reweighting scheme
itself. This will be checked and reported plainly, not glossed over.

**Cost**: One encoder forward pass over each dataset's test set (shared with
Experiment B) — real GPU time, but a single pass with a frozen bank (no
online updates, no seed sweep), materially cheaper than the original
84-cell adaptation sweep.

**Output**: `outputs/frozen_bank_eval_report.md` — control vs. TCR accuracy
per dataset, plus the Experiment B purity table.

---

## Experiment D — Uncertainty-gated patch fusion

**Context**: Experiments A-C ruled out fixing the bank's *contents*
(purity, specificity-aware reweighting) as a path to recovering accuracy.
A follow-up check split write-impurity into "ambiguous" vs. "confident"
CLIP writes and found that a large share (37-70%, dataset-dependent) of all
impurity comes from writes CLIP made *confidently*, not hesitantly — which
by definition no function of the classifier's own confidence can filter
out, since "confidently right" and "confidently wrong" look identical from
inside that one signal. Given that, the more promising remaining lever
(flagged earlier, Section 8.4) is not to fix *what enters the bank*, but to
gate *how much the patch term is allowed to move the final prediction*,
conditioned on how confident the baseline (CLIP + image-level prototype) is
in its own answer before patch evidence is added. If the baseline is
already sure, patch evidence has little to gain and real room to break a
correct answer; if it's unsure, patch evidence has more legitimate room to
help.

**Method**: Every stored record for the `PatchModPTA-CS` arm (all 3
datasets, all 4 seeds — real full data, not a frozen-bank approximation)
already contains the exact per-class logit components used at inference:
`clip`, `image_proto`, `patch_proto` (pre-squash), and `final`. Confirmed by
direct reconstruction that:

```
final == clip + tau_image_proto * image_proto
       + tau_patch_proto * proto_alpha * tanh(patch_proto / patch_squash_scale)
```

which means a "baseline-only" prediction (`clip + tau_image_proto *
image_proto`, patch term excluded) and its own confidence margin
(top1-top2 softmax probability) can be computed directly from already-saved
data, with no bank access, no encoder forward pass, and no aggregation
simplification — this is the cleanest experiment run in the whole
investigation.

The patch term is then gated by that baseline margin, per image (one gate
applied uniformly to all classes, since the question is how much to trust
the baseline overall):

- **hard gate**: include the patch term only if `baseline_margin < tau`,
  else drop it entirely.
- **soft gate**: linearly ramp the patch term's weight down to 0 as the
  margin approaches `tau` (checks the hard cutoff isn't creating an
  artifact).

`tau` is swept over `[0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]`
— a self-bounding range: `tau=0.0` always drops the patch term (should
reproduce plain PTA), `tau=1.0` never drops it (should reproduce the
original always-fused result). Statistics reuse the exact machinery already
locked in `outputs/patch_benefit_prereg.md` (per-dataset noise-floor
thresholds, paired deltas, sign-permutation p-value, seed-level bootstrap
CI).

**Caveat, stated plainly**: the `tau=0.0` reconstruction is *not* a perfect
stand-in for an independently-run PTA. `PatchModPTA-CS`'s own image-level
prototype trajectory evolved, during the actual run, under predictions that
already included patch influence throughout (the write mask that decides
which image-level updates happen is downstream of the *fused* prediction) —
so gating it off after the fact replays a trajectory that patch fusion
already shaped, not the trajectory a truly patch-free run would have
produced. The sanity check below caught this directly (small but
non-negligible gaps between the `tau=0.0` reconstruction and PTA-CS's own
recorded accuracy). The gaps are modest — see Results — and don't change the
conclusion, but this is a real limitation of doing this offline rather than
as a live rerun, and is reported rather than hidden.

**Cost**: CPU only, no new GPU runs — seconds.

**Output**: `outputs/uncertainty_gated_fusion_report.md` (+ `.json`) — full
`tau` x dataset x gate-type accuracy/verdict table.

## Sequencing

A first (immediate, no GPU). Then B and C together, since they share the
same per-image forward pass over the frozen bank — building that pass once
and computing both experiments' numbers from it. D came later, after a
follow-up discussion once A-C's results were in; it only needed already-
stored records, no new run.

---

## Results

All three experiments ran on the same three datasets as the main study
(dtd, oxford_flowers, oxford_pets). Short version first, details after.

**Short version**: none of the three follow-ups changed the conclusion from
`patch_benefit_report.md`. If anything, they explain *why* the earlier fixes
(appearance weighting, and now TCR too) only go partway.

### A — Variance-aware separability: the confusability finding holds up

Using the real scoring math (center + spread, not just plain similarity)
instead of the simplified version from before didn't change the picture:
every dataset's prototypes are still more similar to their nearest neighbor
in a *different* class than to their own classmates. Full numbers in
`outputs/prototype_separability_v2_report.md`. This rules out "the earlier
finding was just an artifact of using a too-simple similarity check" — it
wasn't.

### B — Per-cluster purity: appearance weight works on dtd, but not where it's needed most

This is the most informative result of the three. For every prototype
("cluster") in every class's bank, we checked: of the images whose patches
actually matched into it, how often were they really that class? Then we
split each dataset's prototypes into three groups by how often they'd been
reinforced (their "appearance weight" — low/medium/high), and asked whether
purity gets better as reinforcement goes up (which is exactly what the
current code — and the TCR idea — both assume).

| Dataset | Least-reinforced third | Middle third | Most-reinforced third |
|---|---|---|---|
| dtd | 29.8% correct | 62.8% correct | **65.7% correct** |
| oxford_flowers | 73.8% correct | **79.7% correct** | 71.6% correct |
| oxford_pets | 83.5% correct | **87.8% correct** | 83.3% correct |

On dtd, the assumption holds cleanly: the more a prototype has been
reinforced, the more likely it's actually correct. That's exactly why
appearance-weighting helped there (Section 9.3).

On oxford_flowers and oxford_pets — the two datasets where patch fusion
actually *hurts* — the pattern breaks. The most-reinforced prototypes are
*not* the most trustworthy; the middle group is actually more reliable than
the top group. In other words, on exactly the datasets where we most need
"reinforced = trustworthy" to be true, it isn't — some of the
most-reinforced prototypes are almost certainly being built by the same
mistake happening over and over (the "systematic confusion" case flagged in
Section 9.3), not by genuinely correct evidence. This is a concrete,
measured confirmation of that concern, not just a theoretical one.

(As a cross-check: the overall purity per dataset from this cluster-level
measurement — 64.0% / 74.1% / 84.9% — lines up closely with the simpler,
class-level purity measured earlier in `prototype_purity_report.md` — 62.6%
/ 74.1% / 85.7%. The two independent measurements agree, which is
reassuring that neither has a hidden bug.)

Full numbers: `outputs/frozen_bank_eval_report.md`.

### C — TCR reweighting: no real accuracy gain

Swapping the current flat appearance weight for the TCR-style weight
(rewards a prototype for being frequent *and* not also frequent in other
classes) and re-scoring every image against the frozen bank:

| Dataset | Appearance-weighted (control) | TCR-weighted | Change |
|---|---|---|---|
| dtd | 46.57% | 46.69% | +0.12pp |
| oxford_flowers | 73.89% | 73.73% | -0.16pp |
| oxford_pets | 90.79% | 90.43% | -0.35pp |

All three changes are small — smaller than the 1.0 percentage-point bar the
main study used to call something a real signal rather than noise. And
notice the pattern: a tiny gain on dtd, a small loss on the other two — the
exact same shape every other patch-related change in this whole
investigation has shown. TCR does not break that pattern. So: no, at least
not in this form, TCR-style reweighting does not recover the accuracy that
patch-level evidence is missing.

One honest caveat, repeated from the Method section above: this comparison
uses a simplified version of the scoring formula (both conditions, so it's
still a fair, apples-to-apples test of appearance-vs-TCR) rather than the
exact live pipeline. That affects how much weight to put on the *exact*
numbers, but not the direction of the result, and Experiment B independently
explains *why* a purely frequency-based specificity correction like TCR
would struggle here: the problem on flowers/pets isn't that prototypes are
generic (spread evenly across many classes, which TCR is built to catch) —
it looks more like a small number of prototypes being wrong in one
consistent, concentrated way, which is precisely the failure mode Section
9.4 already flagged TCR as unable to see.

### D — Uncertainty gating: recovers most of the loss, but never turns into a real win

Gating the patch term off whenever the baseline (CLIP + image-level
prototype) is already confident produces a strikingly consistent pattern
across all three datasets: as the gate is tightened (`tau` lowered, meaning
patch evidence gets used less and less), accuracy rises smoothly and
monotonically back toward — and, on dtd only, slightly past — the plain
baseline. Full sweep in `outputs/uncertainty_gated_fusion_report.md`;
highlights:

| Dataset | Always-fused (tau=1.0, original) | Best gated result | Baseline (tau=0.0, patch off) |
|---|---|---|---|
| dtd | 47.19% | 48.21% (soft gate, tau=0.5) | 47.65% |
| oxford_flowers | 72.27% | 74.59% (hard gate, tau=0.02) | 74.55% |
| oxford_pets | 89.09% | 90.98% (hard gate, tau=0.02) | 90.96% |

Two things stand out. First, gating clearly works *directionally* — it
undoes nearly all of the damage the always-on version does on
oxford_flowers and oxford_pets, and does the same, more modestly, on dtd:
suppressing the patch term whenever the baseline is already confident
consistently claws back accuracy as the gate tightens, exactly as the
"baseline already trusted -> little to gain, real room to lose" reasoning
predicted. Second, and more important: none of this ever adds up to a real
improvement *over doing nothing (plain PTA)*. The
best-performing setting on flowers and pets is essentially indistinguishable
from `tau=0.0` (i.e., "almost never use the patch term") — deltas of
+0.04pp and +0.01pp, both far under the pre-registered 1.0pp noise floor.
dtd does a little better (+0.56pp at its best setting), but that's still
below the noise floor too. **No cell in the entire sweep — 3 datasets x 2
gate types x 10 tau values, 60 cells — reaches the pre-registered SUPPORT
threshold.**

In plain terms: gating is the right instinct for stopping patch fusion from
actively hurting, but it doesn't reveal a hidden benefit that a purity fix
was masking. The most reliable setting found by this sweep is close to
"turn the patch term off" — which is another way of saying the patch signal
itself doesn't have real accuracy value to contribute here, confident-write
contamination or not.

(Caveat, as flagged in the Method section: the `tau=0.0` line in this table
is a post-hoc reconstruction from `PatchModPTA-CS`'s own recorded run, not
an independently-run PTA — its image-level prototype trajectory was itself
shaped by patch-influenced predictions throughout the live run. Sanity checks
confirmed this reconstruction lands close to real PTA-CS accuracy but not
exactly on it, off by roughly 0.1-0.4pp seed-to-seed. That gap is smaller
than every comparison in this table and doesn't change the "no real signal"
conclusion.)

### Bottom line

Nothing here reopens the case for patch-level prototypes helping on these
datasets. Experiment A confirms the confusability diagnosis wasn't a
measurement artifact. Experiment B pinpoints *why* fixing the weighting
scheme has limited reach — the reinforcement signal itself stops being
trustworthy exactly where it's needed most. Experiment C shows that even a
more principled reweighting scheme (TCR), tested directly, doesn't recover
meaningful accuracy and reproduces the same dataset-dependent pattern
(small dtd gain, flowers/pets losses) documented throughout this whole
investigation. Experiment D shows that the other remaining lever —
suppressing the patch term whenever the baseline is already confident,
rather than trying to purify the bank — does what it's supposed to
(mostly undoes the regression) but never produces a statistically real
improvement over simply not using patch evidence at all. Taken together,
these four experiments point at the ceiling/base-rate explanation from
Section 8.3 as the dominant factor: the problem was never really about bank
quality or when to trust it — the patch signal itself, even under its best
achievable treatment, doesn't carry enough independent accuracy value on
these three datasets to beat plain PTA.
