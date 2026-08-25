# Patch-Level Prototype Benefit Study — Report

Scope: dtd, oxford_flowers, oxford_pets — ViT-B/16, CLIP Surgery, seeds 1-4.
Decision rule: see `outputs/patch_benefit_prereg.md`.

## 1. Accuracy Table

| Arm | dtd | oxford_flowers | oxford_pets |
|---|---|---|---|
| PTA-CS | 47.52±0.08 [47.64/47.46/47.46/47.52] | 74.62±0.33 [74.46/74.79/74.26/74.99] | 91.07±0.11 [91.01/91.14/90.95/91.20] |
| PatchModPTA-CS-QGated | 47.21±0.73 [46.75/48.23/47.22/46.63] | 72.25±0.37 [71.70/72.43/72.43/72.43] | 89.10±0.42 [89.53/89.37/88.83/88.66] |
| PatchModPTA-CS-MVote | 48.06±0.48 [48.76/47.81/47.70/47.99] | 74.17±0.24 [74.34/74.22/73.81/74.30] | 90.67±0.16 [90.81/90.76/90.46/90.62] |
| PatchModPTA-CS-AGate | 48.12±0.27 [48.40/48.23/47.75/48.11] | 74.14±0.26 [74.02/74.14/73.89/74.50] | 90.83±0.08 [90.87/90.92/90.76/90.76] |
| PatchModPTA-CS | 47.21±0.73 [46.75/48.23/47.22/46.63] | 72.22±0.37 [71.66/72.39/72.39/72.43] | 89.08±0.40 [89.48/89.34/88.83/88.66] |
| ratio-pta-mv | 48.45±0.74 [49.29/47.99/48.82/47.70] | 74.17±0.27 [74.10/73.89/74.54/74.14] | 90.76±0.08 [90.76/90.68/90.87/90.73] |
| ratio-clip-mv | 48.51±0.28 [48.76/48.11/48.64/48.52] | 74.18±0.18 [74.22/74.18/73.93/74.38] | 90.62±0.15 [90.79/90.54/90.71/90.46] |

## 2. Paired Delta vs PTA-CS

| Arm | Dataset | deltas (s1..s4) | mean | p_sign | seed CI | threshold | verdict |
|---|---|---|---|---|---|---|---|
| PatchModPTA-CS-QGated | dtd | -0.89/+0.77/-0.24/-0.89 | -0.31pp | 0.8750 | [-0.89, +0.35] | 1.00pp | **REFUTE** |
| PatchModPTA-CS-QGated | oxford_flowers | -2.76/-2.35/-1.83/-2.56 | -2.38pp | 1.0000 | [-2.66, -2.01] | 1.00pp | **REFUTE** |
| PatchModPTA-CS-QGated | oxford_pets | -1.47/-1.77/-2.13/-2.53 | -1.98pp | 1.0000 | [-2.34, -1.62] | 1.00pp | **REFUTE** |
| PatchModPTA-CS-MVote | dtd | +1.12/+0.35/+0.24/+0.47 | +0.55pp | 0.0625 | [+0.30, +0.93] | 1.00pp | **INCONCLUSIVE** |
| PatchModPTA-CS-MVote | oxford_flowers | -0.12/-0.57/-0.45/-0.69 | -0.46pp | 1.0000 | [-0.63, -0.23] | 1.00pp | **REFUTE** |
| PatchModPTA-CS-MVote | oxford_pets | -0.19/-0.38/-0.49/-0.57 | -0.41pp | 1.0000 | [-0.53, -0.27] | 1.00pp | **REFUTE** |
| PatchModPTA-CS-AGate | dtd | +0.77/+0.77/+0.30/+0.59 | +0.61pp | 0.0625 | [+0.41, +0.77] | 1.00pp | **INCONCLUSIVE** |
| PatchModPTA-CS-AGate | oxford_flowers | -0.45/-0.65/-0.37/-0.49 | -0.49pp | 1.0000 | [-0.60, -0.40] | 1.00pp | **REFUTE** |
| PatchModPTA-CS-AGate | oxford_pets | -0.14/-0.22/-0.19/-0.44 | -0.25pp | 1.0000 | [-0.37, -0.16] | 1.00pp | **REFUTE** |
| PatchModPTA-CS | dtd | -0.89/+0.77/-0.24/-0.89 | -0.31pp | 0.8750 | [-0.89, +0.35] | 1.00pp | **REFUTE** |
| PatchModPTA-CS | oxford_flowers | -2.80/-2.40/-1.87/-2.56 | -2.41pp | 1.0000 | [-2.70, -2.04] | 1.00pp | **REFUTE** |
| PatchModPTA-CS | oxford_pets | -1.53/-1.80/-2.13/-2.53 | -2.00pp | 1.0000 | [-2.35, -1.66] | 1.00pp | **REFUTE** |
| ratio-pta-mv | dtd | +1.65/+0.53/+1.36/+0.18 | +0.93pp | 0.0625 | [+0.35, +1.51] | 1.00pp | **INCONCLUSIVE** |
| ratio-pta-mv | oxford_flowers | -0.37/-0.89/+0.28/-0.85 | -0.46pp | 0.9375 | [-0.87, +0.00] | 1.00pp | **REFUTE** |
| ratio-pta-mv | oxford_pets | -0.25/-0.46/-0.08/-0.46 | -0.31pp | 1.0000 | [-0.46, -0.16] | 1.00pp | **REFUTE** |
| ratio-clip-mv | dtd | +1.12/+0.65/+1.18/+1.00 | +0.99pp | 0.0625 | [+0.77, +1.15] | 1.00pp | **INCONCLUSIVE** |
| ratio-clip-mv | oxford_flowers | -0.24/-0.61/-0.32/-0.61 | -0.45pp | 1.0000 | [-0.61, -0.28] | 1.00pp | **REFUTE** |
| ratio-clip-mv | oxford_pets | -0.22/-0.60/-0.25/-0.74 | -0.45pp | 1.0000 | [-0.67, -0.23] | 1.00pp | **REFUTE** |

## 3. Tie / Agreement Breakdown (generalizes outputs/tie_breaking_analysis.md)

A tie is `clip.argmax != image_proto.argmax` (per PTA-CS's own logits, pooled over 4 seeds).

| Arm | Dataset | ties/total | PTA acc on ties | arm acc on ties | helps | hurts | agree-image | agree-clip | other | patch-GT-match |
|---|---|---|---|---|---|---|---|---|---|---|
| PatchModPTA-CS-QGated | dtd | 2243/6768 (33.1%) | 27.6% | 28.2% | 184 | 170 | 468 (20.9%) | 524 (23.4%) | 1251 (55.8%) | 424/2243 (18.9%) |
| PatchModPTA-CS-QGated | oxford_flowers | 1696/9852 (17.2%) | 35.7% | 29.8% | 88 | 189 | 345 (20.3%) | 259 (15.3%) | 1092 (64.4%) | 262/1696 (15.4%) |
| PatchModPTA-CS-QGated | oxford_pets | 1288/14676 (8.8%) | 59.9% | 49.5% | 105 | 238 | 327 (25.4%) | 270 (21.0%) | 691 (53.6%) | 301/1288 (23.4%) |
| PatchModPTA-CS-MVote | dtd | 2243/6768 (33.1%) | 27.6% | 29.3% | 140 | 102 | 468 (20.9%) | 524 (23.4%) | 1251 (55.8%) | 424/2243 (18.9%) |
| PatchModPTA-CS-MVote | oxford_flowers | 1696/9852 (17.2%) | 35.7% | 33.0% | 77 | 124 | 345 (20.3%) | 259 (15.3%) | 1092 (64.4%) | 262/1696 (15.4%) |
| PatchModPTA-CS-MVote | oxford_pets | 1288/14676 (8.8%) | 59.9% | 55.1% | 94 | 155 | 327 (25.4%) | 270 (21.0%) | 691 (53.6%) | 301/1288 (23.4%) |
| PatchModPTA-CS-AGate | dtd | 2243/6768 (33.1%) | 27.6% | 29.4% | 113 | 73 | 468 (20.9%) | 524 (23.4%) | 1251 (55.8%) | 424/2243 (18.9%) |
| PatchModPTA-CS-AGate | oxford_flowers | 1696/9852 (17.2%) | 35.7% | 32.7% | 42 | 93 | 345 (20.3%) | 259 (15.3%) | 1092 (64.4%) | 262/1696 (15.4%) |
| PatchModPTA-CS-AGate | oxford_pets | 1288/14676 (8.8%) | 59.9% | 57.1% | 63 | 99 | 327 (25.4%) | 270 (21.0%) | 691 (53.6%) | 301/1288 (23.4%) |
| PatchModPTA-CS | dtd | 2243/6768 (33.1%) | 27.6% | 28.2% | 184 | 170 | 468 (20.9%) | 524 (23.4%) | 1251 (55.8%) | 424/2243 (18.9%) |
| PatchModPTA-CS | oxford_flowers | 1696/9852 (17.2%) | 35.7% | 29.7% | 88 | 191 | 345 (20.3%) | 259 (15.3%) | 1092 (64.4%) | 262/1696 (15.4%) |
| PatchModPTA-CS | oxford_pets | 1288/14676 (8.8%) | 59.9% | 49.5% | 105 | 239 | 327 (25.4%) | 270 (21.0%) | 691 (53.6%) | 301/1288 (23.4%) |
| ratio-pta-mv | dtd | 2243/6768 (33.1%) | 27.6% | 30.4% | 224 | 161 | 468 (20.9%) | 524 (23.4%) | 1251 (55.8%) | 424/2243 (18.9%) |
| ratio-pta-mv | oxford_flowers | 1696/9852 (17.2%) | 35.7% | 33.2% | 116 | 159 | 345 (20.3%) | 259 (15.3%) | 1092 (64.4%) | 262/1696 (15.4%) |
| ratio-pta-mv | oxford_pets | 1288/14676 (8.8%) | 59.9% | 56.5% | 106 | 149 | 327 (25.4%) | 270 (21.0%) | 691 (53.6%) | 301/1288 (23.4%) |
| ratio-clip-mv | dtd | 2243/6768 (33.1%) | 27.6% | 30.2% | 189 | 131 | 468 (20.9%) | 524 (23.4%) | 1251 (55.8%) | 424/2243 (18.9%) |
| ratio-clip-mv | oxford_flowers | 1696/9852 (17.2%) | 35.7% | 32.5% | 83 | 137 | 345 (20.3%) | 259 (15.3%) | 1092 (64.4%) | 262/1696 (15.4%) |
| ratio-clip-mv | oxford_pets | 1288/14676 (8.8%) | 59.9% | 55.0% | 89 | 151 | 327 (25.4%) | 270 (21.0%) | 691 (53.6%) | 301/1288 (23.4%) |

## 4. flip_metrics_v2 Aggregate (pooled over 4 seeds)

| Arm | Dataset | patch_alone corr/reg (net) | patch_alone acc | patch_image corr/reg (net) | patch_image acc |
|---|---|---|---|---|---|
| PatchModPTA-CS-QGated | dtd | 422/768 (-346) | 35.5% | 365/471 (-106) | 43.7% |
| PatchModPTA-CS-QGated | oxford_flowers | 296/3218 (-2922) | 8.4% | 210/1261 (-1051) | 14.3% |
| PatchModPTA-CS-QGated | oxford_pets | 404/6110 (-5706) | 6.2% | 380/1680 (-1300) | 18.4% |
| PatchModPTA-CS-MVote | dtd | 404/681 (-277) | 37.2% | 336/344 (-8) | 49.4% |
| PatchModPTA-CS-MVote | oxford_flowers | 285/3174 (-2889) | 8.2% | 183/1209 (-1026) | 13.1% |
| PatchModPTA-CS-MVote | oxford_pets | 399/6105 (-5706) | 6.1% | 373/1663 (-1290) | 18.3% |
| PatchModPTA-CS-AGate | dtd | 401/681 (-280) | 37.1% | 333/343 (-10) | 49.3% |
| PatchModPTA-CS-AGate | oxford_flowers | 285/3174 (-2889) | 8.2% | 183/1209 (-1026) | 13.1% |
| PatchModPTA-CS-AGate | oxford_pets | 399/6105 (-5706) | 6.1% | 373/1663 (-1290) | 18.3% |
| PatchModPTA-CS | dtd | 422/768 (-346) | 35.5% | 365/471 (-106) | 43.7% |
| PatchModPTA-CS | oxford_flowers | 296/3218 (-2922) | 8.4% | 210/1261 (-1051) | 14.3% |
| PatchModPTA-CS | oxford_pets | 404/6110 (-5706) | 6.2% | 380/1680 (-1300) | 18.4% |
| ratio-pta-mv | dtd | 392/663 (-271) | 37.2% | 321/279 (+42) | 53.5% |
| ratio-pta-mv | oxford_flowers | 289/3177 (-2888) | 8.3% | 191/1090 (-899) | 14.9% |
| ratio-pta-mv | oxford_pets | 399/6105 (-5706) | 6.1% | 330/1557 (-1227) | 17.5% |
| ratio-clip-mv | dtd | 396/678 (-282) | 36.9% | 308/348 (-40) | 47.0% |
| ratio-clip-mv | oxford_flowers | 285/3175 (-2890) | 8.2% | 193/1160 (-967) | 14.3% |
| ratio-clip-mv | oxford_pets | 399/6105 (-5706) | 6.1% | 347/1603 (-1256) | 17.8% |

## 5. Verdict

**Overall study verdict: REFUTE**

| Arm | dtd | oxford_flowers | oxford_pets |
|---|---|---|---|
| PatchModPTA-CS-QGated | REFUTE | REFUTE | REFUTE |
| PatchModPTA-CS-MVote | INCONCLUSIVE | REFUTE | REFUTE |
| PatchModPTA-CS-AGate | INCONCLUSIVE | REFUTE | REFUTE |
| PatchModPTA-CS | REFUTE | REFUTE | REFUTE |
| ratio-pta-mv | INCONCLUSIVE | REFUTE | REFUTE |
| ratio-clip-mv | INCONCLUSIVE | REFUTE | REFUTE |

No (arm, dataset) cell reaches SUPPORT anywhere, so per the pre-registered
overall rule ("REFUTE H1 if no cell reaches SUPPORT"), **H1 is REFUTED**:
patch-level prototype evidence does not bring a genuine, noise-exceeding
accuracy benefit over PTA baseline on dtd/oxford_flowers/oxford_pets, under
any of the 6 fusion/write-rule configurations tested, at seeds 1-4.

The one consistent partial signal is on **dtd**: 4 of 6 arms (MVote, AGate,
ratio-pta-mv, ratio-clip-mv) beat baseline in all 4 seeds each
(`p_sign=0.0625`, the strongest possible signal at n=4), but the effect size
(+0.55pp to +0.99pp) never clears the pre-registered 1.0pp noise floor, so it
stays INCONCLUSIVE rather than SUPPORT. On **oxford_flowers** and
**oxford_pets**, all 6 arms lose to baseline, almost all with `p_sign=1.0`
(unanimous 4/4 seeds against the arm) — clear REFUTE, not noise.

## 6. External Corroborating Evidence

`outputs/experiment_summary.md` (updated 2026-08-19, commit `a31a16e`) reports
the full paper grid — 5 methods x 5 datasets x **5 seeds**, same
clip_surgery/ViT-B/16 defaults — run via `scripts/run_experiments.sh`
(raw per-sample records not present on this machine, so not pooled into the
statistics above; corroboration only). Its per-dataset deltas vs PTA for the
4 arms it shares with this study, on this study's 3 target datasets:

| Dataset | ProtoAlpha | QGated | MVote | AGate |
|---|---|---|---|---|
| dtd | -0.08% | -0.09% | +0.65% | +0.60% |
| oxford_flowers | -2.29% | -2.24% | -0.13% | -0.24% |
| oxford_pets | -2.01% | -2.00% | -0.39% | -0.31% |

This independently-run, larger (5-seed) grid shows the same pattern found
here: small positive MVote/AGate deltas on dtd, negative on both
oxford_flowers and oxford_pets for every arm. The two studies' raw accuracy
numbers themselves are close (e.g. MVote dtd: 48.06% here vs 48.09% there;
AGate oxford_pets: 90.83% here vs 90.85% there — all within ~0.3pp),
which also serves as an independent sanity check on this study's pipeline.

## 7. Bottom Line

Across 6 fusion/write-rule configurations, 3 datasets, and 4 seeds each, with
a pre-registered decision rule and a noise floor set before any patch-arm
data was inspected: **patch-level prototype evidence does not demonstrate a
statistically defensible accuracy benefit over baseline PTA** on
dtd/oxford_flowers/oxford_pets. It actively hurts on the two fine-grained
datasets (oxford_flowers, oxford_pets) regardless of fusion mechanism, and
produces at best a small, sub-threshold, statistically-inconclusive gain on
dtd. The `flip_metrics_v2` diagnostic (Section 4) shows why mechanistically:
`patch_alone`'s own predictions are net-negative everywhere (more regressions
than corrections vs text-only CLIP), and `patch_image` (patch added on top of
PTA) is net-negative on oxford_flowers/oxford_pets for every arm and
only marginally positive on dtd for the best-case `ratio-pta-mv` arm
(net +42). The known instability of `ratio-pta-mv` is not responsible for
this conclusion: its stable control, `ratio-clip-mv`, shows the identical
REFUTE/INCONCLUSIVE pattern.

## 8. Follow-up: Why Doesn't Patch-Level Help? (Root-Cause Diagnostics)

Section 7 establishes *that* patch-level evidence doesn't help. This section
asks *why*, by testing two candidate explanations. Both turned out to be
dead ends — which is itself useful, because it rules out the two most
obvious fixes and points at a third, simpler explanation.

Some vocabulary used below, defined once up front:

- **The "bank"**: for each class (e.g. "dtd texture: banded"), the patch
  method keeps a small set of reference patches it has collected from test
  images it believed belonged to that class. This collection is the "bank."
  It's built on the fly during the test run — there's no ground truth to
  build it from, only the model's own (sometimes wrong) guesses.
- **"Writing" to a bank**: adding a new image's patches into a class's bank.
  This happens automatically, during the run, whenever the model is
  confident enough (above a threshold) that an image belongs to that class.
- **"Purity"**: of all the times something got written into class X's bank,
  what fraction of those images actually *were* class X (per the true test
  label)? High purity = the bank is mostly built from the right images.
  Low purity = the bank is contaminated with patches from other classes.
- **"Separability"**: even if a bank is 100% pure (every patch in it really
  is from the right class), do those patches actually *look different*
  from a neighboring class's patches? Two visually-similar breeds of cat
  can have fur patches that look more alike to each other than to other
  photos of the same breed — that would be a bank with good purity but bad
  separability.
- **"Ceiling / base-rate effect"**: how much room a method has left to
  improve. If a baseline is already right 91% of the time (oxford_pets),
  there are very few wrong answers left to fix, so almost any extra,
  imperfect signal you add is more likely to break a correct answer than
  fix a wrong one. If a baseline is only right 47% of the time (dtd), the
  opposite is true — there's a lot more to gain and less to lose.

### 8.1 Diagnostic 1: is the bank contaminated? (Purity)

Test: replay the exact rule the code uses to decide what gets written into
each class's bank (`models/patch_modulated_pta.py`, using the already-stored
zero-shot CLIP scores and true labels — no new GPU runs needed), and check,
for every write, whether the image really belonged to that class.

Full numbers: `outputs/prototype_purity_report.md`.

| Dataset | Purity (higher = cleaner bank) | Patch fusion's effect on accuracy |
|---|---|---|
| dtd | 62.6% (dirtiest bank) | Mildly **helps** (+0.6 to +0.65pp) |
| oxford_flowers | 74.1% | **Hurts** (-0.13 to -0.24pp) |
| oxford_pets | 85.7% (cleanest bank) | **Hurts** most (-0.31 to -0.39pp) |

**Result: this is backwards from what contamination would predict.** The
dataset with the *dirtiest* bank (dtd) is the one where patch fusion helps;
the dataset with the *cleanest* bank (oxford_pets) is the one it hurts the
most. So a contaminated bank is not the problem — even a clean bank doesn't
help.

### 8.2 Diagnostic 2: is a clean bank still confusable? (Separability)

Test: for each dataset, dump the actual reference patches stored in every
class's bank at the end of a run, and measure two things — how similar a
class's own reference patches are to *each other* ("coherence"), and how
similar they are to their single closest match in a *different* class
("confusability"). If a class's own patches are more alike to each other
than to any other class's nearest patch, that class is separable; if not,
even a perfectly pure bank can't reliably tell it apart from a neighbor.

Full numbers: `outputs/prototype_separability_report.md` (restricted to
each class's most-trusted reference patches — the top half by internal
confidence weight, since those dominate the actual prediction).

| Dataset | Own-class coherence | Closest different-class match | Separable? |
|---|---|---|---|
| dtd | 0.738 | 0.859 | No — negative margin |
| oxford_flowers | 0.799 | 0.929 | No — negative margin (worst) |
| oxford_pets | 0.795 | 0.908 | No — negative margin |

**Result: every dataset fails this test, including dtd.** A class's
reference patches are consistently *less* similar to each other than to
their nearest neighbor in another class — everywhere, not just on the
fine-grained datasets. And the differences between datasets here are small
(-0.114 to -0.130) compared to how differently patch fusion actually
performs on them (+0.6pp vs -2.3pp), so this alone doesn't explain the gap
either. oxford_pets — the dataset hurt the most — isn't even the least
separable of the three by this measure.

### 8.3 What's left: the ceiling / base-rate effect

Neither diagnostic explains why patch fusion helps on dtd and hurts on
oxford_flowers/oxford_pets — both show the patch bank is roughly similar
(mediocre) quality across all three datasets. What *does* differ sharply
between datasets is baseline PTA accuracy: 47.5% on dtd vs. 74.6% on
oxford_flowers and 91.1% on oxford_pets (Section 1).

That difference alone is enough to produce the observed pattern, without
needing the bank itself to be better or worse: adding a noisy, mediocre
extra signal to a very-often-already-correct baseline (oxford_pets, 91%)
has few wrong answers left to fix and many correct ones it can accidentally
break — so the expected effect is negative. Adding the same
similarly-mediocre signal to a much-more-often-wrong baseline (dtd, 47.5%)
has many more wrong answers available to fix and fewer correct ones to
break — so the expected effect can be positive even though the signal
itself is no better in quality. This is consistent with Section 4's
`flip_metrics_v2` numbers: the patch signal's raw correction/regression
counts are similar in character across datasets, but oxford_flowers/
oxford_pets simply have far fewer wrong baseline predictions available for
patch evidence to correct.

### 8.4 Practical takeaway

Improving the bank's construction (stricter write rules, purity filtering)
or its fusion mechanism (gating, voting) is unlikely to help on its own,
since the bank is roughly comparable in quality across datasets already —
the two most obvious "fix the prototypes" directions were tested and ruled
out here. If patch-level evidence is worth pursuing further, the more
promising direction is making its contribution conditional on how much
headroom the baseline has left (e.g. only trust patch evidence more heavily
when the baseline's own top prediction is already uncertain), rather than
trying to make the bank itself cleaner or more separable.

## 9. Follow-up Discussion: Refining the Diagnosis

Section 8 raised more questions than it answered. This section works through
four follow-ups, each grounded in the actual code rather than speculation.

### 9.1 The bank is made of clusters, not raw patches — does that change anything?

Each entry in a class's bank is not a single stored patch. It's a **cluster
centroid**: the code groups similar raw patches together (if they're similar
enough, `match_threshold=0.6`), and keeps a running average position
("center"), a spread ("variance" — how tightly the group's patches cluster
around that center), and a hit count ("appearance" — how many images have
contributed to that cluster so far).

This doesn't change Diagnostic 1 (purity, Section 8.1) — that test asks
"does this image's patches get sent to class X's bank at all," which happens
before any clustering, so it's unaffected. It also doesn't change the core
finding of Diagnostic 2 (separability, Section 8.2), since that test already
compared cluster centroids directly (the right unit to compare). It does
reveal one limitation of that diagnostic: it compared centroids only by
cosine similarity and ignored the "variance" (spread) each cluster carries.
The real scoring formula (`_gaussian_score_for_class` in `utils/kmeans.py`)
uses both — two classes can have similar-looking centroids and still be
distinguishable in practice if their clusters are tight (low variance). So
the -0.11 to -0.13 confusability numbers in Section 8.2 are a reasonable
first pass but likely somewhat overstate the problem for the tightest
clusters. A sharper version of that diagnostic would fold variance in.

### 9.2 Is low purity actually the problem, or is it something narrower?

Low purity (Section 8.1) by itself isn't necessarily bad — some diversity in
a bank, coming from an imperfect base classifier, can be tolerable or even
useful. The narrower, real problem is in how much *say* an impure write gets.

Checked directly in code: a cluster's "appearance" count (hit count) goes up
by exactly **+1 per matching image, with no adjustment for how confident or
how likely-correct that write was**:

```python
all_apps.append(apps[k] + (1.0 if appeared[k] else 0.0))
```

So a confidently-*wrong* write counts exactly as much toward a cluster's
importance as a confidently-*correct* one — there's no way for the system to
tell them apart, since it never sees the true label. And appearance isn't a
minor bookkeeping detail: it decides both which clusters survive when a
class's bank gets pruned down to its size limit, and how much weight each
surviving cluster gets in the final prediction. So the sharper framing is:
it's not that the bank has some wrong patches in it (expected, and not
inherently harmful) — it's that wrong writes compete on **equal footing**
with correct ones for becoming the bank's most-trusted, most-weighted
entries.

### 9.3 Does the existing "appearance weight" mechanism already fix this?

Partially. The code's own design already leans on this idea: a cluster's
final score is weighted by its appearance count (comment in the code:
"appearance count is a reliable proxy for prototype maturity and
discriminative power"), and clusters below an appearance threshold are
dropped from scoring entirely.

Restricting Diagnostic 2's analysis to only the top half of clusters by
appearance (the "most trusted" half) does shrink the confusability problem
substantially — roughly in half on every dataset:

| Dataset | Confusability margin, all clusters | Confusability margin, high-appearance clusters only |
|---|---|---|
| dtd | -0.178 | -0.121 |
| oxford_flowers | -0.233 | -0.130 |
| oxford_pets | -0.218 | -0.114 |

(Margin = how much tighter a class's own clusters are with each other than
with their closest match in another class. Negative means the class isn't
cleanly separable from its nearest neighbor; less negative is better.)

So appearance-weighting is doing real, measurable work. But the margin never
turns positive — even the most-trusted clusters remain, on average, closer
to a neighboring class's nearest cluster than to their own. It helps; it
doesn't fix the underlying problem.

**Does this benefit actually depend on the base classifier being reasonably
accurate**, as intuition would suggest (a more accurate classifier → more of
the frequently-reinforced clusters are genuinely correct → filtering to
"frequently reinforced" comes closer to filtering to "correct")? Comparing
against how often each dataset's writes are actually correct (purity, from
Section 8.1) does show the expected direction:

| Dataset | Write purity | Margin improvement from filtering to high-appearance |
|---|---|---|
| dtd | 62.6% (least accurate) | 0.057pp (smallest benefit) |
| oxford_flowers | 74.1% | 0.103pp |
| oxford_pets | 85.7% (most accurate) | 0.104pp (largest benefit) |

This is suggestive, not conclusive: with only three datasets, and no way (in
this data) to hold "how many images fed a cluster" fixed while separately
varying "how many of them were correct," there's a competing explanation
that fits equally well — a cluster built from more images always produces a
statistically less noisy centroid, purely from averaging more data, whether
or not those images were the right class. Appearance-weighting would help
by that mechanism too, independent of correctness. There's also a specific
edge case worth flagging: a class that the model *consistently* confuses
with one particular neighbor (always predicting B for images that are
really A) would also build a high-appearance, low-variance, "confident
looking" cluster — appearance weight can't distinguish a cluster that's
frequently reinforced because it's correct from one that's frequently
reinforced because the same mistake keeps repeating.

### 9.4 A different idea from prior work: TCR (Term Confidence Ratio)

TCR is a scoring idea from a different, unrelated project, originally used
to score words/terms in a document-classification setting. In that setting:
for each class, you count how often each candidate term appears in that
class's documents ("term frequency," normalized into a density), then
downweight terms that *also* show up frequently in *other* classes (a
class-specificity correction), and additionally downweight terms whose
occurrence is spread evenly across all classes rather than concentrated in
one (an entropy penalty for genericness). Multiplying density × specificity
× (1 − genericness) produces a score that's high only for terms that are
both common *and* distinctive to one class.

The question was whether this can be adapted to this codebase's prototype
clusters (treating each cluster as a "term") in place of, or alongside, the
existing appearance weight — and whether it needs ground-truth labels to do
so.

**It does not need ground-truth labels.** It would use exactly the same
substitution the current code already makes: "class" = whichever bank the
model's own confidence-thresholded decision assigned an image to, not a true
label. Density is already computed today (`appearance / n_images`). The
missing pieces — a class-specificity correction and a genericness penalty —
don't exist in the current code at all, and are exactly what Section 8.2's
confusability finding suggests is missing: two classes' clusters looking
alike is never currently penalized. The natural way to add the missing
pieces here: for each cluster, compare it (by cosine similarity) against
every other class's clusters to estimate how much a similar cluster also
exists elsewhere, then compute a specificity/genericness correction from
that, and use it to reweight the cluster instead of (or in addition to) the
current flat appearance weight.

**But this does not close the gap identified in 9.3.** TCR-style specificity
checks whether a cluster's evidence is spread across *many* classes
(catching generic, non-discriminative patches). It does not catch the
systematic-confusion case above — a cluster built entirely from one
consistent A→B misclassification looks perfectly concentrated and specific
to B under this method, indistinguishable from a genuinely correct B
cluster, because it never shows up anywhere else to be penalized for. So TCR
would plausibly help with one specific failure mode (generic, overly broad
clusters) without addressing the other (a consistent, confidently-wrong
classifier bias).

**This needs to be checked, not reasoned about further.** Whether either
mechanism (TCR-style reweighting, or a finer per-cluster purity check) would
actually recover accuracy is an empirical question. The cheapest way to
check it doesn't require a full re-run of the adaptation study: reuse the
already-collected bank snapshots (`outputs/patch_bank_dumps/`) and the
already-stored predictions from this study's records, re-score each image
against a reweighted (frozen) bank, and recompute accuracy — this needs one
fresh pass to extract each image's raw patch features (not currently saved),
but not a repeat of the full multi-seed adaptation sweep. Not yet built;
flagged here as the natural next step if this direction is worth pursuing.
