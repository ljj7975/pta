# Phase 8: Prototype Repulsion — Active Geometry Correction Instead of Write Gating/Reweighting

## Context

Phase 7 closed out four write-time trust mechanisms (patch-vote / multi-view / confusability, hard gate
/ two-sided reweight / permanent freeze / read-time reweight) — all **defensive**, all net negative,
converging on one conclusion: filtering or discounting writes throws away information the EMA needs,
regardless of how well the filtering signal correlates with correctness on average.

This phase tests a mechanistically different family: **never touch the write. Instead, directly correct
prototype geometry** when the confusability diagnostic (Phase 4-6's strongest, cheapest signal) says two
class prototypes have drifted too close — an explicit repulsive nudge pushing them apart along their own
connecting direction, applied *in addition to* an always-firing, unmodified top-1 EMA write. Every prior
use of this signal blocked information flow into the confusable class; this is the first attempt to
correct the resulting geometry instead.

**Dev set**: `dtd`, `oxford_flowers`, `oxford_pets`, ViT-B/16, CLIP Surgery, seeds 1-4, single-class top-1
EMA write (same reference as every prior study this round: dtd 46.85 / flowers 74.22 / pets 90.75).

---

## Mechanism

Unmodified top-1 EMA write always fires. Immediately after, a repulsion step (`models/repulsive_pta.py`)
checks the target class's nearest-other-class prototype and, if triggered, nudges both apart:

```python
proto_norm = _safe_normalize(prototype_state, dim=-1)
sims = proto_norm @ proto_norm.t(); sims.fill_diagonal_(-1.0)
nearest_val, nearest_idx = sims.max(dim=1)
# trigger == "confusable": only fires when nearest_val[top1] exceeds the
#   66th-percentile threshold over already-written classes (Phase 4-6/7's tercile check)
# trigger == "always": fires unconditionally once top1 and its nearest-other class are both written
direction = proto_norm[top1] - proto_norm[other]
unit_dir = direction / direction.norm().clamp(min=1e-8)
prototype_state[top1]  += repulsion_lr * prototype_state[top1].norm()  * unit_dir
prototype_state[other] -= repulsion_lr * prototype_state[other].norm() * unit_dir
```
`repulsion_lr=0.0` is a hard no-op control. Grid: control + `{0.02, 0.05, 0.1} × {confusable, always}` = 7
settings × 3 datasets × 4 seeds = 84 runs, all completed with 0 failures.

Calibration (pre-registered in the plan, run before the full sweep): a 300-batch smoke test on dtd
confirmed the control reproduces the reference exactly (40.00% at 300 batches, matching every prior
study's control at the same slice), the correction fires at a sensible rate (66% for `confusable`, 99%
for `always`, matching Phase 7's `clean-conf` write-rate range), and produces no numerical instability.

---

## Results

### Accuracy (4-seed mean) vs. `control`

| Setting | dtd | oxford_flowers | oxford_pets | Avg | Δ vs control |
|---|---|---|---|---|---|
| control | 46.85 | 74.22 | 90.75 | 70.608 | +0.000 |
| lr0.02-conf | 47.03 | 74.48 | 89.88 | 70.465 | −0.142 |
| lr0.05-conf | 47.33 | 74.61 | 87.10 | 69.680 | −0.927 |
| lr0.1-conf | 46.38 | 73.90 | 82.75 | 67.676 | −2.932 |
| lr0.02-always | 47.02 | 74.07 | 89.58 | 70.222 | −0.385 |
| lr0.05-always | 46.78 | 72.98 | 85.60 | 68.453 | −2.155 |
| lr0.1-always | 36.85 | 66.73 | 81.64 | 61.740 | −8.868 |

All 6 settings: **no-go** (bar: >0.3pp avg gain, no >0.5pp regression on any dataset).

### Per-dataset delta (the average above hides a sharp, consistent split)

| Setting | dtd Δ | oxford_flowers Δ | oxford_pets Δ |
|---|---|---|---|
| lr0.02-conf | **+0.18** | **+0.26** | −0.86 |
| lr0.05-conf | **+0.47** | **+0.39** | −3.64 |
| lr0.1-conf | −0.47 | −0.32 | −8.00 |
| lr0.02-always | **+0.16** | −0.16 | −1.16 |
| lr0.05-always | −0.07 | −1.24 | −5.15 |
| lr0.1-always | −10.00 | −7.50 | −9.10 |

At low dose (`lr=0.02`, `confusable` trigger), `dtd` and `oxford_flowers` are both *slightly positive* —
the only positive deltas anywhere in this round's write-time campaign. `oxford_pets` regresses at every
single dose tested, monotonically and severely (−0.86 → −3.64 → −8.00pp as `repulsion_lr` increases), and
is what drives every averaged verdict to no-go. At high dose with the unconditional `always` trigger, the
correction compounds every step and collapses all three datasets (−7.5 to −10.0pp) — worse than most of
Phase 7's view-based settings.

### Mechanism: does the correction actually reduce confusability?

| Setting | Fire rate | Mean pre-confusability | Mean post-confusability | Δ |
|---|---|---|---|---|
| lr0.02-conf | 0.726 | 0.7912 | 0.7826 | −0.0087 |
| lr0.05-conf | 0.720 | 0.5866 | 0.5702 | −0.0164 |
| lr0.1-conf | 0.591 | 0.4440 | 0.4192 | −0.0248 |
| lr0.02-always | 1.000 | 0.7353 | 0.7260 | −0.0093 |
| lr0.05-always | 0.999 | 0.4661 | 0.4501 | −0.0160 |
| lr0.1-always | 0.993 | 0.2900 | 0.2702 | −0.0198 |

**Yes, unambiguously.** The correction fires at a nontrivial rate in every setting and reduces the target
class's nearest-other-class similarity every single time it fires (confirmed at the individual-sample
level during calibration: 199/199 in the smoke test). The reduction also scales with dose, and the
*pre*-confusability mean itself drops as `repulsion_lr` increases (0.79 → 0.44 for `confusable`) — later
samples in the same run are hitting an already-separated bank, showing the effect compounds correctly
across the online stream, not just per-nudge.

This decisively separates two questions Phase 4-6/7 could not: the geometric diagnosis (two prototypes
drifting close together predicts errors) was correct, **and** this mechanism does fix that specific
geometric fact — but fixing it does not translate into better classification, and for one dataset it
makes things substantially worse.

---

## Why it fails: repulsion assumes closeness is drift, but closeness is sometimes real similarity

The oxford_pets-specific collapse is the key diagnostic. Oxford Pets is a fine-grained breed dataset —
many class pairs (e.g. visually similar cat or dog breeds) sit close in CLIP's embedding space *because
they genuinely look alike*, not because either prototype has drifted from noisy writes. The confusability
signal cannot distinguish "these two prototypes are close because of drift/noise" from "these two classes
are close because CLIP's own feature space puts visually similar categories close together" — it only
measures the current geometric fact. Freezing/gating (Phase 4-6/7) responded to this ambiguity by
withholding information from both classes symmetrically, which is merely suboptimal. Repulsion actively
worsens it: forcing two legitimately-similar prototypes apart distorts the *decision boundary* between
them, and every subsequent classification near that boundary inherits the distortion — a boundary that
was well-calibrated to real visual similarity is now offset by an artificial correction with no
supporting evidence that the offset is in the right direction for any specific pair. `dtd` (46 texture
classes, less inherent fine-grained visual overlap) and `oxford_flowers` (102 classes, but botanically
more separable in CLIP's space than dog/cat breeds) show the opposite pattern at low dose precisely
because their close pairs are more likely to be genuine drift rather than genuine similarity — but the
mechanism cannot tell the difference in advance, so a dose small enough to help the drift cases is also
applied uniformly to the genuine-similarity cases, and larger doses (needed to fix drift further) make
the genuine-similarity damage compound faster than the drift-case gains ever accumulate.

---

## Comparison to Phase 7's confusability gate (same signal, opposite intervention)

| Signal use | Best case | Worst case |
|---|---|---|
| Permanent freeze (Phase 4-6) | — | −2.2 to −10.0pp (monotonic) |
| Per-sample gate, `clean` (Phase 7) | −0.77pp (dtd) | −2.1pp (`confident_and_clean`) |
| Per-sample reweight, mildest point (Phase 7) | +0.006pp (noise) | −0.82pp |
| **Repulsion, `lr=0.02-conf` (this phase)** | **+0.18 to +0.47pp (dtd, flowers)** | **−8.0pp (pets, at higher dose)** |

Repulsion is the only mechanism in the entire confusability-signal line of investigation that produces a
genuinely positive delta on any dataset at any dose — but it is also the only one capable of a
double-digit single-dataset regression, because unlike gating (which can only ever suppress information,
bounded by "the write never happens") a geometric nudge compounds every step it fires and has no natural
ceiling on how far it can push two prototypes apart. The mechanism is higher-variance in both directions:
more promising where the signal is right (drift), more damaging where the signal is ambiguous (genuine
similarity).

---

## Go/no-go

All 6 settings: **no-go** by the standard bar. No held-out validation triggered.

The `confusable`-trigger, low-dose regime (`lr=0.02`) is the one setting in this study worth flagging as a
**partial/mechanistic finding, not a promotion**: it is the only per-sample correction in this round's
entire write-time campaign to post a positive accuracy delta on two of three dev datasets, and it does so
while measurably fixing the geometry the diagnostic claims to detect. A follow-up that could plausibly
convert this into a real gain would need a way to discriminate "confusable because of drift" from
"confusable because of genuine visual similarity" *before* choosing whether to repel — e.g., gating the
repulsion step on a corroborating second signal (drift velocity — has this class's prototype moved
recently? — rather than static closeness), which is a natural next direction but out of scope here.

## Reproducing

```bash
bash scripts/run_repulsion_sweep.sh --run   # 84 runs, single-stage, ~15-20 min
python scripts/analyze_repulsion.py
```
