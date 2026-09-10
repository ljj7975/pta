# Where Things Stand — PTA Improvement Experiments

**What this is**: A plain-language summary of ~6 months of experiments trying to improve PTA. Written for a direction-revisit discussion.

---

## The One-Sentence Version

**We tried everything we could think of to improve PTA's prototype, and nothing works. The problem is structural: CLIP's own biased guesses are the only signal available at test time, and no amount of filtering or correction can overcome that.**

---

## What PTA Does (Quick Refresher)

PTA improves CLIP's predictions by maintaining a running "prototype" for each class. When CLIP sees a test image, it makes a guess. That guess gets blended into the class's prototype. On the next image, the prototype's opinion competes with CLIP's own.

PTA beats CLIP by about 2–3% on average. The question was: can we do better?

---

## What We Tried

### 1. Patch-Level Features

**The idea**: Instead of just using the whole-image embedding (CLS token), use the 196 patch-level features to get a richer signal — like looking at individual parts of the image instead of just the overall impression.

**What happened**: Patch features from different classes look more alike than features from the same class. The separability is negative everywhere — meaning a cat's fur patches look more like a dog's fur patches than like other cat patches. No amount of fusion can fix that.

**One useful thing we found**: Whether CLIP and the patch vote agree is a strong signal that CLIP is correct (30+ percentage point accuracy gap). But this is a *diagnostic*, not a *fix* — we can tell when CLIP is uncertain, but we can't do anything about it.

**Tried in**: write-gating, write-reweighting, read-time reweighting. All failed.

**Verdict**: Dead end. Closed.

---

### 2. Trust Signals (Phases 1–10)

**The idea**: If we can identify *when* CLIP is wrong, we can either skip those writes, down-weight them, or correct the prototype geometry afterward.

**What we tried** (10 phases, hundreds of GPU runs):

| Approach | Result |
|----------|--------|
| Reweight the prototype at read-time based on trust | Did nothing — the trust signal and "tie" samples overlap 88%, so there's nowhere for the reweighting to act |
| Skip writes when the prototype looks confused | Made things worse — freezing a class starves it of the good writes it needs |
| Nudge confused prototypes apart | Fixed the geometry exactly as designed, but made accuracy worse — can't distinguish "close because of drift" from "close because they genuinely look alike" |
| Add a drift detector to only nudge drifted pairs | Mildest result ever (zero regression anywhere), but the gain was +0.03 to +0.07% — not enough to matter |
| Multi-view consistency (run the image through 4 augmented copies, check if CLIP agrees with itself) | Strongest diagnostic signal in the entire campaign (+27pp purity gap). Still zero improvement as an intervention. |

**The key paradox**: The *better* a diagnostic signal is at identifying wrong predictions, the *worse* it performs as a write-time lever. This isn't a coincidence — it's structural. Good diagnostics flag the same samples that are genuinely ambiguous, and there's no reliable way to know what the right answer is for those samples.

**Verdict**: Exhaustively explored. Close this line entirely.

---

### 3. Better Prototype Representations (Waves 1–3)

**The idea**: PTA uses one average vector per class. What if we used something richer — a Gaussian (mean + variance), a bank of K prototypes, or a gated write that only accepts nearby writes?

**What happened**: Every method failed. Offline diagnostics passed; online methods failed.

| Method | Why it failed |
|--------|---------------|
| Gaussians | Variance estimated from biased stream amplifies errors instead of reducing them |
| K-prototype bank (v1) | All K prototypes collapse to one — nearest-assignment can't create diversity from CLIP's guesses |
| K-prototype bank (v2) | Diversity achieved, but the new prototypes are seeded from *wrong* CLIP writes — error modes, not genuine sub-classes |
| Distance-based write gate | Gate removes correct writes along with wrong ones — same problem as trust gating |
| Augmentation ensembling | Averaging multiple augmented views produces *worse* predictions than a single view |
| Text-anchor damping | After a short warm-up, 98–100% of writes get damped — becomes uniform scaling, identical to base PTA |

**Why offline gates passed but online failed**: Offline tests used ground-truth labels to validate the mechanism. Online, everything must be estimated from CLIP's own guesses, and those guesses are biased ~50% of the time on hard datasets. The gap between "works with perfect information" and "works with CLIP's information" is the whole problem.

**Hyperparameter check**: We swept α and T (PTA's two main parameters). Default values sit at the top of the grid. The baseline is not a strawman.

**Verdict**: Closed.

---

### 4. DEC Certainty Regularizer (Not Yet Run)

**The idea**: Use prediction entropy to modulate write strength — uncertain predictions get written more weakly.

**Status**: Built, committed, not tested on GPU. Offline checks show the committed implementation is near-constant (doesn't actually adapt), and the direction is inverted relative to the strongest purity signal we found. Would need rework before running.

**Verdict**: On hold. Reparameterize or close.

---

## Five Things We Know for Sure

1. **The prototype is the right thing to improve.** *(A "prototype" is a single running average vector PTA maintains per class, distilled from the test images it has seen so far; on any given image, PTA blends the prototype's opinion with CLIP's.)* It beats CLIP when they disagree. The method works; we just can't make it work *better*.

2. **The problem is the input, not the mechanism.** *(Every test image is fed through CLIP, which produces a guess with no ground-truth label to validate it against; that guess — a "write" — is what gets folded into the prototype. So the prototype is only ever as good as CLIP's own biased guesses.)* All writes come from CLIP's own guesses with no ground truth. Every method that tries to estimate quality from these guesses fails because the guesses themselves are biased.

3. **Good diagnostics don't make good interventions.** *(A "diagnostic" is a signal that tells us which predictions are likely wrong; a "lever" is a mechanism that acts on that signal to improve accuracy. We can reliably *detect* bad predictions, but every attempt to *react* to that detection fails.)* We found three strong signals (patch agreement, confusability, multi-view consistency) that reliably identify wrong predictions. None of them improves accuracy when used as a lever. This isn't a failure of engineering — it's a structural limitation: the same samples that carry a useful warning are the genuinely ambiguous ones, for which there is no way to know the correct answer at test time.

4. **The scalar trust-lever space is exhausted.** *(A "trust signal" tries to estimate how much to trust CLIP's guess on the current image — e.g. its own confidence, or how consistent it is across views. The "write side" is when the guess is being folded into the prototype; the "read side" is when the prototype's opinion is combined with CLIP's to produce the final prediction. A "lever" is any knob that scales that trust — gating (drop uncertain guesses entirely), up-weighting (trust uncertain ones *more*), down-weighting (trust them *less*), or two-sided reweighting.)* We tested gating, up-weighting, down-weighting, and two-sided reweighting on the write side and read side, with three different signal sources, in all combinations. Consistent null results.

5. **PTA's default is already optimal within this class.** *(The "α/T sweep" is a grid search over PTA's only two hyperparameters — α, how much weight the original text description keeps vs. the accumulating prototype, and T, a temperature controlling how slowly the prototype updates. We swept both across a range; the default values already score at the top of the grid, so every deviation makes things worse.)* The α/T sweep confirms it. Every modification makes things worse.

---

## Where Does This Leave Us?

If we want to beat PTA, we need a fundamentally different approach — one that doesn't rely on filtering or correcting CLIP's own guesses. Some options:

| Direction | What it means | Effort |
|-----------|---------------|--------|
| External grounding | Use a second model, a retrieval database, or self-training to get better pseudo-labels than CLIP's zero-shot | High |
| Batch-aware methods | Process groups of images jointly instead of one-at-a-time (contrastive objectives, memory buffers) | Medium-High |
| Stronger backbone | Move beyond ViT-B/16 CLIP Surgery features entirely | Different question |
| Accept the ceiling | PTA's +2–3% is the limit for this adaptation mechanism; pivot to a different TTA paradigm | Zero effort |

---

*For detailed numbers and methodology, see the individual phase reports in this directory.*
