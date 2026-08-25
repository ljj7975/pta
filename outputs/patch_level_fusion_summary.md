# PTA Limitations & Patch-Level Fusion — Full Investigation Summary

**The starting point:** plain PTA (test-time adaptation via a single
running memory per class, updated from CLIP's own confident guesses) has
real limitations. The whole investigation below — roughly 30 distinct
experiments across three earlier rounds plus this session — asks: can
adding *patch-level* signal (evidence from small pieces of the image, not
just the whole image) fix those limitations?

**Short answer:** No, not yet. Across every round, patch *content* almost
never helps once it's fused into the actual prediction — several attempts
actively hurt, sometimes badly. The one thing that keeps showing up as
genuinely useful, independently rediscovered twice, is a much narrower
idea: using patch evidence as a **meta-signal** (does it agree with CLIP?
does it look trustworthy right now?) rather than as a vote on the class
itself. That signal is real and well-validated — but we haven't yet
managed to turn it into an accuracy win.

---

## Part 1 — Earlier rounds: does patch-level signal fix PTA? (before this session)

### Round 1: First attempt — many prototypes per class, patch-based (12+ variants)

Added multiple patch-level "memory slots" per class (found via clustering)
alongside PTA's existing single memory, then tried many ways to combine
them.

| Variant | Idea | Result |
|---|---|---|
| Patch-only prototypes | Classify using only patch memory, no PTA | 60.4 vs PTA 67.1 — badly hurt |
| Fixed diverse patch set | 15 hand-picked representative patches/class | Worst variant tested |
| 50/50 blend | Equal weight patch + image signal | Still far below PTA |
| Gaussian-shaped memory | Soft clusters instead of hard ones | Still far below PTA |
| Full 3-way fusion (fixed/tunable weights) | Text + image + patch, weighted | ≈ PTA — patch adds ~nothing |
| Downweight image memory when patch data is abundant | — | **Catastrophic** (-25pp on one dataset) |
| Upweight image instead | Opposite of above | Also clearly worse |
| Boost patch weight 25× | — | Barely moved the needle at all |
| Patch weight modulated by prediction uncertainty | — | **Catastrophic collapse** (single digits on 2 datasets) |
| Soft proportional update gate | — | ≈ PTA — harmless, no benefit |
| Patch quality gates the *image* memory's update speed | — | Best of all 12 — ties PTA |

Documented conclusion at the time: **no variant beat PTA.**

### Round 2: Rebuilt cleanly to isolate *where* any benefit comes from

| Test | Idea | Result |
|---|---|---|
| Downweight image branch when patch data is abundant | Retest of Round 1's worst idea | -11pp vs PTA — hypothesis directly disproved |
| Soft gating variant | — | Exactly matched PTA — neutral |
| **Quality-gate only** — remove patch content from the prediction entirely, keep *only* a signal that speeds up/slows down the image memory's update rate | Isolates whether patch *content* matters at all | **Statistically identical to using full patch content** |
| Best single-patch match vs. averaging several patches | Aggregation method sweep | Averaging beat picking one best patch |

**Key finding:** the entire benefit of the patch system came from a
meta-signal about how trustworthy the evidence looked *right now* — not
from patch content actually voting on the class. Removing the patch
content from the prediction and keeping only that meta-signal produced
the same result. Documented recommendation at the time: drop patch-level
content entirely, or simplify down to just this quality-gate mechanism.

### Round 3: A properly powered, multi-seed retest (immediately before this session)

Tested 5 different ways to fuse patch evidence, across 5 datasets and 5
seeds each — the first adequately-powered test in the whole investigation.

| Fusion mechanism | Idea | Result vs. PTA |
|---|---|---|
| Always blend in a weighted patch term | Two variants of this | Both **collapsed on fine-grained datasets** (~-1.2pp average) |
| Only trust patch evidence when ≥2 of 3 signals agree | Majority vote | **Best of the five, +0.14pp** |
| Only count patch evidence when it agrees with CLIP's own guess | Agreement gate | Avoided the collapse (-0.04pp) |

Pattern: **always-on fusion of patch content collapses; gating on
agreement avoids the collapse.**

A follow-up diagnostic looked specifically at "tie" cases — images where
CLIP's own guess and PTA's memory disagree (about 31% of samples).
Memory turned out to be *wrong* 75% of the time on exactly these cases.
But: **when the patch-only vote happened to agree with CLIP's guess,
CLIP's accuracy on those cases jumped 13 points; when it agreed with
memory, memory's accuracy jumped 9 points.** This is the first sighting of
the idea that later became the central finding of this whole
investigation.

---

## Part 2 — This session: chasing the agreement signal properly (Experiments A–K)

This session picked up exactly where Round 3 left off, using PTA's
current, refactored patch-level-memory design (not the older
MultiProtoPTA architecture from Round 1).

### Root-cause diagnosis (A–D)

- Confirmed both PTA and patch-level memory trust CLIP's own guess to
  decide what to remember, with no ground truth.
- PTA blends every new guess into one running average per class, so a
  wrong guess gets diluted away. Patch-level memory instead groups
  patches into separate clusters and lets the most-repeated one win — a
  *repeatedly* wrong guess can build its own trusted cluster instead of
  being diluted.
- Confirmed a large share of bad memory writes happen when CLIP was
  *confident* and still wrong — no confidence-based filter can ever catch
  that, since confident-right and confident-wrong look identical from
  inside CLIP's own output.

### First fix round (E–H)

| # | Idea | Result |
|---|---|---|
| E | Let old memory "trust" fade over time, like PTA does | Made things worse |
| F | Weight memory by visual distinctiveness | Mixed, net negative |
| G | Only let patches vote among CLIP's top few guesses | No benefit found |
| H | Check whether a second, patches-only "vote" agrees with CLIP's usual guess | **Real signal — rediscovers Round 3's tie-breaking finding, this time rigorously** |

### Second round (I–J) — sharpening the H signal

You pointed out H's patch-vote (a simple average over all patches) could
be diluted by background. Tried alternatives — best-matching patches only,
a smoother version of the same idea, and, at your request, **Otsu's
method** (an automatic, no-tuning way to decide how many patches "count").

- Confirmed: on oxford_pets, switching from "average every patch" to
  "average just the best few" roughly **doubled** the signal's usefulness.
- The automatic (Otsu) version matched or beat the fixed-count version,
  with nothing to tune.
- Stress-tested four ways (different seeds, graded confidence, per-class
  breakdown, checking it wasn't secretly re-detecting something we already
  knew). Held up on every check.

### Third round (K) — turning the signal into a fix

Built a live version: trust a memory write more when the second vote
agrees with CLIP, less when it disagrees.

- A quick single-seed test looked promising (+0.6 points on one dataset).
- The full, proper test (all seeds, both patch-picking methods) showed
  that result was noise — the four-seed average was flat.
  **Net result: no real improvement.**

---

## Where things stand, across the whole investigation

- **Patch content, fused directly into the prediction, has never reliably
  beaten plain PTA** — not in Round 1 (12+ variants), Round 2, Round 3, or
  this session's fix attempts. Several variants actively hurt, sometimes
  catastrophically.
- **The one repeating pattern across every round is that meta-signals beat
  content.** Round 2 found the patch system's entire benefit lived in a
  quality-gate, not patch content. Round 3 found gated/agreement-based
  fusion avoided collapse where always-on fusion didn't. This session
  independently rediscovered and rigorously validated the same
  "agreement" signal (H/I/J) — real, robust, survives heavy stress-testing.
- **We have never yet successfully converted that meta-signal into an
  accuracy win.** Round 2's version tied PTA. Round 3's version tied or
  barely edged it. This session's live implementation (K) was a clean
  null result.

## Options for what to try next

1. **Lean fully into the meta-signal, not patch content.** Every round
   that used patch evidence as a trust/agreement signal did better than
   every round that used it as a vote on the class. K tried this via
   write-time down-weighting and got a null result — worth trying other
   mechanisms in that same spirit (e.g. gating the *image*-level EMA rate,
   like Round 2's best variant did, using this session's more rigorously
   validated agreement signal instead of Round 2's cruder version).
2. **Retest more carefully with multiple seeds from the start.** This
   session's K pilot showed single-seed tuning can look like a real trend
   and not be one.
3. **Treat H/I/J as a finding in its own right**, independent of fixing
   patch-level fusion — a validated confidence/agreement signal for
   CLIP-based test-time adaptation, useful even if it never becomes part
   of PTA itself.
4. **Step back from patch-level fusion as a direction.** After roughly 30
   experiments across three earlier rounds and this session, all pointing
   the same way, it may be a structurally hard problem to solve by fusing
   patch content into the prediction.

## Where to find full detail

- `docs/experiments_v1.md`, `docs/experiments_v3.md` — Round 1 (MultiProtoPTA, 12+ variants)
- `docs/experiments_v2.md` — Round 2 (quality-gate isolation)
- `outputs/experiment_summary.md`, `outputs/session_summary.md`, `outputs/tie_breaking_analysis.md` — Round 3 (5-fusion-mechanism test, tie-breaking diagnostic)
- `outputs/patch_benefit_report.md` — this session's original benefit test
- `outputs/offline_experiments_plan.md` — this session's root-cause diagnosis (A–D)
- `outputs/patch_fix_experiments_plan.md` — this session's fix attempts (E–K), full data, statistics, and two bugs caught and fixed along the way
