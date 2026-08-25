# Pre-Registration: Does Patch-Level Prototype Evidence Bring Genuine Accuracy Benefit?

**Locked (Stage A): 2026-08-19, before any seed-2/3/4 run for this study.**
**Scope**: dtd, oxford_flowers, oxford_pets — ViT-B/16, CLIP Surgery, seeds 1-4.

## Prior evidence motivating this study

- `outputs/tie_breaking_analysis.md` (dtd, seed 1 only): on the 31.1% of samples
  where `clip.argmax != image_proto.argmax` ("ties"), the patch-level prediction
  matches ground truth directly in only 16.5% of ties.
- Earlier 5-arm accuracy comparisons (wrong dataset mix, single seed) showed only
  ~0.1-0.6pp gains over baseline — plausibly seed noise, not signal.
- `ratio-pta-mv` (write_source=pta, write_rule=ratio, fusion.type=MajorityVoteFusion)
  — the best setting found in the write-rule x write-source sweep
  (`results_dev_writerule.md`, 60.68% avg across 5 datasets) — is measurably
  **unstable**: on eurosat, accuracy = 50.46 / 60.06 / 66.15 across seeds 1-3
  (15.7pp range; `outputs/result_eurosat_seed_check.txt`), vs PTA baseline's
  61.28 / 61.62 / 61.30 (0.34pp range) on the same dataset/seeds. `ratio-clip-mv`
  (write_source=clip, otherwise identical) is the 2nd-best, more stable setting
  from that sweep (60.42 avg) and is included here specifically as a control to
  separate "write_source=pta instability" from "genuine patch-level benefit."

### Addendum — 2026-08-19 (after Stage B lock, before this study's own patch-arm results were inspected)

`outputs/experiment_summary.md` was corrected (commit `a31a16e`) to report the
full `scripts/run_experiments.sh` grid (5 methods x 5 datasets x **5 seeds**,
same `clip_surgery`/ViT-B/16 defaults as this study) on the correct 5 dev
datasets, superseding the earlier wrong-dataset-mix numbers cited above. Raw
per-sample records for that grid are not present on this machine
(`outputs/records/` does not exist locally), so it cannot be pooled into this
study's paired seed-level statistics — it is corroborating context only. Its
per-dataset deltas vs PTA for the 3 datasets this study targets:

| Dataset | ProtoAlpha | QGated | MVote | AGate |
|---|---|---|---|---|
| dtd | -0.08% | -0.09% | **+0.65%** | **+0.60%** |
| oxford_flowers | -2.29% | -2.24% | -0.13% | -0.24% |
| oxford_pets | -2.01% | -2.00% | -0.39% | -0.31% |

At 5 seeds, even the two gating arms this study also runs (MVote, AGate) are
**negative** on oxford_flowers and oxford_pets, positive only on dtd. This
does not change this study's arms, datasets, seeds, or the locked Stage
A/B decision rule/threshold (revising those now would be post-hoc tuning) —
it is noted here for the audit trail and will be discussed alongside this
study's own independently-computed verdict in the final report.

## Arms (7)

| # | Label prefix | Method | Config | Override |
|---|---|---|---|---|
| 1 | `PTA-CS` | pta | configs/PTA | — (baseline) |
| 2 | `PatchModPTA-CS` | patch_modulated_pta | configs/patch_modulated_pta | — (default ProtoAlphaFusion) |
| 3 | `PatchModPTA-CS-QGated` | patch_modulated_pta | configs/patch_modulated_pta | fusion.type=QualityGatedFusion |
| 4 | `PatchModPTA-CS-MVote` | patch_modulated_pta | configs/patch_modulated_pta | fusion.type=MajorityVoteFusion |
| 5 | `PatchModPTA-CS-AGate` | patch_modulated_pta | configs/patch_modulated_pta | fusion.type=AgreementGateFusion |
| 6 | `ratio-pta-mv` | patch_modulated_pta | configs/patch_modulated_pta | write_source=pta write_rule=ratio fusion.type=MajorityVoteFusion **[KNOWN UNSTABLE]** |
| 7 | `ratio-clip-mv` | patch_modulated_pta | configs/patch_modulated_pta | write_source=clip write_rule=ratio fusion.type=MajorityVoteFusion **[stability control]** |

## Hypothesis H1

"At least one of arms 2-7 achieves a mean accuracy improvement over PTA baseline
(arm 1), paired by seed, on ≥2 of the 3 target datasets, that exceeds the
pre-registered noise-floor threshold with the pre-registered statistical
evidence — i.e. patch-level evidence brings a genuine, not noise-explainable,
accuracy benefit."

## Noise floor — two-stage lock

**Stage A (design lock, now)**:

```
threshold(dataset) = max(1.0pp, 3 * sigma_PTA(dataset))
```

where `sigma_PTA(dataset)` is the sample std (ddof=1) of PTA-CS accuracy across
its 4 seeds on that dataset. The 1.0pp floor reuses this repo's existing
`DECISION_DELTA_PP` convention (`scripts/analyze_records.py`) for cross-study
consistency. The 3x multiplier is chosen because the previously-observed
baseline-only seed swing (eurosat: 0.34pp over 3 seeds) is the same order of
magnitude as the "marginal" 0.1-0.6pp patch gains being questioned here — a 3x
margin is the minimum needed to call an observed gain signal rather than an
echo of ordinary seed-to-seed noise.

**Stage B (numeric lock)**: to be computed and appended below, timestamped,
once PTA-CS seeds 1-4 exist for all 3 target datasets — **before any patch-arm
accuracy number is inspected**.

<!-- STAGE_B_LOCK: done -->

### Stage B lock — 2026-08-19T05:19:27.284404+00:00

Computed from PTA-CS seeds 1-4, before any patch-arm accuracy was inspected.

| Dataset | PTA-CS accuracies (s1..s4) | sigma_PTA (ddof=1) | threshold(dataset) |
|---|---|---|---|
| dtd | 47.64 / 47.46 / 47.46 / 47.52 | 0.084pp | 1.000pp |
| oxford_flowers | 74.46 / 74.79 / 74.26 / 74.99 | 0.326pp | 1.000pp |
| oxford_pets | 91.01 / 91.14 / 90.95 / 91.20 | 0.115pp | 1.000pp |


## Statistical evidence per (arm, dataset) cell

- `delta_s = acc(arm, seed s) - acc(PTA-CS, seed s)`, s=1..4. Paired: the same
  `--seed` value drives the same deterministic test-loader shuffle order for
  both arms (`runner.py`'s `torch.use_deterministic_algorithms(True)` +
  `seed=args.seed` passed to the loader), so same-seed cells are legitimately
  paired.
- `delta_mean`, `delta_std` (n=4, ddof=1) — reported transparently as the 4 raw
  pairs, not just the summary.
- Exact sign-permutation test: enumerate all 2^4=16 sign assignments of
  `|delta_s|`; one-sided `p_sign` = fraction of the 16 permutation means >=
  observed mean. Finest achievable resolution at n=4 is p=0.0625 (all four
  seeds favor the arm).
- Seed-level percentile bootstrap: 10,000 resamples (with replacement) of the 4
  deltas -> 95% CI on `delta_mean`. Explicitly caveated as coarse at n=4 (only
  4^4=256 distinct resamples exist) — reported for transparency, not as
  primary evidence.
- Sample-level paired bootstrap (secondary): resample the paired per-sample
  correct/incorrect outcomes (joined by `batch_idx`, same seed => same sample
  order for both arms) with replacement, 10,000 resamples -> CI. This captures
  only within-seed sampling noise, not seed-to-seed (init/order) noise — a
  lower bound on true uncertainty, used only as a corroborating/red-flag check.
- `flip_metrics_v2` (`scripts/analyze_records.py`, reused as-is, imported not
  reimplemented) pooled across the arm's 4-seed record sets per dataset:
  `patch_alone.net` and `patch_image.net` (corrections - regressions) must
  both be > 0 for confirmatory (mechanistic, not just outcome-level) evidence.

## Decision rule per (arm, dataset) cell

```
SUPPORT   if delta_mean > threshold(dataset)
             AND p_sign <= 0.0625
             AND seed-level bootstrap CI lower bound > 0
             AND flip_metrics_v2 patch_image.net > 0 (pooled over that
                 arm x dataset's 4 seeds)
REFUTE    if delta_mean <= 0 OR p_sign > 0.5
INCONCLUSIVE otherwise
```

## Overall study verdict

- **SUPPORT H1** only if >= 1 of arms 2-7 reaches per-cell SUPPORT on >= 2 of
  the 3 datasets.
- A SUPPORT verdict driven **solely** by arm 6 (`ratio-pta-mv`), without at
  least one of arms 2-5 or 7 also reaching SUPPORT on at least one dataset,
  does **not** count as sufficient evidence for H1 — its known instability is
  excluded by design from being mistaken for signal. Arm 7 (`ratio-clip-mv`)
  exists specifically to disambiguate this case.
- **REFUTE H1** if no (arm, dataset) cell reaches SUPPORT.
- **INCONCLUSIVE** otherwise, with a stated recommendation (e.g. "needs more
  seeds").

## Non-goals / explicit exclusions

- No oracle/ground-truth write-rule exists in the code or will be built for
  this study; "write rule is fair" (motivating this study's exclusion of an
  oracle arm) refers to `results_dev_writerule.md`'s write-source sweep
  behaving sensibly, not a GT-cheating mechanism.
- The existing `hypothesis`/`decide()` machinery in `scripts/analyze_records.py`
  (SUPPORT if delta_mean>1.0pp AND rho_mean>0.2) tests a **different**,
  within-run temporal hypothesis (does accuracy improve as the prototype bank
  accumulates within one stream) and is **not** reused as this study's verdict
  logic — only the `flip_metrics_v2` computation itself is reused.
