# Learnings: Patch-Prototype Revalidation Follow-Up

Created 2026-08-06 (Task 10, plan `patch-proto-revalidation`, STOP branch). This file records what the follow-up revalidation established. Companion docs: `outputs/hypothesis_findings.md` Part B (narrative + verdict), `outputs/hypothesis_metrics.md` Part B (tables). Every claim traces to `.omo/evidence/` (wave2-gate.md, config-read-audit.md, gate-noise-sizing.md, task-5-sweep-attest.txt, task-6-gate-check.txt, task-6-cluster-count.txt).

## 1. multi_gate fix effect

- Before the fix, `multi_gate` was read top-level only (`self.cfg.get("multi_gate", False)`) and silently defaulted to False; the intended value lived nested (`patch_level.multi_gate: true` in `configs/base.yaml:16`). Follow-up Task 1 made the read nested-first (`models/patch_modulated_pta.py:168`), so the flag is now EFFECTIVE.
- Evidence of the fix in records: all three PatchModPTA-fixed runs carry `gate_mode=="multi"` on 180/180 records (task-6-gate-check.txt).
- What the fix did NOT change: on the fixed path the patch term still under-performed PTA at every tau (best 50.00 vs 51.67). The prior plan's multigate ablation (+0.29 pp on full-DTD) was measured through the buggy path and is not directly comparable.

## 2. Alpha-drop effect

- The `proto_alpha` machinery (top-level `proto_alpha_max` read, `n_half` evidence schedule, `_alpha_from_evidence` call) was removed in Task 1; the fusion default `proto_alpha=1.0` is now a constant. The patch term is `tau_patch_proto * patch_proto` wherever clusters exist; the `proto_stats_min_count: 5` guard is preserved.
- Effect: the uncapped patch term is strictly monotonic worse with tau (subset mean 50.00 / 47.22 / 43.89 at tau 2.5 / 5.0 / 10.0). The prior plan's "raise the cap" lever (alpha10 ablation, -2.90 pp) is moot: removing the cap entirely did not help either. The mechanism itself, not its scale, is the problem.

## 3. Determinism fix

- `build_data_loader` gained a `seed` parameter: seeded runs use `num_workers=0` plus a seeded generator, so the same seed replays the exact same stream (byte-identical sha256 in the test) and different seeds give distinct streams.
- Consequence for the gate: the single-seed Wave 2 numbers are reproducible. The Task 4 noise floor (±2 pp at n=180) is now genuine stream variation, not loader nondeterminism.
- Cross-code reproducibility anchor: ZeroShot-fixed subset mean 48.33 equals the old-code ZeroShot subset mean 48.33 exactly (ZeroShot has no adaptation state, so it is order-independent).

## 4. Dotted-override discipline

- Sweep overrides MUST be dotted: `--override fusion.tau_patch_proto=<v>`. A top-level `tau_patch_proto=<v>` override is silently ignored (the nested-first read at patch_modulated_pta.py:132-134 never sees it).
- Proven by headers: `fusion.tau_patch_proto` = 2.5 / 5.0 / 10.0 for the three PMP runs, top-level key ABSENT (task-5-sweep-attest.txt §2).
- Red-flag check: strictly monotonic sweep means the tau read was LIVE; a flat sweep would have flagged the override as inert.

## 5. Fresh-records-root practice

- Never write to `outputs/records/` or `outputs/records_subset/` (old-code, append-mode, historical). All follow-up runs went to a fresh root `outputs/records_v2/` (`subset/` for the gate, `full/` reserved for Wave 3).
- Pre-run absence asserted on the head node and re-asserted inside the Slurm wrapper; post-run line counts verified (181 lines = 1 header + 180 records each).

## 6. Gate FAILURE result (the headline of the follow-up)

- Pre-registered loose gate (gate-noise-sizing.md §5): proceed iff `max over tau of (PatchModPTA-fixed-sub mean) >= PTA-fixed-sub mean`; tie counts as proceed. Wave 2 (seed 1, fixed code): max_tau = 50.00 (tau 2.5) < PTA-fixed-sub = 51.67 → **FAIL** (difference -1.67 pp, not a tie). STOP branch: Wave 3 (Tasks 7/8/9) NOT run.
- Per-class at best tau: 3 worsened (bumpy -8.3, flecked -2.8, lacelike -5.6), 2 improved (lined +2.8, pitted +5.6). lacelike collapses monotonically with tau (-5.6 / -22.2 / -36.1 pp).
- Cluster context: every PatchModPTA class forms clusters on ≥32 of 36 samples (flecked most starved at 4/36, and hardest at 0.0-2.8% for every method). The gate miss is NOT attributable to missing clusters.
- The point-5 bar (`rho_mean > 0 AND delta_mean > 0`) was NOT tested: it needs full-DTD per-seed runs, which only Wave 3 would have produced. Do not substitute subset means for it.
- Flip diagnostics (flips-v2 on the subset records): text-only 48.33 for every run (determinism check); the image-EMA component delivers the entire gain (+6 net); patch-alone is a net detractor (-21/-22); fused patch contribution worsens with tau (-4/-8/-14). lacelike is where the patch term does its worst damage.

## 7. What did NOT change / guardrails honored

- No code edits in Task 10; fusion.py docstrings untouched; `experiments_v1-3.md` and all other docs untouched.
- Old INCONCLUSIVE verdict (delta_mean=+5.02 pp, rho_mean=-0.44, old code, full-DTD) preserved in both findings and metrics docs with explicit "old code" labels; the Part B gate failure supersedes its verdict context, not the record itself.
- Commit discipline: Task 10 commits exactly 3 docs (`outputs/hypothesis_findings.md`, `outputs/hypothesis_metrics.md`, `outputs/learnings.md`); records, evidence dirs, and scripts are not staged. All commits in this plan are `[auto]`-prefixed (686cf0b, 3fac709, b9c0ccb, then the Task 10 docs commit).
