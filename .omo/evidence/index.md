# Evidence Index — perclass-difficult-classes-validation

Plan: `.omo/plans/perclass-difficult-classes-validation.md`
Generated: 2026-08-06 (Task 15 QA sweep)

Every file below is a captured run/assertion proving a task deliverable.
Files are committed via `git add -f` (`.omo/` is gitignored).

## Task 1 — utils/records.py writer
- `task-1-help.txt` — runner.py `--help` shows `--method`/`--config`/`--datasets`/`--backbone` flags.
- `task-1-import-check.txt` — `tests/capture_fidelity.py` module-level imports resolve, no ImportError.
- `task-1-noop.txt` — writer is a strict no-op (zero files) when `RECORD_DIR` is unset/empty.
- `task-1-records-roundtrip.txt` — records.jsonl + summary.json round-trip, header + schema verified.

## Task 2 — build_subset_test_data_loader
- `task-2-base-import.txt` — `models.image_level.base.BaseImageLevel` instantiates.
- `task-2-subset-loader.txt` — 2-class closed-set subset loader remaps labels to {0,1}, len==72 on dtd.
- `task-2-unknown-class.txt` — unknown class name raises real `ValueError`.

## Task 3 — runner.py `--class-file`
- `task-3-help.txt` — `--class-file` flag present in runner.py `--help`.
- `task-3-missing-file.txt` — missing class file raises `FileNotFoundError` before any GPU work, exit 1.
- `task-3-pta-image-import.txt` — `PTAImageLevel` import/instantiate/build/init_state (2026-07-04).
- `task-3-pta-image-update.txt` — `PTAImageLevel.update_prototypes` output shapes (2026-07-04).
- `task-3-sample-classfile.txt` — 2-class dtd sample class file (bumpy/flecked).

## Task 4 — scripts/analyze_records.py
- `task-4-image-init.txt` — models/image_level symbols exported (2026-07-04).
- `task-4-parity-and-hypothesis.txt` — parity + hypothesis subcommands tested on synthetic records.
- `task-4-subcommands.txt` — table/parity/select/hypothesis CLI entry points verified.

## Task 5 — PatchModPTA per-sample recording
- `task-5-behavior-neutral.txt` — GPU behavior-neutrality: bit-identical final acc with vs without RECORD_DIR (66.66666666666667 == 66.66666666666667).
- `task-5-normalize-check.txt` — `_safe_normalize`/`_alpha_from_evidence` correctness (2026-07-04).
- `task-5-schema.txt` — patchmod JSONL schema dump (FULL_HEADER/FULL_RECORD0).
- `task-5-utils-import.txt` — models/patch_level/base symbols import (2026-07-04).

## Task 6 — PTA per-sample recording
- `task-6-behavior-neutral.txt` — GPU behavior-neutrality for PTA (bit-identical acc, exit 0).
- `task-6-gaussian-import.txt` — `GaussianPatchLevel` import/init_state shapes (2026-07-04).
- `task-6-schema.txt` — PTA JSONL schema dump.

## Task 7 — ZeroShot per-sample recording
- `task-7-behavior-neutral.txt` — GPU behavior-neutrality for ZeroShot (bit-identical acc, exit 0).
- `task-7-patch-init.txt` — models/patch_level/__init__ exports (2026-07-04).
- `task-7-schema.txt` — ZeroShot JSONL schema dump.
- `task-7-smoke-exp4.txt` — Exp4FullFusion caltech101 smoke run, MAX_BATCHES=10, exit 0 (2026-07-02).
- `task-7-smoke-exp5.txt` — Exp5TunableFusion caltech101 smoke run, MAX_BATCHES=10, exit 0 (2026-07-02).

## Task 8 — scripts/slurm_perclass.sh
- `task-8-fusion-formula.txt` — WeightedFusion / QualityGatedFusion formula tests PASS (2026-07-04).
- `task-8-fusion-import.txt` — models/fusion symbols import OK (2026-07-04).
- `task-8-script.txt` — slurm_perclass.sh bash -n + stub-python dry-run across array 0-7 (CMD + env correct).

## Task 9 — scripts/slurm_subset.sh
- `task-9-missing-classfile.txt` — missing/empty CLASS_FILE guard fires before python, exit 1.
- `task-9-models-init.txt` — **STALE LEFTOVER** from a previous plan (dated 2026-07-04); NOT part of this plan's task-9 deliverable; excluded from index.
- `task-9-subset-script.txt` — slurm_subset.sh bash -n + dry-run across 9 exps (3 methods × 3 seeds).

## Task 10 — full-DTD per-class batch + parity gate
- `task-10-batch.txt` — 8/8 full-DTD runs COMPLETED 0:0, result_perclass.txt = 8 lines, records 1692 samples each.
- `task-10-failure-triage.txt` — node2 CUDA driver init failure triage (job 13933_2), re-submission, PTA RESULT_FILE fix.
- `task-10-parity-gate.txt` — **PARITY GATE PASS**: PatchModPTA-CS-s1=48.35==ref 48.35, PTA-CS-s1=46.99==ref 46.99.

## Task 11 — class selection
- `task-11-empty-records.txt` — `select` on empty records → exit 1, no partial output.
- `task-11-selection.txt` — bottom-5 selection = bumpy/flecked/lacelike/lined/pitted (pre-registered rule, re-run deterministic).

## Task 12 — hypothesis metrics + verdict
- `task-12-hypothesis.txt` — hypothesis run on parity-validated records, per-seed deltas/rhos.
- `task-12-verdict.txt` — **DECISION: INCONCLUSIVE (delta_mean=5.02 pp, rho_mean=-0.44)**; hand cross-check matches.

## Task 13 — subset batch (3 methods × 3 seeds)
- `task-13-ablations.txt` — ablation records verified present (alpha10 45.45, multigate 48.64 vs baseline 48.35).
- `task-13-failure-triage.txt` — node2 failures on 13949_2/3/4 + 13958_4, re-submission churn.
- `task-13-subset-batch.txt` — 9/9 subset runs COMPLETED 0:0, means 50.37/50.37/48.33.

## Task 14 — hypothesis findings report
- `task-14-report.txt` — `outputs/hypothesis_findings.md` QA: REPORT_EXISTS, DECISION×1, bugs, ablations, parity values.
- `task-14-scope.txt` — scope fidelity: DTD-only, confound/caveat/limited sections present.

## Task 15 — full QA sweep (this task)
- `task-15-sweep.txt` — non-GPU tests (3/3 PASS), GPU tests (3/3 PASS, behavior-neutral), parity gate PASS, outputs present, stray-records=0.
- `task-15-evidence.txt` — evidence completeness check: every indexed task-1..14 file exists; no MISSING lines.

## Not indexed (deliberately)
- `task-9-models-init.txt` — stale leftover from a previous plan (dated 2026-07-04), excluded from this index, NOT deleted.
- `final-fusion-integration.txt` — unrelated legacy artifact (2026-07-04), not a task-1..15 evidence file.
