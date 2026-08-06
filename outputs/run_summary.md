# Run Summary — perclass-difficult-classes-validation

Plan: `.omo/plans/perclass-difficult-classes-validation.md`
Date: 2026-08-05/06 · Backbone: ViT-B/16 · Dataset: dtd · 3 seeds (subset) / 2 seeds (full-DTD)

## What ran (Tasks 1-15)

| # | What | Where | Result |
|---|---|---|---|
| T1 | `utils/records.py` writer (env-gated, no-op when RECORD_DIR unset) | `utils/records.py` | tests pass |
| T2 | `build_subset_test_data_loader` closed-set subset loader | `utils/data.py` | tests pass |
| T3 | `runner.py --class-file` flag | `runner.py` | tests pass |
| T4 | `scripts/analyze_records.py` (table/parity/select/hypothesis) | `scripts/analyze_records.py` | tests pass |
| T5 | PatchModPTA per-sample recording + behavior-neutrality | `models/patch_modulated_pta.py` | GPU test PASS |
| T6 | PTA per-sample recording + behavior-neutrality | `models/pta.py` | GPU test PASS |
| T7 | ZeroShot per-sample recording + behavior-neutrality | `models/zeroshot.py` | GPU test PASS |
| T8 | `scripts/slurm_perclass.sh` (8-expr full-DTD batch) | `scripts/slurm_perclass.sh` | dry-run verified |
| T9 | `scripts/slurm_subset.sh` (9-expr subset batch) | `scripts/slurm_subset.sh` | dry-run verified |
| T10 | Full-DTD batch 13933 (+13947) + parity gate | `outputs/records/` | 8/8 COMPLETED, parity PASS |
| T11 | Bottom-5 class selection (bumpy/flecked/lacelike/lined/pitted) | `outputs/subset_classes.txt` | deterministic |
| T12 | Hypothesis metrics + verdict | `outputs/hypothesis_metrics.md` | INCONCLUSIVE |
| T13 | Subset batch 13949 (+13958/13961) | `outputs/records_subset/` | 9/9 COMPLETED |
| T14 | Findings report | `outputs/hypothesis_findings.md` | written |
| T15 | **Full QA sweep (this task)** | `.omo/evidence/task-15-*` | all green |

Tests: 6/6 (3 non-GPU + 3 GPU, see `tests/test_records*.py`). GPU jobs for T15 re-run: **13962** (patchmod), **13963** (pta), **13964** (zeroshot) — all behavior-neutral, bit-identical accuracies with vs without RECORD_DIR.

## Key numbers

Full-DTD per-class accuracies (T10, seed 1/2):

| Method | s1 | s2 |
|---|---|---|
| ZeroShot-CS | 44.39 | 44.39 |
| PatchModPTA-CS | **48.35** | 47.70 |
| PTA-CS | 46.99 | 46.39 |
| PatchModPTA-CS-alpha10-s1 (ablation) | 45.45 | — |
| PatchModPTA-CS-multigate-s1 (ablation) | 48.64 | — |

Ablations vs baseline s1 48.35: alpha10 **45.45** (−2.90 pp), multigate **48.64** (+0.29 pp).

Subset (5-class closed-set, seeds 1-3):

| Method | s1 | s2 | s3 | mean |
|---|---|---|---|---|
| PatchModPTA-CS-sub | 52.22 | 50.00 | 48.89 | **50.37** |
| PTA-CS-sub | 50.56 | 51.67 | 48.89 | **50.37** |
| ZeroShot-CS-sub | 48.33 | 48.33 | 48.33 | **48.33** |

Selected difficult classes: **bumpy, flecked, lacelike, lined, pitted**.

**Parity gate (T15 re-run): PASS** — PatchModPTA-CS-s1 acc=48.35 == ref 48.35; PTA-CS-s1 acc=46.99 == ref 46.99 (exit 0).

**Hypothesis verdict: INCONCLUSIVE (delta_mean=5.02 pp, rho_mean=-0.44)** — delta leg positive (+5.44/+4.61 per seed, ~45.5%→50.5% across stream) but rho leg negative (−0.51/−0.37): classes with more cluster growth end at lower last-half accuracy. SUPPORT requires delta>+1.0 AND rho>0.2 — rho leg fails. Two config bugs documented: `multi_gate` read top-level (recorded "single"), `proto_alpha_max` 0.2 default vs nested 1.0.

## Where results live

- Per-class tables: `outputs/perclass_tables.md`
- Selection: `outputs/subset_classes.txt` (5 classes)
- Gap analysis: `outputs/gap_analysis.md`
- Hypothesis metrics: `outputs/hypothesis_metrics.md`
- Findings report: `outputs/hypothesis_findings.md`
- Raw accs: `outputs/result_perclass.txt` (8 lines), `outputs/result_subset.txt` (9 lines)
- Per-sample records: `outputs/records/<LABEL>/records.jsonl` (1692 samples/run), `outputs/records_subset/<LABEL>/records.jsonl` (180 samples/run)
- Slurm logs: `logs/`

## Evidence

Every task-1..14 evidence file + this sweep's task-15 files are indexed in
`.omo/evidence/index.md`. Completeness check: no MISSING entries.
Stale leftover `task-9-models-init.txt` (previous plan, 2026-07-04) explicitly excluded, not deleted.
