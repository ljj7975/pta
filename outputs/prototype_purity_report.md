# Patch-Level Prototype Bank — Write-Time Purity Diagnostic

Replays the patch bank's `multi_gate` write mask (`models/patch_modulated_pta.py` run(), `conf_source="text"`, `conf_threshold=0.3`) from stored `logits.clip` + `target` in `outputs/records_patch_benefit/` (canonical arm `PatchModPTA-CS` — clip_logits is state-independent zero-shot CLIP, identical across all 6 patch arms for a given dataset/seed). No new GPU runs.

**Scope note**: this gate is *not* touched by the `write_source`/`write_rule` sweep in `results_dev_writerule.md` — that sweep only governs the separate image-level EMA prototype. The patch bank's write purity has not been previously measured.

## Per-dataset write-purity summary (pooled over 4 seeds)

| Dataset | Images | Writes/image | Multi-write rate | Zero-write rate | Overall purity | Primary-write purity | Collateral-write purity | Collateral share |
|---|---|---|---|---|---|---|---|---|
| dtd | 6768 | 0.60 | 2.7% | 43.0% | 62.6% | 63.4% | 46.7% | 4.5% |
| oxford_flowers | 9852 | 0.97 | 7.7% | 10.5% | 74.1% | 77.2% | 38.1% | 7.9% |
| oxford_pets | 14676 | 1.08 | 8.9% | 1.2% | 85.7% | 89.9% | 40.1% | 8.4% |

## Per-class purity distribution

| Dataset | Classes ever written | Median per-class purity |
|---|---|---|
| dtd | 39 | 62.5% |
| oxford_flowers | 93 | 83.3% |
| oxford_pets | 37 | 90.9% |

## Cross-reference: does purity track the observed accuracy pattern?

`outputs/patch_benefit_report.md` / `outputs/experiment_summary.md` found: dtd mildly **positive** for MVote/AGate (+0.6-0.99pp), oxford_flowers/oxford_pets consistently **negative** across every arm (-0.1 to -2.3pp).

| Dataset | Overall write purity | Collateral-write purity | Known MVote/AGate delta |
|---|---|---|---|
| dtd | 62.6% | 46.7% | MVote +0.65pp / AGate +0.60pp |
| oxford_flowers | 74.1% | 38.1% | MVote -0.13pp / AGate -0.24pp |
| oxford_pets | 85.7% | 40.1% | MVote -0.39pp / AGate -0.31pp |

