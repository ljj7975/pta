# Session Summary: Patch-Level Prototype Experiments

**Date**: 2026-08-18
**Goal**: Run and analyze patch-level prototype experiments to show PTA limitations and how patch-level prototypes address them.

---

## What We Did

### 1. Experiment Runner Scripts

Created `scripts/run_experiments.sh` to run 5 methods × 5 datasets × 5 seeds:

| Method | Fusion Type |
|--------|------------|
| PTA | (baseline, no patch) |
| PatchModPTA | ProtoAlphaFusion (default) |
| PatchModPTA-QGated | QualityGatedFusion |
| PatchModPTA-MVote | MajorityVoteFusion |
| PatchModPTA-AGate | AgreementGateFusion |

**Bug Fixed**: Original script had wrong datasets (`caltech101 dtd eurosat fgvc food101`). Corrected to `dtd eurosat fgvc oxford_flowers oxford_pets`.

### 2. Missing Datasets Script

Created `scripts/run_missing_datasets.sh` for 50 missing runs:
- oxford_flowers (25 runs)
- oxford_pets (25 runs)

Usage: `bash scripts/run_missing_datasets.sh --run --launcher sbatch --concurrency 5`

### 3. Analysis Scripts

Created 6 analysis scripts:

| Script | Purpose |
|--------|---------|
| `analyze_perclass.py` | Per-class accuracy breakdown |
| `analyze_ties.py` | Tie-breaking analysis (clip vs image_proto disagreement) |
| `analyze_spatial.py` | Spatial sensitivity (proxy using existing records) |
| `analyze_voting_ablation.py` | Voting mechanism ablation study |
| `analyze_component_ablation.py` | Component contribution analysis |
| `analyze_hyperparam_sensitivity.py` | Hyperparameter sensitivity (alpha, T) |

### 4. Unified Plotting Pipeline

Created `scripts/plot_figures.py` for paper figures:
- `perclass.pdf` — Per-class accuracy heatmap
- `ties.pdf` — Tie-breaking analysis
- `spatial.pdf` — Spatial sensitivity
- `voting.pdf` — Voting ablation
- `component.pdf` — Component ablation

### 5. Config Fix

Created `configs/patch_modulated_pta/food101.yaml` (was missing, caused experiment failures).

### 6. Test Fixtures

Created `tests/fixtures/records_sample/` with 6 fixture directories for offline script verification.

---

## Key Findings

### Overall Accuracy (4 datasets: dtd, eurosat, fgvc, caltech101)

| Method | Mean Accuracy | vs PTA |
|--------|---------------|--------|
| PTA (baseline) | 57.29% | — |
| PatchModPTA (ProtoAlpha) | 56.82% | −0.47% |
| PatchModPTA-QGated | 56.82% | −0.47% |
| PatchModPTA-MVote | 57.56% | +0.27% |
| PatchModPTA-AGate | 57.40% | +0.11% |

**Note**: These are from wrong datasets (caltech101 instead of oxford_flowers/oxford_pets). Need to regenerate with correct data.

### ProtoAlphaFusion Collapse on FGVC

| Method | FGVC Accuracy | vs PTA |
|--------|---------------|--------|
| PTA | 25.65% | — |
| PatchModPTA (ProtoAlpha) | 22.90% | **−2.75%** |
| PatchModPTA-QGated | 22.93% | **−2.72%** |
| PatchModPTA-MVote | 25.61% | −0.04% |
| PatchModPTA-AGate | 25.55% | −0.10% |

**Insight**: Always-on patch fusion (ProtoAlpha/QGated) COLLAPSES on fine-grained datasets. Discrete gating (MVote/AGate) prevents collapse.

### Tie-Breaking Analysis (dtd, seed 1)

- **31.1% of samples** have ties (clip.argmax != image_proto.argmax)
- **75.3% of ties are wrong** (image_proto disagrees with CLIP and is incorrect)
- This is the **prototype drift** problem

### Patch Agreement as Quality Signal

| Patch agrees with | Source is correct | Δ vs baseline |
|-----------------|-------------------|---------------|
| image_proto | 33.9% | +9.3% |
| clip | 29.9% | +13.2% |
| other (independent) | 5.8% | — |

**Insight**: Patch agreement boosts confidence in both sources. Voting can leverage this signal.

---

## Files Created This Session

### Scripts
- `scripts/run_experiments.sh` — Main experiment runner
- `scripts/run_food101.sh` — Food101补 runs
- `scripts/run_missing_datasets.sh` — Missing datasets (oxford_flowers, oxford_pets)
- `scripts/analyze_perclass.py` — Per-class analysis
- `scripts/analyze_ties.py` — Tie-breaking analysis
- `scripts/analyze_spatial.py` — Spatial sensitivity
- `scripts/analyze_voting_ablation.py` — Voting ablation
- `scripts/analyze_component_ablation.py` — Component ablation
- `scripts/analyze_hyperparam_sensitivity.py` — Hyperparameter sensitivity
- `scripts/plot_figures.py` — Unified plotting pipeline

### Config
- `configs/patch_modulated_pta/food101.yaml`

### Tests
- `tests/fixtures/records_sample/` — 6 fixture directories

### Outputs
- `outputs/experiment_summary.md` — Analysis report (needs regeneration)

---

## Next Steps

1. Run `bash scripts/run_missing_datasets.sh --run --launcher sbatch --concurrency 5`
2. Re-run analysis scripts on correct 5 datasets
3. Regenerate `outputs/experiment_summary.md`
4. Commit all files
