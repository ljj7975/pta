#!/bin/bash
#SBATCH --job-name=pta_patch_filter_full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-24
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_patch_filter_full_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_patch_filter_full_%x-%A_%a.err

# ============================================================================
# Patch Filter Mode Full Benchmark: 5 filter modes × 5 CD datasets = 25 jobs
#
# Experiment: patch_modulated_pta / configs/patch_modulated_pta / clip_surgery
# (same backbone as "PatchModPTA-CLIPSurgery" in slurm_dev.sh)
#
# HOW TO USE
# ----------
# 1. Comment/uncomment individual exp() lines below.
#
# 2. Update --array to match N_EXP * N_DS - 1:
#      N_EXP = count of uncommented exp() lines   (currently 5)
#      N_DS  = ${#DATASETS[@]}                    (currently 5)
#      --array=0-24  (5*5-1)
#
# 3. Submit:
#      sbatch scripts/slurm_patch_filter_full.sh
#
# Task mapping:
#   exp_idx = SLURM_ARRAY_TASK_ID / N_DS
#   ds_idx  = SLURM_ARRAY_TASK_ID % N_DS
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Datasets (shared across all experiments)
# ---------------------------------------------------------------------------
DATASETS=(dtd eurosat fgvc oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}

# ---------------------------------------------------------------------------
# Experiment registry
#
# exp LABEL OVERRIDE
#
# OVERRIDE is passed as --override <value>; use "" for no override (none mode).
# ---------------------------------------------------------------------------
_EXP_LABELS=()
_OVERRIDES=()

exp() {
    _EXP_LABELS+=("$1")
    _OVERRIDES+=("$2")
}

# ── Patch filter modes: all five, same backbone as PatchModPTA-CLIPSurgery ──
exp "PatchModPTA-CLIPSurgery-FilterNone"           ""
exp "PatchModPTA-CLIPSurgery-FilterCosineLabels"   "patch_level.patch_filter_mode=cosine_with_labels"
exp "PatchModPTA-CLIPSurgery-FilterCosineNoLabels" "patch_level.patch_filter_mode=cosine_no_labels"
exp "PatchModPTA-CLIPSurgery-FilterSurgeryLabels"  "patch_level.patch_filter_mode=surgery_with_labels"
exp "PatchModPTA-CLIPSurgery-FilterSurgeryNoLabels" "patch_level.patch_filter_mode=surgery_no_labels"

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
EXP_LABELS=("${_EXP_LABELS[@]}")
OVERRIDES=("${_OVERRIDES[@]}")

N_EXP=${#EXP_LABELS[@]}
N_TOTAL=$((N_EXP * N_DS))

if (( N_EXP == 0 )); then
    echo "ERROR: No experiments enabled. Uncomment at least one exp() line above."
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

exp_idx=$((SLURM_ARRAY_TASK_ID / N_DS))
ds_idx=$((SLURM_ARRAY_TASK_ID % N_DS))

DATASET=${DATASETS[$ds_idx]}
EXP_LABEL=${EXP_LABELS[$exp_idx]}
OVERRIDE=${OVERRIDES[$exp_idx]}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $((N_TOTAL - 1))"
echo "  Experiment : $EXP_LABEL (exp_idx=$exp_idx)"
echo "  Dataset    : $DATASET  (ds_idx=$ds_idx)"
echo "  Override   : ${OVERRIDE:-<none>}"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="${EXP_LABEL}"

CMD=(python -u runner.py
    --method patch_modulated_pta
    --config configs/patch_modulated_pta
    --clip-model clip_surgery
    --datasets "$DATASET"
    --backbone ViT-B/16)

if [[ -n "$OVERRIDE" ]]; then
    CMD+=(--override $OVERRIDE)
fi

"${CMD[@]}"
