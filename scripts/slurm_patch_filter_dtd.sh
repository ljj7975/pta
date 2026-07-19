#!/bin/bash
#SBATCH --job-name=pta_patch_filter_dtd
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --output=/share_98/projects/brandon/repos/pta/dev_logs/pta_patch_filter_dtd_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/dev_logs/pta_patch_filter_dtd_%x-%A_%a.err

# ============================================================================
# Patch Filter Mode Quick Test: 5 filter modes × dtd only = 5 jobs
#
# Experiment: patch_modulated_pta / configs/patch_modulated_pta / clip_surgery
#
# HOW TO USE
# ----------
# 1. Comment/uncomment individual exp() lines below.
#
# 2. Update --array to match N_EXP - 1:
#      --array=0-4  (5 modes, 1 dataset)
#
# 3. Submit:
#      sbatch scripts/slurm_patch_filter_dtd.sh
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

# ── Five filter modes on DTD ─────────────────────────────────────────────────
exp "PatchModPTA-CLIPSurgery-FilterNone"            ""
exp "PatchModPTA-CLIPSurgery-FilterCosineLabels"    "patch_level.patch_filter_mode=cosine_with_labels"
exp "PatchModPTA-CLIPSurgery-FilterCosineNoLabels"  "patch_level.patch_filter_mode=cosine_no_labels"
exp "PatchModPTA-CLIPSurgery-FilterSurgeryLabels"   "patch_level.patch_filter_mode=surgery_with_labels"
exp "PatchModPTA-CLIPSurgery-FilterSurgeryNoLabels" "patch_level.patch_filter_mode=surgery_no_labels"

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
EXP_LABELS=("${_EXP_LABELS[@]}")
OVERRIDES=("${_OVERRIDES[@]}")

N_EXP=${#EXP_LABELS[@]}

if (( N_EXP == 0 )); then
    echo "ERROR: No experiments enabled. Uncomment at least one exp() line above."
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= N_EXP )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_EXP ($N_EXP). Nothing to do."
    exit 0
fi

EXP_LABEL=${EXP_LABELS[$SLURM_ARRAY_TASK_ID]}
OVERRIDE=${OVERRIDES[$SLURM_ARRAY_TASK_ID]}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $((N_EXP - 1))"
echo "  Experiment : $EXP_LABEL"
echo "  Dataset    : dtd"
echo "  Override   : ${OVERRIDE:-<none>}"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="${EXP_LABEL}"

CMD=(python -u runner.py
    --method patch_modulated_pta
    --config configs/patch_modulated_pta
    --clip-model clip_surgery
    --datasets dtd
    --backbone ViT-B/16)

if [[ -n "$OVERRIDE" ]]; then
    CMD+=(--override "$OVERRIDE")
fi

"${CMD[@]}"
