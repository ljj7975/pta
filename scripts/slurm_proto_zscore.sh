#!/bin/bash
#SBATCH --job-name=pta_proto_zscore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-9
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_proto_zscore_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_proto_zscore_%x-%A_%a.err

# ============================================================================
# Prototype Score Normalization — Final Clean Sweep
#
# Four active methods, all with appearance_min_weight=0.5:
#   zscore                  — raw signed z-score, appearance-weighted sum.
#   zscore_cdf              — Φ(z), appearance-weighted sum.
#   zscore-MG               — same as zscore but with multi_gate=true
#   zscore_cdf-MG           — same as CDF but with multi_gate=true
#
# Multi-gate mode updates ALL classes whose confidence exceeds the threshold
# (no margin check), so several prototypes may update per image. Default
# single-gate only updates the single top-1 class if confidence-margin passes.
#
# Setting  diagnostic_pta + fusion.mode=patch_only
#          final_logits ARE the patch scores, so the comparison is a pure
#          argmax over aggregated prototype scores — scale-invariant, no tau
#          to tune. This is the cleanest read on whether normalisation helps.
#
# HOW TO USE
# ----------
# 1. Set CLIP_MODEL / EXTRA_OVERRIDES below.
# 2. Update --array to match N_EXP x N_DS (currently 6 x 5 = 30 → 0-29).
# 3. sbatch scripts/slurm_proto_zscore.sh
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
# Configuration
# ---------------------------------------------------------------------------
CLIP_MODEL=clip_surgery
EXTRA_OVERRIDES=""

# Examples:
# EXTRA_OVERRIDES="patch_level.patch_filter_mode=surgery_with_labels"
# EXTRA_OVERRIDES="patch_level.proto_stats_log_every=500"

# ---------------------------------------------------------------------------
# Datasets — the 5-dataset cross-domain core
# ---------------------------------------------------------------------------
DATASETS=(dtd eurosat fgvc oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}

# ---------------------------------------------------------------------------
# Experiment registry
# exp METHOD CONFIG_DIR CLIP_MODEL CLIP_CHECKPOINT LABEL OVERRIDE
# ---------------------------------------------------------------------------
_METHODS=()
_CONFIG_DIRS=()
_CLIP_MODELS=()
_CLIP_CHECKPOINTS=()
_EXP_LABELS=()
_OVERRIDES=()

exp() {
    _METHODS+=("$1")
    _CONFIG_DIRS+=("$2")
    _CLIP_MODELS+=("$3")
    _CLIP_CHECKPOINTS+=("$4")
    _EXP_LABELS+=("$5")
    _OVERRIDES+=("$6")
}

exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "PatchOnly-CS" "$EXTRA_OVERRIDES"

exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "PMP-CS" "$EXTRA_OVERRIDES"

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
METHODS=("${_METHODS[@]}")
CONFIG_DIRS=("${_CONFIG_DIRS[@]}")
CLIP_MODELS=("${_CLIP_MODELS[@]}")
CLIP_CHECKPOINTS=("${_CLIP_CHECKPOINTS[@]}")
EXP_LABELS=("${_EXP_LABELS[@]}")
OVERRIDES=("${_OVERRIDES[@]}")

N_EXP=${#METHODS[@]}
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

METHOD=${METHODS[$exp_idx]}
CONFIG=${CONFIG_DIRS[$exp_idx]}
CLIP_CKPT=${CLIP_CHECKPOINTS[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL=${EXP_LABELS[$exp_idx]}

echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL ($exp_idx / $N_EXP)"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  CLIP model : $CLIP_MODEL"
echo "  Dataset    : $DATASET  (ds_idx=$ds_idx)"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="${EXP_LABEL}"
export RESULT_FILE="outputs/result.txt"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --clip-model "$CLIP_MODEL"
    --datasets "$DATASET"
    --backbone ViT-B/16)

if [[ -n "$CLIP_CKPT" ]]; then
    CMD+=(--clip-checkpoint "$CLIP_CKPT")
fi

OVERRIDE=${_OVERRIDES[$exp_idx]}
if [[ -n "$OVERRIDE" ]]; then CMD+=(--override $OVERRIDE); fi

"${CMD[@]}"
