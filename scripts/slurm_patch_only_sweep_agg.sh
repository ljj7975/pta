#!/bin/bash
#SBATCH --job-name=pta_patch_only
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-24
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_patch_only_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_patch_only_%x-%A_%a.err

# ============================================================================
# Patch-Only Classification Sweep
#
# HOW TO USE
# ----------
# 1. Set CLIP_MODEL, EXTRA_OVERRIDES below to configure the run.
# 2. Update --array to match N_EXP x N_DS.
# 3. sbatch scripts/slurm_patch_only_sweep.sh
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
# Configuration — change these to switch CLIP model / filtering
# ---------------------------------------------------------------------------
CLIP_MODEL=clip_surgery
EXTRA_OVERRIDES="patch_level.patch_filter_mode=surgery_with_labels"

# Examples:
# CLIP_MODEL=clip_surgery
# CLIP_MODEL=clip
# CLIP_MODEL=detail-clip

# EXTRA_OVERRIDES=""
# EXTRA_OVERRIDES="patch_level.patch_filter_mode=surgery_no_labels patch_level.patch_filter_threshold=0.5"
# EXTRA_OVERRIDES="patch_level.patch_filter_mode=surgery_with_labels patch_level.patch_filter_threshold=0.3"

# ---------------------------------------------------------------------------
# Datasets
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

exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "with_labels-Max"          "fusion.mode=patch_only patch_level.aggregation=max $EXTRA_OVERRIDES"
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "with_labels-Mean"         "fusion.mode=patch_only patch_level.aggregation=mean $EXTRA_OVERRIDES"
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "with_labels-WeightedMean" "fusion.mode=patch_only patch_level.aggregation=weighted_mean $EXTRA_OVERRIDES"
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "with_labels-TopM3"        "fusion.mode=patch_only patch_level.aggregation=top_m_mean patch_level.soft_nn_top_m=3 $EXTRA_OVERRIDES"
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "with_labels-TopM5"        "fusion.mode=patch_only patch_level.aggregation=top_m_mean patch_level.soft_nn_top_m=5 $EXTRA_OVERRIDES"

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
export RESULT_FILE="outputs/patch_only_results.txt"

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
