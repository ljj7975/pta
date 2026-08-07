#!/bin/bash
#SBATCH --job-name=pta_perclass
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-7
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_perclass_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_perclass_%x-%A_%a.err

# ============================================================================
# Per-class recording benchmark (8 full DTD runs) for the difficult-classes
# validation. Records per-batch per-class stats into outputs/records/<LABEL>/
# via utils.records (RECORD_DIR hook, see T1) and appends the final accuracy
# to outputs/result_perclass.txt (RESULT_FILE hook).
#
# HOW TO USE
# ----------
# 1. Comment/uncomment INDIVIDUAL experiment lines below.
#    Each line is one experiment — independent of the others.
#
# 2. Update --array to match the number of active experiments × datasets:
#
#      N_EXP = count of uncommented exp() lines
#      N_DS  = ${#DATASETS[@]}   (currently 1: dtd)
#      --array=0-$(( N_EXP * N_DS - 1 ))
#
#    Example: all 8 active → --array=0-7
#
# 3. Labels are configurable per-line for re-running with different names.
#    Each label gets a unique RECORD_DIR (outputs/records/$LABEL).
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
mkdir -p outputs/records
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Datasets (shared across all experiments)
# ---------------------------------------------------------------------------
DATASETS=(dtd)
N_DS=${#DATASETS[@]}

# ---------------------------------------------------------------------------
# Experiment registry
#
# exp METHOD CONFIG_DIR CLIP_MODEL CLIP_CHECKPOINT LABEL SEED [OVERRIDE]
#
# Comment/uncomment individual lines to enable/disable experiments.
# OVERRIDE is a single top-level KEY=VALUE (e.g. multi_gate=1).
# NOTE: multi_gate must be overridden numerically (multi_gate=1), NOT as
# "true" — bool("false") == True in Python, so a string override is a footgun.
# ---------------------------------------------------------------------------
_METHODS=()
_CONFIG_DIRS=()
_CLIP_MODELS=()
_CLIP_CHECKPOINTS=()
_EXP_LABELS=()
_EXP_SEEDS=()
_OVERRIDES=()

exp() {
    _METHODS+=("$1")
    _CONFIG_DIRS+=("$2")
    _CLIP_MODELS+=("$3")
    _CLIP_CHECKPOINTS+=("$4")
    _EXP_LABELS+=("$5")
    _EXP_SEEDS+=("$6")
    _OVERRIDES+=("${7:-}")
}

# ── Main experiments (seeds 1 & 2, full DTD) ─────────────────────────
exp patch_modulated_pta configs/patch_modulated_pta clip_surgery "" "PatchModPTA-CS-s1" 1 ""
exp patch_modulated_pta configs/patch_modulated_pta clip_surgery "" "PatchModPTA-CS-s2" 2 ""
exp pta                 configs/PTA                   clip_surgery "" "PTA-CS-s1"        1 ""
exp pta                 configs/PTA                   clip_surgery "" "PTA-CS-s2"        2 ""
exp zeroshot            configs/PTA                   clip_surgery "" "ZeroShot-CS-s1"   1 ""
exp zeroshot            configs/PTA                   clip_surgery "" "ZeroShot-CS-s2"   2 ""

# ── Ablations (seed 1 only) ──────────────────────────────────────────
# multi_gate is now read nested-first (patch_level.multi_gate); base.yaml
# sets patch_level.multi_gate: true, so it is ON by default. Gate-off
# requires patch_level.multi_gate=false (or top-level multi_gate=0).
exp patch_modulated_pta configs/patch_modulated_pta clip_surgery "" "PatchModPTA-CS-multigate-s1" 1 "multi_gate=1"

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
METHODS=("${_METHODS[@]}")
CONFIG_DIRS=("${_CONFIG_DIRS[@]}")
CLIP_MODELS=("${_CLIP_MODELS[@]}")
CLIP_CHECKPOINTS=("${_CLIP_CHECKPOINTS[@]}")
EXP_LABELS=("${_EXP_LABELS[@]}")
EXP_SEEDS=("${_EXP_SEEDS[@]}")
OVERRIDES=("${_OVERRIDES[@]}")

N_EXP=${#METHODS[@]}
N_TOTAL=$((N_EXP * N_DS))

if (( N_EXP == 0 )); then
    echo "ERROR: No experiments enabled. Uncomment at least one exp() line above."
    exit 1
fi

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

exp_idx=$((SLURM_ARRAY_TASK_ID / N_DS))
ds_idx=$((SLURM_ARRAY_TASK_ID % N_DS))

METHOD=${METHODS[$exp_idx]}
CONFIG=${CONFIG_DIRS[$exp_idx]}
CLIP_MODEL=${CLIP_MODELS[$exp_idx]}
CLIP_CKPT=${CLIP_CHECKPOINTS[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL=${EXP_LABELS[$exp_idx]}
SEED=${EXP_SEEDS[$exp_idx]}
OVERRIDE=${OVERRIDES[$exp_idx]}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL ($exp_idx / $N_EXP)"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  CLIP model : $CLIP_MODEL"
echo "  Dataset    : $DATASET  (ds_idx=$ds_idx)"
echo "  Seed       : $SEED"
echo "  Override   : ${OVERRIDE:-<none>}"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="$EXP_LABEL"
export RECORD_DIR="outputs/records/$EXP_LABEL"
export RESULT_FILE="outputs/result_perclass.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"

# Build command — only pass --clip-checkpoint when non-empty
CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --clip-model "$CLIP_MODEL"
    --datasets "$DATASET"
    --backbone ViT-B/16
    --seed "$SEED")

if [[ -n "$CLIP_CKPT" ]]; then
    CMD+=(--clip-checkpoint "$CLIP_CKPT")
fi

if [[ -n "$OVERRIDE" ]]; then CMD+=(--override "$OVERRIDE"); fi

"${CMD[@]}"
