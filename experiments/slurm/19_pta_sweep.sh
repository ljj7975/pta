#!/bin/bash
#SBATCH --job-name=pta_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-23%10
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_sweep_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_sweep_%x-%A_%a.err

# ============================================================================
# SLURM array job: WAVE 3, PTA BASELINE alpha/T SENSITIVITY SWEEP.
#
# Purpose: show the alpha=0.01 / T=20 baseline (configs/PTA) is already
# near-optimal — wave-3 claims are vs a WELL-TUNED baseline, not a strawman.
#
# 24 tasks = 4 off-center settings x 3 datasets x 2 seeds
# (the center itself, alpha=0.01/T=20, is the existing PTA baseline.)
#
#   set_idx  = SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)
#   ds_seed  = SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)
#   ds_idx   = ds_seed / N_SEEDS
#   seed_idx = ds_seed % N_SEEDS
#
# Settings (DOTTED image_level overrides — REQUIRED: configs/PTA inherits
# base.yaml, whose nested image_level.alpha/T already exist; PTAAdapter's
# flat-key propagation runs only when nested keys are ABSENT, so flat
# alpha=.../T=... would silently no-op):
#   A0.003-T10, A0.003-T40  (low alpha: near-zero-shot)
#   A0.1-T10,   A0.1-T40    (high alpha: prototype-dominated)
#
# Environment variables:
#   RECORD_DIR   — outputs_repr_upgrade/records_pta_sweep/{label}
#   RESULT_LABEL — encodes alpha/T + dataset + seed
#   RESULT_FILE  — outputs_repr_upgrade/result_pta_sweep.txt
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/records_pta_sweep

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=pta
CONFIG=configs/PTA

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2)
N_SEEDS=${#SEEDS[@]}

SETS_NAMES=(A0.003-T10 A0.003-T40 A0.1-T10 A0.1-T40)
SETS_OVR=("image_level.alpha=0.003 image_level.T=10" "image_level.alpha=0.003 image_level.T=40" "image_level.alpha=0.1 image_level.T=10" "image_level.alpha=0.1 image_level.T=40")
N_SETS=${#SETS_NAMES[@]}

N_TOTAL=$((N_SETS * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (setting, dataset, seed)
# ---------------------------------------------------------------------------
set_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

SET_NAME=${SETS_NAMES[$set_idx]}
# shellcheck disable=SC2206
OVR=(${SETS_OVR[$set_idx]})
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

RESULT_LABEL="PTASW-${SET_NAME}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/records_pta_sweep/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/result_pta_sweep.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Setting      : $SET_NAME  (set_idx=$set_idx/$N_SETS)  override: ${OVR[*]}"
echo "  Dataset      : $DATASET  (ds_idx=$ds_idx/$N_DS)"
echo "  Seed         : $SEED  (seed_idx=$seed_idx/$N_SEEDS)"
echo "  Method       : $METHOD"
echo "  Config       : $CONFIG"
echo "  RECORD_DIR   : $RECORD_DIR"
echo "  RESULT_LABEL : $RESULT_LABEL"
echo "  RESULT_FILE  : $RESULT_FILE"
echo "  Node         : $(hostname)"
echo "========================================================================"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --override "${OVR[@]}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"