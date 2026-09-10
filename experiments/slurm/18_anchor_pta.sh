#!/bin/bash
#SBATCH --job-name=pta_anchor
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-23%10
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_anchor_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_anchor_%x-%A_%a.err

# ============================================================================
# SLURM array job: WAVE 3, METHOD ANCHORPTA (text-anchor-referenced EMA
# rate damping, write-purity lever #2).
#
# 24 tasks = 2 damp thresholds x 3 datasets x 4 seeds
#
#   set_idx  = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed  = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx   = ds_seed / 4
#   seed_idx = ds_seed % 4
#
# Settings:
#   d096 — damp_threshold=0.96 (healthy classes sim ~0.97-0.99 never damp)
#   d093 — damp_threshold=0.93 (heavier damping window)
#
# Mechanism (models/anchor_pta.py): for each class whose refined vector
# cos(refine[c], text[c]) falls below damp_threshold relative to its
# IMMUTABLE text anchor, the per-class EMA write rate is damped toward
# damp_floor (0.5): w_new[c] *= damp_floor + (1-damp_floor)*sim/threshold.
# The referee is the fixed text anchor — never estimated from CLIP's biased
# stream. damp_threshold=0 disables damping => base PTA exactly.
#
# Environment variables:
#   RECORD_DIR   — outputs_repr_upgrade/records_anchor_pta/{label}
#   RESULT_LABEL — encodes threshold + dataset + seed
#   RESULT_FILE  — outputs_repr_upgrade/result_anchor_pta.txt
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/records_anchor_pta

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=anchor_pta
CONFIG=configs/anchor_pta

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2 3 4)
N_SEEDS=${#SEEDS[@]}

SETS_NAMES=(d096 d093)
SETS_OVR=("damp_threshold=0.96 damp_floor=0.5" "damp_threshold=0.93 damp_floor=0.5")
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

RESULT_LABEL="ANCHOR-${SET_NAME}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/records_anchor_pta/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/result_anchor_pta.txt"
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