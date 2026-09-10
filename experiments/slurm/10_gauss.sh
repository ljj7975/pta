#!/bin/bash
#SBATCH --job-name=pta_gauss
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-23%10
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_gauss_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_gauss_%x-%A_%a.err

# ============================================================================
# SLURM array job: Representation upgrade M1a — per-class Gaussian prototype
# (mean + shrunk diagonal covariance, variance-aware scoring; TCA-style).
#
# 24 tasks = 2 shrinkage settings x 3 datasets x 4 seeds
#
#   set_idx  = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed  = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx   = ds_seed / 4
#   seed_idx = ds_seed % 4
#
# Mechanism (models/gauss_pta.py): write rule identical to base PTA
# (multi-class EMA, all classes with softmax >= 0.1, w = 1 - exp(-p/T)).
# State per class: mean (normalized EMA) + per-dim variance EMA. Score:
#   d_c = (1/D) * sum_d (x - mu_c)^2 / (v_c + shrink * Vbar)
#   logit_c = -d_c
# Fusion: final = 1.0*clip + 100.0*(logit_c - max_c logit)  (PTA-mirrored).
#
# Convergence: every run writes $RECORD_DIR/convergence.json.
#
# Environment variables:
#   RECORD_DIR   — outputs_repr_upgrade/records_gauss/{label}
#   RESULT_LABEL — encodes shrinkage + dataset + seed
#   RESULT_FILE  — outputs_repr_upgrade/result_gauss.txt
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/records_gauss

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=gauss_pta
CONFIG=configs/gauss_pta

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2 3 4)
N_SEEDS=${#SEEDS[@]}

SHRINKS=(0.1 0.5)
N_SETS=${#SHRINKS[@]}

N_TOTAL=$((N_SETS * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (shrinkage, dataset, seed)
# ---------------------------------------------------------------------------
set_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

SHRINK=${SHRINKS[$set_idx]}
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

RESULT_LABEL="GAU-s${SHRINK}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/records_gauss/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/result_gauss.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  shrink       : $SHRINK  (set_idx=$set_idx/$N_SETS)"
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
    --override "shrink=${SHRINK}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
