#!/bin/bash
#SBATCH --job-name=pta_baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_baseline_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_baseline_%x-%A_%a.err

# ============================================================================
# SLURM array job: PTA baseline — 3 datasets × 4 seeds with per-sample recording
#
# 12 tasks = 3 datasets × 4 seeds
#
#   dataset_idx = SLURM_ARRAY_TASK_ID / 4
#   seed_idx    = SLURM_ARRAY_TASK_ID % 4
#
# Datasets: dtd, oxford_flowers, oxford_pets
# Seeds:    1, 2, 3, 4
#
# Environment variables for per-sample recording:
#   RECORD_DIR   — output path for records.jsonl
#   RESULT_LABEL — method label in output results
#   SEED         — random seed
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
# Definitions
# ---------------------------------------------------------------------------
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

# ---------------------------------------------------------------------------
# Map task ID → dataset + seed
# ---------------------------------------------------------------------------
dataset_idx=$((SLURM_ARRAY_TASK_ID / 4))
seed_idx=$((SLURM_ARRAY_TASK_ID % 4))

DATASET=${DATASETS[$dataset_idx]}
SEED=${SEEDS[$seed_idx]}

export RECORD_DIR="outputs/records/PTA-CS-${DATASET}-s${SEED}"
export RESULT_LABEL="PTA-CS-${DATASET}-s${SEED}"
export SEED="${SEED}"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Dataset    : $DATASET  (dataset_idx=$dataset_idx)"
echo "  Seed       : $SEED  (seed_idx=$seed_idx)"
echo "  RECORD_DIR : $RECORD_DIR"
echo "  RESULT_LABEL: $RESULT_LABEL"
echo "  Node       : $(hostname)"
echo "========================================================================"

python -u runner.py \
    --method pta \
    --config configs/PTA \
    --datasets "$DATASET" \
    --backbone ViT-B/16 \
    --seed "$SEED"
