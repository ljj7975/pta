#!/bin/bash
#SBATCH --job-name=pta_sep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_sep_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_sep_%x-%A_%a.err

# ============================================================================
# SLURM array job: Part 2.2 — PatchModPTA-CS bank dump + separability analysis
#
# Runs PatchModPTA-CS with DUMP_PATCH_BANK enabled to collect Gaussian bank
# state, then runs analyze_prototype_separability.py on the dumps.
#
# 12 tasks = 3 datasets × 4 seeds
#
#   dataset_idx = SLURM_ARRAY_TASK_ID / 4
#   seed_idx    = SLURM_ARRAY_TASK_ID % 4
#
# Datasets: dtd, oxford_flowers, oxford_pets
# Seeds:    1, 2, 3, 4
#
# Environment variables:
#   DUMP_PATCH_BANK — path for bank dump (.pt), triggers bank saving in adapter
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/patch_bank_dumps
mkdir -p outputs/prototype_separability

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

export DUMP_PATCH_BANK="outputs/patch_bank_dumps/${DATASET}-s${SEED}.pt"

# ---------------------------------------------------------------------------
# Run PatchModPTA-CS with bank dump
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Dataset    : $DATASET  (dataset_idx=$dataset_idx)"
echo "  Seed       : $SEED  (seed_idx=$seed_idx)"
echo "  DUMP_PATCH_BANK: $DUMP_PATCH_BANK"
echo "  Node       : $(hostname)"
echo "========================================================================"

python -u runner.py \
    --method patch_modulated_pta \
    --config configs/patch_modulated_pta \
    --datasets "$DATASET" \
    --backbone ViT-B/16 \
    --seed "$SEED"

# ---------------------------------------------------------------------------
# Run separability analysis on the collected bank dumps
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Running separability analysis for $DATASET seed=$SEED"
echo "========================================================================"

python -u scripts/analyze_prototype_separability.py \
    --dumps outputs/patch_bank_dumps \
    --out "outputs/prototype_separability/${DATASET}-s${SEED}.md"

echo "Done."
