#!/bin/bash
#SBATCH --job-name=pta_agree
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_agree_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_agree_%x-%A_%a.err

# ============================================================================
# SLURM array job: Part 3.1 + 3.3 — Patch vote agreement signal validation
#
# Runs validate_patch_vote_signal.py which reads stored records and patch bank
# dumps to validate that patch-level vote agreement carries predictive signal
# for correctness.  Aggregates across all 4 seeds per dataset (no seed
# dimension — each task handles one dataset across seeds 1-4).
#
# 3 tasks = 3 datasets
#
#   dataset_idx = SLURM_ARRAY_TASK_ID  (0-2)
#
# Datasets: dtd, oxford_flowers, oxford_pets
#
# Dependencies:
#   - 02_part2_purity.sh (PatchModPTA records)
#   - 03_part2_separability.sh (patch bank dumps)
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/patch_vote_validation

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
DATASETS=(dtd oxford_flowers oxford_pets)

# ---------------------------------------------------------------------------
# Map task ID → dataset
# ---------------------------------------------------------------------------
dataset_idx=$SLURM_ARRAY_TASK_ID
DATASET=${DATASETS[$dataset_idx]}

# ---------------------------------------------------------------------------
# Run patch vote agreement signal validation
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Dataset    : $DATASET  (dataset_idx=$dataset_idx)"
echo "  Node       : $(hostname)"
echo "========================================================================"

python -u scripts/validate_patch_vote_signal.py \
    --dataset "$DATASET" \
    --seeds 1,2,3,4

echo "Done."
