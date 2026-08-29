#!/bin/bash
#SBATCH --job-name=pta_tie_break
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --exclude=node1
#SBATCH --time=01:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/tie_break_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/tie_break_%x-%A_%a.err

# ============================================================================
# Part 1 — Tie-breaking analysis (Exp 1.2)
#
# Post-hoc analysis of PTA baseline records: accuracy on samples where
# clip and image_proto disagree (component disagreement / "ties").
#
# 3 tasks = 3 datasets (aggregated across seeds, no seed dimension)
#
#   dataset_idx = SLURM_ARRAY_TASK_ID
#
# Datasets: dtd, oxford_flowers, oxford_pets
#
# Depends on: 00_baseline_pta.sh (produces PTA baseline records)
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs/tie_breaking
mkdir -p logs

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
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Dataset    : $DATASET  (dataset_idx=$dataset_idx)"
echo "  Records dir: outputs/records/"
echo "  Output     : outputs/tie_breaking/${DATASET}.md"
echo "  Node       : $(hostname)"
echo "========================================================================"

python -u scripts/analyze_ties.py \
    --records outputs/records/ \
    --out "outputs/tie_breaking/${DATASET}.md"
