#!/bin/bash
#SBATCH --job-name=pta_exp1_vit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=5:00:00
#SBATCH --array=0-9
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/exp1_vit_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/exp1_vit_%x-%A_%a.err

# ============================================================================
# Slurm array job: exp1 (Fixed Unique Patches) CD benchmark (ViT-B/16)
#
# Each array task processes one of 10 cross-domain datasets.
# Method: exp1_fixed_unique_patches  (models/exp1_fixed_unique_patches.py)
# Config: configs_exp1/
# ============================================================================

set -euo pipefail

DATASETS=(caltech101 dtd eurosat fgvc oxford_flowers oxford_pets ucf101)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Running Exp1 CD benchmark for dataset: $DATASET (array task $SLURM_ARRAY_TASK_ID)"

python -u runner.py \
    --method exp1_fixed_unique_patches \
    --config configs_exp1 \
    --datasets "$DATASET" \
    --backbone ViT-B/16
