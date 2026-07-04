#!/bin/bash
#SBATCH --job-name=pta_exp3_vit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=5:00:00
#SBATCH --array=0-9
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/exp3_vit_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/exp3_vit_%x-%A_%a.err

# ============================================================================
# Slurm array job: exp3 (Gaussian Prototypes) CD benchmark (ViT-B/16)
#
# Each array task processes one of 10 cross-domain datasets.
# Method: exp3_gaussian_prototypes  (models/exp3_gaussian_prototypes.py)
# Config: configs_exp3/
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Running Exp3 CD benchmark for dataset: $DATASET (array task $SLURM_ARRAY_TASK_ID)"

python -u runner.py \
    --method exp3_gaussian_prototypes \
    --config configs_exp3 \
    --datasets "$DATASET" \
    --backbone ViT-B/16
