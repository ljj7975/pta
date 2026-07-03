#!/bin/bash
#SBATCH --job-name=exp6_cd_benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=5:00:00
#SBATCH --array=0-9
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/exp6_vit_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/exp6_vit_%x-%A_%a.err

# ============================================================================
# Slurm array job: exp6 (Adaptive Per-Class Image-Level Weighting) CD benchmark (ViT-B/16)
#
# Each array task processes one of 10 cross-domain datasets.
# Method: exp6_adaptive_image_weight  (models/exp6_adaptive_image_weight.py)
# Config: configs_exp6/
#
# Key difference from exp5:
#   tau_image[c] = tau_image_max - (tau_image_max - tau_image_min) * proto_alpha[c]
#   Per-class adaptive image-level weight based on prototype quality.
# ============================================================================

set -euo pipefail

DATASETS=(caltech101 dtd eurosat fgvc oxford_flowers oxford_pets Ducf101)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Running Exp6 CD benchmark for dataset: $DATASET (array task $SLURM_ARRAY_TASK_ID)"

python -u runner.py \
    --method exp6_adaptive_image_weight \
    --config configs_exp6 \
    --datasets "$DATASET" \
    --backbone ViT-B/16
