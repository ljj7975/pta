#!/bin/bash
#SBATCH --job-name=exp3_cd_vit_rem
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=3:00:00
#SBATCH --array=0-3
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/cd_vit_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/cd_vit_%x-%A_%a.err

set -euo pipefail

# Remaining Exp3 datasets for full 7-CD evaluation
# Already run: oxford_flowers (6306_0), dtd (6306_1), eurosat (6306_2)
DATASETS=(caltech101 fgvc oxford_pets ucf101)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Running Exp3 (GaussianPrototypes) CD benchmark for dataset: $DATASET (array task $SLURM_ARRAY_TASK_ID)"

python -u runner.py \
    --method exp3_gaussian_prototypes \
    --config configs_exp3 \
    --datasets "$DATASET" \
    --backbone ViT-B/16
