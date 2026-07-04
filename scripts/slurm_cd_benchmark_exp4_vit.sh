#!/bin/bash
#SBATCH --job-name=exp4_cd_benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=24:00:00
#SBATCH --array=0-29
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/exp4_cd_benchmark_%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/exp4_cd_benchmark_%A_%a.err

# ============================================================================
# Slurm array job: exp4/exp5 CD benchmark (ViT-B/16)
#
# Array index mapping: 0-29 = 3 methods × 10 datasets
#   Methods: pta, exp4_full_fusion, exp5_tunable_fusion
#   Datasets: caltech101, dtd, eurosat, fgvc, food101,
#             oxford_flowers, oxford_pets, stanford_cars, sun397, ucf101
#
# Config:
#   pta               → configs/
#   exp4_full_fusion  → configs_exp4/
#   exp5_tunable_fusion → configs_exp4/
# ============================================================================

set -euo pipefail

METHODS=("pta" "exp4_full_fusion" "exp5_tunable_fusion")
DATASETS=("caltech101" "dtd" "eurosat" "fgvc" "food101" "oxford_flowers" "oxford_pets" "stanford_cars" "sun397" "ucf101")

idx=$SLURM_ARRAY_TASK_ID
method_idx=$((idx / 10))
dataset_idx=$((idx % 10))
method=${METHODS[$method_idx]}
dataset=${DATASETS[$dataset_idx]}

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [ "$method" = "pta" ]; then
    config_dir="configs"
else
    config_dir="configs_exp4"
fi

echo "Running ${method} on ${dataset} (array task ${SLURM_ARRAY_TASK_ID}, method_idx=${method_idx}, dataset_idx=${dataset_idx})"

python -u runner.py \
    --method "$method" \
    --config "$config_dir" \
    --datasets "$dataset" \
    --backbone ViT-B/16
