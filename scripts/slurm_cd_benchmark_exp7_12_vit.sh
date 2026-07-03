#!/bin/bash
#SBATCH --job-name=exp7_12_cd_benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=5:00:00
#SBATCH --array=0-41
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/exp7_12_vit_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/exp7_12_vit_%x-%A_%a.err

# ============================================================================
# Slurm array job: Exp7-Exp12 CD benchmark (ViT-B/16)
#
# 42 tasks: 6 experiments × 7 datasets
#   exp_idx = SLURM_ARRAY_TASK_ID / 7
#   ds_idx  = SLURM_ARRAY_TASK_ID % 7
#
# Experiments:
#   0 = exp7_inverted_image_weight  (configs_exp7)
#   1 = exp8_boosted_patch          (configs_exp8)
#   2 = exp9_entropy_fusion         (configs_exp9)
#   3 = exp10_soft_patch_gate       (configs_exp10)
#   4 = exp5_tunable_fusion         (configs_exp11) — Exp11 REUSES exp5 method
#   5 = exp12_patch_quality_modulation (configs_exp12)
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

DATASETS=(caltech101 dtd eurosat fgvc oxford_flowers oxford_pets ucf101)

METHODS=(
    exp7_inverted_image_weight
    exp8_boosted_patch
    exp9_entropy_fusion
    exp10_soft_patch_gate
    exp5_tunable_fusion_new
    exp12_patch_quality_modulation
)

CONFIG_DIRS=(
    configs_exp7
    configs_exp8
    configs_exp9
    configs_exp10
    configs_exp11
    configs_exp12
)

EXP_IDX=$((SLURM_ARRAY_TASK_ID / 7))
DS_IDX=$((SLURM_ARRAY_TASK_ID % 7))

METHOD=${METHODS[$EXP_IDX]}
CONFIG=${CONFIG_DIRS[$EXP_IDX]}
DATASET=${DATASETS[$DS_IDX]}

echo "Running Exp7-12 CD benchmark: method=$METHOD config=$CONFIG dataset=$DATASET (exp_idx=$EXP_IDX ds_idx=$DS_IDX array_task=$SLURM_ARRAY_TASK_ID)"

python -u runner.py \
    --method "$METHOD" \
    --config "$CONFIG" \
    --datasets "$DATASET" \
    --backbone ViT-B/16
