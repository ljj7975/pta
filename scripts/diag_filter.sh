#!/bin/bash
#SBATCH --job-name=diag_filter
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=0:10:00
#SBATCH --output=/share_98/projects/brandon/repos/pta/dev_logs/diag_filter_%j_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/dev_logs/diag_filter_%j_%a.err
#SBATCH --array=0-4c

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

_MODES=(
    ""
    "patch_level.patch_filter_mode=cosine_with_labels"
    "patch_level.patch_filter_mode=cosine_no_labels"
    "patch_level.patch_filter_mode=surgery_with_labels"
    "patch_level.patch_filter_mode=surgery_no_labels"
)

_MODE_LABELS=(
    "none"
    "cosine_with_labels"
    "cosine_no_labels"
    "surgery_with_labels"
    "surgery_no_labels"
)

IDX=${SLURM_ARRAY_TASK_ID}
MODE=${_MODES[$IDX]}
LABEL=${_MODE_LABELS[$IDX]}

echo "========================================================================"
echo "  DIAG FILTER: mode=$LABEL"
echo "  Override: ${MODE:-<none>}"
echo "========================================================================"

MAX_BATCHES=3 python -u runner.py \
    --method patch_modulated_pta \
    --config configs/patch_modulated_pta \
    --clip-model clip_surgery \
    --datasets dtd \
    --backbone ViT-B/16 \
    ${MODE:+--override "$MODE"}
