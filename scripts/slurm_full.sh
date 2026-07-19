#!/bin/bash
#SBATCH --job-name=pta_vs_patch_mod
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=5:00:00
#SBATCH --array=0-13
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_vs_patch_mod_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_vs_patch_mod_%x-%A_%a.err

# ============================================================================
# Slurm array job: PTA vs PatchModulatedPTA CD benchmark (ViT-B/16)
#
# 14 tasks = 2 experiments × 7 CD core datasets
#
#   exp_idx = SLURM_ARRAY_TASK_ID / 7
#   ds_idx  = SLURM_ARRAY_TASK_ID % 7
#
#   ID  Experiment           Method                 Config
#   --  --------------------- ---------------------- ---------------
#    0  PTA (baseline)       pta                    configs
#    1  PatchModulatedPTA    patch_modulated_pta    configs
#
# CD core datasets (same 7 for every experiment):
#   caltech101 dtd eurosat fgvc oxford_flowers oxford_pets ucf101
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
DATASETS=(caltech101 dtd eurosat fgvc oxford_flowers oxford_pets ucf101)

METHODS=(
    pta
    patch_modulated_pta
)

CONFIG_DIRS=(
    configs
    configs/patch_modulated_pta
)

EXP_LABELS=(
    "PTA-baseline"
    "PatchModulatedPTA"
)

# ---------------------------------------------------------------------------
# Map task ID → experiment + dataset
# ---------------------------------------------------------------------------
exp_idx=$((SLURM_ARRAY_TASK_ID / 7))
ds_idx=$((SLURM_ARRAY_TASK_ID % 7))

METHOD=${METHODS[$exp_idx]}
CONFIG=${CONFIG_DIRS[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL=${EXP_LABELS[$exp_idx]}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL ($exp_idx)"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  Dataset    : $DATASET  (ds_idx=$ds_idx)"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="${EXP_LABEL}"

python -u runner.py \
    --method "$METHOD" \
    --config "$CONFIG" \
    --datasets "$DATASET" \
    --backbone ViT-B/16
