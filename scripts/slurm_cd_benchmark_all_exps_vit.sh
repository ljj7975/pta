#!/bin/bash
#SBATCH --job-name=all_exps_cd_vit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=24:00:00
#SBATCH --array=0-55
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/all_exps_vit_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/all_exps_vit_%x-%A_%a.err

# ============================================================================
# Slurm array job: All CD benchmark experiments (ViT-B/16)
#
# 56 tasks = 8 experiments × 7 CD core datasets
#
#   exp_idx = SLURM_ARRAY_TASK_ID / 7
#   ds_idx  = SLURM_ARRAY_TASK_ID % 7
#
#   ID  Experiment               Method                     Config
#   --  ------------------------ -------------------------- ---------------
#    0  PTA (baseline)           pta                        configs
#    1  Exp4FullFusion           exp4_full_fusion           configs_exp4
#    2  Exp5TunableFusion        exp5_tunable_fusion        configs_exp4
#    3  Exp7InvertedImageWeight  exp7_inverted_image_weight configs_exp7
#    4  Exp8BoostedPatch         exp8_boosted_patch         configs_exp8
#    5  Exp10SoftPatchGate       exp10_soft_patch_gate      configs_exp10
#    6  Exp11TunableFusionNew    exp11_tunable_fusion_new   configs_exp11
#    7  Exp12PatchQualityMod     exp12_patch_quality_modulation configs_exp12
#
# CD core datasets (all same 7 for every experiment):
#   caltech101 dtd eurosat fgvc oxford_flowers oxford_pets ucf101
#
# Run after randomness fix (commit 95c1855) for reproducible results.
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
    exp4_full_fusion
    exp5_tunable_fusion
    exp7_inverted_image_weight
    exp8_boosted_patch
    exp10_soft_patch_gate
    exp11_tunable_fusion_new
    exp12_patch_quality_modulation
)

CONFIG_DIRS=(
    configs
    configs_exp4
    configs_exp4
    configs_exp7
    configs_exp8
    configs_exp10
    configs_exp11
    configs_exp12
)

EXP_LABELS=(
    "PTA-baseline"
    "Exp4-FullFusion"
    "Exp5-TunableFusion"
    "Exp7-InvertedImageWeight"
    "Exp8-BoostedPatch"
    "Exp10-SoftPatchGate"
    "Exp11-TunableFusionNew"
    "Exp12-PatchQualityModulation"
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

python -u runner.py \
    --method "$METHOD" \
    --config "$CONFIG" \
    --datasets "$DATASET" \
    --backbone ViT-B/16
