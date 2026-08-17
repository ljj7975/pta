#!/bin/bash
#SBATCH --job-name=pta_dev_vote
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-14
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_dev_vote_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_dev_vote_%x-%A_%a.err

# ============================================================================
# Vote-gating fusion comparison on the 5 DEV datasets.
# Default config chain (tanh squash is now baked into base.yaml):
#   configs/patch_modulated_pta/<ds>.yaml -> patch_modulated_pta.yaml -> base.yaml
#
#   exp_idx = SLURM_ARRAY_TASK_ID / N_DS   (fusion setting)
#   ds_idx  = SLURM_ARRAY_TASK_ID % N_DS   (dataset)
#
#   ID  Fusion type            Experiment
#   --  ---------------------  ------------------------------------------------
#    0  ProtoAlphaFusion       baseline: always-on patch (tanh squash, tau 1/80/10)
#    1  MajorityVoteFusion     2-of-3 vote; no majority -> clip+image fallback
#    2  AgreementGateFusion    mute patch when patch.argmax != clip.argmax
#
# MajorityVoteFusion and AgreementGateFusion were validated offline against
# recorded logits (scripts/counterfactual_gate_analysis.py): +1.48/+1.30 on
# full DTD, +2.43/+2.19 on flowers vs the always-on baseline.
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/records_dev_vote

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=patch_modulated_pta
CONFIG=configs/patch_modulated_pta
SEED=1

DATASETS=(dtd eurosat fgvc oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}

FUSION_TYPES=(
    "ProtoAlphaFusion"     # baseline: always-on patch (tanh squash)
    "MajorityVoteFusion"   # 2-of-3 majority vote, no-majority -> clip+image
    "AgreementGateFusion"  # mute patch when it disagrees with CLIP
)
N_EXP=${#FUSION_TYPES[@]}
N_TOTAL=$((N_EXP * N_DS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (fusion setting, dataset)
# ---------------------------------------------------------------------------
exp_idx=$((SLURM_ARRAY_TASK_ID / N_DS))
ds_idx=$((SLURM_ARRAY_TASK_ID % N_DS))

FUSION_TYPE=${FUSION_TYPES[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL="${FUSION_TYPE}-${DATASET}-s${SEED}"

export RESULT_LABEL="$EXP_LABEL"
export RECORD_DIR="outputs/records_dev_vote/$EXP_LABEL"
export RESULT_FILE="outputs/result_dev_vote.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL (setting=$exp_idx/$N_EXP, ds=$ds_idx/$N_DS)"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  Dataset    : $DATASET"
echo "  Fusion     : $FUSION_TYPE"
echo "  Seed       : $SEED"
echo "  Record dir : $RECORD_DIR"
echo "  Node       : $(hostname)"
echo "========================================================================"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --override "fusion.type=${FUSION_TYPE}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
