#!/bin/bash
#SBATCH --job-name=pta_dev_writerule
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-49
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_dev_writerule_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_dev_writerule_%x-%A_%a.err

# ============================================================================
# Write-rule x write-source comparison on the 5 DEV datasets.
# Default config chain (tanh squash is baked into base.yaml):
#   configs/patch_modulated_pta/<ds>.yaml -> patch_modulated_pta.yaml -> base.yaml
#
#   exp_idx = SLURM_ARRAY_TASK_ID / N_DS   (write setting)
#   ds_idx  = SLURM_ARRAY_TASK_ID % N_DS   (dataset)
#
# write_source (which logits pick the classes to write):
#   clip   — frozen zero-shot CLIP logits (previous default)
#   pta    — CLIP + tau_image*image_proto (PTA-style fused logits)
#   image  — image-level prototype logits only
# For pta/image the prototype bank is read BEFORE the sample's write
# (causal mask), matching the .half() matmul convention of the post-update
# logits. Hypothesis: pta source is best since PTA outperforms zero-shot.
#
# write_rule (which classes get written, given that source's softmax):
#   top1   — only the argmax class
#   ratio  — w >= 0.5 * w.max()   (rate-based, class-count-independent)
#   cpm    — smallest top-k with cumulative mass >= 0.5
#   thresh — w >= 0.1 (legacy; control)
#
#   ID  Setting       write_source  write_rule
#   --  ------------  ------------  -----------
#    0  thresh-clip   clip          thresh   (control = previous default)
#    1  top1-clip     clip          top1
#    2  top1-pta      pta           top1
#    3  top1-image    image         top1
#    4  ratio-clip    clip          ratio
#    5  ratio-pta     pta           ratio
#    6  ratio-image   image         ratio
#    7  cpm-clip      clip          cpm
#    8  cpm-pta       pta           cpm
#    9  cpm-image     image         cpm
#
# Literature context (why this grid):
#   * Soft/multi-class weighting dominates TTA practice (TPT, TENT, SAR,
#     RoTTA, EcoTTA, EATA); hard top-1-only updates are the documented
#     bias-accumulation risk (LMR 2026 warns against them; ProtoTTA uses
#     top-1 only under strong confidence filtering). Our counterfactual
#     evidence on dtd/flowers showed hard-winner policies can still beat
#     soft baselines, so top1 is tested — but alongside ratio/cpm, which
#     are top-1-like without the single-class exclusivity.
#   * Write-source landscape: zero-shot CLIP logits alone is the most
#     common source (PTA reference: softmax(clip_logits), w>=0.1);
#     CLIP+image-proto fused logits is emerging (Tip-Adapter, TDA, MCP);
#     image-proto-only as a write source appears in NO published method —
#     the image-source arm is a novel contribution if it wins.
#   * pta-source is the safety net for the risky top1-clip arm: it writes
#     with the same fused logits used for prediction, whose accuracy is
#     already validated (MajorityVoteFusion: +1.51 avg over softmax).
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/records_dev_writerule

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

WRITE_SOURCES=(clip clip pta image clip pta image clip pta image)
WRITE_RULES=(thresh top1 top1 top1 ratio ratio ratio cpm cpm cpm)
EXP_LABELS=(thresh-clip-mv top1-clip-mv top1-pta-mv top1-image-mv ratio-clip-mv ratio-pta-mv ratio-image-mv cpm-clip-mv cpm-pta-mv cpm-image-mv)
N_EXP=${#EXP_LABELS[@]}
N_TOTAL=$((N_EXP * N_DS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (write setting, dataset)
# ---------------------------------------------------------------------------
exp_idx=$((SLURM_ARRAY_TASK_ID / N_DS))
ds_idx=$((SLURM_ARRAY_TASK_ID % N_DS))

WRITE_SOURCE=${WRITE_SOURCES[$exp_idx]}
WRITE_RULE=${WRITE_RULES[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL="${EXP_LABELS[$exp_idx]}-${DATASET}-s${SEED}"

export RESULT_LABEL="$EXP_LABEL"
export RECORD_DIR="outputs/records_dev_writerule/$EXP_LABEL"
export RESULT_FILE="outputs/result_dev_writerule.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment   : $EXP_LABEL (setting=$exp_idx/$N_EXP, ds=$ds_idx/$N_DS)"
echo "  Method       : $METHOD"
echo "  Config       : $CONFIG"
echo "  Dataset      : $DATASET"
echo "  write_source : $WRITE_SOURCE"
echo "  write_rule   : $WRITE_RULE"
echo "  Seed         : $SEED"
echo "  Record dir   : $RECORD_DIR"
echo "  Node         : $(hostname)"
echo "========================================================================"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --override "write_source=${WRITE_SOURCE}" "write_rule=${WRITE_RULE}" "fusion.type=MajorityVoteFusion"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
