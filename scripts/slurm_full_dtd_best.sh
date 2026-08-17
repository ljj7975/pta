#!/bin/bash
#SBATCH --job-name=pta_full_dtd_best
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=5:00:00
#SBATCH --array=0-1
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_full_dtd_best_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_full_dtd_best_%x-%A_%a.err

# ============================================================================
# Slurm array job: write-rule comparison on FULL DTD (47 classes),
# default config chain:
#   configs/patch_modulated_pta/dtd.yaml -> patch_modulated_pta.yaml -> base.yaml
#   (patch_squash=tanh scale=1.0, patch_filter_mode=surgery_with_labels)
#
# write_source=final and patch_layer (block10) have been REMOVED from the code
# (job 15251 findings): the fused-logits write source is state-coupled and
# collapses the prototype bank; the block10 early-exit loses patch
# discriminability.  The write mask always derives from the frozen zero-shot
# CLIP logits.  Remaining knob: write_rule.
#
#   ID  Experiment   Overrides                  Expectation
#   --  -----------  -------------------------  -------------------------
#    0  cpm-clip     write_rule=cpm cpm_p=0.5   ~45% (class-count-independent rule)
#    1  ratio-clip   write_rule=ratio ratio_r=0.5  ~45% (class-count-independent rule)
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
DATASET=dtd
BACKBONE=ViT-B/16
METHOD=patch_modulated_pta
CONFIG=configs/patch_modulated_pta

EXP_LABELS=(
    "PatchModPTA-dtd-cpm-clip"
    "PatchModPTA-dtd-ratio-clip"
)

OVERRIDES=(
    "write_rule=cpm cpm_p=0.5"
    "write_rule=ratio ratio_r=0.5"
)

SEED=1

# ---------------------------------------------------------------------------
# Map task ID -> experiment
# ---------------------------------------------------------------------------
exp_idx=$SLURM_ARRAY_TASK_ID

EXP_LABEL=${EXP_LABELS[$exp_idx]}
OVERRIDE=${OVERRIDES[$exp_idx]}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  Dataset    : $DATASET"
echo "  Override   : ${OVERRIDE:-<none>}"
echo "  Seed       : $SEED"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="${EXP_LABEL}"
export SEED="$SEED"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

if [[ -n "$OVERRIDE" ]]; then CMD+=(--override $OVERRIDE); fi

"${CMD[@]}"
