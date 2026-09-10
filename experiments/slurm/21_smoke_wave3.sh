#!/bin/bash
#SBATCH --job-name=pta_w3_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=01:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_w3_smoke_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_w3_smoke_%x-%A_%a.err

# ============================================================================
# SLURM array job: WAVE 3 PRE-FLIGHT SMOKE — AugPTA + AnchorPTA on
# caltech101, seed 1, MAX_BATCHES=200. Writes to outputs_repr_upgrade/smoke3/
# (NEVER the real result files).
#
# Purpose: (1) catch runtime bugs (imports, config chain, the fp16/fp32
# matmul path, record writes, NaN); (2) verify AugPTA's augmented branch
# reaches the entropy-selection code (V>1); (3) sanity-check the V=1
# identity cell reproduces base PTA on the same seed end-to-end.
#
#   task 0: aug_pta     V=8 K_views=4   (main setting)
#   task 1: aug_pta     V=4 K_views=2   (cheaper variant)
#   task 2: anchor_pta  damp 0.96 f0.5  (main setting)
#   task 3: anchor_pta  damp 0.93 f0.5  (heavier damping)
#   task 4: aug_pta     V=1 K_views=1   (identity == base PTA)
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/smoke3

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAX_BATCHES=200

BACKBONE=ViT-B/16
DATASET=caltech101
SEED=1

# --override is argparse nargs="*": one flag takes every following KEY=VALUE
# token, and repeating the flag keeps only the LAST. Pass all tokens once.
case $SLURM_ARRAY_TASK_ID in
    0) METHOD=aug_pta;     CONFIG=configs/aug_pta;    OVR=(V=8 K_views=4);       TAG=AUG-V8K4 ;;
    1) METHOD=aug_pta;     CONFIG=configs/aug_pta;    OVR=(V=4 K_views=2);       TAG=AUG-V4K2 ;;
    2) METHOD=anchor_pta;  CONFIG=configs/anchor_pta; OVR=(damp_threshold=0.96 damp_floor=0.5); TAG=ANCH-d096 ;;
    3) METHOD=anchor_pta;  CONFIG=configs/anchor_pta; OVR=(damp_threshold=0.93 damp_floor=0.5); TAG=ANCH-d093 ;;
    4) METHOD=aug_pta;     CONFIG=configs/aug_pta;    OVR=(V=1 K_views=1);       TAG=AUG-V1K1 ;;
    *) echo "SKIP: unknown task $SLURM_ARRAY_TASK_ID"; exit 0 ;;
esac

RESULT_LABEL="SMOKE3-${TAG}-${DATASET}-s${SEED}"
export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/smoke3/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/smoke3/result.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Method       : $METHOD"
echo "  Override     : ${OVR[*]}"
echo "  Dataset      : $DATASET  Seed: $SEED  MAX_BATCHES=$MAX_BATCHES"
echo "  RECORD_DIR   : $RECORD_DIR"
echo "  Node         : $(hostname)"
echo "========================================================================"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --override "${OVR[@]}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"