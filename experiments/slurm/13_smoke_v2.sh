#!/bin/bash
#SBATCH --job-name=pta_repr_smoke2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=01:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_repr_smoke2_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_repr_smoke2_%x-%A_%a.err

# ============================================================================
# SLURM array job: PRE-FLIGHT SMOKE for the proper-bank + combinations wave.
# 5 tasks, dtd seed 1, MAX_BATCHES=400. Writes to outputs_repr_upgrade/smoke2/
# (NEVER the real result files).
#
# Purpose: (1) catch runtime bugs (import, config, NaN, record writes) for the
#   3 new methods; (2) confirm the proper growing bank actually SPREADS
#   active_k across slots on real CLIP features (no rich-get-richer collapse);
#   (3) compare create_threshold 0.85 vs 0.95 to pick the full-wave value.
#
#   task 0: bank_v2_pta      K=3 ct=0.85
#   task 1: bank_v2_pta      K=5 ct=0.85
#   task 2: bank_v2_pta      K=3 ct=0.95   (threshold comparison)
#   task 3: bank_compact_pta K=3 ct=0.85 gate=soft
#   task 4: gauss_compact_pta shrink=0.5 gate=soft
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/smoke2

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAX_BATCHES=400

BACKBONE=ViT-B/16
DATASET=dtd
SEED=1

# --override is argparse nargs="*": one flag takes every following KEY=VALUE
# token, and repeating the flag keeps only the LAST. Pass all tokens once.
case $SLURM_ARRAY_TASK_ID in
    0) METHOD=bank_v2_pta;      CONFIG=configs/bank_v2_pta;      OVR=(K=3 create_threshold=0.85); TAG=BNKV2-K3-ct085 ;;
    1) METHOD=bank_v2_pta;      CONFIG=configs/bank_v2_pta;      OVR=(K=5 create_threshold=0.85); TAG=BNKV2-K5-ct085 ;;
    2) METHOD=bank_v2_pta;      CONFIG=configs/bank_v2_pta;      OVR=(K=3 create_threshold=0.95); TAG=BNKV2-K3-ct095 ;;
    3) METHOD=bank_compact_pta; CONFIG=configs/bank_compact_pta; OVR=(K=3 create_threshold=0.85 gate_mode=soft); TAG=BNKCMP-K3-soft ;;
    4) METHOD=gauss_compact_pta; CONFIG=configs/gauss_compact_pta; OVR=(shrink=0.5 gate_mode=soft); TAG=GAUCMP-s0.5-soft ;;
    *) echo "SKIP: unknown task $SLURM_ARRAY_TASK_ID"; exit 0 ;;
esac

RESULT_LABEL="SMOKE2-${TAG}-${DATASET}-s${SEED}"
export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/smoke2/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/smoke2/result.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Method       : $METHOD"
echo "  Override     : $OVR"
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
