#!/bin/bash
#SBATCH --job-name=pta_repr_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_repr_smoke_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_repr_smoke_%x-%A_%a.err

# ============================================================================
# SLURM array job: PRE-FLIGHT SMOKE for the representation-upgrade wave.
# 4 tasks = one per method (first setting), dtd seed 1, MAX_BATCHES=50.
# Writes to outputs_repr_upgrade/smoke/ (NEVER the real result files).
# Purpose: catch runtime bugs (import, config, NaN, record writes) before
# submitting the full 72-task wave.
#
#   task 0: gauss_pta    shrink=0.5
#   task 1: bank_pta     K=5
#   task 2: compact_pta  gate_mode=soft
#   task 3: compact_pta  gate_mode=hard
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/smoke

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MAX_BATCHES=50

BACKBONE=ViT-B/16
DATASET=dtd
SEED=1

case $SLURM_ARRAY_TASK_ID in
    0) METHOD=gauss_pta;   CONFIG=configs/gauss_pta;   OVR="shrink=0.5";   TAG=GAU-s0.5 ;;
    1) METHOD=bank_pta;    CONFIG=configs/bank_pta;    OVR="K=5";          TAG=BNK-K5 ;;
    2) METHOD=compact_pta; CONFIG=configs/compact_pta; OVR="gate_mode=soft"; TAG=CMP-soft ;;
    3) METHOD=compact_pta; CONFIG=configs/compact_pta; OVR="gate_mode=hard"; TAG=CMP-hard ;;
    *) echo "SKIP: unknown task $SLURM_ARRAY_TASK_ID"; exit 0 ;;
esac

RESULT_LABEL="SMOKE-${TAG}-${DATASET}-s${SEED}"
export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/smoke/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/smoke/result_smoke.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

echo "========================================================================"
echo "  SMOKE task   : $SLURM_ARRAY_TASK_ID"
echo "  Method       : $METHOD  (override: $OVR)"
echo "  MAX_BATCHES  : $MAX_BATCHES"
echo "  RECORD_DIR   : $RECORD_DIR"
echo "  Node         : $(hostname)"
echo "========================================================================"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --override "$OVR"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
