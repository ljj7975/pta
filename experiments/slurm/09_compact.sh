#!/bin/bash
#SBATCH --job-name=pta_compact
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-23%10
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_compact_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_compact_%x-%A_%a.err

# ============================================================================
# SLURM array job: Representation upgrade M2 — intra-class compactness write
# gate on the image prototype (models/compact_pta.py).
#
# 24 tasks = 2 gate modes x 3 datasets x 4 seeds
#
#   set_idx  = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed  = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx   = ds_seed / 4
#   seed_idx = ds_seed % 4
#
# Mechanism: write rule identical to base PTA (multi-class EMA, all classes
# with softmax >= 0.1, w = 1 - exp(-p/T)); only the write weight is modulated
# by the distance d_c = 1 - cos(x, proto_c) to the current class prototype:
#   soft: w_c *= exp(-d_c^2 / (2 sigma_c^2))   (sigma_c = running EMA of d_c)
#   hard: w_c = 0  if d_c > mean_c + 0.5 * std_c
# Pre-validation (archived base-PTA records): write purity falls steeply with
# distance (dtd 71%->13% across deciles; flowers 87%->12%; pets 89%->38%),
# so far writes are disproportionately wrong.
#
# Convergence: every run writes $RECORD_DIR/convergence.json (per-sample
# online-accuracy trajectory, acc@fractions, norm-AUC).
#
# Environment variables:
#   RECORD_DIR   — outputs_repr_upgrade/records_compact/{label}
#   RESULT_LABEL — encodes gate mode + dataset + seed
#   RESULT_FILE  — outputs_repr_upgrade/result_compact.txt
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/records_compact

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=compact_pta
CONFIG=configs/compact_pta

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2 3 4)
N_SEEDS=${#SEEDS[@]}

GATE_MODES=(soft hard)
N_SETS=${#GATE_MODES[@]}

N_TOTAL=$((N_SETS * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (gate mode, dataset, seed)
# ---------------------------------------------------------------------------
set_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

GATE_MODE=${GATE_MODES[$set_idx]}
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

RESULT_LABEL="CMP-${GATE_MODE}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/records_compact/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/result_compact.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  gate_mode    : $GATE_MODE  (set_idx=$set_idx/$N_SETS)"
echo "  Dataset      : $DATASET  (ds_idx=$ds_idx/$N_DS)"
echo "  Seed         : $SEED  (seed_idx=$seed_idx/$N_SEEDS)"
echo "  Method       : $METHOD"
echo "  Config       : $CONFIG"
echo "  RECORD_DIR   : $RECORD_DIR"
echo "  RESULT_LABEL : $RESULT_LABEL"
echo "  RESULT_FILE  : $RESULT_FILE"
echo "  Node         : $(hostname)"
echo "========================================================================"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --override "gate_mode=${GATE_MODE}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
