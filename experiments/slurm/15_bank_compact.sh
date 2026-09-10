#!/bin/bash
#SBATCH --job-name=pta_bnkcompact
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-23%10
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_bnkcompact_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_bnkcompact_%x-%A_%a.err

# ============================================================================
# SLURM array job: COMBINATION — proper GROWING bank (BankV2) + intra-class
# compactness write gate.
#
# 24 tasks = 2 bank sizes x 3 datasets x 4 seeds
#
#   set_idx  = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed  = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx   = ds_seed / 4
#   seed_idx = ds_seed % 4
#
# Mechanism (models/bank_compact_pta.py): BankV2 routing (nearest active
# prototype; new mode grows/recycles a slot UNGATED); when a feature is
# ABSORBED into slot k*, its EMA weight is modulated by the compactness gate
# on d = 1 - cos(x, P[c,k*]) (soft attenuation). Score = max-of-active cosine;
# fusion = exact PTA mirror. K=1 + gate=off = base PTA.
#
# Environment variables:
#   RECORD_DIR   — outputs_repr_upgrade/records_combos/{label}
#   RESULT_LABEL — encodes K + dataset + seed
#   RESULT_FILE  — outputs_repr_upgrade/result_combos.txt
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs_repr_upgrade/records_combos

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=bank_compact_pta
CONFIG=configs/bank_compact_pta
CREATE_THRESHOLD=0.85
GATE_MODE=soft

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2 3 4)
N_SEEDS=${#SEEDS[@]}

KS=(3 5)
N_SETS=${#KS[@]}

N_TOTAL=$((N_SETS * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (K, dataset, seed)
# ---------------------------------------------------------------------------
set_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

K=${KS[$set_idx]}
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

RESULT_LABEL="BNKCMP-K${K}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs_repr_upgrade/records_combos/${RESULT_LABEL}"
export RESULT_FILE="outputs_repr_upgrade/result_combos.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  K            : $K  (set_idx=$set_idx/$N_SETS)  ct=$CREATE_THRESHOLD gate=$GATE_MODE"
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
    --override "K=${K}" "create_threshold=${CREATE_THRESHOLD}" "gate_mode=${GATE_MODE}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
