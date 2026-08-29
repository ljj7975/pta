#!/bin/bash
#SBATCH --job-name=pta_write_gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-47%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_write_gate_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_write_gate_%x-%A_%a.err

# ============================================================================
# SLURM array job: Part 4 (redesigned) — write-rule study on the image prototype
#
# 48 tasks = 4 write-gates x 3 datasets x 4 seeds
#
#   gate_idx  = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed   = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx    = ds_seed / 4
#   seed_idx  = ds_seed % 4
#
# The "%4" in the #SBATCH --array line is the concurrency limit (max array
# tasks running at once). When submitted via run_all.sh it is overridden by
# --array=0-47%<WRITE_GATE_CONCURRENCY>; set WRITE_GATE_CONCURRENCY on the
# run_all.sh command line to change it (e.g. WRITE_GATE_CONCURRENCY=10).
#
# write_gate modes (gates the image-prototype WRITE; prediction rule is fixed
# as clip + 100*image_proto in all variants):
#   baseline            — always write argmax(clip)               (new top-1 baseline)
#   confident           — write iff softmax(clip).top1-top2 >= 0.2
#   agree               — write iff patch_vote_pred(topk20) == argmax(clip)
#   confident_and_agree — write iff both confident AND agree
#
# The agreement signal uses the stateless topk20 CLIP patch vote
# (utils.patch_vote.compute_patch_vote) — NOT the patch_proto bank.
#
# Adapter / config:
#   --method write_gate_pta        (models/write_gate_pta.py)
#   --config configs/write_gate_pta
#   --override write_gate=<mode>   (gate mode varies per array task)
#
# Environment variables for per-sample recording:
#   RECORD_DIR   — output path for records.jsonl (one dir per gate+ds+seed)
#   RESULT_LABEL — method label in output results (encodes the gate mode)
#   RESULT_FILE  — result file for accuracy output
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/records_wg

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=write_gate_pta
CONFIG=configs/write_gate_pta

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2 3 4)
N_SEEDS=${#SEEDS[@]}

GATES=(baseline confident agree confident_and_agree)
GATE_LABELS=(baseline confident agree confident-and-agree)
N_GATES=${#GATES[@]}

N_TOTAL=$((N_GATES * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (write gate, dataset, seed)
# ---------------------------------------------------------------------------
gate_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

GATE=${GATES[$gate_idx]}
GATE_LABEL=${GATE_LABELS[$gate_idx]}
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

RESULT_LABEL="WG-${GATE_LABEL}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs/records_wg/${RESULT_LABEL}"
export RESULT_FILE="outputs/result_write_gate.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  write_gate   : $GATE  (gate_idx=$gate_idx/$N_GATES)"
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
    --override "write_gate=${GATE}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
