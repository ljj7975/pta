#!/bin/bash
#SBATCH --job-name=pta_reweight
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-47%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_reweight_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_reweight_%x-%A_%a.err

# ============================================================================
# SLURM array job: Part 4b — write-reweight study on the image prototype
#
# 48 tasks = 4 boost levels x 3 datasets x 4 seeds
#
#   boost_idx = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed   = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx    = ds_seed / 4
#   seed_idx  = ds_seed % 4
#
# The "%4" in the #SBATCH --array line is the concurrency limit (max array
# tasks running at once). When submitted via run_all.sh it is overridden by
# --array=0-47%<REWEIGHT_CONCURRENCY>; set REWEIGHT_CONCURRENCY on the
# run_all.sh command line to change it (e.g. REWEIGHT_CONCURRENCY=10).
#
# boost levels (scales the single-class EMA write weight when the image is
# confident AND the topk20 patch vote agrees; write rule is always argmax(clip),
# identical to base PTA — no write is ever dropped):
#   boost=1  — no boost (== base PTA control)
#   boost=2  — 2x weight on confident-and-agree writes
#   boost=5  — 5x weight on confident-and-agree writes
#   boost=10 — 10x weight on confident-and-agree writes
#
# The agreement signal uses the stateless topk20 CLIP patch vote
# (utils.patch_vote.compute_patch_vote) — NOT the patch_proto bank.
#
# Adapter / config:
#   --method reweight_pta        (models/reweight_pta.py)
#   --config configs/reweight_pta
#   --override boost=<B>         (boost level varies per array task)
#
# Environment variables for per-sample recording:
#   RECORD_DIR   — output path for records.jsonl (one dir per boost+ds+seed)
#   RESULT_LABEL — method label in output results (encodes the boost level)
#   RESULT_FILE  — result file for accuracy output
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/records_rw

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
BACKBONE=ViT-B/16
METHOD=reweight_pta
CONFIG=configs/reweight_pta

DATASETS=(dtd oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}
SEEDS=(1 2 3 4)
N_SEEDS=${#SEEDS[@]}

BOOSTS=(1 2 5 10)
N_BOOSTS=${#BOOSTS[@]}

N_TOTAL=$((N_BOOSTS * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (boost, dataset, seed)
# ---------------------------------------------------------------------------
boost_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

BOOST=${BOOSTS[$boost_idx]}
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

RESULT_LABEL="RW-b${BOOST}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs/records_rw/${RESULT_LABEL}"
export RESULT_FILE="outputs/result_reweight.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  boost        : $BOOST  (boost_idx=$boost_idx/$N_BOOSTS)"
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
    --override "boost=${BOOST}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
