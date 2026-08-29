#!/bin/bash
#SBATCH --job-name=pta_down
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-59%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_down_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_down_%x-%A_%a.err

# ============================================================================
# SLURM array job: Part 4b (down-weight) — two-sided write re-weighting +
# convergence-speed tracking on the image prototype.
#
# 60 tasks = 5 (boost_up x boost_down) settings x 3 datasets x 4 seeds
#
#   set_idx  = SLURM_ARRAY_TASK_ID / (N_DS * 4)
#   ds_seed  = SLURM_ARRAY_TASK_ID % (N_DS * 4)
#   ds_idx   = ds_seed / 4
#   seed_idx = ds_seed % 4
#
# The "%4" in the #SBATCH --array line is the concurrency limit (max array
# tasks running at once). When submitted via run_all.sh it is overridden by
# --array=0-59%<REWEIGHT5_CONCURRENCY>; set REWEIGHT5_CONCURRENCY on the
# run_all.sh command line to change it (e.g. REWEIGHT5_CONCURRENCY=10).
#
# Mechanism (models/reweight_pta.py): write rule identical to base PTA — always
# write argmax(clip) into the image prototype (single-class EMA), never drop.
# Only the write weight is re-scaled along the trust axis:
#   trusted   (confident AND topk20 vote agrees) -> weight x boost       (>1)
#   untrusted (ambiguous OR vote disagrees)      -> weight x boost_down  (<1)
# Trusted = clip margin >= conf_margin_thresh AND patch_vote_pred == argmax(clip).
# (boost=1, boost_down=1) == base PTA control.
#
# Settings (up, down):
#   (1.0, 1.0)  base PTA control
#   (1.0, 0.5)  down-weight only, mild
#   (1.0, 0.1)  down-weight only, strong
#   (2.0, 0.5)  combined, mild
#   (2.0, 0.1)  combined, strong
#
# Convergence: every run writes $RECORD_DIR/convergence.json with the per-sample
# online-accuracy trajectory, acc@fractions, time-to-80%-of-final, and norm-AUC
# (utils/records.write_convergence) — the evidence for whether re-weighting
# changes how fast adaptation converges, not just where it lands.
#
# Environment variables:
#   RECORD_DIR   — outputs/records_dw/{label}
#   RESULT_LABEL — encodes (up,down) + dataset + seed
#   RESULT_FILE  — outputs/result_reweight_down.txt
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs
mkdir -p outputs/records_dw

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

# (boost_up, boost_down) settings — index-aligned
BOOST_UP=(1.0 1.0 1.0 2.0 2.0)
BOOST_DOWN=(1.0 0.5 0.1 0.5 0.1)
N_SETS=${#BOOST_UP[@]}

N_TOTAL=$((N_SETS * N_DS * N_SEEDS))

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Map task ID -> (setting, dataset, seed)
# ---------------------------------------------------------------------------
set_idx=$((SLURM_ARRAY_TASK_ID / (N_DS * N_SEEDS)))
ds_seed=$((SLURM_ARRAY_TASK_ID % (N_DS * N_SEEDS)))
ds_idx=$((ds_seed / N_SEEDS))
seed_idx=$((ds_seed % N_SEEDS))

BOOST=${BOOST_UP[$set_idx]}
BOOST_DOWN=${BOOST_DOWN[$set_idx]}
DATASET=${DATASETS[$ds_idx]}
SEED=${SEEDS[$seed_idx]}

# Label encodes the full (up, down) pair, e.g. DW-u1.0-d0.5-dtd-s1
RESULT_LABEL="DW-u${BOOST}-d${BOOST_DOWN}-${DATASET}-s${SEED}"

export RESULT_LABEL="$RESULT_LABEL"
export RECORD_DIR="outputs/records_dw/${RESULT_LABEL}"
export RESULT_FILE="outputs/result_reweight_down.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl" "$RECORD_DIR/convergence.json" "$RECORD_DIR/summary.json"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID      : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  boost_up     : $BOOST  (set_idx=$set_idx/$N_SETS)"
echo "  boost_down   : $BOOST_DOWN"
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
    --override "boost=${BOOST}" "boost_down=${BOOST_DOWN}"
    --datasets "$DATASET"
    --backbone "$BACKBONE"
    --seed "$SEED")

echo "  Command    : ${CMD[*]}"
"${CMD[@]}"
