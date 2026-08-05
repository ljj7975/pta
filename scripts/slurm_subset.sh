#!/bin/bash
#SBATCH --job-name=pta_subset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-8
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_subset_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_subset_%x-%A_%a.err

# ============================================================================
# Closed-set subset benchmark (9 full DTD runs: 3 methods x 3 seeds) for the
# difficult-classes validation. Filters dtd's test set down to the difficult
# classes listed in CLASS_FILE (one classname per line) via runner.py's
# --class-file flag (build_subset_test_data_loader, T2), records per-batch
# per-class stats into outputs/records_subset/<LABEL>/ (RECORD_DIR hook) and
# appends the final accuracy to outputs/result_subset.txt (RESULT_FILE hook).
#
# CLASS_FILE SEMANTICS
# --------------------
# - Defaults to outputs/subset_classes.txt but is overridable via the
#   CLASS_FILE env var (e.g. CLASS_FILE=/tmp/my_classes.txt sbatch ...).
# - The file is PRODUCED by Task 11 (analyze_records.py select --k ...) and
#   CONSUMED by Task 13 (sbatch submission of this script). It must exist and
#   be non-empty BEFORE this batch runs — the script fails fast (exit 1) if
#   the file is missing or empty, BEFORE any python invocation.
# - NO class names are hardcoded in this script; the subset is defined solely
#   by CLASS_FILE.
#
# HOW TO USE
# ----------
# 1. Ensure outputs/subset_classes.txt exists (Task 11) or export CLASS_FILE.
#
# 2. Update --array to match the number of active experiments x datasets:
#
#      N_EXP = count of uncommented exp() lines   (currently 9)
#      N_DS  = ${#DATASETS[@]}                     (currently 1: dtd)
#      --array=0-$(( N_EXP * N_DS - 1 ))
#
#    Example: all 9 active → --array=0-8
#
# 3. Labels are configurable per-line for re-running with different names.
#    Each label gets a unique RECORD_DIR (outputs/records_subset/$LABEL).
#
# Task mapping:
#   exp_idx = SLURM_ARRAY_TASK_ID / N_DS
#   ds_idx  = SLURM_ARRAY_TASK_ID % N_DS
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p outputs/records_subset
mkdir -p logs

# ---------------------------------------------------------------------------
# Closed-set class list (env-overridable)
# ---------------------------------------------------------------------------
CLASS_FILE=${CLASS_FILE:-outputs/subset_classes.txt}

if [[ ! -f "$CLASS_FILE" ]]; then
    echo "ERROR: CLASS_FILE not found: $CLASS_FILE"
    echo "ERROR: The subset class list does not exist yet. Task 11 (analyze_records.py select) must produce it before this batch is submitted (Task 13)."
    exit 1
fi

if [[ ! -s "$CLASS_FILE" ]]; then
    echo "ERROR: CLASS_FILE is empty: $CLASS_FILE"
    echo "ERROR: Task 11 must write at least one classname per line before this batch is submitted (Task 13)."
    exit 1
fi

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Datasets (shared across all experiments)
# ---------------------------------------------------------------------------
DATASETS=(dtd)
N_DS=${#DATASETS[@]}

# ---------------------------------------------------------------------------
# Experiment registry
#
# exp METHOD CONFIG_DIR CLIP_MODEL CLIP_CHECKPOINT LABEL SEED [OVERRIDE]
#
# Comment/uncomment individual lines to enable/disable experiments.
# OVERRIDE is a single top-level KEY=VALUE (e.g. proto_alpha_max=1.0).
# NOTE: multi_gate must be overridden numerically (multi_gate=1), NOT as
# "true" — bool("false") == True in Python, so a string override is a footgun.
# ---------------------------------------------------------------------------
_METHODS=()
_CONFIG_DIRS=()
_CLIP_MODELS=()
_CLIP_CHECKPOINTS=()
_EXP_LABELS=()
_EXP_SEEDS=()
_OVERRIDES=()

exp() {
    _METHODS+=("$1")
    _CONFIG_DIRS+=("$2")
    _CLIP_MODELS+=("$3")
    _CLIP_CHECKPOINTS+=("$4")
    _EXP_LABELS+=("$5")
    _EXP_SEEDS+=("$6")
    _OVERRIDES+=("${7:-}")
}

# ── Subset runs: 3 methods x 3 seeds (1, 2, 3) on the CLASS_FILE subset ──
exp patch_modulated_pta configs/patch_modulated_pta clip_surgery "" "PatchModPTA-CS-sub-s1" 1 ""
exp patch_modulated_pta configs/patch_modulated_pta clip_surgery "" "PatchModPTA-CS-sub-s2" 2 ""
exp patch_modulated_pta configs/patch_modulated_pta clip_surgery "" "PatchModPTA-CS-sub-s3" 3 ""
exp pta                 configs/PTA                   clip_surgery "" "PTA-CS-sub-s1"        1 ""
exp pta                 configs/PTA                   clip_surgery "" "PTA-CS-sub-s2"        2 ""
exp pta                 configs/PTA                   clip_surgery "" "PTA-CS-sub-s3"        3 ""
exp zeroshot            configs/PTA                   clip_surgery "" "ZeroShot-CS-sub-s1"   1 ""
exp zeroshot            configs/PTA                   clip_surgery "" "ZeroShot-CS-sub-s2"   2 ""
exp zeroshot            configs/PTA                   clip_surgery "" "ZeroShot-CS-sub-s3"   3 ""

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
METHODS=("${_METHODS[@]}")
CONFIG_DIRS=("${_CONFIG_DIRS[@]}")
CLIP_MODELS=("${_CLIP_MODELS[@]}")
CLIP_CHECKPOINTS=("${_CLIP_CHECKPOINTS[@]}")
EXP_LABELS=("${_EXP_LABELS[@]}")
EXP_SEEDS=("${_EXP_SEEDS[@]}")
OVERRIDES=("${_OVERRIDES[@]}")

N_EXP=${#METHODS[@]}
N_TOTAL=$((N_EXP * N_DS))

if (( N_EXP == 0 )); then
    echo "ERROR: No experiments enabled. Uncomment at least one exp() line above."
    exit 1
fi

# Guard: skip if this task ID exceeds the experiment grid
if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

exp_idx=$((SLURM_ARRAY_TASK_ID / N_DS))
ds_idx=$((SLURM_ARRAY_TASK_ID % N_DS))

METHOD=${METHODS[$exp_idx]}
CONFIG=${CONFIG_DIRS[$exp_idx]}
CLIP_MODEL=${CLIP_MODELS[$exp_idx]}
CLIP_CKPT=${CLIP_CHECKPOINTS[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL=${EXP_LABELS[$exp_idx]}
SEED=${EXP_SEEDS[$exp_idx]}
OVERRIDE=${OVERRIDES[$exp_idx]}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL ($exp_idx / $N_EXP)"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  CLIP model : $CLIP_MODEL"
echo "  Dataset    : $DATASET  (ds_idx=$ds_idx)"
echo "  Seed       : $SEED"
echo "  Override   : ${OVERRIDE:-<none>}"
echo "  CLASS_FILE : $CLASS_FILE"
echo "  RECORD_DIR : outputs/records_subset/$EXP_LABEL"
echo "  RESULT_FILE: outputs/result_subset.txt"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="$EXP_LABEL"
export RECORD_DIR="outputs/records_subset/$EXP_LABEL"
export RESULT_FILE="outputs/result_subset.txt"
export SEED="$SEED"

mkdir -p "$RECORD_DIR"

# Build command — only pass --clip-checkpoint when non-empty
CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --clip-model "$CLIP_MODEL"
    --datasets "$DATASET"
    --backbone ViT-B/16
    --seed "$SEED"
    --class-file "$CLASS_FILE")

if [[ -n "$CLIP_CKPT" ]]; then
    CMD+=(--clip-checkpoint "$CLIP_CKPT")
fi

if [[ -n "$OVERRIDE" ]]; then CMD+=(--override "$OVERRIDE"); fi

"${CMD[@]}"
