#!/bin/bash
#SBATCH --job-name=pta_proxy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-6%3
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_proxy_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_proxy_%x-%A_%a.err

# ============================================================================
# DTD proxy validation + flowers pivot (plan task 3).
#
# 7 jobs, max 3 concurrent (#SBATCH --array=0-6%3), one case branch per task id.
# Each branch invokes python runner.py directly.
#
# PURPOSE
# -------
# Validates that the DTD-based proxy conclusions transfer to the flowers pivot:
#   * Jobs 0,3,4,6  : Combo1 = patch_modulated_pta with the canonical bounded
#                     patch overrides (tanh squash, scale 1.0).
#   * Jobs 0-4      : FULL-dataset per-class records on dtd (reference) and
#                     the flowers pivot, seeds 1 and 2.
#   * Jobs 5-6      : Closed-set subset on dtd restricted to CLASS_FILE,
#                     comparing PTA-CS vs Combo1 on the proxy subset.
#
# JOB TABLE
# ---------
#   id | method               | dataset  | scope  | seed | record dir
#   ---+----------------------+----------+--------+------+-------------------------
#   0  | patch_modulated_pta  | dtd      | full   | 1    | Combo1-full-dtd-s1
#   1  | pta                  | flowers  | full   | 1    | PTA-CS-flowers-s1
#   2  | pta                  | flowers  | full   | 2    | PTA-CS-flowers-s2
#   3  | patch_modulated_pta  | flowers  | full   | 1    | Combo1-flowers-s1
#   4  | patch_modulated_pta  | flowers  | full   | 2    | Combo1-flowers-s2
#   5  | pta                  | dtd      | subset | 1    | subset-PTA-CS-s1
#   6  | patch_modulated_pta  | dtd      | subset | 1    | subset-Combo1-s1
#
# CLASS_FILE SEMANTICS
# --------------------
# - Defaults to outputs/subset_classes_proxy.txt but is overridable via the
#   CLASS_FILE env var (e.g. CLASS_FILE=/tmp/classes.txt sbatch ...).
# - The file is PRODUCED by Task 2 and CONSUMED by this batch. It must exist
#   and be non-empty BEFORE this batch runs — the script fails fast (exit 1)
#   if the file is missing or empty, BEFORE any python invocation.
#
# HOW TO SUBMIT
# -------------
#   sbatch scripts/slurm_proxy_validation.sh
#   (or CLASS_FILE=/path/to/classes.txt sbatch scripts/slurm_proxy_validation.sh)
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p outputs/records_proxy_validation
mkdir -p logs

# ---------------------------------------------------------------------------
# Closed-set class list (env-overridable, produced by Task 2)
# ---------------------------------------------------------------------------
CLASS_FILE=${CLASS_FILE:-outputs/subset_classes_proxy.txt}

if [[ ! -f "$CLASS_FILE" ]]; then
    echo "ERROR: CLASS_FILE not found: $CLASS_FILE" >&2
    echo "ERROR: Task 2 must produce outputs/subset_classes_proxy.txt before this batch is submitted." >&2
    exit 1
fi

if [[ ! -s "$CLASS_FILE" ]]; then
    echo "ERROR: CLASS_FILE is empty: $CLASS_FILE" >&2
    echo "ERROR: Task 2 must write at least one classname per line before this batch is submitted." >&2
    exit 1
fi

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Dispatch: one branch per SLURM_ARRAY_TASK_ID (0-6).
# Every branch sets METHOD/CONFIG/DS_LABEL/SEED/EXP_LABEL plus the RECORD_DIR
# and RESULT_FILE env hooks consumed by utils/records.py.
# ---------------------------------------------------------------------------
METHOD=
CONFIG=
DS_LABEL=
SEED=
EXP_LABEL=
CMD=()

case "${SLURM_ARRAY_TASK_ID}" in

    0)
        # Combo1 reference on the FULL dtd test set (per-class records, seed 1)
        METHOD=patch_modulated_pta
        CONFIG=configs/patch_modulated_pta
        DS_LABEL=full-dtd
        SEED=1
        EXP_LABEL=Combo1-full-dtd-s1
        export RECORD_DIR=outputs/records_proxy_validation/Combo1-full-dtd-s1
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method patch_modulated_pta --config configs/patch_modulated_pta --override fusion.patch_squash=tanh fusion.patch_squash_scale=1.0 --datasets dtd --backbone ViT-B/16 --seed 1)
        ;;

    1)
        # PTA closed-set flowers pivot, seed 1
        METHOD=pta
        CONFIG=configs/PTA
        DS_LABEL=flowers
        SEED=1
        EXP_LABEL=PTA-CS-flowers-s1
        export RECORD_DIR=outputs/records_proxy_validation/PTA-CS-flowers-s1
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method pta --config configs/PTA --datasets oxford_flowers --backbone ViT-B/16 --seed 1)
        ;;

    2)
        # PTA closed-set flowers pivot, seed 2
        METHOD=pta
        CONFIG=configs/PTA
        DS_LABEL=flowers
        SEED=2
        EXP_LABEL=PTA-CS-flowers-s2
        export RECORD_DIR=outputs/records_proxy_validation/PTA-CS-flowers-s2
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method pta --config configs/PTA --datasets oxford_flowers --backbone ViT-B/16 --seed 2)
        ;;

    3)
        # Combo1 flowers pivot, seed 1
        METHOD=patch_modulated_pta
        CONFIG=configs/patch_modulated_pta
        DS_LABEL=flowers
        SEED=1
        EXP_LABEL=Combo1-flowers-s1
        export RECORD_DIR=outputs/records_proxy_validation/Combo1-flowers-s1
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method patch_modulated_pta --config configs/patch_modulated_pta --override fusion.patch_squash=tanh fusion.patch_squash_scale=1.0 --datasets oxford_flowers --backbone ViT-B/16 --seed 1)
        ;;

    4)
        # Combo1 flowers pivot, seed 2
        METHOD=patch_modulated_pta
        CONFIG=configs/patch_modulated_pta
        DS_LABEL=flowers
        SEED=2
        EXP_LABEL=Combo1-flowers-s2
        export RECORD_DIR=outputs/records_proxy_validation/Combo1-flowers-s2
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method patch_modulated_pta --config configs/patch_modulated_pta --override fusion.patch_squash=tanh fusion.patch_squash_scale=1.0 --datasets oxford_flowers --backbone ViT-B/16 --seed 2)
        ;;

    5)
        # PTA closed-set on the proxy subset (dtd restricted to CLASS_FILE), seed 1
        METHOD=pta
        CONFIG=configs/PTA
        DS_LABEL=dtd-subset
        SEED=1
        EXP_LABEL=subset-PTA-CS-s1
        export RECORD_DIR=outputs/records_proxy_validation/subset-PTA-CS-s1
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method pta --config configs/PTA --datasets dtd --class-file "$CLASS_FILE" --backbone ViT-B/16 --seed 1)
        ;;

    6)
        # Combo1 on the proxy subset (dtd restricted to CLASS_FILE), seed 1
        METHOD=patch_modulated_pta
        CONFIG=configs/patch_modulated_pta
        DS_LABEL=dtd-subset
        SEED=1
        EXP_LABEL=subset-Combo1-s1
        export RECORD_DIR=outputs/records_proxy_validation/subset-Combo1-s1
        export RESULT_FILE=outputs/result_proxy_validation.txt
        CMD=(python -u runner.py --method patch_modulated_pta --config configs/patch_modulated_pta --override fusion.patch_squash=tanh fusion.patch_squash_scale=1.0 --datasets dtd --class-file "$CLASS_FILE" --backbone ViT-B/16 --seed 1)
        ;;

    *)
        echo "ERROR: unknown SLURM_ARRAY_TASK_ID '${SLURM_ARRAY_TASK_ID:-<unset>}'. Expected 0-6." >&2
        exit 1
        ;;

esac

# ---------------------------------------------------------------------------
# Dispatch table (per-job stdout for agent verification)
# ---------------------------------------------------------------------------
echo "========================================================================="
echo "  Task ID    : ${SLURM_ARRAY_TASK_ID} / ${SLURM_ARRAY_TASK_MAX}"
echo "  Experiment : ${EXP_LABEL}"
echo "  Method     : ${METHOD}"
echo "  Config     : ${CONFIG}"
echo "  Dataset    : ${DS_LABEL}"
echo "  Seed       : ${SEED}"
echo "  Class file : ${CLASS_FILE}"
echo "  Record dir : ${RECORD_DIR}"
echo "  Result file: ${RESULT_FILE}"
echo "  Node       : $(hostname)"
echo "========================================================================="

export RESULT_LABEL="$EXP_LABEL"
mkdir -p "$RECORD_DIR"

# Run. Exit status propagates (set -e): 0 on success, nonzero on failure so
# sacct / slurm accounting reflects the outcome per array task.
"${CMD[@]}"
