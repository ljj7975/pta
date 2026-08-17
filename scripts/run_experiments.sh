#!/bin/bash
# ============================================================================
# run_experiments.sh — Run all method/dataset/seed combinations for the paper.
#
# Grid: 5 methods × 5 datasets × 5 seeds = 125 total runs.
#
# Methods:
#   1. pta                                — image-level baseline
#   2. patch_modulated_pta                — default ProtoAlphaFusion
#   3. patch_modulated_pta --override fusion.type=QualityGatedFusion
#   4. patch_modulated_pta --override fusion.type=MajorityVoteFusion
#   5. patch_modulated_pta --override fusion.type=AgreementGateFusion
#
# Usage:
#   bash scripts/run_experiments.sh --dry-run              # print all 125 cmds
#   bash scripts/run_experiments.sh --method pta            # dry-run for one method
#   bash scripts/run_experiments.sh --run --launcher sbatch # submit all via sbatch array
#   bash scripts/run_experiments.sh --run --launcher srun   # run sequentially via srun
#   bash scripts/run_experiments.sh --run --concurrency 3   # run 3 jobs at a time
#
# Options:
#   --dry-run          Print commands without executing (default)
#   --run              Execute commands (requires cluster or local GPU)
#   --launcher TYPE    Force launcher: sbatch|srun|direct (auto-detected if omitted)
#   --concurrency N    Max parallel jobs for sbatch array (default: unlimited)
#   --method METHOD    Filter to a single method (match by label substring)
#   --dataset DATASET  Filter to a single dataset
#   --seed SEED        Filter to a single seed
#   --backbone NAME    Backbone (default: ViT-B/16)
#   --partition NAME   SLURM partition (default: auto)
#   --time HH:MM:SS   Job time limit (default: 05:00:00)
# ============================================================================

set -euo pipefail

# ── Resolve project root (same convention as other slurm scripts) ─────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs

# ── Parse arguments ──────────────────────────────────────────────────────
DRY_RUN=true
RUN_MODE=false
LAUNCHER=""
CONCURRENCY=""
FILTER_METHOD=""
FILTER_DATASET=""
FILTER_SEED=""
BACKBONE="ViT-B/16"
PARTITION=""
TIME_LIMIT="05:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=true;  RUN_MODE=false; shift ;;
        --run)         DRY_RUN=false; RUN_MODE=true;  shift ;;
        --launcher)    LAUNCHER="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --method)      FILTER_METHOD="$2";  shift 2 ;;
        --dataset)     FILTER_DATASET="$2"; shift 2 ;;
        --seed)        FILTER_SEED="$2";    shift 2 ;;
        --backbone)    BACKBONE="$2";       shift 2 ;;
        --partition)   PARTITION="$2";      shift 2 ;;
        --time)        TIME_LIMIT="$2";     shift 2 ;;
        -h|--help)
            sed -n '2,/^# ====/{ /^# /s/^# //p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Detect launcher if not specified ─────────────────────────────────────
if [[ -z "$LAUNCHER" ]]; then
    if command -v srun &>/dev/null; then
        LAUNCHER="srun"
    elif command -v sbatch &>/dev/null; then
        LAUNCHER="sbatch"
    else
        LAUNCHER="direct"
    fi
fi

echo "========================================================================"
echo "  Experiment Runner"
echo "  Project dir : $PROJECT_DIR"
echo "  Launcher    : $LAUNCHER"
echo "  Dry run     : $DRY_RUN"
echo "  Concurrency : ${CONCURRENCY:-unlimited}"
echo "  Backbone    : $BACKBONE"
echo "  Time limit  : $TIME_LIMIT"
if [[ -n "$FILTER_METHOD" ]]; then echo "  Method filter : $FILTER_METHOD"; fi
if [[ -n "$FILTER_DATASET" ]]; then echo "  Dataset filter: $FILTER_DATASET"; fi
if [[ -n "$FILTER_SEED" ]]; then echo "  Seed filter   : $FILTER_SEED"; fi
if [[ -n "$PARTITION" ]]; then echo "  Partition   : $PARTITION"; fi
echo "========================================================================"

# ── Experiment registry ──────────────────────────────────────────────────
#   METHOD_LABEL  METHOD  CONFIG_DIR  OVERRIDE
METHODS=(
    "PTA|pta|configs/PTA|"
    "PatchModPTA|patch_modulated_pta|configs/patch_modulated_pta|"
    "PatchModPTA-QGated|patch_modulated_pta|configs/patch_modulated_pta|fusion.type=QualityGatedFusion"
    "PatchModPTA-MVote|patch_modulated_pta|configs/patch_modulated_pta|fusion.type=MajorityVoteFusion"
    "PatchModPTA-AGate|patch_modulated_pta|configs/patch_modulated_pta|fusion.type=AgreementGateFusion"
)

DATASETS=(caltech101 dtd eurosat fgvc food101)
SEEDS=(1 2 3 4 5)

# ── Count total and filtered ─────────────────────────────────────────────
TOTAL=0
for entry in "${METHODS[@]}"; do
    IFS='|' read -r LABEL METHOD CMD_CONFIG OVERRIDE <<< "$entry"

    if [[ -n "$FILTER_METHOD" && "$METHOD" != "$FILTER_METHOD" ]]; then
        continue
    fi

    for DS in "${DATASETS[@]}"; do
        if [[ -n "$FILTER_DATASET" && "$DS" != "$FILTER_DATASET" ]]; then
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            if [[ -n "$FILTER_SEED" && "$SEED" != "$FILTER_SEED" ]]; then
                continue
            fi
            TOTAL=$((TOTAL + 1))
        done
    done
done

echo "  Total runs  : $TOTAL"
echo "========================================================================"
echo ""

# ── Build experiment list ────────────────────────────────────────────────
EXPERIMENTS=()

for entry in "${METHODS[@]}"; do
    IFS='|' read -r LABEL METHOD CMD_CONFIG OVERRIDE <<< "$entry"

    if [[ -n "$FILTER_METHOD" && "$METHOD" != "$FILTER_METHOD" ]]; then
        continue
    fi

    for DS in "${DATASETS[@]}"; do
        if [[ -n "$FILTER_DATASET" && "$DS" != "$FILTER_DATASET" ]]; then
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            if [[ -n "$FILTER_SEED" && "$SEED" != "$FILTER_SEED" ]]; then
                continue
            fi

            LABEL_FULL="${LABEL}-${DS}-s${SEED}"
            RECORD_DIR="outputs/records/${LABEL_FULL}"

            PY_CMD="python -u runner.py --method ${METHOD} --config ${CMD_CONFIG} --datasets ${DS} --backbone ${BACKBONE} --seed ${SEED}"
            [[ -n "$OVERRIDE" ]] && PY_CMD+=" --override ${OVERRIDE}"

            FULL_CMD="export RECORD_DIR='${RECORD_DIR}' && mkdir -p '${RECORD_DIR}' && rm -f '${RECORD_DIR}/records.jsonl' && ${PY_CMD}"

            EXPERIMENTS+=("${LABEL_FULL}|${FULL_CMD}")
        done
    done
done

RUN_COUNT=${#EXPERIMENTS[@]}

# ── Execute or submit ────────────────────────────────────────────────────
if [[ "$DRY_RUN" == true ]]; then
    for i in "${!EXPERIMENTS[@]}"; do
        IFS='|' read -r LABEL CMD <<< "${EXPERIMENTS[$i]}"
        echo "[$((i + 1))/${RUN_COUNT}] ${LABEL}"
        echo "  ${CMD}"
        echo ""
    done

else
    if [[ "$LAUNCHER" == "sbatch" ]]; then
        # Generate sbatch array script
        ARRAY_SCRIPT="logs/array_job.sh"
        cat > "$ARRAY_SCRIPT" << 'ARRAY_HEADER'
#!/bin/bash
#SBATCH --job-name=exp_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/exp_%x_%a.out
#SBATCH --error=logs/exp_%x_%a.err
ARRAY_HEADER

        echo "#SBATCH --time=${TIME_LIMIT}" >> "$ARRAY_SCRIPT"
        [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=${PARTITION}" >> "$ARRAY_SCRIPT"

        if [[ -n "$CONCURRENCY" ]]; then
            echo "#SBATCH --array=1-${RUN_COUNT}%${CONCURRENCY}" >> "$ARRAY_SCRIPT"
        else
            echo "#SBATCH --array=1-${RUN_COUNT}" >> "$ARRAY_SCRIPT"
        fi

        echo "EXP_COUNT=${RUN_COUNT}" >> "$ARRAY_SCRIPT"
        echo "" >> "$ARRAY_SCRIPT"
        echo "EXP_LIST=(" >> "$ARRAY_SCRIPT"

        for exp in "${EXPERIMENTS[@]}"; do
            echo "  \"${exp}\"" >> "$ARRAY_SCRIPT"
        done

        echo ")" >> "$ARRAY_SCRIPT"
        cat >> "$ARRAY_SCRIPT" << 'ARRAY_BODY'

IDX=$((SLURM_ARRAY_TASK_ID - 1))
IFS='|' read -r LABEL CMD <<< "${EXP_LIST[$IDX]}"

echo "========================================================================"
echo "  [${SLURM_ARRAY_TASK_ID}/${EXP_COUNT}] ${LABEL}"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"

eval "${CMD}"
ARRAY_BODY

        JOB_ID=$(sbatch "$ARRAY_SCRIPT" | grep -oP 'Submitted batch job \K\d+')
        echo "========================================================================"
        echo "  Submitted array job: ${JOB_ID}"
        echo "  Tasks: 1-${RUN_COUNT}${CONCURRENCY:+%${CONCURRENCY}}"
        echo "  Script: ${ARRAY_SCRIPT}"
        echo "========================================================================"
        echo ""
        echo "Monitor with: squeue -j ${JOB_ID}"

    elif [[ "$LAUNCHER" == "srun" ]]; then
        for i in "${!EXPERIMENTS[@]}"; do
            IFS='|' read -r LABEL CMD <<< "${EXPERIMENTS[$i]}"
            echo "========================================================================"
            echo "  [$((i + 1))/${RUN_COUNT}] ${LABEL}"
            echo "========================================================================"

            srun --job-name="exp_${LABEL}" \
                 --time="${TIME_LIMIT}" \
                 bash -c "$CMD"
            echo "  Completed."
            echo ""
        done

    else
        for i in "${!EXPERIMENTS[@]}"; do
            IFS='|' read -r LABEL CMD <<< "${EXPERIMENTS[$i]}"
            echo "========================================================================"
            echo "  [$((i + 1))/${RUN_COUNT}] ${LABEL}"
            echo "========================================================================"

            eval "$CMD"
            echo "  Completed."
            echo ""
        done
    fi
fi

echo "Done. ${RUN_COUNT} experiment(s) processed."
