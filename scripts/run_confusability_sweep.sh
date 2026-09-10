#!/bin/bash
# ============================================================================
# run_confusability_sweep.sh — Phase 2b: prototype confusability write-freeze
# sweep. See /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# freeze_percentile=100 is the control (in-grid, must reproduce base PTA).
# Lower values freeze classes more aggressively. Grid widened past the
# plan's original {100,90,80,70} with a gentler 95 point after a single-seed
# smoke test showed p=90 already regresses dtd by -5.4pp -- checking whether
# a much gentler freeze still regresses or whether there's a sweet spot.
#
# Grid: 5 settings x 3 datasets x 4 seeds = 60 runs.
#
# Usage:
#   bash scripts/run_confusability_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_confusability

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

# (label, freeze_percentile)
SETTINGS=(
    "control|100.0"
    "p95|95.0"
    "p90|90.0"
    "p80|80.0"
    "p66|66.0"
)

RESULT_FILE="outputs/result_confusability.txt"
RECORD_ROOT="outputs/records_confusability"

MAX_PARALLEL="${MAX_PARALLEL:-3}"
RUN="${1:-dry}"

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" freeze_pct="$2" dataset="$3" seed="$4"
    local result_label="CGP-${setting_label}-s${seed}"
    local record_label="CGP-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method confusability_gated_pta
        --config configs/confusability_gated_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "freeze_percentile=${freeze_pct}")

    if [[ "$RUN" != "--run" ]]; then
        echo "[dry-run] RESULT_LABEL=$result_label RECORD_DIR=$record_dir ${cmd[*]}"
        return
    fi

    wait_for_slot
    (
        export RESULT_LABEL="$result_label"
        export RESULT_FILE="$RESULT_FILE"
        export RECORD_DIR="$record_dir"
        echo "[$record_label] ${cmd[*]}" >&2
        "${cmd[@]}" > "$log_file" 2>&1
        echo "[$record_label] done (exit $?)" >&2
    ) &
}

for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label freeze_pct <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$freeze_pct" "$dataset" "$seed"
        done
    done
done

if [[ "$RUN" == "--run" ]]; then
    wait
    echo "All jobs finished. Logs in logs/, raw results in $RESULT_FILE"
fi
