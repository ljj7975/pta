#!/bin/bash
# ============================================================================
# run_bilateral_drift_sweep.sh -- bilateral-drift prototype-repulsion study
# (Phase 10, Exp4). See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# No extra CLIP forward passes for any setting (drift tracking is one dot
# product per write, repulsion reuses the confusability matmul), so this is
# a single-stage sweep like Phase 9's.
#
# Grid: control (repulsion_lr=0.0) + bilateral_drift x {0.02, 0.05, 0.1}
#       = 4 settings x 3 datasets x 4 seeds = 48 runs.
#
# bilateral_drift fires iff confusable AND floor both pass AND
# (drift_ema[top1] > thresh OR drift_ema[other] > thresh) -- the OR-condition
# is strictly more permissive than Phase 9's top1-only drift_confusable.
#
# Usage:
#   bash scripts/run_bilateral_drift_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_bilateral_drift

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)
MAX_PARALLEL=2

RESULT_FILE="outputs/result_bilateral_drift.txt"
RECORD_ROOT="outputs/records_bilateral_drift"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" repulsion_lr="$2" trigger="$3" dataset="$4" seed="$5"
    local result_label="BD-${setting_label}-s${seed}"
    local record_label="BD-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method drift_gated_repulsion_pta
        --config configs/drift_gated_repulsion_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "repulsion_lr=${repulsion_lr}" "trigger=${trigger}")

    if [[ "$RUN" != "--run" ]]; then
        echo "[dry-run] RESULT_LABEL=$result_label RECORD_DIR=$record_dir ${cmd[*]}"
        return
    fi

    wait_for_slot "$MAX_PARALLEL"
    (
        export RESULT_LABEL="$result_label"
        export RESULT_FILE="$RESULT_FILE"
        export RECORD_DIR="$record_dir"
        echo "[$record_label] ${cmd[*]}" >&2
        "${cmd[@]}" > "$log_file" 2>&1
        echo "[$record_label] done (exit $?)" >&2
    ) &
}

# (label, repulsion_lr, trigger)
SETTINGS=(
    "control|0.0|bilateral_drift"
    "lr0.02-bd|0.02|bilateral_drift"
    "lr0.05-bd|0.05|bilateral_drift"
    "lr0.1-bd|0.1|bilateral_drift"
)

echo "=== Bilateral-drift repulsion sweep: 4 settings x 3 datasets x 4 seeds, MAX_PARALLEL=$MAX_PARALLEL ==="
for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label lr trigger <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$lr" "$trigger" "$dataset" "$seed"
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "All jobs done. Logs in logs/, raw results in $RESULT_FILE"
fi
