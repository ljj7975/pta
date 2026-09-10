#!/bin/bash
# ============================================================================
# run_drift_gated_repulsion_sweep.sh -- drift-gated prototype-repulsion
# study (Phase 9 follow-up to Phase 8's repulsion study). See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# No extra CLIP forward passes for any setting (drift tracking is one dot
# product per write, repulsion reuses the confusability matmul), so this is
# a single-stage sweep like Phase 8's.
#
# Grid: control (repulsion_lr=0.0) + {drift_confusable, stable_confusable}
#       x {0.02, 0.05, 0.1} = 7 settings x 3 datasets x 4 seeds = 84 runs.
#
# Usage:
#   bash scripts/run_drift_gated_repulsion_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_drift_gated_repulsion

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)
MAX_PARALLEL=4

RESULT_FILE="outputs/result_drift_gated_repulsion.txt"
RECORD_ROOT="outputs/records_drift_gated_repulsion"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" repulsion_lr="$2" trigger="$3" dataset="$4" seed="$5"
    local result_label="DGR-${setting_label}-s${seed}"
    local record_label="DGR-${setting_label}-${dataset}-s${seed}"
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
    "control|0.0|drift_confusable"
    "lr0.02-drift|0.02|drift_confusable"
    "lr0.05-drift|0.05|drift_confusable"
    "lr0.1-drift|0.1|drift_confusable"
    "lr0.02-stable|0.02|stable_confusable"
    "lr0.05-stable|0.05|stable_confusable"
    "lr0.1-stable|0.1|stable_confusable"
)

echo "=== Drift-gated repulsion sweep: 7 settings x 3 datasets x 4 seeds, MAX_PARALLEL=$MAX_PARALLEL ==="
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
