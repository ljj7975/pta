#!/bin/bash
# ============================================================================
# run_repulsion_sweep.sh -- prototype-repulsion study (geometry correction,
# not write gating/reweighting). See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# No extra CLIP forward passes for any setting (repulsion is one [C,C]
# matmul reused from the confusability computation), so this is a
# single-stage sweep, unlike the two-stage trust-gate/reweight sweeps.
#
# Grid: control (repulsion_lr=0.0) + {0.02, 0.05, 0.1} x {always, confusable}
#       = 7 settings x 3 datasets x 4 seeds = 84 runs.
#
# Usage:
#   bash scripts/run_repulsion_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_repulsion

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)
MAX_PARALLEL=4

RESULT_FILE="outputs/result_repulsion.txt"
RECORD_ROOT="outputs/records_repulsion"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" repulsion_lr="$2" trigger="$3" dataset="$4" seed="$5"
    local result_label="REP-${setting_label}-s${seed}"
    local record_label="REP-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method repulsive_pta
        --config configs/repulsive_pta
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
    "control|0.0|confusable"
    "lr0.02-conf|0.02|confusable"
    "lr0.05-conf|0.05|confusable"
    "lr0.1-conf|0.1|confusable"
    "lr0.02-always|0.02|always"
    "lr0.05-always|0.05|always"
    "lr0.1-always|0.1|always"
)

echo "=== Repulsion sweep: 7 settings x 3 datasets x 4 seeds, MAX_PARALLEL=$MAX_PARALLEL ==="
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
