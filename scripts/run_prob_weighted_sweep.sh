#!/bin/bash
# ============================================================================
# run_prob_weighted_sweep.sh -- probability-weighted top-1 EMA write study
# (Phase 10, Exp3). See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# No extra CLIP forward passes for any setting (the write weight is scaled by
# w_top1 ** gamma, one scalar per write), so this is a single-stage sweep.
#
# Grid: control (gamma=0.0, bit-identical to the canonical top-1 write) +
#       {g0.5, g1.0, g2.0} = 4 settings x 3 datasets x 4 seeds = 48 runs.
#
# Usage:
#   bash scripts/run_prob_weighted_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_prob_weighted

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)
MAX_PARALLEL=2

RESULT_FILE="outputs/result_prob_weighted.txt"
RECORD_ROOT="outputs/records_prob_weighted"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" gamma="$2" dataset="$3" seed="$4"
    local result_label="PW-${setting_label}-s${seed}"
    local record_label="PW-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method prob_weighted_pta
        --config configs/prob_weighted_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "gamma=${gamma}")

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

# (label, gamma)
SETTINGS=(
    "control|0.0"
    "g0.5|0.5"
    "g1.0|1.0"
    "g2.0|2.0"
)

echo "=== Probability-weighted EMA sweep: 4 settings x 3 datasets x 4 seeds, MAX_PARALLEL=$MAX_PARALLEL ==="
for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label gamma <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$gamma" "$dataset" "$seed"
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "All jobs done. Logs in logs/, raw results in $RESULT_FILE"
fi