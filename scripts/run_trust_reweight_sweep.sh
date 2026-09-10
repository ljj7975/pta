#!/bin/bash
# ============================================================================
# run_trust_reweight_sweep.sh — Study B: two-sided write reweight driven by
# view-consistency / prototype-confusability / both. See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# Two stages, same rationale as run_trust_write_gate_sweep.sh:
#   Stage 1 (cheap):  (1,1) control [signal-independent] + confusability's
#                      4 (boost, boost_down) points -- MAX_PARALLEL=3
#   Stage 2 (expensive): view's + both's 4 (boost, boost_down) points each
#                      -- 4x forward passes/sample, MAX_PARALLEL=2
#
# Grid: 13 settings x 3 datasets x 4 seeds = 156 runs (60 cheap, 96 expensive).
#
# Usage:
#   bash scripts/run_trust_reweight_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_trust_reweight

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

RESULT_FILE="outputs/result_trust_reweight.txt"
RECORD_ROOT="outputs/records_trust_reweight"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" boost="$2" boost_down="$3" signal_source="$4" dataset="$5" seed="$6" max_parallel="$7"
    local result_label="TRW-${setting_label}-s${seed}"
    local record_label="TRW-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method trust_reweight_pta
        --config configs/trust_reweight_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "boost=${boost}" "boost_down=${boost_down}" "signal_source=${signal_source}")

    if [[ "$RUN" != "--run" ]]; then
        echo "[dry-run] RESULT_LABEL=$result_label RECORD_DIR=$record_dir ${cmd[*]}"
        return
    fi

    wait_for_slot "$max_parallel"
    (
        export RESULT_LABEL="$result_label"
        export RESULT_FILE="$RESULT_FILE"
        export RECORD_DIR="$record_dir"
        echo "[$record_label] ${cmd[*]}" >&2
        "${cmd[@]}" > "$log_file" 2>&1
        echo "[$record_label] done (exit $?)" >&2
    ) &
}

# (label, boost, boost_down, signal_source)
CHEAP_SETTINGS=(
    "control|1.0|1.0|view"
    "b1.0d0.5-conf|1.0|0.5|confusability"
    "b1.0d0.1-conf|1.0|0.1|confusability"
    "b2.0d0.5-conf|2.0|0.5|confusability"
    "b2.0d0.1-conf|2.0|0.1|confusability"
)
EXPENSIVE_SETTINGS=(
    "b1.0d0.5-view|1.0|0.5|view"
    "b1.0d0.1-view|1.0|0.1|view"
    "b2.0d0.5-view|2.0|0.5|view"
    "b2.0d0.1-view|2.0|0.1|view"
    "b1.0d0.5-both|1.0|0.5|both"
    "b1.0d0.1-both|1.0|0.1|both"
    "b2.0d0.5-both|2.0|0.5|both"
    "b2.0d0.1-both|2.0|0.1|both"
)

echo "=== Stage 1: cheap settings (control + confusability), MAX_PARALLEL=3 ==="
for setting in "${CHEAP_SETTINGS[@]}"; do
    IFS='|' read -r label boost down source <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$boost" "$down" "$source" "$dataset" "$seed" 3
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "Stage 1 finished."
fi

echo "=== Stage 2: expensive settings (view + both), MAX_PARALLEL=2 ==="
for setting in "${EXPENSIVE_SETTINGS[@]}"; do
    IFS='|' read -r label boost down source <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$boost" "$down" "$source" "$dataset" "$seed" 2
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "Stage 2 finished. All jobs done. Logs in logs/, raw results in $RESULT_FILE"
fi
