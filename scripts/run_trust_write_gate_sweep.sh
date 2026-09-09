#!/bin/bash
# ============================================================================
# run_trust_write_gate_sweep.sh — Study A: write-gate driven by
# view-consistency / prototype-confusability / both. See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# Two stages to respect the different compute cost per signal source:
#   Stage 1 (cheap):  baseline + confusability's {clean, confident_and_clean}
#                      -- no extra CLIP forward passes, MAX_PARALLEL=3
#   Stage 2 (expensive): view's + both's {clean, confident_and_clean}
#                      -- 4x forward passes/sample (n_views=4), MAX_PARALLEL=2
#
# Grid: 7 settings x 3 datasets x 4 seeds = 84 runs (36 cheap, 48 expensive).
#
# Usage:
#   bash scripts/run_trust_write_gate_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_trust_write_gate

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

RESULT_FILE="outputs/result_trust_write_gate.txt"
RECORD_ROOT="outputs/records_trust_write_gate"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" write_gate="$2" signal_source="$3" dataset="$4" seed="$5" max_parallel="$6"
    local result_label="TWG-${setting_label}-s${seed}"
    local record_label="TWG-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method trust_write_gate_pta
        --config configs/trust_write_gate_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "write_gate=${write_gate}" "signal_source=${signal_source}")

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

# (label, write_gate, signal_source)
CHEAP_SETTINGS=(
    "baseline|baseline|view"
    "clean-conf|clean|confusability"
    "cac-conf|confident_and_clean|confusability"
)
EXPENSIVE_SETTINGS=(
    "clean-view|clean|view"
    "cac-view|confident_and_clean|view"
    "clean-both|clean|both"
    "cac-both|confident_and_clean|both"
)

echo "=== Stage 1: cheap settings (confusability + baseline), MAX_PARALLEL=3 ==="
for setting in "${CHEAP_SETTINGS[@]}"; do
    IFS='|' read -r label gate source <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$gate" "$source" "$dataset" "$seed" 3
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "Stage 1 finished."
fi

echo "=== Stage 2: expensive settings (view + both), MAX_PARALLEL=2 ==="
for setting in "${EXPENSIVE_SETTINGS[@]}"; do
    IFS='|' read -r label gate source <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$gate" "$source" "$dataset" "$seed" 2
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "Stage 2 finished. All jobs done. Logs in logs/, raw results in $RESULT_FILE"
fi
