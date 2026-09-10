#!/bin/bash
# ============================================================================
# run_text_anchored_sweep.sh -- text-anchored EMA study (Phase 10, Exp 2).
# See /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# The top-1 EMA write anchors the written feature toward the current refined
# text row of the predicted class:
#
#   blend = anchor_mix * image_feature + (1.0 - anchor_mix) * text_features[top1]
#
# anchor_mix=1.0 is the canonical write (bit-identical) and must reproduce the
# full-run reference (dtd 46.85 / flowers 74.22 / pets 90.75). No extra CLIP
# forward passes for any setting, so this is a single-stage sweep.
#
# Grid: control (anchor_mix=1.0) + {0.99, 0.95, 0.90, 0.80}
#       = 5 settings x 3 datasets x 4 seeds = 60 runs.
#
# MAX_PARALLEL=2: two sibling sweeps (floor-ablation, prob-weighted) run
# concurrently on the same single GPU (RTX 5070 Ti 16GB); each run peaks
# ~2.2GB, 6 concurrent jobs total is the safe ceiling.
#
# Usage:
#   bash scripts/run_text_anchored_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_text_anchored

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)
MAX_PARALLEL=2

RESULT_FILE="outputs/result_text_anchored.txt"
RECORD_ROOT="outputs/records_text_anchored"

RUN="${1:-dry}"

wait_for_slot() {
    local max="$1"
    while (( $(jobs -rp | wc -l) >= max )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" anchor_mix="$2" dataset="$3" seed="$4"
    local result_label="TA-${setting_label}-s${seed}"
    local record_label="TA-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method text_anchored_pta
        --config configs/text_anchored_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "anchor_mix=${anchor_mix}")

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

# (label, anchor_mix)
SETTINGS=(
    "control|1.0"
    "a0.99|0.99"
    "a0.95|0.95"
    "a0.90|0.90"
    "a0.80|0.80"
)

echo "=== Text-anchored sweep: 5 settings x 3 datasets x 4 seeds, MAX_PARALLEL=$MAX_PARALLEL ==="
for setting in "${SETTINGS[@]}"; do
    IFS='|' read -r label mix <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$mix" "$dataset" "$seed"
        done
    done
done
if [[ "$RUN" == "--run" ]]; then
    wait
    echo "All jobs done. Logs in logs/, raw results in $RESULT_FILE"
fi