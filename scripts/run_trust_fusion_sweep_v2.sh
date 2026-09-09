#!/bin/bash
# ============================================================================
# run_trust_fusion_sweep_v2.sh — Phase 1b follow-up: downweight the UNTRUSTED
# regime instead of upweighting it.
#
# The v1 sweep (run_trust_fusion_sweep.sh) found near-zero deltas everywhere.
# Post-hoc analysis of the control records showed why: only 4.4% of
# "trusted" samples are ties (where reweighting can even change the argmax),
# but 87.6% of ties ARE "untrusted" -- and v1 only tried upweighting that
# regime (1.5x/2.0x), which just reinforces whichever term already wins the
# tie. This v2 sweep tests the untested direction: pulling image_proto's
# weight DOWN specifically on untrusted (ambiguous/disagreeing) samples, to
# see whether letting CLIP's own vote compete more on exactly the samples
# where reweighting has leverage changes tie-breaking outcomes.
#
# Grid: 3 settings x 3 datasets x 4 seeds = 36 runs, appended to the same
# outputs/result_trust_fusion.txt / outputs/records_trust_fusion/ as v1 so
# scripts/analyze_trust_fusion.py picks them up automatically.
#
# Usage:
#   bash scripts/run_trust_fusion_sweep_v2.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_trust_fusion

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

# (label, tau_scale_trusted, tau_scale_untrusted)
SETTINGS=(
    "downU-0.3|1.0|0.3"
    "downU-0.5|1.0|0.5"
    "downU-0.7|1.0|0.7"
)

RESULT_FILE="outputs/result_trust_fusion.txt"
RECORD_ROOT="outputs/records_trust_fusion"

MAX_PARALLEL="${MAX_PARALLEL:-3}"
RUN="${1:-dry}"

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" trusted="$2" untrusted="$3" dataset="$4" seed="$5"
    local result_label="TFP-${setting_label}-s${seed}"
    local record_label="TFP-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method trust_fusion_pta
        --config configs/trust_fusion_pta
        --datasets "$dataset"
        --backbone "$BACKBONE"
        --seed "$seed"
        --override "fusion.tau_scale_trusted=${trusted}" "fusion.tau_scale_untrusted=${untrusted}")

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
    IFS='|' read -r label trusted untrusted <<< "$setting"
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            run_job "$label" "$trusted" "$untrusted" "$dataset" "$seed"
        done
    done
done

if [[ "$RUN" == "--run" ]]; then
    wait
    echo "All jobs finished. Logs in logs/, raw results in $RESULT_FILE"
fi
