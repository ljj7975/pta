#!/bin/bash
# ============================================================================
# run_view_consistency_sweep.sh — Phase 3b: multi-view-consistency
# trust-adaptive fusion sweep. See
# /home/brandon/.claude/plans/toasty-pondering-eagle.md.
#
# Re-sweeps the EXACT (tau_scale_trusted, tau_scale_untrusted) grid used in
# the completed patch-vote-based trust_fusion_pta study
# (outputs/trust_fusion_report.md) for direct comparability -- only the
# trust-signal source differs (multi-view agreement instead of patch vote).
#
# NOTE: each sample now costs n_views=4 forward passes (vs. 1 for the
# original trust_fusion study), so this uses MAX_PARALLEL=2 by default
# (not 3) to stay within the 16GB/no-slurm GPU budget.
#
# Grid: 7 settings x 3 datasets x 4 seeds = 84 runs.
#
# Usage:
#   bash scripts/run_view_consistency_sweep.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_view_consistency

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

# (label, tau_scale_trusted, tau_scale_untrusted) -- identical grid to
# scripts/run_trust_fusion_sweep.sh
SETTINGS=(
    "control|1.0|1.0"
    "downT-0.5|0.5|1.0"
    "downT-0.3|0.3|1.0"
    "upU-1.5|1.0|1.5"
    "upU-2.0|1.0|2.0"
    "both-mod|0.5|1.5"
    "both-agg|0.3|2.0"
)

RESULT_FILE="outputs/result_view_consistency.txt"
RECORD_ROOT="outputs/records_view_consistency"

MAX_PARALLEL="${MAX_PARALLEL:-2}"
RUN="${1:-dry}"

wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n
    done
}

run_job() {
    local setting_label="$1" trusted="$2" untrusted="$3" dataset="$4" seed="$5"
    local result_label="VCP-${setting_label}-s${seed}"
    local record_label="VCP-${setting_label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    local cmd=(python -u runner.py
        --method view_consistency_pta
        --config configs/view_consistency_pta
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
