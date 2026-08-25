#!/bin/bash
# ============================================================================
# Non-slurm local runner: reproduce, on the 5 DEV datasets (dtd, eurosat,
# fgvc, oxford_flowers, oxford_pets):
#
#   1. "best"     — patch_modulated_pta, write_source=pta, write_rule=ratio,
#                    fusion.type=MajorityVoteFusion.
#                    Best single setting from the write-rule x write-source
#                    sweep (see results_dev_writerule.md): 60.68% avg.
#
#   2. "baseline" — plain PTA (models/pta.py, configs/PTA): image-level only,
#                    no patch-level evidence, no quality gate. This is the
#                    "no quality gate setting of PTA (i.e. no patch-level
#                    stuffs)" control used as the PTA row in
#                    scripts/analyze_component_ablation.py.
#
# Both settings use clip_surgery (the runner's default --clip-model) and
# ViT-B/16, matching how result_dev_writerule.txt / results_dev_writerule.md
# were produced.
#
# Runs are local (no slurm) and executed with bounded parallelism so
# multiple jobs share one 16GB GPU at once — set MAX_PARALLEL to tune.
#
# Usage:
#   bash scripts/run_dev_local.sh              # both settings, all 5 datasets
#   MAX_PARALLEL=6 bash scripts/run_dev_local.sh
#   bash scripts/run_dev_local.sh best          # only the best setting
#   bash scripts/run_dev_local.sh baseline      # only the baseline
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_dev_local

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
SEED=1
DATASETS=(dtd eurosat fgvc oxford_flowers oxford_pets)

RESULT_FILE="outputs/result_dev_local.txt"
RECORD_ROOT="outputs/records_dev_local"

MAX_PARALLEL="${MAX_PARALLEL:-4}"

MODE="${1:-all}"   # all | best | baseline

# ---------------------------------------------------------------------------
# Bounded-parallelism job runner
# ---------------------------------------------------------------------------
wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n
    done
}

run_job() {
    # result_label is constant across datasets for a given setting, so
    # summarize_results.py groups all 5 dev datasets into one table row
    # (matching the convention used by run_cd_benchmark_vit.sh / slurm_dev.sh).
    # record_label is per-dataset so RECORD_DIR never collides across datasets.
    local result_label="$1" record_label="$2" method="$3" config="$4"; shift 4
    local overrides=("$@")

    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"
    mkdir -p "$record_dir"

    wait_for_slot

    (
        export RESULT_LABEL="$result_label"
        export RESULT_FILE="$RESULT_FILE"
        export RECORD_DIR="$record_dir"

        CMD=(python -u runner.py
            --method "$method"
            --config "$config"
            --datasets "$DATASET"
            --backbone "$BACKBONE"
            --seed "$SEED")
        if (( ${#overrides[@]} > 0 )); then
            CMD+=(--override "${overrides[@]}")
        fi

        echo "[$record_label] ${CMD[*]}" >&2
        "${CMD[@]}" > "$log_file" 2>&1
        echo "[$record_label] done (exit $?)" >&2
    ) &
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
for DATASET in "${DATASETS[@]}"; do
    if [[ "$MODE" == "all" || "$MODE" == "best" ]]; then
        run_job "ratio-pta-mv-s${SEED}" "ratio-pta-mv-${DATASET}-s${SEED}" \
            patch_modulated_pta configs/patch_modulated_pta \
            "write_source=pta" "write_rule=ratio" "fusion.type=MajorityVoteFusion"
    fi

    if [[ "$MODE" == "all" || "$MODE" == "baseline" ]]; then
        run_job "PTA-CS-s${SEED}" "PTA-CS-${DATASET}-s${SEED}" \
            pta configs/PTA
    fi
done

# Wait for all remaining jobs
wait

echo "All jobs finished. Logs in logs/, raw results in $RESULT_FILE"

# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
python scripts/summarize_results.py \
    --file "$RESULT_FILE" \
    --out outputs/exp_results_dev_local.txt

echo "Summary table written to outputs/exp_results_dev_local.txt"
