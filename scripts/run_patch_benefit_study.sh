#!/bin/bash
# ============================================================================
# run_patch_benefit_study.sh — Patch-level prototype benefit study.
#
# Scope: dtd, oxford_flowers, oxford_pets x 4 seeds x 7 method arms.
# See outputs/patch_benefit_prereg.md for the pre-registered hypothesis and
# decision rule this data feeds into.
#
# Grid: 3 datasets x 4 seeds x 7 arms = 84 run-slots. 6 of those (PTA-CS and
# ratio-pta-mv, seed 1, all 3 datasets) are REUSED via symlink from
# outputs/records_dev_local/ instead of rerun. Remaining 78 run fresh.
#
# Arms:
#   1. PTA-CS                 — pta, configs/PTA (baseline)
#   2. PatchModPTA-CS          — patch_modulated_pta, default (ProtoAlphaFusion)
#   3. PatchModPTA-CS-QGated   — fusion.type=QualityGatedFusion
#   4. PatchModPTA-CS-MVote    — fusion.type=MajorityVoteFusion
#   5. PatchModPTA-CS-AGate    — fusion.type=AgreementGateFusion
#   6. ratio-pta-mv             — write_source=pta write_rule=ratio fusion.type=MajorityVoteFusion [KNOWN UNSTABLE]
#   7. ratio-clip-mv            — write_source=clip write_rule=ratio fusion.type=MajorityVoteFusion [stability control]
#
# Concurrency: loop over arms sequentially; within an arm, launch all
# non-reused (dataset x seed) jobs in background, bounded by MAX_PARALLEL
# (default 12 = 3 datasets x 4 seeds), then `wait` fully before the next arm.
#
# Usage:
#   bash scripts/run_patch_benefit_study.sh --dry-run          # default; print all commands
#   bash scripts/run_patch_benefit_study.sh --run              # execute
#   bash scripts/run_patch_benefit_study.sh --run --arm PTA-CS # filter to one arm
#   bash scripts/run_patch_benefit_study.sh --run --dataset dtd
#   bash scripts/run_patch_benefit_study.sh --run --seed 2
#   MAX_PARALLEL=6 bash scripts/run_patch_benefit_study.sh --run
# ============================================================================

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p outputs logs outputs/records_patch_benefit

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

BACKBONE=ViT-B/16
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

RECORD_ROOT="outputs/records_patch_benefit"
DEV_LOCAL_ROOT="outputs/records_dev_local"
RESULT_FILE="outputs/result_patch_benefit.txt"

MAX_PARALLEL="${MAX_PARALLEL:-12}"

DRY_RUN=true
FILTER_ARM=""
FILTER_DATASET=""
FILTER_SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true;  shift ;;
        --run)      DRY_RUN=false; shift ;;
        --arm)      FILTER_ARM="$2";     shift 2 ;;
        --dataset)  FILTER_DATASET="$2"; shift 2 ;;
        --seed)     FILTER_SEED="$2";    shift 2 ;;
        -h|--help)
            sed -n '2,/^# ====/{ /^# /s/^# //p }' "$0"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Arm registry: LABEL|METHOD|CONFIG|OVERRIDE ─────────────────────────────
ARMS=(
    "PTA-CS|pta|configs/PTA|"
    "PatchModPTA-CS|patch_modulated_pta|configs/patch_modulated_pta|"
    "PatchModPTA-CS-QGated|patch_modulated_pta|configs/patch_modulated_pta|fusion.type=QualityGatedFusion"
    "PatchModPTA-CS-MVote|patch_modulated_pta|configs/patch_modulated_pta|fusion.type=MajorityVoteFusion"
    "PatchModPTA-CS-AGate|patch_modulated_pta|configs/patch_modulated_pta|fusion.type=AgreementGateFusion"
    "ratio-pta-mv|patch_modulated_pta|configs/patch_modulated_pta|write_source=pta write_rule=ratio fusion.type=MajorityVoteFusion"
    "ratio-clip-mv|patch_modulated_pta|configs/patch_modulated_pta|write_source=clip write_rule=ratio fusion.type=MajorityVoteFusion"
)

# Arms whose seed-1 x 3-dataset records already exist in outputs/records_dev_local
REUSABLE_ARMS=(PTA-CS ratio-pta-mv)

is_reusable() {
    local label="$1" seed="$2"
    [[ "$seed" != "1" ]] && return 1
    for a in "${REUSABLE_ARMS[@]}"; do
        [[ "$a" == "$label" ]] && return 0
    done
    return 1
}

echo "========================================================================"
echo "  Patch-Level Prototype Benefit Study"
echo "  Project dir : $PROJECT_DIR"
echo "  Dry run     : $DRY_RUN"
echo "  Max parallel: $MAX_PARALLEL"
[[ -n "$FILTER_ARM" ]]     && echo "  Arm filter    : $FILTER_ARM"
[[ -n "$FILTER_DATASET" ]] && echo "  Dataset filter: $FILTER_DATASET"
[[ -n "$FILTER_SEED" ]]    && echo "  Seed filter   : $FILTER_SEED"
echo "========================================================================"

# ── Bounded-parallelism job runner ─────────────────────────────────────────
wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= MAX_PARALLEL )); do
        wait -n
    done
}

run_job() {
    local label="$1" method="$2" config="$3" dataset="$4" seed="$5"; shift 5
    local overrides=("$@")

    local record_label="${label}-${dataset}-s${seed}"
    local record_dir="$RECORD_ROOT/$record_label"
    local log_file="logs/${record_label}.log"

    CMD=(python -u runner.py --method "$method" --config "$config"
         --datasets "$dataset" --backbone "$BACKBONE" --seed "$seed")
    if (( ${#overrides[@]} > 0 )); then
        CMD+=(--override "${overrides[@]}")
    fi

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $record_label"
        echo "  RECORD_DIR=$record_dir RESULT_LABEL=${label}-s${seed} SEED=$seed"
        echo "  ${CMD[*]}"
        echo ""
        return
    fi

    mkdir -p "$record_dir"
    rm -f "$record_dir/records.jsonl"   # idempotency guard: writer only appends

    wait_for_slot
    (
        export RESULT_LABEL="${label}-s${seed}"
        export RESULT_FILE="$RESULT_FILE"
        export RECORD_DIR="$record_dir"
        export SEED="$seed"   # fixes header-seed decorative-metadata mismatch

        echo "[$record_label] ${CMD[*]}" >&2
        "${CMD[@]}" > "$log_file" 2>&1
        echo "[$record_label] done (exit $?)" >&2
    ) &
}

# ── Reuse step: symlink seed-1 PTA-CS / ratio-pta-mv from records_dev_local ─
echo ""
echo "── Reuse (symlink) ─────────────────────────────────────────────────"
for label in "${REUSABLE_ARMS[@]}"; do
    [[ -n "$FILTER_ARM" && "$label" != "$FILTER_ARM" ]] && continue
    [[ -n "$FILTER_SEED" && "1" != "$FILTER_SEED" ]] && continue
    for dataset in "${DATASETS[@]}"; do
        [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue
        src="$DEV_LOCAL_ROOT/${label}-${dataset}-s1"
        dst="$RECORD_ROOT/${label}-${dataset}-s1"
        if [[ -d "$src" && ! -e "$dst" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "[REUSE] $dst -> $src"
            else
                ln -s "$(realpath --relative-to="$RECORD_ROOT" "$src")" "$dst"
                echo "[REUSE] $dst -> $src"
            fi
        elif [[ -e "$dst" ]]; then
            echo "[SKIP] $dst already exists"
        else
            echo "[WARN] expected reusable dir not found: $src"
        fi
    done
done

# ── Launch: loop over arms sequentially, dataset x seed concurrent within ──
echo ""
echo "── Runs ────────────────────────────────────────────────────────────"
for entry in "${ARMS[@]}"; do
    IFS='|' read -r LABEL METHOD CFG OVERRIDES <<< "$entry"
    [[ -n "$FILTER_ARM" && "$LABEL" != "$FILTER_ARM" ]] && continue
    read -ra OVR_ARR <<< "$OVERRIDES"

    for DATASET in "${DATASETS[@]}"; do
        [[ -n "$FILTER_DATASET" && "$DATASET" != "$FILTER_DATASET" ]] && continue
        for SEED in "${SEEDS[@]}"; do
            [[ -n "$FILTER_SEED" && "$SEED" != "$FILTER_SEED" ]] && continue
            if is_reusable "$LABEL" "$SEED"; then
                continue
            fi
            run_job "$LABEL" "$METHOD" "$CFG" "$DATASET" "$SEED" "${OVR_ARR[@]}"
        done
    done

    if [[ "$DRY_RUN" == false ]]; then
        wait   # drain this arm's jobs fully before moving to the next arm
        echo "── Arm $LABEL complete. Checking logs for suspect (empty/truncated) files ──"
        for f in logs/${LABEL}-*-s*.log; do
            [[ -e "$f" ]] || continue
            if ! grep -q "Top1\|%" "$f" 2>/dev/null; then
                echo "  SUSPECT: $f (no accuracy line found)"
            fi
        done
    fi
done

if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo "All jobs finished. Logs in logs/, raw results in $RESULT_FILE"
    echo "Records in $RECORD_ROOT/"
fi
