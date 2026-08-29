#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Submit all experiment slurm scripts in dependency order.
#
# Dependency graph:
#
#   00_baseline_pta ──────────→ 01_part1_tie_breaking
#         │
#         └── (parallel) ──→ 02_part2_purity ──→ 04_part3_agreement
#                           │                    └──→ 05_part3_agreement
#                           │
#                           └──→ 03_part2_separability ──→ 04_part3_agreement
#                                                         └──→ 05_part3_agreement
#
#   06_part4_trust_signal (fully independent)
#   07_part4_reweight    (fully independent)
#   08_part4b_downweight (fully independent)
#
# Usage:
#   bash run_all.sh              # Dry run (default) — prints commands only
#   bash run_all.sh --run        # Actually submit jobs
#   bash run_all.sh --dry-run    # Explicit dry run
#   WRITE_GATE_CONCURRENCY=10 REWEIGHT_CONCURRENCY=10 REWEIGHT5_CONCURRENCY=10 bash run_all.sh --run
#     # raise the 06/07/08 array-job concurrency above the default %4
# =============================================================================

set -euo pipefail

# Number of 06 write-gate array tasks to run concurrently (overrides the
# script's default "%4" via --array). Override: WRITE_GATE_CONCURRENCY=10.
WRITE_GATE_CONCURRENCY="${WRITE_GATE_CONCURRENCY:-4}"
ARRAY_SPEC_06="0-47%${WRITE_GATE_CONCURRENCY}"

# Number of 07 write-reweight array tasks to run concurrently (overrides the
# script's default "%4" via --array). Override: REWEIGHT_CONCURRENCY=10.
REWEIGHT_CONCURRENCY="${REWEIGHT_CONCURRENCY:-4}"
ARRAY_SPEC_07="0-47%${REWEIGHT_CONCURRENCY}"

# Number of 08 write-reweight-down array tasks to run concurrently (overrides
# the script's default "%4" via --array). Override: REWEIGHT5_CONCURRENCY=10.
REWEIGHT5_CONCURRENCY="${REWEIGHT5_CONCURRENCY:-4}"
ARRAY_SPEC_08="0-59%${REWEIGHT5_CONCURRENCY}"

DRY_RUN=true

for arg in "$@"; do
  case "$arg" in
    --dry-run)  DRY_RUN=true  ;;
    --run)      DRY_RUN=false ;;
    -h|--help)
      echo "Usage: $0 [--dry-run | --run]"
      echo "  --dry-run  Print commands without executing (default)"
      echo "  --run      Actually submit jobs to Slurm"
      exit 0
      ;;
    *)
      echo "Unknown flag: $arg" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Pre-flight: verify all scripts exist ------------------------------------
REQUIRED_SCRIPTS=(
  00_baseline_pta.sh
  01_part1_tie_breaking.sh
  02_part2_purity.sh
  03_part2_separability.sh
  04_part3_agreement.sh
  05_part3_aggregation.sh
  06_part4_trust_signal.sh
  07_part4_reweight.sh
  08_part4b_downweight.sh
)

MISSING=0
for script in "${REQUIRED_SCRIPTS[@]}"; do
  if [[ ! -x "$SCRIPT_DIR/$script" ]]; then
    echo "ERROR: Missing or not executable: $SCRIPT_DIR/$script" >&2
    MISSING=1
  fi
done
if (( MISSING )); then
  echo "Aborting. Fix the missing scripts above." >&2
  exit 1
fi

submit() {
  local script="$1"
  local dep="${2:-}"
  local extra="${3:-}"

  if $DRY_RUN; then
    local cmd="sbatch"
    [[ -n "$dep" ]] && cmd="$cmd $dep"
    cmd="$cmd $extra"
    cmd="$cmd $script"
    echo "  [dry-run] $cmd" >&2
    echo "0"
  else
    local -a cmd=(sbatch)
    [[ -n "$dep" ]] && cmd+=(--dependency="${dep#--dependency=}")
    [[ -n "$extra" ]] && cmd+=($extra)
    cmd+=("$script")
    echo "  [submit]  ${cmd[*]}" >&2
    "${cmd[@]}" | grep -oP 'Submitted batch job \K\d+'
  fi
}

check_00_done() {
  [[ -f "$REPO_ROOT/outputs/records/PTA-CS-dtd-s1/records.jsonl" ]]
}

check_01_done() {
  [[ -f "$REPO_ROOT/outputs/tie_breaking/dtd.md" ]] &&
  [[ -f "$REPO_ROOT/outputs/tie_breaking/oxford_flowers.md" ]] &&
  [[ -f "$REPO_ROOT/outputs/tie_breaking/oxford_pets.md" ]]
}

check_02_done() {
  [[ -f "$REPO_ROOT/outputs/patch_bank_dumps/dtd-s1.pt" ]] &&
  [[ -f "$REPO_ROOT/outputs/records/PatchModPTA-CS-dtd-s1/records.jsonl" ]] &&
  [[ -f "$REPO_ROOT/outputs/prototype_purity/purity_report.md" ]]
}

check_03_done() {
  [[ -f "$REPO_ROOT/outputs/prototype_separability/dtd-s1.md" ]] &&
  [[ -f "$REPO_ROOT/outputs/prototype_separability/oxford_flowers-s1.md" ]] &&
  [[ -f "$REPO_ROOT/outputs/prototype_separability/oxford_pets-s1.md" ]]
}

check_04_done() {
  [[ -f "$REPO_ROOT/outputs/patch_vote_validation/dtd.json" ]] &&
  [[ -f "$REPO_ROOT/outputs/patch_vote_validation/oxford_flowers.json" ]] &&
  [[ -f "$REPO_ROOT/outputs/patch_vote_validation/oxford_pets.json" ]]
}

check_05_done() {
  [[ -f "$REPO_ROOT/outputs/patch_vote_aggregation/dtd.json" ]] &&
  [[ -f "$REPO_ROOT/outputs/patch_vote_aggregation/oxford_flowers.json" ]] &&
  [[ -f "$REPO_ROOT/outputs/patch_vote_aggregation/oxford_pets.json" ]]
}

check_06_done() {
  # Part 4 (redesigned) write-gate study: one record dir per (gate × dataset × seed).
  [[ -f "$REPO_ROOT/outputs/records_wg/WG-baseline-dtd-s1/records.jsonl" ]] &&
  [[ -f "$REPO_ROOT/outputs/result_write_gate.txt" ]]
}

check_07_done() {
  # Part 4b write-reweight study: one record dir per (boost × dataset × seed).
  [[ -f "$REPO_ROOT/outputs/records_rw/RW-b1-dtd-s1/records.jsonl" ]] &&
  [[ -f "$REPO_ROOT/outputs/result_reweight.txt" ]]
}

check_08_done() {
  # Part 4b down-weight study: one record dir per ((up,down) × dataset × seed).
  [[ -f "$REPO_ROOT/outputs/records_dw/DW-u1.0-d1.0-dtd-s1/records.jsonl" ]] &&
  [[ -f "$REPO_ROOT/outputs/result_reweight_down.txt" ]]
}

# --- Submit jobs -------------------------------------------------------------
declare -A JOB_IDS
declare -A SKIPPED

echo "=== Submitting experiment jobs ==="
echo

echo "[1/4] Submitting independent jobs (00, 06, 07, 08)..."
if check_00_done; then
  echo "  [skip] 00_baseline_pta.sh — outputs already exist"
  JOB_IDS[00]="SKIP"
  SKIPPED[00]=1
else
  JOB_IDS[00]=$(submit "$SCRIPT_DIR/00_baseline_pta.sh")
fi
if check_06_done; then
  echo "  [skip] 06_part4_trust_signal.sh — outputs already exist"
  JOB_IDS[06]="SKIP"
  SKIPPED[06]=1
else
  JOB_IDS[06]=$(submit "$SCRIPT_DIR/06_part4_trust_signal.sh" "" "--array=${ARRAY_SPEC_06}")
fi
if check_07_done; then
  echo "  [skip] 07_part4_reweight.sh — outputs already exist"
  JOB_IDS[07]="SKIP"
  SKIPPED[07]=1
else
  JOB_IDS[07]=$(submit "$SCRIPT_DIR/07_part4_reweight.sh" "" "--array=${ARRAY_SPEC_07}")
fi
if check_08_done; then
  echo "  [skip] 08_part4b_downweight.sh — outputs already exist"
  JOB_IDS[08]="SKIP"
  SKIPPED[08]=1
else
  JOB_IDS[08]=$(submit "$SCRIPT_DIR/08_part4b_downweight.sh" "" "--array=${ARRAY_SPEC_08}")
fi
echo

echo "[2/4] Submitting 01 (depends on 00), 02, 03 (no deps)..."
if check_01_done; then
  echo "  [skip] 01_part1_tie_breaking.sh — outputs already exist"
  JOB_IDS[01]="SKIP"
  SKIPPED[01]=1
else
  if [[ "${JOB_IDS[00]}" == "SKIP" ]]; then
    JOB_IDS[01]=$(submit "$SCRIPT_DIR/01_part1_tie_breaking.sh")
  else
    JOB_IDS[01]=$(submit "$SCRIPT_DIR/01_part1_tie_breaking.sh" "afterok:${JOB_IDS[00]}")
  fi
fi
if check_02_done; then
  echo "  [skip] 02_part2_purity.sh — outputs already exist"
  JOB_IDS[02]="SKIP"
  SKIPPED[02]=1
else
  JOB_IDS[02]=$(submit "$SCRIPT_DIR/02_part2_purity.sh")
fi
if check_03_done; then
  echo "  [skip] 03_part2_separability.sh — outputs already exist"
  JOB_IDS[03]="SKIP"
  SKIPPED[03]=1
else
  JOB_IDS[03]=$(submit "$SCRIPT_DIR/03_part2_separability.sh")
fi
echo

echo "[3/4] Submitting 04, 05 (depends on 02 + 03)..."
if check_04_done; then
  echo "  [skip] 04_part3_agreement.sh — outputs already exist"
  JOB_IDS[04]="SKIP"
  SKIPPED[04]=1
else
  DEP_04_05=""
  [[ "${JOB_IDS[02]}" != "SKIP" && "${JOB_IDS[03]}" != "SKIP" ]] && DEP_04_05="afterok:${JOB_IDS[02]}:${JOB_IDS[03]}"
  [[ "${JOB_IDS[02]}" != "SKIP" && "${JOB_IDS[03]}" == "SKIP" ]] && DEP_04_05="afterok:${JOB_IDS[02]}"
  [[ "${JOB_IDS[02]}" == "SKIP" && "${JOB_IDS[03]}" != "SKIP" ]] && DEP_04_05="afterok:${JOB_IDS[03]}"
  JOB_IDS[04]=$(submit "$SCRIPT_DIR/04_part3_agreement.sh" "$DEP_04_05")
fi
if check_05_done; then
  echo "  [skip] 05_part3_aggregation.sh — outputs already exist"
  JOB_IDS[05]="SKIP"
  SKIPPED[05]=1
else
  DEP_05=""
  [[ "${JOB_IDS[02]}" != "SKIP" && "${JOB_IDS[03]}" != "SKIP" ]] && DEP_05="afterok:${JOB_IDS[02]}:${JOB_IDS[03]}"
  [[ "${JOB_IDS[02]}" != "SKIP" && "${JOB_IDS[03]}" == "SKIP" ]] && DEP_05="afterok:${JOB_IDS[02]}"
  [[ "${JOB_IDS[02]}" == "SKIP" && "${JOB_IDS[03]}" != "SKIP" ]] && DEP_05="afterok:${JOB_IDS[03]}"
  JOB_IDS[05]=$(submit "$SCRIPT_DIR/05_part3_aggregation.sh" "$DEP_05")
fi
echo

echo "=== Summary ==="
echo
printf "  %-8s %-35s %s\n" "Job" "Script" "Status"
printf "  %-8s %-35s %s\n" "---" "------" "------"
for idx in 00 01 02 03 04 05 06 07 08; do
  script_name="${idx}_*.sh"
  script_name=$(ls "$SCRIPT_DIR"/${script_name} 2>/dev/null | head -1)
  script_name=$(basename "$script_name")
  if [[ "${SKIPPED[$idx]:-0}" == "1" ]]; then
    printf "  %-8s %-35s %s\n" "—" "$script_name" "SKIP (already done)"
  else
    printf "  %-8s %-35s %s\n" "${JOB_IDS[$idx]}" "$script_name" "SUBMITTED"
  fi
done
echo

if $DRY_RUN; then
  echo "(Dry run — no jobs were submitted. Use --run to submit.)"
else
  N_SKIPPED=${#SKIPPED[@]}
  N_SUBMITTED=$((9 - N_SKIPPED))
  echo "$N_SUBMITTED job(s) submitted, $N_SKIPPED skipped (already done)."
fi
