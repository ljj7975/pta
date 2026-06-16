#!/bin/bash
# run_all_benchmarks.sh
# Schedules all Slurm benchmark jobs (CD + OOD), waits for completion,
# then runs aggregate_results.py and prints the final comparison table.
#
# Usage:
#   bash scripts/run_all_benchmarks.sh [--dry-run]
#
# Output:
#   outputs/result.txt      (appended by each job)
#   comparison_table.txt    (overwritten at the end)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DRY_RUN=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
    esac
done

cd "$PROJECT_DIR"
mkdir -p outputs logs

echo "========================================================"
echo " PTA Benchmark Runner — ViT-B/16"
echo " Project: $PROJECT_DIR"
echo " Started: $(date)"
echo "========================================================"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRY-RUN] Would submit:"
    echo "  sbatch scripts/slurm_cd_benchmark_vit.sh"
    echo "  sbatch scripts/slurm_ood_benchmark_vit.sh"
    echo "[DRY-RUN] Would then call:"
    echo "  python aggregate_results.py --output comparison_table.txt"
    exit 0
fi

# Clear previous results so we get a clean run
> outputs/result.txt

echo ""
echo "--- Submitting CD benchmark (8 datasets) ---"
CD_JOB=$(sbatch --parsable scripts/slurm_cd_benchmark_vit.sh)
echo "  CD job ID: $CD_JOB"

echo ""
echo "--- Submitting OOD benchmark (3 datasets: R/S/A) ---"
OOD_JOB=$(sbatch --parsable scripts/slurm_ood_benchmark_vit.sh)
echo "  OOD job ID: $OOD_JOB"

echo ""
echo "--- Waiting for jobs to complete ---"
echo "  CD  job: $CD_JOB"
echo "  OOD job: $OOD_JOB"
echo "  (checking every 60 s ...)"

while true; do
    RUNNING=$(squeue --me --noheader --format="%i" 2>/dev/null \
        | grep -E "^(${CD_JOB}|${OOD_JOB})(_[0-9]+)?$" | wc -l)
    if [[ $RUNNING -eq 0 ]]; then
        break
    fi
    echo "  $(date '+%H:%M:%S') — $RUNNING task(s) still running ..."
    sleep 60
done

echo ""
echo "--- All jobs finished at $(date) ---"
echo ""
echo "--- Aggregating results ---"

# Determine which Python to use
PTA_PYTHON="${HOME}/envs/pta/bin/python"
if [[ ! -x "$PTA_PYTHON" ]]; then
    PTA_PYTHON="/share_98/projects/brandon/envs/pta/bin/python"
fi
if [[ ! -x "$PTA_PYTHON" ]]; then
    PTA_PYTHON="python3"
fi

"$PTA_PYTHON" aggregate_results.py \
    --results outputs/result.txt \
    --output comparison_table.txt

echo ""
echo "========================================================"
echo " FINAL RESULTS"
echo "========================================================"
cat comparison_table.txt
echo "========================================================"
echo " Done: $(date)"
echo "========================================================"
