#!/bin/bash
#SBATCH --job-name=pta_validate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_validate_%x-%A.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_validate_%x-%A.err

# ============================================================================
# PTA Implementation Validation Script
#
# Compares the user's PTA implementation (pta repo) against the original
# (PTA-main) on a small set of test samples from caltech101.
#
# HOW TO USE
# ----------
#   sbatch scripts/slurm_validate_pta.sh
#
# Output:
#   - Console output (see .out file)
#   - Detailed results: outputs/validation/pta_validation_results.txt
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs/validation
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "========================================================================"
echo "  Job ID     : $SLURM_JOB_ID"
echo "  Job Name   : $SLURM_JOB_NAME"
echo "  Node       : $(hostname)"
echo "  GPUs       : $SLURM_GPUS_ON_NODE"
echo "  Date       : $(date)"
echo "========================================================================"

python -u tests/test_vs_original.py

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "  Validation complete (exit code: $EXIT_CODE)"
echo "  Results: outputs/validation/pta_validation_results.txt"
echo "========================================================================"

exit $EXIT_CODE
