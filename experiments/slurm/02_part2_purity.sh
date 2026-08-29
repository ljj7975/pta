#!/bin/bash
#SBATCH --job-name=part2_purity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --exclude=node1
#SBATCH --time=05:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/part2_purity_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/part2_purity_%x-%A_%a.err

# ============================================================================
# Part 2.1 — Bank Purity Analysis
#
# SLURM array job: PatchModPTA-CS on 3 datasets × 4 seeds with per-sample
# recording.  The purity analysis itself runs as a separate post-step
# (depends on all array tasks completing) since it scans ALL datasets.
#
# 12 tasks = 3 datasets × 4 seeds
#
#   dataset_idx = SLURM_ARRAY_TASK_ID / 4
#   seed_idx    = SLURM_ARRAY_TASK_ID % 4
#
# Datasets: dtd, oxford_flowers, oxford_pets
# Seeds:    1, 2, 3, 4
#
# Environment variables for per-sample recording:
#   RECORD_DIR   — output path for records.jsonl
#   RESULT_LABEL — method label in output results
#   SEED         — random seed
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs outputs/records logs outputs/prototype_purity

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------
DATASETS=(dtd oxford_flowers oxford_pets)
SEEDS=(1 2 3 4)

# ---------------------------------------------------------------------------
# Map task ID → dataset + seed
# ---------------------------------------------------------------------------
dataset_idx=$((SLURM_ARRAY_TASK_ID / 4))
seed_idx=$((SLURM_ARRAY_TASK_ID % 4))

DATASET=${DATASETS[$dataset_idx]}
SEED=${SEEDS[$seed_idx]}

export RECORD_DIR="outputs/records/PatchModPTA-CS-${DATASET}-s${SEED}"
export RESULT_LABEL="PatchModPTA-CS-${DATASET}-s${SEED}"
export SEED="${SEED}"

mkdir -p "$RECORD_DIR"
rm -f "$RECORD_DIR/records.jsonl"

# ---------------------------------------------------------------------------
# Run PatchModPTA-CS
# ---------------------------------------------------------------------------
echo "========================================================================"
echo "  Part 2.1 — Bank Purity Analysis"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Dataset    : $DATASET  (dataset_idx=$dataset_idx)"
echo "  Seed       : $SEED  (seed_idx=$seed_idx)"
echo "  RECORD_DIR : $RECORD_DIR"
echo "  RESULT_LABEL: $RESULT_LABEL"
echo "  Node       : $(hostname)"
echo "========================================================================"

python -u runner.py \
    --method patch_modulated_pta \
    --config configs/patch_modulated_pta \
    --datasets "$DATASET" \
    --backbone ViT-B/16 \
    --seed "$SEED"

echo ""
echo "========================================================================"
echo "  Done: $DATASET seed=$SEED"
echo "========================================================================"

# ---------------------------------------------------------------------------
# Submit purity analysis as a dependent job (runs after ALL array tasks)
# ---------------------------------------------------------------------------
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
    echo ""
    echo "Submitting purity analysis job (depends on this array)..."
    ANALYSIS_JOB=$(sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        --job-name=purity_analysis \
        --partition=gpu \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --mem=8G \
        --time=01:00:00 \
        --output=/share_98/projects/brandon/repos/pta/logs/purity_analysis_%j.out \
        --error=/share_98/projects/brandon/repos/pta/logs/purity_analysis_%j.err \
        --wrap="bash -c 'cd $PROJECT_DIR && source /shared/miniconda3/etc/profile.d/conda.sh && conda activate $HOME_DIR/envs/pta && export PYTHONPATH=$PROJECT_DIR && python -u scripts/analyze_prototype_purity.py --records outputs/records/ --out outputs/prototype_purity/purity_report.md'" \
        | grep -oP 'Submitted batch job \K\d+')
    echo "  Purity analysis job ID: $ANALYSIS_JOB"
fi
