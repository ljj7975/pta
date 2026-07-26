#!/bin/bash
#SBATCH --job-name=pta_proto_zscore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=5:00:00
#SBATCH --array=0-44
#SBATCH --output=/share_98/projects/brandon/repos/pta/logs/pta_proto_zscore_%x-%A_%a.out
#SBATCH --error=/share_98/projects/brandon/repos/pta/logs/pta_proto_zscore_%x-%A_%a.err

# ============================================================================
# Prototype Score Normalization (z-score) Sweep
#
# Compares raw prototype-score aggregation against per-prototype z-score
# normalization in two settings:
#
#   Setting 2  diagnostic_pta + fusion.mode=patch_only
#              final_logits ARE the patch scores, so the comparison is a pure
#              argmax over aggregated prototype scores — scale-invariant, no tau
#              to tune. This is the cleanest read on whether z-scoring helps.
#
#   Setting 1  patch_modulated_pta
#              patch scores enter a linear fusion as
#                  tau_patch_proto * proto_alpha * patch_logits
#              tau=20 was tuned for raw Gaussian scores in [0, 1]; z-scores are
#              unbounded (~[-3, +10]), so tau is swept and a bounded tanh
#              variant is included.
#
# CAVEAT (not swept here): quality_gate = var / (var + quality_eps) with
# quality_eps=1e-3 was also tuned for [0,1] scores and saturates to ~1.0 on the
# z scale. Harmless for ProtoAlphaFusion (the configured default, which ignores
# the gate), but it also modulates the image-level EMA in patch_modulated_pta
# via quality_modulation. Worth a follow-up sweep on patch_level.quality_eps.
#
# HOW TO USE
# ----------
# 1. Set CLIP_MODEL / EXTRA_OVERRIDES below.
# 2. Update --array to match N_EXP x N_DS (currently 9 x 5 = 45 → 0-44).
# 3. sbatch scripts/slurm_proto_zscore.sh
#
# Task mapping:
#   exp_idx = SLURM_ARRAY_TASK_ID / N_DS
#   ds_idx  = SLURM_ARRAY_TASK_ID % N_DS
# ============================================================================

set -euo pipefail

HOME_DIR=/share_98/projects/$USER
PROJECT_DIR=$HOME_DIR/repos/pta

cd "$PROJECT_DIR"

mkdir -p outputs
mkdir -p logs

source /shared/miniconda3/etc/profile.d/conda.sh
conda activate "$HOME_DIR/envs/pta"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLIP_MODEL=clip_surgery
EXTRA_OVERRIDES=""

# Examples:
# EXTRA_OVERRIDES="patch_level.patch_filter_mode=surgery_with_labels"
# EXTRA_OVERRIDES="patch_level.proto_stats_log_every=500"

# ---------------------------------------------------------------------------
# Datasets — the 5-dataset cross-domain core
# ---------------------------------------------------------------------------
DATASETS=(dtd eurosat fgvc oxford_flowers oxford_pets)
N_DS=${#DATASETS[@]}

# ---------------------------------------------------------------------------
# Experiment registry
# exp METHOD CONFIG_DIR CLIP_MODEL CLIP_CHECKPOINT LABEL OVERRIDE
# ---------------------------------------------------------------------------
_METHODS=()
_CONFIG_DIRS=()
_CLIP_MODELS=()
_CLIP_CHECKPOINTS=()
_EXP_LABELS=()
_OVERRIDES=()

exp() {
    _METHODS+=("$1")
    _CONFIG_DIRS+=("$2")
    _CLIP_MODELS+=("$3")
    _CLIP_CHECKPOINTS+=("$4")
    _EXP_LABELS+=("$5")
    _OVERRIDES+=("$6")
}

# ── Setting 2: diagnostic_pta, patch_only (scale-invariant argmax) ──────────
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "patchonly-Raw-WeightedMean" \
    "fusion.mode=patch_only patch_level.aggregation=weighted_mean $EXTRA_OVERRIDES"
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "patchonly-Z-WeightedMean" \
    "fusion.mode=patch_only patch_level.aggregation=zscore_weighted_mean $EXTRA_OVERRIDES"
exp diagnostic_pta configs/diagnostic_pta $CLIP_MODEL "" "patchonly-Z-WeightedMean-MinC20" \
    "fusion.mode=patch_only patch_level.aggregation=zscore_weighted_mean patch_level.proto_stats_min_count=20 $EXTRA_OVERRIDES"

# ── Setting 1: patch_modulated_pta (z enters a linear fusion) ───────────────
exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "pm-Raw-TopM" \
    "$EXTRA_OVERRIDES"
exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "pm-Z-Tau0.5" \
    "patch_level.aggregation=zscore_weighted_mean fusion.tau_patch_proto=0.5 $EXTRA_OVERRIDES"
exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "pm-Z-Tau1" \
    "patch_level.aggregation=zscore_weighted_mean fusion.tau_patch_proto=1.0 $EXTRA_OVERRIDES"
exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "pm-Z-Tau2" \
    "patch_level.aggregation=zscore_weighted_mean fusion.tau_patch_proto=2.0 $EXTRA_OVERRIDES"
exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "pm-Z-Tau5" \
    "patch_level.aggregation=zscore_weighted_mean fusion.tau_patch_proto=5.0 $EXTRA_OVERRIDES"
exp patch_modulated_pta configs/patch_modulated_pta $CLIP_MODEL "" "pm-Z-Tanh-Tau20" \
    "patch_level.aggregation=zscore_weighted_mean fusion.patch_squash=tanh fusion.tau_patch_proto=20.0 $EXTRA_OVERRIDES"

# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
METHODS=("${_METHODS[@]}")
CONFIG_DIRS=("${_CONFIG_DIRS[@]}")
CLIP_MODELS=("${_CLIP_MODELS[@]}")
CLIP_CHECKPOINTS=("${_CLIP_CHECKPOINTS[@]}")
EXP_LABELS=("${_EXP_LABELS[@]}")
OVERRIDES=("${_OVERRIDES[@]}")

N_EXP=${#METHODS[@]}
N_TOTAL=$((N_EXP * N_DS))

if (( N_EXP == 0 )); then
    echo "ERROR: No experiments enabled. Uncomment at least one exp() line above."
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= N_TOTAL )); then
    echo "SKIP: task $SLURM_ARRAY_TASK_ID >= N_TOTAL ($N_TOTAL). Nothing to do."
    exit 0
fi

exp_idx=$((SLURM_ARRAY_TASK_ID / N_DS))
ds_idx=$((SLURM_ARRAY_TASK_ID % N_DS))

METHOD=${METHODS[$exp_idx]}
CONFIG=${CONFIG_DIRS[$exp_idx]}
CLIP_CKPT=${CLIP_CHECKPOINTS[$exp_idx]}
DATASET=${DATASETS[$ds_idx]}
EXP_LABEL=${EXP_LABELS[$exp_idx]}

echo "========================================================================"
echo "  Task ID    : $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_MAX"
echo "  Experiment : $EXP_LABEL ($exp_idx / $N_EXP)"
echo "  Method     : $METHOD"
echo "  Config     : $CONFIG"
echo "  CLIP model : $CLIP_MODEL"
echo "  Dataset    : $DATASET  (ds_idx=$ds_idx)"
echo "  Node       : $(hostname)"
echo "========================================================================"

export RESULT_LABEL="${EXP_LABEL}"
export RESULT_FILE="outputs/proto_zscore_results.txt"

CMD=(python -u runner.py
    --method "$METHOD"
    --config "$CONFIG"
    --clip-model "$CLIP_MODEL"
    --datasets "$DATASET"
    --backbone ViT-B/16)

if [[ -n "$CLIP_CKPT" ]]; then
    CMD+=(--clip-checkpoint "$CLIP_CKPT")
fi

OVERRIDE=${_OVERRIDES[$exp_idx]}
if [[ -n "$OVERRIDE" ]]; then CMD+=(--override $OVERRIDE); fi

"${CMD[@]}"
