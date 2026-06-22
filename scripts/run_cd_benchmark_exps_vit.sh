#!/bin/bash
# CD benchmark: exp1 / exp2 / exp3 sequentially, ViT-B/16.
# Dataset sets are bounded by each exp's configs_expN coverage.
# Runs from repo root; activate venv before calling, or use the ssh pattern in CLAUDE.md.

set -euo pipefail

cd /home/brandon/repos/pta
source pta/bin/activate
mkdir -p outputs

PROGRESS=outputs/cd_exps_run.log

run() {
    local method=$1 config=$2 datasets=$3 logfile=$4
    echo "=== START $method $(date) ===" | tee -a "$PROGRESS"
    python runner.py \
        --method "$method" \
        --config "$config" \
        --datasets "$datasets" \
        --backbone ViT-B/16 \
        > "outputs/$logfile" 2>&1
    local exit_code=$?
    echo "=== DONE $method exit=$exit_code $(date) ===" | tee -a "$PROGRESS"
    return $exit_code
}

# Full 10-dataset CD suite for all three exps
CD_ALL="caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101"

run exp1_fixed_unique_patches  configs_exp1 "$CD_ALL" cd_exp1_vit.log
run exp2_adaptive_tau_proto    configs_exp2 "$CD_ALL" cd_exp2_vit.log
run exp3_gaussian_prototypes   configs_exp3 "$CD_ALL" cd_exp3_vit.log

echo "=== ALL DONE $(date) ===" | tee -a "$PROGRESS"
