#!/bin/bash
# CD benchmark: exp1 / exp2 / exp3 sequentially, ViT-B/16.
# Dataset sets are bounded by each exp's configs_expN coverage.
# Runs from repo root; activate venv before calling, or use the ssh pattern in CLAUDE.md.

# set -euo pipefail

mkdir -p outputs

PROGRESS=outputs/cd_exps_run.log
MAX_JOBS=3

run() {
    local method=$1 config=$2 datasets=$3 logfile=$4
    echo "=== START $method $(date) ===" | tee -a "$PROGRESS"

    # Run python, combine stdout/stderr, and split to both terminal and file
    python runner.py \
        --method "$method" \
        --config "$config" \
        --datasets "$datasets" \
        --backbone ViT-B/16 \
        2>&1 | tee "outputs/$logfile"

    local exit_code=${PIPESTATUS[0]}
    echo "=== DONE $method exit=$exit_code $(date) ===" | tee -a "$PROGRESS"
    return $exit_code
}

# Full CD suite (stanford_cars requires manual image download if ai.stanford.edu is down)
CD_ALL="caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101"

# Define the experiments to run as an array of arguments
# Format: "method config datasets logfile"
experiments=(
    "exp1_fixed_unique_patches configs_exp1 $CD_ALL cd_exp1_vit.log"
    "exp2_adaptive_tau_proto   configs_exp2 $CD_ALL cd_exp2_vit.log"
    "exp3_gaussian_prototypes  configs_exp3 $CD_ALL cd_exp3_vit.log"
    # You can easily add more experiment lines here in the future
)

current_jobs=0

for exp in "${experiments[@]}"; do
    # Read the arguments out of the string
    read -r method config datasets logfile <<< "$exp"
    
    # Run in the background
    run "$method" "$config" "$datasets" "$logfile" &
    ((current_jobs++))

    # If we hit the max job limit, wait for ANY single job to finish before spawning the next
    if (( current_jobs >= MAX_JOBS )); then
        wait -n
        ((current_jobs--))
    fi
done

# Wait for all remaining background jobs to finish completely
wait

echo "=== ALL DONE $(date) ===" | tee -a "$PROGRESS"
