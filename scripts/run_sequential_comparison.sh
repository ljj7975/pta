#!/bin/bash
# Sequential MPTA vs exp1/exp2/exp3 comparison on caltech101/oxford_flowers/oxford_pets.
# Runs one method at a time (Jetson has only 7.4GB RAM; concurrent jobs OOM-kill).
# Launch detached (nohup + disown) so it survives SSH disconnects.
cd /home/brandon/repos/pta
source pta/bin/activate
mkdir -p outputs

DATASETS="caltech101/oxford_flowers/oxford_pets"
PROGRESS=outputs/sequential_run.log

run() {
    local method=$1 config=$2 logfile=$3
    echo "=== START $method $(date) ===" >> "$PROGRESS"
    python runner.py --method "$method" --config "$config" --datasets "$DATASETS" --backbone ViT-B/16 > "outputs/$logfile" 2>&1
    echo "=== DONE $method exit=$? $(date) ===" >> "$PROGRESS"
}

run multi_proto_pta            configs_multi_proto rerun_mpta.log
run exp1_fixed_unique_patches  configs_exp1         rerun_exp1.log
run exp2_adaptive_tau_proto    configs_exp2         rerun_exp2.log
run exp3_gaussian_prototypes   configs_exp3         rerun_exp3.log

echo "=== ALL DONE $(date) ===" >> "$PROGRESS"
