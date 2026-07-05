#!/bin/bash
# PTA OOD benchmark — runs ImageNet and 4 distribution-shift variants
# Usage: bash scripts/run_pta_ood.sh [backbone]
# Example: bash scripts/run_pta_ood.sh ViT-B/16
BACKBONE=${1:-ViT-B/16}
CUDA_VISIBLE_DEVICES=0 python runner.py \
    --method pta \
    --config configs \
    --datasets I/V/R/S/A \
    --backbone "$BACKBONE"
