#!/bin/bash
# PTA CD benchmark — runs all 10 CD datasets with configurable backbone
# Usage: bash scripts/run_pta_benchmark.sh [backbone]
# Example: bash scripts/run_pta_benchmark.sh ViT-B/16
BACKBONE=${1:-ViT-B/16}
CUDA_VISIBLE_DEVICES=0 python runner.py \
    --method pta \
    --config configs \
    --datasets caltech101/dtd/eurosat/fgvc/oxford_flowers/oxford_pets/ucf101/stanford_cars/food101/sun397 \
    --backbone "$BACKBONE"
