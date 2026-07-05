#!/bin/bash
# PatchModulatedPTA CD benchmark — runs 7 CD datasets, ViT-B/16 only
# (RN50 is not supported — adapter requires ViT patch embeddings)
# Usage: bash scripts/run_patch_modulated_pta_benchmark.sh
CUDA_VISIBLE_DEVICES=0 python runner.py \
    --method patch_modulated_pta \
    --config configs \
    --datasets caltech101/dtd/eurosat/fgvc/oxford_flowers/oxford_pets/ucf101 \
    --backbone ViT-B/16
