#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python runner.py \
    --method pta \
    --config configs \
    --datasets I/V/R/S/A \
    --backbone RN50