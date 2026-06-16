#!/bin/bash
# Full CD suite: caltech101/dtd/eurosat/fgvc/oxford_flowers/oxford_pets/ucf101/stanford_cars/food101/sun397
CUDA_VISIBLE_DEVICES=0 python runner.py \
    --method pta \
    --config configs \
    --datasets caltech101/oxford_flowers/oxford_pets \
    --backbone ViT-B/16