#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python runner.py \
    --method pta \
    --config configs \
    --datasets caltech101/dtd/eurosat/fgvc/oxford_flowers/oxford_pets/ucf101/stanford_cars/food101/sun397 \
    --backbone ViT-B/16