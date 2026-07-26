#!/usr/bin/env python3
"""
Determinism test: run PTA twice on the same dataset with the same seed
and verify that the results are identical.

Usage (local):
    python tests/test_pta_determinism.py

Usage (srun on slurm):
    srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=16G bash -c '
        source /shared/miniconda3/etc/profile.d/conda.sh
        conda activate /share_98/projects/$USER/envs/pta
        cd /share_98/projects/$USER/repos/pta
        export PYTHONPATH="$PWD:${PYTHONPATH:-}"
        python -u tests/test_determinism.py
    '

Usage (sbatch):
    sbatch scripts/slurm_test_determinism.sh
"""

import os
import sys
import re

# Must be set BEFORE torch import for deterministic CuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Ensure repo root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import random
import numpy as np
import torch

import clip
from utils import get_config_file, build_test_data_loader, clip_classifier
from models.pta import PTAAdapter


SEED = 42
DATASET = "dtd"
BACKBONE = "ViT-B/16"
CONFIG_DIR = "configs"


def set_full_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def run_pta_once(seed: int, dataset: str, backbone: str, config_dir: str) -> float:
    """Run PTA on a single dataset and return the final accuracy."""
    set_full_seed(seed)

    # Load CLIP
    clip_model, preprocess = clip.load(backbone)
    clip_model.eval()

    # Load config
    cfg = get_config_file(config_dir, dataset)

    # Build adapter
    adapter = PTAAdapter(cfg)

    # Build data loader — shuffle=True is now deterministic with seed + worker_init_fn
    test_loader, classnames, template = build_test_data_loader(
        dataset, "./data", preprocess, shuffle=True
    )

    # Build CLIP weights
    clip_weights = clip_classifier(classnames, template, clip_model)

    # Run
    acc = adapter.run(test_loader, clip_model, clip_weights, dataset)

    return acc


def main():
    print("=" * 60)
    print(f"  Determinism Test: PTA on {DATASET} ({BACKBONE})")
    print(f"  Seed: {SEED}")
    print("=" * 60)

    # Run twice with the same seed
    print("\n--- Run 1 ---")
    acc1 = run_pta_once(SEED, DATASET, BACKBONE, CONFIG_DIR)
    print(f"  Accuracy (run 1): {acc1:.4f}%")

    print("\n--- Run 2 ---")
    acc2 = run_pta_once(SEED, DATASET, BACKBONE, CONFIG_DIR)
    print(f"  Accuracy (run 2): {acc2:.4f}%")

    # Compare
    print("\n" + "=" * 60)
    diff = abs(acc1 - acc2)
    print(f"  Difference: {diff:.6f}%")

    if diff < 1e-4:
        print("  ✅ PASS — Results are identical (deterministic)")
        print("=" * 60)
        return 0
    else:
        print("  ❌ FAIL — Results differ (non-deterministic)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    code = main()
    sys.exit(code)
