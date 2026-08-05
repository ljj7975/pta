#!/usr/bin/env python3
"""
GPU test for T7: ZeroShot per-sample recording via utils.records.

Verifies (dtd, ViT-B/16, MAX_BATCHES=30):
1. With RECORD_DIR set: records.jsonl has exactly 1 header line + 30 records.
2. logits['final'] == logits['clip'] (zero-shot: no fusion).
3. logits['image_proto'] is None and logits['patch_proto'] is None.
4. Per-sample logits lists have length C == 47 (dtd class count).
5. summary.json total == 30 and acc matches the returned final acc.
6. Behavior neutrality: final accuracy is bit-identical with and without
   RECORD_DIR set, and no records.jsonl appears in the repo tree.

NOTE: the comparison uses shuffle=False + num_workers=0 (deterministic
sequential iteration). The stock 8-worker shuffled loader yields batches in
nondeterministic completion order on this cluster, which would make a
with/without-RECORD_DIR accuracy comparison meaningless.

Usage (sbatch):
    sbatch scripts/slurm_test_records_zeroshot.sh
"""

import json
import os
import random
import subprocess
import sys
import tempfile

# Must be set BEFORE torch import for deterministic CuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch

from encoder import create_encoder_instance
from utils import get_config_file, build_test_data_loader, clip_classifier
from models.zeroshot import ZeroShotAdapter

SEED = 42
DATASET = "dtd"
BACKBONE = "ViT-B/16"
CLIP_MODEL = "clip_surgery"  # matches runner.py default
CONFIG_DIR = "configs/PTA"
MAX_BATCHES = 30
N_CLASSES = 47  # dtd


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


def run_zeroshot(record_dir):
    """Run ZeroShot on dtd with MAX_BATCHES=30; return (final_acc, targets).

    Uses shuffle=False + num_workers=0 so the batch sequence is fully
    deterministic: the multi-worker shuffled loader (num_workers=8) is
    nondeterministic in yield order on this cluster (probe-verified), which
    would make the with/without-RECORD_DIR accuracy comparison meaningless.
    """
    set_full_seed(SEED)
    os.environ["MAX_BATCHES"] = str(MAX_BATCHES)
    if record_dir:
        os.environ["RECORD_DIR"] = record_dir
    else:
        os.environ.pop("RECORD_DIR", None)

    encoder = create_encoder_instance(CLIP_MODEL, model_type=BACKBONE, device="cuda")
    preprocess = encoder.preprocess
    encoder.eval()

    cfg = get_config_file(CONFIG_DIR, DATASET)
    adapter = ZeroShotAdapter(cfg)

    test_loader, classnames, template = build_test_data_loader(
        DATASET, "./data", preprocess, shuffle=False
    )
    test_loader.num_workers = 0
    clip_weights = clip_classifier(classnames, template, encoder)

    acc = adapter.run(test_loader, encoder, clip_weights, DATASET)
    return acc


def capture_targets():
    """Deterministic reference: first MAX_BATCHES targets in dataset order."""
    set_full_seed(SEED)
    encoder = create_encoder_instance(CLIP_MODEL, model_type=BACKBONE, device="cuda")
    loader, classnames, template = build_test_data_loader(
        DATASET, "./data", encoder.preprocess, shuffle=False
    )
    loader.num_workers = 0
    targets = []
    for i, (images, target) in enumerate(loader):
        if i >= MAX_BATCHES:
            break
        targets.append(int(target.item()))
    return targets


def find_repo_records_files():
    """All records.jsonl under the repo (excluding pre-existing output dir)."""
    out = subprocess.run(
        ["find", ".", "-name", "records.jsonl", "-not", "-path", "./outputs/records/*"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    return set(p for p in out.stdout.split() if p)


def check_records(record_dir: str, expected_acc: float):
    records_file = os.path.join(record_dir, "records.jsonl")
    lines = open(records_file).read().splitlines()
    assert len(lines) == 1 + MAX_BATCHES, f"expected 1+{MAX_BATCHES} lines, got {len(lines)}"

    header = json.loads(lines[0])
    assert header["__header__"] is True, header
    assert header["method"] == "ZeroShot", header
    assert header["dataset"] == DATASET, header
    assert header["seed"] == SEED, header
    assert header["C"] == N_CLASSES, header
    assert header["classnames"] == [], header
    assert isinstance(header["resolved_config"], dict), header

    records = [json.loads(line) for line in lines[1:]]
    for rec in records:
        assert isinstance(rec["batch_idx"], int), rec
        assert isinstance(rec["target"], int), rec
        assert isinstance(rec["pred"], int), rec
        assert isinstance(rec["correct"], bool), rec
        assert isinstance(rec["conf"], float), rec
        assert rec["quality_gate"] is None, rec
        assert rec["gate_mode"] is None, rec
        assert rec["proto_alpha"] is None, rec
        lg = rec["logits"]
        assert lg["image_proto"] is None, rec
        assert lg["patch_proto"] is None, rec
        assert lg["final"] == lg["clip"], rec
        assert len(lg["final"]) == N_CLASSES, rec
        assert rec["proto_stats"] == {"true": None, "pred": None}, rec

    summary = json.load(open(os.path.join(record_dir, "summary.json")))
    assert summary["method"] == "ZeroShot", summary
    assert summary["dataset"] == DATASET, summary
    assert summary["seed"] == SEED, summary
    assert summary["total"] == MAX_BATCHES, summary
    assert abs(summary["acc"] - expected_acc) < 1e-9, summary
    assert summary["per_class"] == {}, summary

    print(f"records.jsonl: {len(lines)} lines (1 header + {MAX_BATCHES} records)")
    print(f"header: method={header['method']} dataset={header['dataset']} "
          f"seed={header['seed']} C={header['C']} classnames={header['classnames']}")
    print("FULL_HEADER=" + json.dumps(header))
    print("FULL_RECORD0=" + json.dumps(records[0]))
    print(f"summary.json: total={summary['total']} acc={summary['acc']:.6f}")
    return header, records[0], [rec["target"] for rec in records]


def main():
    print("=" * 60)
    print(f"  T7 ZeroShot records test: {DATASET} ({BACKBONE}), MAX_BATCHES={MAX_BATCHES}")
    print("=" * 60)

    os.environ["SEED"] = str(SEED)
    os.environ["RESULT_LABEL"] = "ZeroShot"
    os.environ["RESULT_FILE"] = os.path.join(tempfile.gettempdir(), "t7_result.txt")

    record_dir = tempfile.mkdtemp(prefix="rec_zs_")
    expected_targets = capture_targets()

    print(f"\n--- Run A (RECORD_DIR={record_dir}) ---")
    acc_with = run_zeroshot(record_dir)
    print(f"  acc with RECORD_DIR:  {acc_with:.6f}")
    _, _, recorded_targets = check_records(record_dir, acc_with)
    assert recorded_targets == expected_targets, "recorded targets != deterministic reference"
    print("  recorded target sequence == deterministic reference: PASS")

    print(f"\n--- Run B (RECORD_DIR unset) ---")
    before = find_repo_records_files()
    acc_without = run_zeroshot(None)
    print(f"  acc without RECORD_DIR: {acc_without:.6f}")
    after = find_repo_records_files()
    assert before == after, f"stray records.jsonl appeared: {after - before}"
    assert capture_targets() == expected_targets, "run B consumed a different sequence"

    print("\n--- Behavior neutrality ---")
    print(f"  acc_with    = {acc_with:.10f}")
    print(f"  acc_without = {acc_without:.10f}")
    assert acc_with == acc_without, f"acc differs: {acc_with} vs {acc_without}"
    print("  acc identical (bit-for-bit): PASS")

    print("\nzeroshot records + behavior-neutral: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
