#!/usr/bin/env python3
"""
Loader determinism test (Task 2 of patch-proto-revalidation).

Verifies the seeded-test-loader fix in ``datasets/utils.py build_data_loader``
and ``utils/data.py``:

1. Same seed, run twice -> byte-identical records.jsonl (sample stream fully
   reproducible: num_workers=0 + ``torch.Generator().manual_seed(seed)``).
2. Different seeds -> the stream genuinely varies (not a no-op): the subset
   records files differ in full-stream sha256 AND the full-DTD loader's first
   batch target differs between seeds 1 and 2.
3. The loader is actually single-worker with an explicit generator when a seed
   is passed (proves the fix applied, not the legacy 8-worker path).

Rationale for num_workers=0 (codebase's own documented conclusion,
tests/test_records_zeroshot.py:14-17): the stock 8-worker shuffled loader
"yields batches in nondeterministic completion order on this cluster", which
makes same-seed runs differ regardless of any generator. With a single worker
and an explicit generator the shuffled stream is deterministic for a seed and
varies across seeds — exactly what the 3-seed sweep / delta_mean needs.

Note on the "first-line target" check: on the 5-class subset, seeds 1 and 2
happen to draw a class-4 sample first (a ~20% label-level collision — the
streams already diverge at record 1: 2 vs 1). The cross-seed "first target
differs" proof therefore runs on the full DTD loader (47 classes), where
seeds 1/2 yield first targets 16 and 2 (verified). The subset records are
still the cross-seed no-op detector via full-stream sha256.

The test uses no model forward: the sample stream is entirely loader-derived,
so it is CPU-only and fast (a reliable pre-commit gate).

Usage:
    python tests/test_determinism.py
"""

import hashlib
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

from utils import build_test_data_loader  # noqa: E402
from utils.data import build_subset_test_data_loader  # noqa: E402

DATASET = "dtd"
DATA_ROOT = os.path.join(REPO_ROOT, "data")
CLASS_FILE = os.path.join(REPO_ROOT, "outputs", "subset_classes.txt")
SEED_A = 1
SEED_B = 2

# Deterministic (RNG-free) preprocess — only the stream order is under test.
_PREPROCESS = transforms.Compose(
    [transforms.Resize(224), transforms.ToTensor()]
)


def run_subset_loader(seed: int, record_path: str):
    """Iterate the seeded 5-class subset loader once, writing a deterministic
    records.jsonl (header without timestamp + one record per batch).

    Returns (record_path, first_target, target_sequence).
    """
    loader, classnames, template = build_subset_test_data_loader(
        DATASET,
        DATA_ROOT,
        _PREPROCESS,
        class_file=CLASS_FILE,
        shuffle=True,
        seed=seed,
    )
    assert loader.num_workers == 0, f"expected num_workers=0, got {loader.num_workers}"
    assert loader.generator is not None, "expected explicit generator for seeded shuffle"

    records = []
    for i, (images, target) in enumerate(loader):
        records.append({"batch_idx": i, "target": int(target.item())})

    header = {
        "__header__": True,
        "method": "ZeroShot",
        "dataset": DATASET,
        "seed": seed,
        "C": len(classnames),
        "classnames": classnames,
        "subset": True,
    }
    with open(record_path, "w") as f:
        for line in [header] + records:
            f.write(json.dumps(line) + "\n")

    targets = [r["target"] for r in records]
    return record_path, targets[0], targets


def first_batch_target_full_dtd(seed: int) -> int:
    """First batch target of the full-DTD seeded loader (probes one image)."""
    loader, _, _ = build_test_data_loader(
        DATASET, DATA_ROOT, _PREPROCESS, shuffle=True, seed=seed
    )
    assert loader.num_workers == 0, f"expected num_workers=0, got {loader.num_workers}"
    assert loader.generator is not None, "expected explicit generator for seeded shuffle"
    for images, target in loader:
        return int(target.item())


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def first_divergence(seq_a, seq_b):
    """Index of the first position where two sequences differ, or None."""
    for i, (x, y) in enumerate(zip(seq_a, seq_b)):
        if x != y:
            return i
    return None if len(seq_a) == len(seq_b) else min(len(seq_a), len(seq_b))


def main():
    print("=" * 60)
    print(f"  Loader determinism: {DATASET}, shuffle=True, seeds {SEED_A}/{SEED_B}")
    print("=" * 60)

    tmp = tempfile.mkdtemp(prefix="det_")
    run_a = os.path.join(tmp, "seed1-a-records.jsonl")
    run_b = os.path.join(tmp, "seed1-b-records.jsonl")
    run_c = os.path.join(tmp, "seed2-records.jsonl")

    # --- 1. Same seed, run twice -> byte-identical records.jsonl ------------
    print("\n--- Same seed, run twice (subset, seed=1) ---")
    _, first_a, seq_a = run_subset_loader(SEED_A, run_a)
    _, first_b, seq_b = run_subset_loader(SEED_A, run_b)
    print(f"  records: {len(seq_a)} samples, first_target={first_a} (both runs)")
    print(f"  sha256(A) = {sha256(run_a)}")
    print(f"  sha256(B) = {sha256(run_b)}")
    assert first_a == first_b, f"same-seed first target differs: {first_a} vs {first_b}"
    assert sha256(run_a) == sha256(run_b), "records.jsonl differs across same-seed runs"
    print("  records.jsonl byte-identical (same seed): PASS")

    # --- 2. Different seeds -> stream genuinely differs (not a no-op) -------
    print("\n--- Different seeds (subset, 1 vs 2) ---")
    _, first_c, seq_c = run_subset_loader(SEED_B, run_c)
    print(f"  seed {SEED_A} first10 targets: {seq_a[:10]}")
    print(f"  seed {SEED_B} first10 targets: {seq_c[:10]}")
    print(f"  sha256(C) = {sha256(run_c)}")
    assert sha256(run_a) != sha256(run_c), "cross-seed records byte-identical — seed not wired"
    diverge = first_divergence(seq_a, seq_c)
    assert diverge is not None, "target sequences identical across seeds — seed not wired"
    print(f"  first diverging record: batch {diverge} (target {seq_a[diverge]} vs {seq_c[diverge]})")
    if diverge == 0:
        print("  first-line target differs (subset): PASS")
    else:
        print(f"  note: subset first targets coincide ({first_a}) — a 5-class label-level "
              "collision; streams diverge at record " + str(diverge))
    print("  cross-seed stream differs (sha256 + target sequence): PASS")

    # --- 3. First-line target differs on full DTD (literal criterion) -------
    print("\n--- First-line target differs (full DTD, 47 classes) ---")
    full_a = first_batch_target_full_dtd(SEED_A)
    full_b = first_batch_target_full_dtd(SEED_B)
    print(f"  seed {SEED_A} first target: {full_a}")
    print(f"  seed {SEED_B} first target: {full_b}")
    assert full_a != full_b, f"full-DTD first target identical across seeds: {full_a}"
    print("  first-line target differs (full DTD): PASS")

    print("\n" + "=" * 60)
    print("  Loader determinism: ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
