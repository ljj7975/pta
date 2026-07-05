#!/usr/bin/env python3
"""Fidelity check script for post-refactoring bit-exact verification.

Runs capture_fidelity.py for a given method and compares the resulting
per-sample accuracies against a pre-captured snapshot.  Exits with
code 0 if all samples match, code 1 on any mismatch.

Snapshot mapping
----------------
  pta                      -> tests/fidelity_snapshots/pta_oxford_pets.json
  patch_modulated_pta      -> tests/fidelity_snapshots/exp12_oxford_pets.json
  <other>                  -> tests/fidelity_snapshots/<method>_<dataset>.json

Usage
-----
  python tests/check_fidelity.py --method pta \\
      --dataset oxford_pets --backbone ViT-B/16 --seed 1 --max-batches 50

  python tests/check_fidelity.py --method patch_modulated_pta \\
      --dataset oxford_pets --backbone ViT-B/16 --seed 1 --max-batches 50

  # Dry-run: just verify the snapshot file exists, don't run inference
  python tests/check_fidelity.py --method pta --dataset oxford_pets --dry-run

  # Re-capture expected snapshot (after deliberate algorithm change)
  python tests/check_fidelity.py --method pta --dataset oxford_pets \\
      --update-snapshots
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# Repo root on path
try:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _REPO_ROOT = os.getcwd()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SNAPSHOTS_DIR = os.path.join(_REPO_ROOT, "tests", "fidelity_snapshots")

# Where the expected snapshot lives for each method
_SNAPSHOT_KEY = {
    "pta": "pta",
    "patch_modulated_pta": "exp12",
}


def snapshot_path(method: str, dataset: str) -> str:
    key = _SNAPSHOT_KEY.get(method, method)
    return os.path.join(SNAPSHOTS_DIR, f"{key}_{dataset}.json")


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Compare new code output against fidelity snapshot.",
    )
    parser.add_argument(
        "--method",
        default="pta",
        help="Adapter method name (default: pta).",
    )
    parser.add_argument(
        "--config",
        default="configs",
        help="Config directory to pass to capture_fidelity.py (default: configs).",
    )
    parser.add_argument(
        "--dataset",
        default="oxford_pets",
        help="Dataset name (default: oxford_pets).",
    )
    parser.add_argument(
        "--backbone",
        default="ViT-B/16",
        choices=["RN50", "ViT-B/16"],
        help="CLIP backbone (default: ViT-B/16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Max batches to process (default: 50).",
    )
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Root directory containing datasets (default: ./data).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify snapshot file exists; do not run inference.",
    )
    parser.add_argument(
        "--update-snapshots",
        action="store_true",
        help="Re-capture expected snapshot from current code and overwrite.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample comparison details on mismatch.",
    )
    return parser.parse_args()


def run_capture(args, output_dir: str) -> str:
    """Run capture_fidelity.py and return path to the resulting JSON."""
    capture_script = os.path.join(_REPO_ROOT, "tests", "capture_fidelity.py")
    cmd = [
        sys.executable, capture_script,
        "--method", args.method,
        "--config", args.config,
        "--dataset", args.dataset,
        "--backbone", args.backbone,
        "--seed", str(args.seed),
        "--max-batches", str(args.max_batches),
        "--output", output_dir,
        "--data-root", args.data_root,
    ]
    env = os.environ.copy()
    env["MAX_BATCHES"] = str(args.max_batches)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    result = subprocess.run(cmd, env=env, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"ERROR: capture_fidelity.py exited with code {result.returncode}")
        sys.exit(1)

    captured_path = os.path.join(output_dir, f"{args.method}_{args.dataset}.json")
    if not os.path.exists(captured_path):
        print(f"ERROR: Expected captured snapshot at {captured_path} not found")
        sys.exit(1)
    return captured_path


def compare(expected: list, actual: list, verbose: bool) -> bool:
    """Return True if all samples match exactly."""
    if len(expected) != len(actual):
        print(f"MISMATCH: length {len(expected)} expected vs {len(actual)} actual")
        return False

    mismatches = []
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e != a:
            mismatches.append((i, e, a))

    if mismatches:
        print(f"MISMATCH: {len(mismatches)}/{len(expected)} samples differ")
        if verbose:
            for i, e, a in mismatches[:20]:
                print(f"  Sample {i:4d}: expected={e:.2f}  actual={a:.2f}")
            if len(mismatches) > 20:
                print(f"  ... and {len(mismatches) - 20} more")
        return False

    print(f"PASS: {len(expected)}/{len(expected)} samples match exactly")
    return True


def main():
    args = get_arguments()

    expected_path = snapshot_path(args.method, args.dataset)

    # ── Dry-run mode ──────────────────────────────────────────────────────
    if args.dry_run:
        if os.path.exists(expected_path):
            data = json.load(open(expected_path))
            n = len(data.get("per_sample_accuracies", []))
            print(f"Snapshot found: {expected_path} ({n} samples)")
            sys.exit(0)
        else:
            print(f"Snapshot not found: {expected_path}")
            sys.exit(1)

    # ── Update-snapshots mode ─────────────────────────────────────────────
    if args.update_snapshots:
        print(f"Re-capturing snapshot for {args.method} / {args.dataset} ...")
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        # Capture into a temp dir, then move to the canonical path
        with tempfile.TemporaryDirectory() as tmpdir:
            captured = run_capture(args, tmpdir)
            # read data and write to expected path with canonical key name
            data = json.load(open(captured))
            with open(expected_path, "w") as f:
                json.dump(data, f, indent=2)
        print(f"Snapshot updated: {expected_path}")
        sys.exit(0)

    # ── Normal fidelity check ─────────────────────────────────────────────
    if not os.path.exists(expected_path):
        print(f"ERROR: Expected snapshot not found at {expected_path}")
        print("  Run capture_fidelity.py with the OLD code first, or use --update-snapshots")
        sys.exit(1)

    expected_data = json.load(open(expected_path))
    expected_acc = expected_data.get("per_sample_accuracies", [])
    print(f"Expected snapshot: {expected_path} ({len(expected_acc)} samples)")

    print(f"Running {args.method} on {args.dataset} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        captured = run_capture(args, tmpdir)
        actual_data = json.load(open(captured))

    actual_acc = actual_data.get("per_sample_accuracies", [])
    print(f"Captured:          {len(actual_acc)} samples")

    ok = compare(expected_acc, actual_acc, args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
