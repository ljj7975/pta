#!/usr/bin/env python3
"""Script-style GPU test for PatchModPTA per-sample recording (plan task 5).

Runs PatchModulatedPTA on dtd with MAX_BATCHES=30 twice — once with
RECORD_DIR=/tmp/rec_pm_a, once with RECORD_DIR unset — and verifies:

  (a) final_acc is bit-identical between the two runs (behavior neutrality);
  (b) records.jsonl has exactly header + 30 per-sample lines;
  (c) every line carries the full logits dict with 4 arrays of length C=47;
  (d) proto_stats.true/pred n_clusters + n_images match states[c] captured
      at record time (spot-check via per-batch snapshots);
  (e) gate_mode == "single" (effective mode; multi_gate config is mis-read).

GPU required for the bit-identical comparison.

Usage (sbatch):
    sbatch scripts/slurm_test_patchmod_records.sh
"""
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch

from encoder import create_encoder_instance
from models.patch_modulated_pta import PatchModulatedPTAAdapter
from utils import build_test_data_loader, clip_classifier, get_config_file

SEED = 42
DATASET = "dtd"
BACKBONE = "ViT-B/16"
CONFIG_DIR = "configs/patch_modulated_pta"
MAX_BATCHES = 30
RECORD_DIR_A = "/tmp/rec_pm_a"
C_EXPECTED = 47
RESULT_FILE = "/tmp/rec_pm_result.txt"
RESULT_LABEL = "PatchModPTA-T5"


def check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def set_full_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def capture_batch_states(adapter):
    """Snapshot per-class (n_clusters, n_images) at each batch's start.

    The adapter records proto_stats from ``states`` BEFORE the update for that
    batch; compute_patch_logits is the first states-touching call per batch, so
    its entry snapshot is exactly the state the record line describes.
    """
    orig = adapter.patch_level.compute_patch_logits
    snapshots = []

    def wrapped(images, encoder, states, *args, **kwargs):
        snapshots.append(
            [(int(s["centers"].shape[0]), int(s["n_images"])) for s in states]
        )
        return orig(images, encoder, states, *args, **kwargs)

    adapter.patch_level.compute_patch_logits = wrapped
    return snapshots


def run_patchmod(encoder, record_dir, capture=False):
    set_full_seed(SEED)
    cfg = get_config_file(CONFIG_DIR, DATASET)
    adapter = PatchModulatedPTAAdapter(cfg)
    snapshots = capture_batch_states(adapter) if capture else None
    test_loader, classnames, template = build_test_data_loader(
        DATASET, "./data", encoder.preprocess, shuffle=True
    )
    text_embeddings = clip_classifier(classnames, template, encoder)
    if record_dir is None:
        os.environ.pop("RECORD_DIR", None)
    else:
        os.environ["RECORD_DIR"] = record_dir
    acc = adapter.run(test_loader, encoder, text_embeddings, DATASET)
    return acc, snapshots


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if not torch.cuda.is_available():
        print("FAIL: GPU required")
        return 1

    os.environ["MAX_BATCHES"] = str(MAX_BATCHES)
    os.environ["SEED"] = str(SEED)
    os.environ["RESULT_LABEL"] = RESULT_LABEL
    os.environ["RESULT_FILE"] = RESULT_FILE
    shutil.rmtree(RECORD_DIR_A, ignore_errors=True)

    encoder = create_encoder_instance("clip", model_type=BACKBONE, device="cuda")

    print("--- Run 1: RECORD_DIR set ---")
    acc_a, snapshots = run_patchmod(encoder, RECORD_DIR_A, capture=True)

    print("--- Run 2: RECORD_DIR unset ---")
    acc_b, _ = run_patchmod(encoder, None)

    # (a) behavior neutrality — bit-identical final accuracy
    check(
        acc_a == acc_b,
        f"(a) final_acc bit-identical with vs without recording: {acc_a} == {acc_b}",
    )

    # (b) JSONL: header + 30 per-sample lines
    jsonl = _read_jsonl(os.path.join(RECORD_DIR_A, "records.jsonl"))
    check(
        len(jsonl) == 1 + MAX_BATCHES,
        f"(b) records.jsonl has header + {MAX_BATCHES} lines (got {len(jsonl)})",
    )
    hdr = jsonl[0]
    check(hdr.get("__header__") is True, "line 0 is the __header__ object")
    check(hdr.get("method") == RESULT_LABEL, "header.method == RESULT_LABEL")
    check(hdr.get("dataset") == DATASET, "header.dataset == dtd")
    check(hdr.get("seed") == SEED, "header.seed == 42")
    check(hdr.get("C") == C_EXPECTED, "header.C == 47")
    check(hdr.get("classnames") == [], "header.classnames == []")
    check(
        isinstance(hdr.get("resolved_config"), dict) and hdr["resolved_config"],
        "header.resolved_config is the resolved cfg dict",
    )

    lines = jsonl[1:]
    check(
        [l["batch_idx"] for l in lines] == list(range(MAX_BATCHES)),
        "per-sample batch_idx is 0..29",
    )

    # (c) full logits dict, 4 arrays of length C=47 on every line
    for idx, line in enumerate(lines):
        lg = line.get("logits")
        if not (
            isinstance(lg, dict)
            and set(lg.keys()) == {"clip", "image_proto", "patch_proto", "final"}
        ):
            check(False, f"(c) line {idx} logits keys {lg and set(lg.keys())}")
        for k in ("clip", "image_proto", "patch_proto", "final"):
            if not (isinstance(lg[k], list) and len(lg[k]) == C_EXPECTED):
                check(False, f"(c) line {idx} logits.{k} length {len(lg[k])}")
    check(
        True,
        f"(c) every line has full logits dict with 4 arrays of length C={C_EXPECTED}",
    )

    # (d) proto_stats.true/pred == states[c] at record time
    ok = len(snapshots) == MAX_BATCHES
    for i, line in enumerate(lines):
        if not ok:
            break
        t, p = line["target"], line["pred"]
        for which, cls_idx in (("true", t), ("pred", p)):
            stats = line["proto_stats"][which]
            if stats["n_clusters"] != snapshots[i][cls_idx][0]:
                ok = False
                break
            if stats["n_images"] != snapshots[i][cls_idx][1]:
                ok = False
                break
    check(
        ok,
        "(d) proto_stats.true/pred n_clusters+n_images match states[c] at record time",
    )

    # (e) effective gate mode is single
    check(
        all(l["gate_mode"] == "single" for l in lines),
        "(e) gate_mode == 'single' on every line",
    )

    # summary sanity: total/acc + per-class n_* keys present
    with open(os.path.join(RECORD_DIR_A, "summary.json")) as f:
        summary = json.load(f)
    check(
        summary["total"] == MAX_BATCHES and summary["acc"] == acc_a,
        "summary.total/acc consistent",
    )
    check(
        len(summary["per_class"]) == C_EXPECTED,
        "summary.per_class has C=47 entries",
    )
    check(
        all(
            "n_images_end" in v and "n_clusters_end" in v
            for v in summary["per_class"].values()
        ),
        "summary.per_class entries carry n_images_end/n_clusters_end",
    )

    print("patchmod records + behavior-neutral: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
