#!/usr/bin/env python3
"""Script-style GPU tests for PTA env-gated per-sample recording (plan task 6).

Covers:
  (a) RECORD_DIR set -> header line + one JSONL line per batch; each line has
      logits.clip / image_proto / final of length C (= 47 for dtd),
      patch_proto = null, quality_gate / gate_mode / proto_alpha = null,
      proto_stats.true / pred = null; summary.json written with total=30.
  (b) RECORD_DIR unset -> final accuracy bit-identical to the RECORD_DIR run
      (behavior-neutral invariant: recording never perturbs results).

Usage:
    python -u tests/test_records_pta.py

Exits 0 and prints "pta records + behavior-neutral: ALL TESTS PASSED" on
success, 1 otherwise. Requires a GPU (dtd test split with CLIP ViT-B/16).

sbatch example:
    sbatch scripts/slurm_test_records_pta.sh
"""
import json
import os
import shutil
import sys
import tempfile

# Must be set BEFORE torch import for deterministic CuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import random  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import clip  # noqa: E402
from utils import get_config_file, build_test_data_loader  # noqa: E402
from models.pta import PTAAdapter  # noqa: E402

SEED = 42
DATASET = "dtd"
BACKBONE = "ViT-B/16"
CONFIG_DIR = "configs/PTA"
MAX_BATCHES = 30
C_CLASSES = 47  # dtd has 47 classes


def check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def build_text_embeddings(classnames, template, encoder):
    """Clip_classifier equivalent: tokenize before encode_text (repo convention).

    The vendored CLIP's ``encode_text`` takes already-tokenized tensors, so the
    texts are tokenized here (mirrors how every other test in this repo drives
    the text encoder). Returns a ``(D, C)`` CUDA tensor.
    """
    with torch.no_grad():
        embeddings = []
        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in template]
            tokens = clip.tokenize(texts).cuda()
            class_embedding = encoder.encode_text(tokens)
            class_embedding /= class_embedding.norm(dim=-1, keepdim=True)
            class_embedding = class_embedding.mean(dim=0)
            class_embedding /= class_embedding.norm()
            embeddings.append(class_embedding)
        return torch.stack(embeddings, dim=1).cuda()


def set_full_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def run_pta_once() -> float:
    """Run PTA on dtd (MAX_BATCHES=30), respecting current RECORD_DIR env."""
    set_full_seed(SEED)
    clip_model, preprocess = clip.load(BACKBONE)
    clip_model.eval()
    cfg = get_config_file(CONFIG_DIR, DATASET)
    adapter = PTAAdapter(cfg)
    test_loader, classnames, template = build_test_data_loader(
        DATASET, "./data", preprocess, shuffle=True
    )
    clip_weights = build_text_embeddings(classnames, template, clip_model)
    return adapter.run(test_loader, clip_model, clip_weights, DATASET)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def test_records(record_dir):
    lines = read_jsonl(os.path.join(record_dir, "records.jsonl"))
    check(len(lines) == MAX_BATCHES + 1, f"JSONL has header + {MAX_BATCHES} lines")

    hdr = lines[0]
    check(hdr.get("__header__") is True, "line 0 is the __header__ object")
    check(hdr.get("method") == "PTA", "header.method == PTA (RESULT_LABEL default)")
    check(hdr.get("dataset") == "dtd", "header.dataset == dtd")
    check(hdr.get("seed") == SEED, "header.seed == 42 (SEED env)")
    check(hdr.get("C") == C_CLASSES, "header.C == 47 (dtd classes)")
    check(hdr.get("classnames") == [], "header.classnames == []")
    check(isinstance(hdr.get("resolved_config"), dict), "header.resolved_config is dict")
    check(isinstance(hdr.get("created_at"), str), "header.created_at is iso string")

    for i, r in enumerate(lines[1:], start=1):
        check(r["batch_idx"] == i - 1, f"line {i}: batch_idx == {i - 1}")
        check(isinstance(r["target"], int), f"line {i}: target is int")
        check(isinstance(r["pred"], int), f"line {i}: pred is int")
        check(isinstance(r["correct"], bool), f"line {i}: correct is bool")
        check(isinstance(r["conf"], float), f"line {i}: conf is float")
        check(r["quality_gate"] is None, f"line {i}: quality_gate is null")
        check(r["gate_mode"] is None, f"line {i}: gate_mode is null")
        check(r["proto_alpha"] is None, f"line {i}: proto_alpha is null")
        check(
            set(r["logits"].keys()) == {"clip", "image_proto", "patch_proto", "final"},
            f"line {i}: logits has exact 4-field schema",
        )
        check(len(r["logits"]["clip"]) == C_CLASSES, f"line {i}: logits.clip len 47")
        check(
            len(r["logits"]["image_proto"]) == C_CLASSES,
            f"line {i}: logits.image_proto len 47",
        )
        check(r["logits"]["patch_proto"] is None, f"line {i}: logits.patch_proto is null")
        check(len(r["logits"]["final"]) == C_CLASSES, f"line {i}: logits.final len 47")
        check(
            set(r["proto_stats"].keys()) == {"true", "pred"},
            f"line {i}: proto_stats has true/pred",
        )
        check(r["proto_stats"]["true"] is None, f"line {i}: proto_stats.true is null")
        check(r["proto_stats"]["pred"] is None, f"line {i}: proto_stats.pred is null")

    with open(os.path.join(record_dir, "summary.json")) as f:
        summary = json.load(f)
    check(summary["method"] == "PTA", "summary.method == PTA")
    check(summary["dataset"] == "dtd", "summary.dataset == dtd")
    check(summary["seed"] == SEED, "summary.seed == 42")
    check(summary["total"] == MAX_BATCHES, "summary.total == 30")
    check(isinstance(summary["acc"], float), "summary.acc is float")
    check(len(summary["per_class"]) == C_CLASSES, "summary.per_class has C=47 entries")
    for c, v in summary["per_class"].items():
        check(
            set(v.keys()) == {"total", "correct", "acc"},
            f"summary.per_class.{c} carries total/correct/acc",
        )
        if v["total"] > 0:
            check(
                abs(v["acc"] - 100.0 * v["correct"] / v["total"]) < 1e-9,
                f"summary.per_class.{c}.acc == 100*correct/total",
            )


def main():
    print("=" * 60)
    print("  PTA records test: dtd, ViT-B/16, MAX_BATCHES=30")
    print("=" * 60)

    os.environ["MAX_BATCHES"] = str(MAX_BATCHES)
    os.environ["SEED"] = str(SEED)

    # ── Run 1: RECORD_DIR set ─────────────────────────────────────────────
    record_dir = tempfile.mkdtemp(prefix="rec_pta_")
    os.environ["RECORD_DIR"] = record_dir
    print("\n--- Run 1 (RECORD_DIR set) ---")
    acc_rec = run_pta_once()
    print(f"  accuracy (with records): {acc_rec:.4f}%")
    test_records(record_dir)

    # ── Run 2: RECORD_DIR unset (behavior-neutral) ────────────────────────
    os.environ.pop("RECORD_DIR", None)
    print("\n--- Run 2 (RECORD_DIR unset) ---")
    acc_plain = run_pta_once()
    print(f"  accuracy (no records): {acc_plain:.4f}%")

    print("\n" + "=" * 60)
    check(acc_rec == acc_plain,
          "accuracy bit-identical with vs without RECORD_DIR (behavior-neutral)")
    print("  pta records + behavior-neutral: ALL TESTS PASSED")
    print("=" * 60)
    shutil.rmtree(record_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: {e}")
        sys.exit(1)
