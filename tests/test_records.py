#!/usr/bin/env python3
"""Script-style tests for utils/records.py (plan task 1).

Covers:
  (a) fixture records with a fake config dict — schema fields + header
      ``resolved_config`` round-trip;
  (b) fp16 tensor → float32 handling (JSON contains floats, not tensor repr);
  (c) ``RECORD_DIR`` unset → zero files created anywhere (no-op);
  (d) ``None`` fields serialized as JSON ``null`` (never empty lists);
  (e) ``os.makedirs(RECORD_DIR, exist_ok=True)`` for nested paths.

Usage:
    python tests/test_records.py                              # all checks
    env -u RECORD_DIR python tests/test_records.py --check-nodir  # no-op only

Exits 0 and prints "records: ALL TESTS PASSED" on success, 1 otherwise.
"""
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402  # only the TEST may import torch; the writer must not

from utils.records import (  # noqa: E402
    write_record,
    write_record_header,
    write_summary,
)

SAMPLE_SCHEMA_FIELDS = {
    "batch_idx", "target", "pred", "correct", "conf", "quality_gate",
    "gate_mode", "proto_alpha", "logits", "proto_stats",
}
LOGITS_SCHEMA_FIELDS = {"clip", "image_proto", "patch_proto", "final"}
PROTO_STATS_SCHEMA_FIELDS = {"true", "pred"}

FAKE_CONFIG = {
    "alpha": 0.01,
    "T": 20.0,
    "multi_gate": False,
    "nested": {"proto_alpha_max": 0.5, "enabled": True, "values": [1, 2, 3]},
}


def check(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── (a) + (d) happy path: full schema round-trip, None → null ──────────────
def test_happy_path(record_dir):
    os.environ["RECORD_DIR"] = record_dir

    write_record_header(
        method="FixtureMethod", dataset="dtd", seed=7, C=3,
        classnames=["a", "b", "c"], resolved_config=FAKE_CONFIG,
    )
    write_record(
        batch_idx=0, target=0, pred=0, correct=True, conf=0.9,
        quality_gate=0.5, gate_mode="single", proto_alpha=[0.1, 0.2, 0.3],
        logits={
            "clip": [0.1, 0.2, 0.3],
            "image_proto": [0.4, 0.5, 0.6],
            "patch_proto": None,
            "final": [0.7, 0.8, 0.9],
        },
        proto_stats={
            "true": {"n_images": 5, "n_clusters": 2},
            "pred": {"n_images": 3, "n_clusters": 1},
        },
    )
    write_record(
        batch_idx=1, target=2, pred=1, correct=False, conf=0.4,
        quality_gate=None, gate_mode=None, proto_alpha=None,
        logits={"clip": None, "image_proto": None, "patch_proto": None,
                "final": [0.9, 0.8, 0.7]},
        proto_stats={"true": None, "pred": None},
    )
    write_summary(
        method="FixtureMethod", dataset="dtd", seed=7, total=2, acc=50.0,
        per_class={
            "a": {"total": 1, "correct": 1, "acc": 100.0,
                  "n_images_end": 9, "n_clusters_end": 3},
            "b": {"total": 1, "correct": 0, "acc": 0.0},
        },
    )

    jsonl_path = os.path.join(record_dir, "records.jsonl")
    summary_path = os.path.join(record_dir, "summary.json")
    check(os.path.exists(jsonl_path), "records.jsonl created in RECORD_DIR")
    check(os.path.exists(summary_path), "summary.json created in RECORD_DIR")

    lines = _read_jsonl(jsonl_path)
    check(len(lines) == 3, "JSONL has exactly header + 2 sample lines")

    # header
    hdr = lines[0]
    check(hdr.get("__header__") is True, "line 0 is the __header__ object")
    check(hdr.get("method") == "FixtureMethod", "header.method")
    check(hdr.get("dataset") == "dtd", "header.dataset")
    check(hdr.get("seed") == 7 and isinstance(hdr.get("seed"), int),
          "header.seed is int")
    check(hdr.get("C") == 3, "header.C")
    check(hdr.get("classnames") == ["a", "b", "c"], "header.classnames")
    check(hdr.get("resolved_config") == FAKE_CONFIG,
          "header.resolved_config round-trips the fixture config")
    check(isinstance(hdr.get("created_at"), str), "header.created_at is iso string")

    # sample line 1 — full schema
    r = lines[1]
    check(set(r.keys()) == SAMPLE_SCHEMA_FIELDS, "sample line has exact schema fields")
    check(set(r["logits"].keys()) == LOGITS_SCHEMA_FIELDS, "logits has exactly 4 fields")
    check(set(r["proto_stats"].keys()) == PROTO_STATS_SCHEMA_FIELDS,
          "proto_stats has true/pred")
    check(r["batch_idx"] == 0 and r["target"] == 0 and r["pred"] == 0,
          "batch_idx/target/pred")
    check(r["correct"] is True, "correct serializes as bool true")
    check(r["conf"] == 0.9, "conf")
    check(r["quality_gate"] == 0.5, "quality_gate")
    check(r["gate_mode"] == "single", "gate_mode")
    check(r["proto_alpha"] == [0.1, 0.2, 0.3], "proto_alpha")
    check(r["logits"]["final"] == [0.7, 0.8, 0.9], "logits.final")
    check(r["logits"]["patch_proto"] is None, "logits.patch_proto is null")
    check(r["proto_stats"]["true"] == {"n_images": 5, "n_clusters": 2},
          "proto_stats.true")
    check(r["proto_stats"]["pred"] == {"n_images": 3, "n_clusters": 1},
          "proto_stats.pred")

    # sample line 2 — (d) None fields → null, not empty lists
    r2 = lines[2]
    check(r2["quality_gate"] is None, "None quality_gate serialized as null")
    check(r2["gate_mode"] is None, "None gate_mode serialized as null")
    check(r2["proto_alpha"] is None, "None proto_alpha serialized as null")
    check(r2["logits"]["clip"] is None and r2["logits"]["image_proto"] is None,
          "None logits fields serialized as null")
    check(r2["proto_stats"]["true"] is None and r2["proto_stats"]["pred"] is None,
          "None proto_stats serialized as null")
    check(r2["correct"] is False, "correct serializes as bool false")

    # summary
    with open(summary_path) as f:
        summary = json.load(f)
    check(summary["method"] == "FixtureMethod", "summary.method")
    check(summary["dataset"] == "dtd", "summary.dataset")
    check(summary["seed"] == 7, "summary.seed")
    check(summary["total"] == 2 and summary["acc"] == 50.0, "summary.total/acc")
    check(summary["per_class"]["a"] == {
        "total": 1, "correct": 1, "acc": 100.0,
        "n_images_end": 9, "n_clusters_end": 3,
    }, "summary.per_class.a incl. n_*")
    check(summary["per_class"]["b"] == {"total": 1, "correct": 0, "acc": 0.0},
          "summary.per_class.b (n_* omitted)")


# ── (b) fp16 tensor → float32 handling ─────────────────────────────────────
def test_fp16():
    d = tempfile.mkdtemp(prefix="rec_fp16_")
    os.environ["RECORD_DIR"] = d
    try:
        write_record_header(method="FP16", dataset="dtd", seed=1, C=2,
                            classnames=["a", "b"], resolved_config={"alpha": 0.01})
        write_record(
            batch_idx=0, target=0, pred=0, correct=True,
            conf=torch.tensor(0.5, dtype=torch.float16),
            quality_gate=torch.tensor(0.75, dtype=torch.float16),
            gate_mode="single", proto_alpha=None,
            logits={
                "clip": torch.tensor([0.1, 0.2], dtype=torch.float16),
                "image_proto": torch.tensor([0.3, 0.4], dtype=torch.float16),
                "patch_proto": None,
                "final": torch.tensor([0.6, 0.7], dtype=torch.float16),
            },
            proto_stats={"true": None, "pred": None},
        )
        jsonl_path = os.path.join(d, "records.jsonl")
        raw = open(jsonl_path).read()
        check("tensor" not in raw.lower(),
              "serialized JSONL contains no tensor repr")
        r = _read_jsonl(jsonl_path)[1]
        check(all(isinstance(v, float) for v in r["logits"]["clip"]),
              "fp16 logits are python floats")
        check(all(isinstance(v, float) for v in r["logits"]["final"]),
              "fp16 final logits are python floats")
        check(isinstance(r["conf"], float) and abs(r["conf"] - 0.5) < 1e-3,
              "fp16 scalar conf → python float via .item()")
        check(isinstance(r["quality_gate"], float) and abs(r["quality_gate"] - 0.75) < 1e-3,
              "fp16 scalar quality_gate → python float")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── (c) RECORD_DIR unset → strict no-op ────────────────────────────────────
def test_noop():
    saved = os.environ.get("RECORD_DIR")
    os.environ.pop("RECORD_DIR", None)
    d = tempfile.mkdtemp(prefix="rec_noop_")
    try:
        write_record_header(method="Noop", dataset="dtd", seed=1, C=2,
                            classnames=["a", "b"], resolved_config={"alpha": 0.01})
        write_record(
            batch_idx=0, target=0, pred=0, correct=True, conf=0.5,
            quality_gate=None, gate_mode=None, proto_alpha=None,
            logits={"clip": [0.1, 0.2], "image_proto": [0.3, 0.4],
                    "patch_proto": None, "final": [0.5, 0.6]},
            proto_stats={"true": None, "pred": None},
        )
        write_summary(method="Noop", dataset="dtd", seed=1, total=1, acc=100.0,
                      per_class={"a": {"total": 1, "correct": 1, "acc": 100.0}})
        check(os.listdir(d) == [], "zero files created anywhere when RECORD_DIR unset")
        check(not os.path.exists(os.path.join(d, "records.jsonl")),
              "no records.jsonl created")
        check(not os.path.exists(os.path.join(d, "summary.json")),
              "no summary.json created")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if saved is not None:
            os.environ["RECORD_DIR"] = saved
        else:
            os.environ.pop("RECORD_DIR", None)


# ── (e) nested-path makedirs ───────────────────────────────────────────────
def test_nested_path():
    d = tempfile.mkdtemp(prefix="rec_nested_")
    nested = os.path.join(d, "nested", "deep", "dir")
    os.environ["RECORD_DIR"] = nested
    try:
        write_record_header(method="Nested", dataset="dtd", seed=1, C=1,
                            classnames=["a"], resolved_config={})
        check(os.path.exists(os.path.join(nested, "records.jsonl")),
              "writer os.makedirs creates nested RECORD_DIR path")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        os.environ.pop("RECORD_DIR", None)


def main():
    print("records: test_records.py")
    if "--check-nodir" in sys.argv:
        test_noop()
        print("no-op verified")
        return 0

    # Happy path honors an externally-set RECORD_DIR (QA scenario reads the
    # fixture back afterwards); otherwise it uses a private temp dir.
    external = os.environ.get("RECORD_DIR")
    record_dir = external or tempfile.mkdtemp(prefix="rec_happy_")
    try:
        test_happy_path(record_dir)
        test_fp16()
        test_noop()
        test_nested_path()
    finally:
        if not external:
            shutil.rmtree(record_dir, ignore_errors=True)

    print("records: ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
