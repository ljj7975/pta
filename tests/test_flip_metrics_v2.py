#!/usr/bin/env python3
"""
Script-style tests for the ground-truth-aware flip metrics v2
(scripts/analyze_records.py::flip_metrics_v2 + flips-v2 CLI).  No GPU, no pytest.

Builds a synthetic fixture with KNOWN targets where:

    * clip (text-only) is wrong and patch is right  -> corrections
    * clip is right and patch is wrong              -> regressions
    * a zero-cluster class (proto_stats.pred.n_clusters == 0) is present
      -> its patch rows must be marked "patch contribution N/A"

The plan's acceptance example is asserted exactly:
3 corrections / 1 regression -> flip accuracy 0.75, net +2 (patch-alone row).

Usage:
    python tests/test_flip_metrics_v2.py
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "analyze_records.py")

# Same fixture convention as tests/test_analyze_records.py.
_FIX_NAMES = [
    "apple", "banana", "cherry", "date", "elderberry", "fig",
]


def _load_analyze():
    spec = importlib.util.spec_from_file_location("analyze_records", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load_analyze()


def rec(target, clip, patch=None, img=None, clusters=0):
    """One per-sample JSONL record (T1 schema) with hand-chosen logits."""
    n = len(_FIX_NAMES)
    return {
        "batch_idx": 0,
        "target": target,
        "pred": target,
        "correct": False,
        "conf": 0.9,
        "quality_gate": None,
        "gate_mode": "single",
        "proto_alpha": [1.0] * n,
        "logits": {
            "clip": list(clip),
            "image_proto": list(img) if img is not None else None,
            "patch_proto": list(patch) if patch is not None else None,
            "final": list(clip),
        },
        "proto_stats": {
            "true": {"n_images": 10, "n_clusters": clusters},
            "pred": {"n_images": 10, "n_clusters": clusters},
        },
    }


def build_fixture():
    """7 records over classes 0-5; classes 0-4 have clusters, class 5 does not.

    Logit design (C=6, tau_img=80, tau_patch_proto=10):
      A t0  clip->1 (wrong)  img->0 (right)  patch->0 (right)
      B t1  clip->2 (wrong)  img->1 (right)  patch->1 (right)
      C t2  clip->3 (wrong)  img->3 (wrong)  patch->2 (right)
      D t3  clip->3 (right)  img->3 (right)  patch->4 (wrong)
      G t4  clip->4 (right)  img->4 (right)  patch->4 (right)
      E t5  clip->5 (right)  img->5 (right)  patch->5 (right)  clusters=0
      F t5  clip->0 (wrong)  img->0 (wrong)  patch->5 (right)  clusters=0
    """
    z = [0.0] * 6
    five = [0.0] * 6
    five[5] = 10.0
    return [
        # class 0 (apple)
        rec(0, [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            patch=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            img=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], clusters=3),
        # class 1 (banana)
        rec(1, [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            patch=[0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            img=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], clusters=3),
        # class 2 (cherry)
        rec(2, [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            patch=[0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            img=list(z), clusters=3),
        # class 3 (date)
        rec(3, [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            patch=[0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            img=list(z), clusters=3),
        # class 4 (elderberry): right everywhere -> no corr/reg, counted in n
        rec(4, [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            patch=[0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            img=list(z), clusters=3),
        # class 5 (fig): NO clusters -> patch rows must be masked
        rec(5, list(five), patch=list(five), img=list(z), clusters=0),
        rec(5, [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            patch=list(five), img=list(z), clusters=0),
    ]


def assert_close(got, want, msg):
    assert abs(got - want) < 1e-6, "{}: got {} want {}".format(msg, got, want)


def test_flip_metrics_v2_rates():
    m = ar.flip_metrics_v2(build_fixture(), tau_img=80.0, tau_patch_proto=10.0)
    a = m["aggregate"]

    # text-only baseline: only D, G, E are clip-correct -> 3/7.
    assert a["text_only"]["n"] == 7, a["text_only"]
    assert a["text_only"]["correct"] == 3, a["text_only"]
    assert_close(a["text_only"]["acc"], 300.0 / 7.0, "text-only acc")

    # +image (PTA): A, B corrected (text wrong -> image right); no regressions.
    assert a["image"]["n"] == 7, a["image"]
    assert a["image"]["corr"] == 2, a["image"]
    assert a["image"]["reg"] == 0, a["image"]
    assert_close(a["image"]["acc"], 100.0, "image flip acc")
    assert a["image"]["net"] == 2, a["image"]

    # +patch alone (cluster-masked): 3 corrections / 1 regression -> 0.75, +2.
    assert a["patch_alone"]["n"] == 5, a["patch_alone"]      # A-D, G
    assert a["patch_alone"]["corr"] == 3, a["patch_alone"]   # A, B, C
    assert a["patch_alone"]["reg"] == 1, a["patch_alone"]    # D
    assert_close(a["patch_alone"]["acc"], 75.0, "patch-alone flip acc")
    assert a["patch_alone"]["net"] == 2, a["patch_alone"]
    assert a["patch_alone"]["masked"] == 2, a["patch_alone"]  # E, F (clusters=0)

    # +patch with image vs image-only: C corrected, D regressed.
    assert a["patch_image"]["n"] == 5, a["patch_image"]
    assert a["patch_image"]["corr"] == 1, a["patch_image"]
    assert a["patch_image"]["reg"] == 1, a["patch_image"]
    assert_close(a["patch_image"]["acc"], 50.0, "patch-image flip acc")
    assert a["patch_image"]["net"] == 0, a["patch_image"]
    assert a["patch_image"]["masked"] == 2, a["patch_image"]
    print("  [ok] v2 rates: 3 corr / 1 reg -> acc 0.75, net +2 on patch-alone")

    # Cluster stratification: class 5 (fig) has zero cluster-bearing records.
    p5 = m["per_class"][5]
    assert p5["n"] == 2 and p5["n_cluster"] == 0, p5
    assert p5["patch_alone"]["acc"] is None, p5["patch_alone"]
    assert p5["patch_image"]["acc"] is None, p5["patch_image"]
    # class 5 still present in text-only and +image rows (mechanisms need no clusters).
    assert p5["text_only"]["correct"] == 1, p5["text_only"]
    assert p5["image"]["n"] == 2, p5["image"]
    # classes with clusters DO get patch attribution.
    assert m["per_class"][0]["patch_alone"]["corr"] == 1, m["per_class"][0]
    assert m["per_class"][2]["patch_alone"]["corr"] == 1, m["per_class"][2]
    assert m["per_class"][3]["patch_alone"]["reg"] == 1, m["per_class"][3]
    print("  [ok] v2 cluster mask: zero-cluster class excluded from patch rows only")


def test_flip_metrics_v2_tau_absent():
    samples = build_fixture()
    # PTA/ZeroShot path: tau_patch_proto absent -> patch-with-image disabled,
    # patch-alone still computed (it only needs patch_proto logits).
    m = ar.flip_metrics_v2(samples, tau_img=80.0, tau_patch_proto=None)
    assert m["aggregate"]["patch_image"]["n"] == 0, m["aggregate"]["patch_image"]
    assert m["aggregate"]["patch_image"]["masked"] == 7, m["aggregate"]["patch_image"]
    assert m["aggregate"]["patch_alone"]["n"] == 5, m["aggregate"]["patch_alone"]
    # tau_img absent -> +image row unavailable too.
    m2 = ar.flip_metrics_v2(samples, tau_img=None, tau_patch_proto=10.0)
    assert m2["aggregate"]["image"]["n"] == 0, m2["aggregate"]["image"]
    assert m2["aggregate"]["image"]["masked"] == 0, m2["aggregate"]["image"]
    print("  [ok] v2 tau-absent: patch-image masked without tau_patch_proto")


def test_resolve_taus():
    cfg = {"fusion": {"tau_image_proto": 80.0, "tau_patch_proto": 10.0}}
    assert ar.resolve_tau_img(cfg) == 80.0
    assert ar.resolve_tau_patch_proto(cfg) == 10.0
    cfg_pta = {"fusion": {"tau_image_proto": 100.0}}
    assert ar.resolve_tau_img(cfg_pta) == 100.0
    assert ar.resolve_tau_patch_proto(cfg_pta) is None
    assert ar.resolve_tau_patch_proto({}) is None
    print("  [ok] v2 tau resolution: nested fusion read + PTA absent -> None")


def _write_fixture_dir(root):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    d = root / "PatchModPTA-fixed-s1"
    d.mkdir(parents=True, exist_ok=True)
    header = {
        "__header__": True,
        "method": "PatchModPTA-fixed",
        "dataset": "dtd",
        "seed": 1,
        "C": len(_FIX_NAMES),
        "classnames": list(_FIX_NAMES),
        "resolved_config": {"fusion": {"tau_image_proto": 80.0, "tau_patch_proto": 10.0}},
        "created_at": "2026-08-05T00:00:00",
    }
    lines = [json.dumps(header)]
    for s in build_fixture():
        lines.append(json.dumps(s))
    (d / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def test_flips_v2_cli():
    root = _write_fixture_dir("/tmp/rec_fix_v2")
    out = "/tmp/rec_fix_v2/flips2.md"
    proc = subprocess.run(
        [sys.executable, SCRIPT_PATH, "flips-v2", "--records", str(root), "--out", out],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = Path(out).read_text(encoding="utf-8")

    # Aggregate rows with exact numbers.
    assert "| text-only (baseline) | 7 | 3 | — | 42.86 | — | — |" in text, text
    assert "| +image (PTA) | 7 | 2 | 0 | 100.00 | +2 | 0 |" in text, text
    assert "| +patch alone | 5 | 3 | 1 | 75.00 | +2 | 2 |" in text, text
    assert "| +patch with image | 5 | 1 | 1 | 50.00 | +0 | 2 |" in text, text
    # Zero-cluster class (fig) marked N/A; still present in text-only/+image.
    assert "| fig | 2 | 0 | 50.00 | — | patch contribution N/A | patch contribution N/A |" in text, text
    # Clustered classes carry patch attribution.
    assert "| apple | 1 | 1 | 0.00 | 1/0 (100.00%, +1) | 1/0 (100.00%, +1) |" in text, text
    # tau resolution surfaced in the label header.
    assert "tau_img = 80.0, tau_patch_proto = 10.0" in text, text
    print("  [ok] v2 CLI: markdown aggregate + per-class (N/A on zero-cluster class)")


def main():
    print("=" * 60)
    print("  analyze_records flip metrics v2 tests (synthetic, no GPU)")
    print("=" * 60)
    test_flip_metrics_v2_rates()
    test_flip_metrics_v2_tau_absent()
    test_resolve_taus()
    test_flips_v2_cli()
    print("=" * 60)
    print("flip metrics v2: ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
