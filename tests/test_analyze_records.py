#!/usr/bin/env python3
"""
Script-style tests for scripts/analyze_records.py (no GPU, no pytest).

Builds synthetic JSONL fixtures (T1 schema) under /tmp and exercises the four
subcommands + offline flip diagnostics:

    (a) table      — hand-computed per-class accuracy on a small fixture
    (b) parity     — PASS and FAIL cases against a stub exp_results mapping
    (c) select     — bottom-k classnames returned in dataset.classnames order
    (d) hypothesis — monotonically increasing stream -> SUPPORT;
                     flat stream -> INCONCLUSIVE; decreasing stream -> REFUTE
    (e) flip_metrics — patch_flip / flip_vs_img / patch_only_flip on crafted logits

Usage:
    python tests/test_analyze_records.py
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

_FIX_NAMES = [
    "apple", "banana", "cherry", "date", "elderberry", "fig",
]


def _load_analyze():
    spec = importlib.util.spec_from_file_location("analyze_records", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ar = _load_analyze()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def reset_dir(path):
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_records(root, label, samples, seed=1, classnames=None, resolved_config=None):
    d = Path(root) / label
    d.mkdir(parents=True, exist_ok=True)
    header = {
        "__header__": True,
        "method": label,
        "dataset": "dtd",
        "seed": seed,
        "C": len(classnames or _FIX_NAMES),
        "classnames": list(classnames or _FIX_NAMES),
        "resolved_config": resolved_config or {"fusion": {"tau_image_proto": 80.0}},
        "created_at": "2026-08-05T00:00:00",
    }
    lines = [json.dumps(header)]
    for s in samples:
        lines.append(json.dumps(s))
    (d / "records.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample(target, correct, batch_idx=0, clusters=None, clip=None, final=None,
           img=None, patch=None):
    """One per-sample JSONL record following the T1 schema."""
    return {
        "batch_idx": batch_idx,
        "target": target,
        "pred": target if correct else (target + 1) % len(_FIX_NAMES),
        "correct": bool(correct),
        "conf": 0.9,
        "quality_gate": None,
        "gate_mode": "single",
        "proto_alpha": [0.0] * len(_FIX_NAMES),
        "logits": {
            "clip": list(clip) if clip is not None else [0.0] * len(_FIX_NAMES),
            "image_proto": list(img) if img is not None else None,
            "patch_proto": list(patch) if patch is not None else None,
            "final": list(final) if final is not None else [0.0] * len(_FIX_NAMES),
        },
        "proto_stats": {
            "true": {"n_images": 10, "n_clusters": clusters if clusters is not None else 0},
            "pred": {"n_images": 10, "n_clusters": clusters if clusters is not None else 0},
        },
    }


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, SCRIPT_PATH] + list(argv),
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# (a) table correctness
# ---------------------------------------------------------------------------

def test_table():
    root = reset_dir("/tmp/rec_fix")
    # class 0 (apple): 2 total / 1 correct -> 50.00%
    # class 1 (banana): 2 total / 2 correct -> 100.00%
    # class 2 (cherry): 1 total / 1 correct -> 100.00%
    samples = [
        sample(0, True, batch_idx=0),
        sample(0, False, batch_idx=1),
        sample(1, True, batch_idx=2),
        sample(1, True, batch_idx=3),
        sample(2, True, batch_idx=4),
    ]
    write_records(root, "PatchModPTA-CS-s1", samples, seed=1)
    write_records(root, "PTA-CS-s1", samples, seed=1)

    out = "/tmp/rec_fix/tables.md"
    rc = ar.cmd_table(ar.argparse.Namespace(records=str(root), out=out))
    assert rc == 0, "table subcommand should exit 0"
    text = Path(out).read_text(encoding="utf-8")

    # Hand-computed per-class rows.
    assert "| PatchModPTA | 1 | apple | 2 | 1 | 50.00 |" in text, text
    assert "| PatchModPTA | 1 | banana | 2 | 2 | 100.00 |" in text, text
    assert "| PatchModPTA | 1 | cherry | 1 | 1 | 100.00 |" in text, text
    # Cross-method diff (same fixture for both methods -> delta +0.00).
    assert "| apple | 50.00 | 50.00 | +0.00 |" in text, text
    print("  [ok] table: per-class rows + cross-method diff")


# ---------------------------------------------------------------------------
# (b) parity gate
# ---------------------------------------------------------------------------

def _write_parity_fixture(root, patch_correct, n=1000):
    root = reset_dir(root)
    # seed-1 PatchModPTA-CS with overall acc = patch_correct/n*100.
    p_samples = [sample(0, i < patch_correct, batch_idx=i) for i in range(n)]
    write_records(root, "PatchModPTA-CS-s1", p_samples, seed=1)
    # seed-1 PTA-CS with acc = 47.0 (470/1000) -> within 0.5 of 46.99.
    q_samples = [sample(0, i < 470, batch_idx=i) for i in range(n)]
    write_records(root, "PTA-CS-s1", q_samples, seed=1)


def _write_stub_exp_results(path, patch_ref=48.35, pta_ref=46.99):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "Method                                        dtd       eurosat          Avg\n"
        "-----------------------------------------------------------------------\n"
        "PatchModPTA-CS-2                            {:.2f}         62.28         59.48\n"
        "PTA-CS                                     {:.2f}         61.51         59.77\n".format(
            patch_ref, pta_ref),
        encoding="utf-8",
    )


def test_parity_pass():
    _write_parity_fixture("/tmp/rec_fix_par", patch_correct=484)  # 48.4%
    exp = "/tmp/rec_fix_par/exp_results.txt"
    _write_stub_exp_results(exp)

    proc = run_cli("parity", "--records", "/tmp/rec_fix_par", "--exp-results", exp)
    assert proc.returncode == 0, "PASS parity must exit 0\n" + proc.stdout + proc.stderr
    assert "PARITY GATE: PASS" in proc.stdout, proc.stdout
    assert "48.40" in proc.stdout and "46.99" in proc.stdout, proc.stdout
    print("  [ok] parity: PASS case exits 0 with values")


def test_parity_fail():
    _write_parity_fixture("/tmp/rec_fix_bad", patch_correct=900)  # 90.0% -> way off
    exp = "/tmp/rec_fix_bad/exp_results.txt"
    _write_stub_exp_results(exp)

    proc = run_cli("parity", "--records", "/tmp/rec_fix_bad", "--exp-results", exp)
    assert proc.returncode == 1, "FAIL parity must exit 1\n" + proc.stdout + proc.stderr
    assert "PARITY GATE: FAIL" in proc.stdout, proc.stdout
    assert "90.00" in proc.stdout, proc.stdout
    print("  [ok] parity: FAIL case exits 1")


# ---------------------------------------------------------------------------
# (c) select bottom-k in dataset order
# ---------------------------------------------------------------------------

def test_select():
    root = reset_dir("/tmp/rec_fix_sel")
    # Mean per-class acc (both seeds identical so means are deterministic):
    #   apple 0.0, banana 100, cherry 100, date 100, elderberry 50, fig 0.0
    def sel_samples(seed_base):
        s = []
        s += [sample(0, False, batch_idx=i) for i in range(2)]          # apple 0/2
        s += [sample(1, True, batch_idx=10 + i) for i in range(2)]      # banana 2/2
        s += [sample(2, True, batch_idx=20 + i) for i in range(2)]      # cherry 2/2
        s += [sample(3, True, batch_idx=30 + i) for i in range(2)]      # date 2/2
        s += [sample(4, False, batch_idx=40)]                           # elderberry 1/2
        s += [sample(4, True, batch_idx=41)]
        s += [sample(5, False, batch_idx=50 + i) for i in range(2)]     # fig 0/2
        return s

    write_records(root, "PatchModPTA-CS-s1", sel_samples(0), seed=1)
    write_records(root, "PatchModPTA-CS-s2", sel_samples(1), seed=2)
    write_records(root, "PTA-CS-s1", sel_samples(0), seed=1)
    write_records(root, "ZeroShot-CS-s1", sel_samples(0), seed=1)

    out = "/tmp/rec_fix_sel/subset.txt"
    proc = run_cli("select", "--records", "/tmp/rec_fix_sel", "--k", "2", "--out", out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    names = [l for l in Path(out).read_text(encoding="utf-8").splitlines() if l.strip()]
    assert names == ["apple", "fig"], names  # bottom-2, dataset order (apple idx0, fig idx5)
    assert "PatchModPTA-PTA" in proc.stdout and "apple" in proc.stdout, proc.stdout
    print("  [ok] select: bottom-k in dataset order + gap table printed")


# ---------------------------------------------------------------------------
# (d) hypothesis decision rule on synthetic streams
# ---------------------------------------------------------------------------

def build_rising_stream(seed):
    """First half all-wrong; last half mostly right with cluster growth
    positively correlated to last-half per-class accuracy."""
    s = []
    for i in range(50):                                   # first half: all wrong
        s.append(sample(0, False, batch_idx=i, clusters=0))
    idx = 50
    # classes 3/4/5: n_clusters_end 10/6/3 vs last-half acc 100/66.67/33.33.
    for t, ncl, ncorr in [(3, 10, 3), (4, 6, 2), (5, 3, 1)]:
        for j in range(3):
            s.append(sample(t, j < ncorr, batch_idx=idx, clusters=ncl))
            idx += 1
    for _ in range(41):                                   # filler last-half correct
        s.append(sample(0, True, batch_idx=idx, clusters=0))
        idx += 1
    return s


def build_flat_stream(seed):
    return [sample(0, True, batch_idx=i, clusters=5) for i in range(100)]


def build_decreasing_stream(seed):
    s = []
    for i in range(50):                                   # first half: all right
        s.append(sample(0, True, batch_idx=i, clusters=5))
    for i in range(50):                                   # last half: all wrong
        s.append(sample(0, False, batch_idx=50 + i, clusters=5))
    return s


def test_hypothesis_support():
    root = reset_dir("/tmp/rec_fix_rise")
    write_records(root, "PatchModPTA-CS-s1", build_rising_stream(1), seed=1)
    write_records(root, "PatchModPTA-CS-s2", build_rising_stream(2), seed=2)
    out = "/tmp/rec_fix_rise/hyp.md"
    proc = run_cli("hypothesis", "--records", "/tmp/rec_fix_rise", "--out", out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    # delta = 94 - 0 = +94 pp > 1.0; rho = 1.0 > 0.2 -> SUPPORT
    assert "DECISION: SUPPORT (delta_mean=94.00 pp, rho_mean=1.00)" in proc.stdout, proc.stdout
    report = Path(out).read_text(encoding="utf-8")
    assert "DECISION: SUPPORT" in report
    print("  [ok] hypothesis: rising stream -> SUPPORT")


def test_hypothesis_flat_and_refute():
    # Flat stream (delta=0) -> INCONCLUSIVE (not >1.0, not <0).
    flat_root = reset_dir("/tmp/rec_fix_flat")
    write_records(flat_root, "PatchModPTA-CS-s1", build_flat_stream(1), seed=1)
    out = "/tmp/rec_fix_flat/hyp.md"
    proc = run_cli("hypothesis", "--records", "/tmp/rec_fix_flat", "--out", out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DECISION: INCONCLUSIVE" in proc.stdout, proc.stdout

    # Decreasing stream (delta = -100 pp < 0) -> REFUTE regardless of rho.
    dec_root = reset_dir("/tmp/rec_fix_dec")
    write_records(dec_root, "PatchModPTA-CS-s1", build_decreasing_stream(1), seed=1)
    out = "/tmp/rec_fix_dec/hyp.md"
    proc = run_cli("hypothesis", "--records", "/tmp/rec_fix_dec", "--out", out)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DECISION: REFUTE (delta_mean=-100.00 pp" in proc.stdout, proc.stdout
    print("  [ok] hypothesis: flat -> INCONCLUSIVE; decreasing -> REFUTE")


# ---------------------------------------------------------------------------
# (e) flip metrics on crafted logit vectors
# ---------------------------------------------------------------------------

def test_flip_metrics():
    clip = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # argmax -> 0
    flipped_final = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # argmax -> 1
    aligned_final = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # argmax -> 0
    zero_img = [0.0] * 6                              # clip+tau*img stays argmax 0
    flip_img = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]        # clip+80*img argmax -> 1
    aligned_patch = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # argmax -> 0
    flipped_patch = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]  # argmax -> 1
    flipped_patch2 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # argmax -> 2

    samples = [
        sample(0, True, batch_idx=0, clusters=3, clip=clip, final=flipped_final,
               img=zero_img, patch=flipped_patch),        # patch_flip, no flip_vs_img, patch_only_flip
        sample(0, True, batch_idx=1, clusters=2, clip=clip, final=aligned_final,
               img=flip_img, patch=aligned_patch),        # no patch_flip, flip_vs_img, no patch_only
        sample(0, True, batch_idx=2, clusters=0, clip=clip, final=flipped_final,
               img=zero_img, patch=flipped_patch2),       # patch_flip, no flip_vs_img, patch_only MASKED (0 clusters)
    ]
    flips = ar.flip_metrics(samples, tau_img=80.0)

    assert 0 in flips
    f = flips[0]
    assert f["n"] == 3
    assert abs(f["patch_flip"] - 66.67) < 0.01, f        # 2/3
    assert abs(f["flip_vs_img"] - 33.33) < 0.01, f       # 1/3
    assert abs(f["patch_only_flip"] - 50.00) < 0.01, f   # 1/2 (third sample masked)
    print("  [ok] flip metrics: patch_flip / flip_vs_img / patch_only_flip rates")


def main():
    print("=" * 60)
    print("  analyze_records tests (synthetic, no GPU)")
    print("=" * 60)
    test_table()
    test_parity_pass()
    test_parity_fail()
    test_select()
    test_hypothesis_support()
    test_hypothesis_flat_and_refute()
    test_flip_metrics()
    print("=" * 60)
    print("analyze_records: ALL TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
