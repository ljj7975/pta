"""CPU headless unit test for models/compact_pta.py.

Verifies the compact-PTA update function (base PTA multi-class EMA write with
an intra-class compactness gate) on synthetic float32 tensors:

  (a) gate disabled ("off") reproduces base PTA's multi-class EMA update
      exactly (reference: PTAImageLevel in models/image_level/pta_image.py);
  (b) soft gate reduces the effective weight for far writes and ~preserves it
      for near writes;
  (c) hard gate zeros far writes;
  (d) zero-prototype rows (never written) cause no NaN and pass the first
      write ungated;
  (e) refined text and logits stay finite over mixed near/far sequences.

Run:
  /share_98/projects/brandon/envs/pta/bin/python scripts/test_compact_pta_unit.py

Prints PASS/FAIL per check and exits nonzero on any failure.
"""
import math
import os
import sys
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.compact_pta import compact_ema_write, init_gate_stats
from models.image_level.pta_image import PTAImageLevel

D = 32          # feature dim
C = 8           # num classes
ALPHA = 0.01
T = 20.0
SIGMA_INIT = 0.1
SIGMA_MIN = 0.05
STATS_EMA = 0.05
HARD_K = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────


def _unit(v):
    return v / v.norm()


def _rowwise_unit(m):
    return m / m.norm(dim=-1, keepdim=True)


def _dominant_logits(c):
    """(1, C) logits whose softmax puts ~all mass on class c."""
    logits = torch.full((1, C), -10.0)
    logits[0, c] = 5.0
    return logits


def _make_near_far(p, cos_near=0.97, seed=1):
    """Unit vectors x_near (cos(x_near, p) = cos_near) and x_far (orthogonal)."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(D, generator=g)
    v = v - (v @ p) * p
    v = v / v.norm()
    theta = math.acos(cos_near)
    x_near = _unit(p * math.cos(theta) + v * math.sin(theta))
    x_far = _unit(v)  # orthogonal to p -> d = 1 - 0 = 1.0
    return x_near, x_far


def _effective_weight(proto_before, proto_after, x):
    """Recover the EMA weight from the prototype change:
    proto_after - proto_before = w * (x - proto_before)."""
    num = (proto_after - proto_before).norm()
    den = (x - proto_before).norm()
    return float(num / den)


def _write(img, logits, text, proto, mode, stats):
    return compact_ema_write(
        img.unsqueeze(0), logits, text, proto,
        ALPHA, T, mode, SIGMA_MIN, STATS_EMA, HARD_K, stats,
    )


# ── Checks ───────────────────────────────────────────────────────────────────


def check_a():
    """(a) gate off == base PTA multi-class EMA update, bit-identical."""
    g = torch.Generator().manual_seed(42)
    text = _rowwise_unit(torch.randn(C, D, generator=g))
    samples = []
    for i in range(50):
        x = _unit(torch.randn(D, generator=g))
        c = int(torch.randint(0, C, (1,), generator=g).item())
        logits = _dominant_logits(c)
        if i % 5 == 4:
            # two-class write: second class also clears the 0.1 threshold
            c2 = (c + 1) % C
            logits[0, c2] = 4.9
        samples.append((x, logits))

    ref = PTAImageLevel({"image_level": {"alpha": ALPHA, "T": T}})
    ref_proto = torch.zeros(C, D)
    ref_text = text.clone()
    stats = init_gate_stats(C, "off", SIGMA_INIT)
    cp_proto = torch.zeros(C, D)
    cp_text = text.clone()

    for x, logits in samples:
        ref_text, ref_proto = ref.update_prototypes(
            x.unsqueeze(0), logits, ref_text, ref_proto
        )
        cp_text, cp_proto, _, _ = _write(x, logits, cp_text, cp_proto, "off", stats)
        assert torch.equal(ref_proto, cp_proto), "prototype mismatch vs base PTA"
        assert torch.equal(ref_text, cp_text), "refined text mismatch vs base PTA"
    return True, "50-sample sequence (incl. two-class writes) bit-identical to PTAImageLevel"


def check_b():
    """(b) soft gate: far writes strongly reduced, near writes ~preserved."""
    g = torch.Generator().manual_seed(7)
    p = _unit(torch.randn(D, generator=g))
    x_near, x_far = _make_near_far(p, cos_near=0.97, seed=8)
    text = _rowwise_unit(torch.randn(C, D, generator=g))
    logits = _dominant_logits(0)
    p_c = float(F.softmax(logits, dim=-1).squeeze(0)[0])
    w_base = 1.0 - math.exp(-p_c / T)

    proto0 = torch.zeros(C, D)
    proto0[0] = p

    stats_off = init_gate_stats(C, "off", SIGMA_INIT)
    _, proto_far_base, _, _ = _write(x_far, logits, text, proto0.clone(), "off", stats_off)
    _, proto_near_base, _, _ = _write(x_near, logits, text, proto0.clone(), "off", stats_off)

    stats_soft = init_gate_stats(C, "soft", SIGMA_INIT)
    _, proto_far_soft, _, _ = _write(x_far, logits, text, proto0.clone(), "soft", stats_soft)
    stats_soft2 = init_gate_stats(C, "soft", SIGMA_INIT)
    _, proto_near_soft, _, _ = _write(x_near, logits, text, proto0.clone(), "soft", stats_soft2)

    w_far_base = _effective_weight(proto0[0], proto_far_base[0], x_far)
    w_far_soft = _effective_weight(proto0[0], proto_far_soft[0], x_far)
    w_near_base = _effective_weight(proto0[0], proto_near_base[0], x_near)
    w_near_soft = _effective_weight(proto0[0], proto_near_soft[0], x_near)

    # base weights match the analytic 1 - exp(-p_c / T)
    assert abs(w_far_base - w_base) <= 1e-6 * w_base, "base far weight != analytic"
    assert abs(w_near_base - w_base) <= 1e-6 * w_base, "base near weight != analytic"

    d_far = 1.0 - float(F.cosine_similarity(x_far, p, dim=0))
    d_near = 1.0 - float(F.cosine_similarity(x_near, p, dim=0))
    sigma = max(SIGMA_INIT, SIGMA_MIN)
    factor_far = math.exp(-(d_far ** 2) / (2.0 * sigma ** 2))
    factor_near = math.exp(-(d_near ** 2) / (2.0 * sigma ** 2))

    # far write: weight strongly reduced (factor ~ exp(-50) ~ 0)
    assert d_far > 0.99, f"test setup: expected d_far ~ 1, got {d_far}"
    assert w_far_soft < 0.01 * w_far_base, (
        f"soft gate did not reduce far write: {w_far_soft} vs base {w_far_base}"
    )
    # near write: weight ~preserved (within 5% of the analytic factor)
    assert 0.9 < factor_near < 1.0, f"test setup: expected factor_near in (0.9,1), got {factor_near}"
    expected_near = factor_near * w_near_base
    assert abs(w_near_soft - expected_near) <= 0.05 * expected_near, (
        f"soft near weight {w_near_soft} deviates from expected {expected_near}"
    )
    assert w_near_soft > 0.9 * w_near_base, (
        f"soft gate over-reduced near write: {w_near_soft} vs base {w_near_base}"
    )
    return True, (
        f"d_far={d_far:.3f} -> w {w_far_base:.4f}->{w_far_soft:.2e} "
        f"(factor {factor_far:.1e}); d_near={d_near:.3f} -> w {w_near_base:.4f}>"
        f"{w_near_soft:.4f} (factor {factor_near:.3f})"
    )


def check_c():
    """(c) hard gate zeros far writes; near writes kept with full weight."""
    g = torch.Generator().manual_seed(11)
    p = _unit(torch.randn(D, generator=g))
    x_near, x_far = _make_near_far(p, cos_near=0.97, seed=12)
    text = _rowwise_unit(torch.randn(C, D, generator=g))
    logits = _dominant_logits(0)

    proto0 = torch.zeros(C, D)
    proto0[0] = p

    # far write: d = 1.0 > mean(0.1) + hard_k * std(0) = 0.1 -> zeroed
    stats = init_gate_stats(C, "hard", SIGMA_INIT)
    _, proto_far, compact_d, _ = _write(x_far, logits, text, proto0.clone(), "hard", stats)
    assert torch.equal(proto_far[0], p), "hard gate should zero the far write"
    assert stats[0]["n_gated"] == 1 and stats[0]["n_modified"] == 1
    assert stats[0]["n_writes_pre"] == 1 and stats[0]["n_writes_post"] == 0
    # stats updated after the zeroed write (with the actual float d_c)
    d_far_actual = 1.0 - float(F.cosine_similarity(x_far, p, dim=0))
    exp_mean = (1.0 - STATS_EMA) * SIGMA_INIT + STATS_EMA * d_far_actual
    assert abs(stats[0]["mean"] - exp_mean) < 1e-12, "hard mean EMA not updated"

    # near write: d = 0.03 < 0.1 -> kept with the full base weight
    stats2 = init_gate_stats(C, "hard", SIGMA_INIT)
    _, proto_near, _, _ = _write(x_near, logits, text, proto0.clone(), "hard", stats2)
    assert not torch.equal(proto_near[0], p), "hard gate should keep the near write"
    stats_off = init_gate_stats(C, "off", SIGMA_INIT)
    _, proto_near_base, _, _ = _write(x_near, logits, text, proto0.clone(), "off", stats_off)
    assert torch.equal(proto_near[0], proto_near_base[0]), (
        "kept hard-gate write must equal base PTA exactly"
    )
    assert stats2[0]["n_modified"] == 0 and stats2[0]["n_writes_post"] == 1
    return True, (
        f"far d=1.0 > thr={SIGMA_INIT + HARD_K * 0.0:.3f} -> zeroed (proto unchanged); "
        f"near d=0.03 < thr -> kept, bit-identical to base PTA"
    )


def check_d():
    """(d) zero-prototype rows: no NaN, first write passes ungated."""
    g = torch.Generator().manual_seed(13)
    x = _unit(torch.randn(D, generator=g))
    logits = _dominant_logits(0)
    text = _rowwise_unit(torch.randn(C, D, generator=g))
    ref = PTAImageLevel({"image_level": {"alpha": ALPHA, "T": T}})

    for mode in ("soft", "hard"):
        proto = torch.zeros(C, D)
        stats = init_gate_stats(C, mode, SIGMA_INIT)
        refined, proto, compact_d, written = _write(x, logits, text, proto, mode, stats)
        assert torch.isfinite(proto).all(), f"{mode}: NaN/Inf in prototype"
        assert torch.isfinite(refined).all(), f"{mode}: NaN/Inf in refined text"
        # first write ungated == base PTA exactly
        ref_proto = torch.zeros(C, D)
        ref_refined, ref_proto = ref.update_prototypes(
            x.unsqueeze(0), logits, text.clone(), ref_proto
        )
        assert torch.equal(proto, ref_proto), f"{mode}: first write != base PTA"
        assert torch.equal(refined, ref_refined), f"{mode}: refined != base PTA"
        assert written == [0]
        assert stats[0]["n_gated"] == 0, f"{mode}: zero-proto write must be ungated"
        assert stats[0]["n_writes_pre"] == 1 and stats[0]["n_writes_post"] == 1
        assert compact_d.get(0) == 1.0, f"{mode}: d of zero-proto write should be 1.0"
    return True, "soft+hard: zero rows finite, first write ungated and bit-identical to base PTA"


def check_e():
    """(e) refined text and logits stay finite over mixed near/far sequences."""
    for mode in ("soft", "hard"):
        g = torch.Generator().manual_seed(21)
        text = _rowwise_unit(torch.randn(C, D, generator=g))
        proto = torch.zeros(C, D)
        refined = text.clone()
        stats = init_gate_stats(C, mode, SIGMA_INIT)
        for step in range(30):
            c = int(torch.randint(0, C, (1,), generator=g).item())
            if proto[c].norm() > 1e-6 and step % 2 == 0:
                # near write: blend toward the current prototype
                x = _unit(0.95 * proto[c] + 0.05 * torch.randn(D, generator=g))
            else:
                # likely-far write
                x = _unit(torch.randn(D, generator=g))
            logits = _dominant_logits(c)
            refined, proto, _, _ = _write(x, logits, refined, proto, mode, stats)
            assert torch.isfinite(refined).all(), (
                f"{mode}: refined text not finite at step {step}"
            )
            # image_proto_logits = image_features @ refined_text.T — the same
            # matmul as PTAImageLevel.compute_logits without the fp16 cast
            # (fp16 matmul is not implemented on this CPU build).
            logits_out = x.unsqueeze(0) @ refined.T
            assert torch.isfinite(logits_out).all(), (
                f"{mode}: logits not finite at step {step}"
            )
        assert torch.isfinite(proto).all(), f"{mode}: prototype not finite at end"
    return True, "30-step mixed near/far sequences (soft+hard): refined text and logits finite at every step"


# ── Runner ───────────────────────────────────────────────────────────────────


def main():
    checks = [
        ("(a) gate off reproduces base PTA multi-class EMA exactly", check_a),
        ("(b) soft gate reduces far writes, ~preserves near writes", check_b),
        ("(c) hard gate zeros far writes", check_c),
        ("(d) zero-prototype rows: no NaN, first write ungated", check_d),
        ("(e) refined text and logits stay finite", check_e),
    ]
    all_ok = True
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"{'PASS' if ok else 'FAIL'} {name}")
        if ok:
            print(f"       {detail}")
        else:
            all_ok = False
            print(f"       {detail}")
    if all_ok:
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
