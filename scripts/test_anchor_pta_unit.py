"""CPU unit tests for models/anchor_pta.py (text-anchor EMA damping, fp32, no GPU).

Run:
    /share_98/projects/brandon/envs/pta/bin/python scripts/test_anchor_pta_unit.py

Checks:
  (a) damp_threshold=0: _damped_update is bit-for-bit identical to base PTA's
      update (models/image_level/pta_image.py) over a 50-step random chain —
      refine and prototype state must match a reference implementation.
  (b) a poisoned prototype (drifted to the ANTIPODE of its text anchor,
      sim ~ -1) gets damped by exactly damp_floor: one correct write moves
      the prototype only floor*delta, while the undamped baseline moves it
      delta. (With sim <= 0 the damp factor saturates at damp_floor.)
  (c) the damp factor is monotone non-decreasing in the anchor cosine sim_c:
      for sim_c in {0.3, 0.6, 0.9, 1.0} with threshold=0.95 / floor=0.4 the
      measured effective write weights satisfy
      damp(0.3) <= damp(0.6) <= damp(0.9) <= damp(1.0) == 1,
      every damp >= damp_floor, and damp(sim_c >= threshold) == 1 exactly.
      (Measured through the REAL _damped_update: effective w = ||dproto||/w_new
      with a zero-initialised prototype and a unit write target.)
  (d) build(cfg) returns an AnchorPTAAdapter instance of BaseAdapter with the
      damp_threshold / damp_floor config surface; damp_floor clamped to [0,1].
  (e) the text anchor (text_ref) is never mutated by _damped_update.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anchor_pta import AnchorPTAAdapter, build  # noqa: E402
from models.base import BaseAdapter  # noqa: E402

ALPHA = 0.01
T = 20.0


def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)


def _rand_unit(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return _norm(torch.randn(shape, generator=g))


def _reference_base_pta(x, clip_logits, refine, proto, alpha, T):
    """Reference: base PTA's exact update (models/image_level/pta_image.py)."""
    w = F.softmax(clip_logits, dim=-1).squeeze(0)       # (C)
    mask = w >= 1e-1                                    # (C)
    w_new = torch.zeros_like(w)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)
    proto[mask] = (1 - w_new[mask]) * proto[mask] + w_new[mask] * x
    refined = ALPHA * refine + (1 - ALPHA) * proto
    refined = _norm(refined)
    return refined, proto


def _cfg(threshold, floor=0.5):
    return {
        "damp_threshold": threshold,
        "damp_floor": floor,
        "image_level": {"alpha": ALPHA, "T": T},
        "fusion": {},
    }


def check_a_disabled_matches_base_pta():
    """(a) threshold=0 -> bit-for-bit base PTA update over a 50-step chain."""
    C, D = 5, 16
    text = _rand_unit((C, D), seed=0)
    adapter = build(_cfg(0.0))

    refine_a = text.clone()
    proto_a = torch.zeros_like(text)
    refine_r = text.clone()
    proto_r = torch.zeros_like(text)
    text_ref = text.clone()

    g = torch.Generator().manual_seed(11)
    for _ in range(50):
        x = _norm(torch.randn(1, D, generator=g))
        logits = torch.randn(1, C, generator=g) * 40 - 10   # diverse masks
        refine_a, proto_a = adapter._damped_update(
            x, logits, text_ref, refine_a, proto_a
        )
        refine_r, proto_r = _reference_base_pta(
            x, logits, refine_r, proto_r, ALPHA, T
        )

    if not torch.equal(refine_a, refine_r):
        return False
    if not torch.equal(proto_a, proto_r):
        return False
    # Text anchor untouched (also check (e) here under threshold=0).
    if not torch.equal(text_ref, text):
        return False
    return True


def check_b_poisoned_proto_damped_to_floor():
    """(b) antipodal proto: one correct write moves it only floor*delta."""
    C, D = 3, 32
    threshold, floor = 0.99, 0.5
    text = _rand_unit((C, D), seed=1)
    c0 = 0

    def _one_step(thr):
        adapter = build(_cfg(thr, floor))
        refine = text.clone()
        proto = torch.zeros_like(text)
        refine[c0] = _norm(ALPHA * text[c0] + (1 - ALPHA) * (-text[c0]))
        # Above hands refine ~ +... no: force the DRIFTED state directly.
        refine[c0] = -text[c0]                          # sim ~ -1 vs anchor
        proto[c0] = -text[c0]                           # captured prototype
        text_ref = text.clone()
        x = text[c0].unsqueeze(0)                       # a "correct" write
        logits = torch.zeros(1, C)
        logits[0, c0] = 30.0                            # confidently class c0
        _, proto_new = adapter._damped_update(x, logits, text_ref, refine, proto)
        return x.squeeze(0), proto_new[c0].clone()

    _, proto_damped = _one_step(threshold)      # floor damping
    _, proto_undamped = _one_step(0.0)          # no damping

    d_undamped = (proto_undamped - (-text[c0])).norm()
    d_damped = (proto_damped - (-text[c0])).norm()
    if d_undamped <= 0:
        return False
    ratio = d_damped / d_undamped
    # sim_c ~ -1 <= 0 => damp factor saturates at damp_floor exactly.
    if abs(ratio - floor) > 1e-3:
        return False
    # The damped write must still move in the correct direction (not zeroed).
    if not bool((proto_damped + text[c0]).norm() > 1e-3):
        return False
    return True


def check_c_damp_monotone_in_sim():
    """(c) damp(sim) non-decreasing, ==1 above threshold, never < floor."""
    C, D = 3, 32
    threshold, floor = 0.95, 0.4
    text = _rand_unit((C, D), seed=2)
    text_ref = text.clone()
    c0 = 0

    def _perp_to(t):
        u = torch.randn_like(t)
        perp = u - (u @ t) * t
        return _norm(perp)

    perp = _perp_to(text[c0])

    def _effective_damp(sim_c):
        adapter = build(_cfg(threshold, floor))
        refine = text.clone()
        refine[c0] = _norm(sim_c * text[c0] + (1 - sim_c**2) ** 0.5 * perp)
        proto = torch.zeros_like(text)                  # zero-init prototype
        x = text[c0].unsqueeze(0)                       # unit write target
        logits = torch.zeros(1, C)
        logits[0, c0] = 30.0
        _, proto_new = adapter._damped_update(x, logits, text_ref, refine, proto)
        # proto[c0] = w_eff * x (zero init) -> w_eff = ||proto_new[c0]||.
        w_eff = float(proto_new[c0].norm().item())
        w = float(F.softmax(logits, dim=-1)[0, c0].item())
        w_new = 1 - float(torch.exp(torch.tensor(-w / T)).item())
        if w_new <= 0:
            raise AssertionError("degenerate w_new")
        return w_eff / w_new

    d_low = _effective_damp(0.3)
    d_mid = _effective_damp(0.6)
    d_high = _effective_damp(0.9)
    d_above = _effective_damp(1.0)                      # >= threshold

    if not (d_low <= d_mid <= d_high <= d_above + 1e-6):
        return False
    if abs(d_above - 1.0) > 1e-4:                       # sim >= thr -> damp 1
        return False
    if min(d_low, d_mid, d_high) < floor - 1e-4:        # floor respected
        return False
    # Analytic spot check for the saturating linear ramp.
    expected_low = floor + (1 - floor) * (0.3 / threshold)
    if abs(d_low - expected_low) > 2e-3:
        return False
    return True


def check_d_build_and_config():
    """(d) build() returns valid AnchorPTAAdapter; config surface parsed."""
    adapter = build(_cfg(0.96, 0.5))
    if not isinstance(adapter, BaseAdapter):
        return False
    if not isinstance(adapter, AnchorPTAAdapter):
        return False
    if adapter.damp_threshold != 0.96 or adapter.damp_floor != 0.5:
        return False
    if adapter.image_level is None:
        return False

    # damp_floor clamped into [0, 1].
    if build(_cfg(0.9, 1.5)).damp_floor != 1.0:
        return False
    if build(_cfg(0.9, -0.2)).damp_floor != 0.0:
        return False
    # Disable path: threshold 0.
    if build(_cfg(0.0)).damp_threshold != 0.0:
        return False
    return True


def check_e_text_ref_immutable():
    """(e) _damped_update never mutates the immutable text anchor."""
    C, D = 4, 16
    threshold, floor = 0.96, 0.5
    text = _rand_unit((C, D), seed=3)
    text_ref = text.clone()
    snapshot = text_ref.clone()
    adapter = build(_cfg(threshold, floor))
    refine = text.clone()
    proto = torch.zeros_like(text)

    g = torch.Generator().manual_seed(9)
    for _ in range(20):
        x = _norm(torch.randn(1, D, generator=g))
        logits = torch.randn(1, C, generator=g) * 30
        refine, proto = adapter._damped_update(x, logits, text_ref, refine, proto)
    if not torch.equal(text_ref, snapshot):
        return False
    # refine/proto moved away from the anchor (adaptation happened).
    if torch.equal(refine, snapshot):
        return False
    return True


def main():
    checks = [
        ("(a) damp_threshold=0 == base PTA bit-for-bit over 50 steps",
         check_a_disabled_matches_base_pta),
        ("(b) antipodal proto: single write damped by exactly damp_floor",
         check_b_poisoned_proto_damped_to_floor),
        ("(c) damp(sim) monotone non-decreasing, 1 above threshold, >= floor",
         check_c_damp_monotone_in_sim),
        ("(d) build() returns valid AnchorPTAAdapter; config + clamps parse",
         check_d_build_and_config),
        ("(e) text_ref is immutable across 20 damped updates",
         check_e_text_ref_immutable),
    ]
    failures = 0
    for name, fn in checks:
        try:
            ok = bool(fn())
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  exception: {type(exc).__name__}: {exc}")
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            failures += 1
    if failures:
        print(f"\n{failures}/{len(checks)} checks FAILED")
        sys.exit(1)
    print(f"\nAll {len(checks)} checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()