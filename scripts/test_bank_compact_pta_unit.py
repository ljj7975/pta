"""CPU unit tests for models/bank_compact_pta.py (growing bank + gate, fp32).

Run:
    /share_98/projects/brandon/envs/pta/bin/python scripts/test_bank_compact_pta_unit.py

Checks:
  (a) with K=1 and gate_mode=off the method reproduces base PTA EXACTLY
      (anchor absorb-only EMA, refine scoring) — the gate is a no-op.
  (b) with K>1 and gate_mode=off a genuinely new mode GROWS a slot
      (active_k increases, no collapse) — routing matches BankV2.
  (c) the SOFT gate reduces the absorb weight for a far sample
      (w_used < ungated w) — tames the drag on the prototype.
  (d) the HARD gate zeros the absorb weight for a far sample (w_used = 0).
  (e) the score is the max over the ACTIVE slots of cos(x, refine[c,k]).
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bank_compact_pta import (
    BankCompactPTAAdapter,
    build,
    init_bank,
    init_gate_stats,
    bank_compact_write,
    bank_compact_score,
    refine_bank,
    GATE_SOFT,
    GATE_HARD,
    GATE_OFF,
)

ALPHA = 0.01
CT = 0.85


def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)


def _rand_unit(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return _norm(torch.randn(shape, generator=g))


def _reference_base_pta(x, clip_logits, text, proto, T):
    """Reference: base PTA's exact update + score (models/image_level/pta_image.py)."""
    w = F.softmax(clip_logits, dim=-1).float().squeeze(0)
    mask = w >= 1e-1
    w_new = torch.zeros_like(w)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)
    proto[mask] = (1 - w_new[mask]) * proto[mask] + w_new[mask] * x
    refined = ALPHA * text + (1 - ALPHA) * proto
    refined = _norm(refined)
    return x @ refined.T


def check_a_k1_off_matches_base_pta():
    """(a) K=1, gate_mode=off reproduces base PTA exactly."""
    C, D, K, T = 5, 16, 1, 20.0
    text = _rand_unit((C, D), seed=0)
    P, active, usage = init_bank(text, K)
    ref_proto = torch.zeros(C, D)
    gate_stats = init_gate_stats(C, 0.1)

    if not torch.equal(P[:, 0, :], ref_proto):
        return False

    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        x = _norm(torch.randn(D, generator=g))
        clip_logits = 2.0 * torch.randn(1, C, generator=g)
        bank_compact_write(x.unsqueeze(0), clip_logits, text, P, active, usage,
                           ALPHA, T, CT, GATE_OFF, 0.05, 0.05, 0.5, gate_stats)
        ref_logit = _reference_base_pta(x, clip_logits, text, ref_proto, T)
        bank_logit = bank_compact_score(x.unsqueeze(0), text, P, active, ALPHA).squeeze(0)
        if not torch.allclose(bank_logit, ref_logit, atol=1e-5):
            return False

    return bool(torch.allclose(P[:, 0, :], ref_proto, atol=1e-5))


def check_b_new_mode_grows():
    """(b) K>1, gate off: a new mode grows a slot (diversity, no collapse)."""
    C, D, K, T = 2, 8, 3, 20.0
    e = torch.eye(D)
    text = _norm(torch.stack([e[0].clone(), e[3].clone()]))
    P, active, usage = init_bank(text, K)
    gate_stats = init_gate_stats(C, 0.1)
    clip0 = torch.tensor([[10.0, -10.0]])

    bank_compact_write(_norm(e[0]).unsqueeze(0), clip0, text, P, active, usage,
                       ALPHA, T, CT, GATE_OFF, 0.05, 0.05, 0.5, gate_stats)
    if int(active[0]) != 1:
        return False

    writes = bank_compact_write(_norm(e[1]).unsqueeze(0), clip0, text, P, active,
                                usage, ALPHA, T, CT, GATE_OFF, 0.05, 0.05, 0.5,
                                gate_stats)
    if int(active[0]) != 2:
        return False
    if not any(a == "grow" for _c, _k, _w, a in writes):
        return False
    return bool(float(usage[0, 0]) > 0.0 and float(usage[0, 1]) > 0.0)


def _far_write_w_used(gate_mode):
    """Return the absorb weight used for a FAR sample after the anchor is set.

    K=1, anchor aligned to e0; write e0 (establishes anchor), then write e1
    (orthogonal -> far). Returns the w_used recorded in `writes` for the far
    absorb.
    """
    C, D, K, T = 1, 8, 1, 20.0
    e = torch.eye(D)
    text = _norm(e[0].clone().unsqueeze(0))
    P, active, usage = init_bank(text, K)
    gate_stats = init_gate_stats(C, 0.1)
    clip0 = torch.tensor([[10.0]])

    # Establish the anchor with a near sample (e0).
    bank_compact_write(_norm(e[0]).unsqueeze(0), clip0, text, P, active, usage,
                       ALPHA, T, CT, gate_mode, 0.05, 0.05, 0.5, gate_stats)
    # Far sample (e1, orthogonal to the anchor) -> absorb branch, gated.
    writes = bank_compact_write(_norm(e[1]).unsqueeze(0), clip0, text, P, active,
                                usage, ALPHA, T, CT, gate_mode, 0.05, 0.05, 0.5,
                                gate_stats)
    # The far write must be an absorb (K=1 -> no growth), return its w_used.
    assert all(a == "absorb" for _c, _k, _w, a in writes)
    return float(writes[0][2])


def check_c_soft_gate_reduces_weight():
    """(c) soft gate reduces the absorb weight for a far sample."""
    w_off = _far_write_w_used(GATE_OFF)
    w_soft = _far_write_w_used(GATE_SOFT)
    # The ungated weight is positive; the soft-gated weight is strictly less.
    return bool(w_off > 0.0 and w_soft < w_off - 1e-6)


def check_d_hard_gate_zeros_weight():
    """(d) hard gate zeros the absorb weight for a far sample."""
    w_hard = _far_write_w_used(GATE_HARD)
    return bool(w_hard < 1e-9)


def check_e_max_of_active():
    """(e) the score is the max over the ACTIVE slots of cos(x, refine[c,k])."""
    C, D, K = 4, 24, 5
    g = torch.Generator().manual_seed(3)
    text = _rand_unit((C, D), seed=9)
    P, active, usage = init_bank(text, K)
    P = torch.randn(C, K, D, generator=g)
    active = torch.tensor([1, 2, 3, K])
    x = _rand_unit((D,), seed=4).unsqueeze(0)

    got = bank_compact_score(x, text, P, active, ALPHA).squeeze(0)
    for c in range(C):
        a = int(active[c])
        refine_c = refine_bank(text[c:c + 1], P[c, :a].unsqueeze(0), ALPHA)[0]
        manual = float((x.squeeze(0) @ refine_c.T).max())
        if abs(float(got[c]) - manual) > 1e-6:
            return False
    return True


def main():
    checks = [
        ("(a) K=1 gate=off reproduces base PTA exactly",
         check_a_k1_off_matches_base_pta),
        ("(b) K>1 gate=off a new mode GROWS a slot (no collapse)",
         check_b_new_mode_grows),
        ("(c) soft gate REDUCES the absorb weight for a far sample",
         check_c_soft_gate_reduces_weight),
        ("(d) hard gate ZEROS the absorb weight for a far sample",
         check_d_hard_gate_zeros_weight),
        ("(e) score is the max over ACTIVE slots of cos(x, refine[c,k])",
         check_e_max_of_active),
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
