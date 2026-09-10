"""CPU unit tests for models/bank_v2_pta.py (proper growing bank, fp32, no GPU).

Run:
    /share_98/projects/brandon/envs/pta/bin/python scripts/test_bank_v2_pta_unit.py

Checks:
  (a) with K=1 the method reproduces base PTA EXACTLY: zero-init anchor,
      absorb-only EMA write (no growth/recycle), and scoring via
      refine = normalize(0.01*text + 0.99*proto).  State and logits must match
      a reference that mirrors models/image_level/pta_image.py bit-for-bit.
  (b) with K>1 a genuinely new mode (dissimilar to every active prototype)
      GROWS a fresh specialized slot -> active_k increases (no collapse).
  (c) the anchor (slot 0) is NEVER recycled: when the bank is full and a new
      mode arrives, a specialized slot (1..K-1) is reset and slot 0 is
      untouched; active stays at K.
  (d) the score is the max over the ACTIVE slots of cos(x, refine[c, k]).
  (e) fused logits are finite and the prediction path matches
      clip + 100 * maxcos.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import BaseAdapter
from models.bank_v2_pta import (
    BankV2PTAAdapter,
    build,
    init_bank_v2,
    bank_v2_write,
    bank_v2_score,
    refine_bank,
)
from models.fusion import WeightedFusion

ALPHA = 0.01
CT = 0.85


def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)


def _rand_unit(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return _norm(torch.randn(shape, generator=g))


def _reference_base_pta(x, clip_logits, text, proto, T):
    """Reference: base PTA's exact update + score (models/image_level/pta_image.py)."""
    w = F.softmax(clip_logits, dim=-1).float().squeeze(0)   # (C)
    mask = w >= 1e-1                                        # (C)
    w_new = torch.zeros_like(w)
    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)
    proto[mask] = (1 - w_new[mask]) * proto[mask] + w_new[mask] * x
    refined = ALPHA * text + (1 - ALPHA) * proto
    refined = _norm(refined)
    return x @ refined.T


def check_a_k1_matches_base_pta():
    """(a) K=1 reproduces base PTA exactly (anchor absorb-only, refine scoring)."""
    C, D, K, T = 5, 16, 1, 20.0
    text = _rand_unit((C, D), seed=0)
    P, active, usage = init_bank_v2(text, K)               # [C,1,D] zeros
    ref_proto = torch.zeros(C, D)                          # base PTA: zero-init

    # Zero-init anchor must match base PTA's init_state; active must be 1.
    if not torch.equal(P[:, 0, :], ref_proto):
        return False
    if not bool((active == 1).all()):
        return False

    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        x = _norm(torch.randn(D, generator=g))
        clip_logits = 2.0 * torch.randn(1, C, generator=g)
        bank_v2_write(x.unsqueeze(0), clip_logits, text, P, active, usage,
                      ALPHA, T, CT)
        ref_logit = _reference_base_pta(x, clip_logits, text, ref_proto, T)

        bank_logit = bank_v2_score(x.unsqueeze(0), text, P, active, ALPHA).squeeze(0)
        if not torch.allclose(bank_logit, ref_logit, atol=1e-5):
            return False

    # State must match base PTA's prototype after the same 50 writes.
    return bool(torch.allclose(P[:, 0, :], ref_proto, atol=1e-5))


def check_b_new_mode_grows():
    """(b) K>1: a feature dissimilar to every active prototype grows a slot."""
    C, D, K, T = 2, 8, 3, 20.0
    e = torch.eye(D)
    # text[0] aligned with mode A (e0) so the first write absorbs into the anchor.
    text = _norm(torch.stack([e[0].clone(), e[3].clone()]))
    P, active, usage = init_bank_v2(text, K)

    clip0 = torch.tensor([[10.0, -10.0]])                  # only class 0 written

    # Write 1: mode A (e0) -> absorbed into anchor (slot 0).
    bank_v2_write(_norm(e[0]).unsqueeze(0), clip0, text, P, active, usage,
                  ALPHA, T, CT)
    if int(active[0]) != 1:
        return False                                        # still just the anchor
    if float(usage[0, 0]) <= 0.0:
        return False

    # Write 2: mode B (e1), orthogonal to the anchor -> must GROW slot 1.
    writes = bank_v2_write(_norm(e[1]).unsqueeze(0), clip0, text, P, active,
                           usage, ALPHA, T, CT)
    if int(active[0]) != 2:
        return False
    if not any(a == "grow" for _c, _k, _w, a in writes):
        return False
    if float(usage[0, 1]) <= 0.0:
        return False
    # Both slots now carry writes -> no collapse to a single active prototype.
    return bool(float(usage[0, 0]) > 0.0 and float(usage[0, 1]) > 0.0)


def check_c_anchor_never_recycled():
    """(c) K>1, bank full: a new mode recycles a specialized slot, not slot 0."""
    C, D, K, T = 1, 8, 3, 20.0
    e = torch.eye(D)
    text = _norm(e[0].clone().unsqueeze(0))                # anchor aligned to e0
    P, active, usage = init_bank_v2(text, K)
    clip0 = torch.tensor([[10.0]])                         # single class, always written

    # Fill the bank with 3 distinct orthogonal modes (e0 anchor + e1 + e2).
    for mode in [0, 1, 2]:
        bank_v2_write(_norm(e[mode]).unsqueeze(0), clip0, text, P, active,
                      usage, ALPHA, T, CT)
    if int(active[0]) != K:
        return False                                        # bank should be full

    anchor_before = P[0, 0].clone()
    # A 4th orthogonal mode (e3) with the bank full -> must recycle a
    # specialized slot (1 or 2); the anchor (slot 0) must be untouched.
    bank_v2_write(_norm(e[3]).unsqueeze(0), clip0, text, P, active, usage,
                  ALPHA, T, CT)
    if int(active[0]) != K:
        return False                                        # active stays K
    if not torch.equal(P[0, 0], anchor_before):
        return False                                        # anchor untouched
    # Some specialized slot was reset to the new mode (e3).
    hit = any(
        float(F.cosine_similarity(P[0, k].float(), _norm(e[3]), dim=0)) > 0.99
        for k in range(1, K)
    )
    return bool(hit)


def check_d_max_of_active():
    """(d) the score is the max over the ACTIVE slots of cos(x, refine[c,k])."""
    C, D, K = 4, 24, 5
    g = torch.Generator().manual_seed(3)
    text = _rand_unit((C, D), seed=9)
    P, active, usage = init_bank_v2(text, K)
    # Manually populate a few slots and set active counts.
    P = torch.randn(C, K, D, generator=g)
    active = torch.tensor([1, 2, 3, K])
    x = _rand_unit((D,), seed=4).unsqueeze(0)

    got = bank_v2_score(x, text, P, active, ALPHA).squeeze(0)
    for c in range(C):
        a = int(active[c])
        refine_c = refine_bank(text[c:c + 1], P[c, :a].unsqueeze(0), ALPHA)[0]
        manual = float((x.squeeze(0) @ refine_c.T).max())
        if abs(float(got[c]) - manual) > 1e-6:
            return False

    # A prototype aligned with x -> refine ~ x -> that class's logit ~ 1.
    P2, act2, _ = init_bank_v2(text, K)
    P2[1, 0] = x.squeeze(0).clone()                        # anchor = x
    got2 = bank_v2_score(x, text, P2, act2, ALPHA).squeeze(0)
    return float(got2[1]) > 0.99


def check_e_fusion_path():
    """(e) fused logits finite; prediction path matches clip + 100*maxcos."""
    C, D, K = 6, 16, 3
    cfg = {
        "image_level": {"alpha": ALPHA, "T": 20.0},
        "fusion": {"tau_image_proto": 100.0, "tau_patch_proto": 0.0},
        "K": K,
        "create_threshold": CT,
    }
    adapter = build(cfg)
    ok = isinstance(adapter, BaseAdapter) and isinstance(adapter, BankV2PTAAdapter)
    ok = ok and adapter.K == K
    ok = ok and isinstance(adapter.fusion, WeightedFusion)
    ok = ok and (
        adapter.fusion.tau_text == 1.0
        and adapter.fusion.tau_image_proto == 100.0
        and adapter.fusion.tau_patch_proto == 0.0
    )
    ok = ok and abs(adapter.create_threshold - CT) < 1e-9
    ok = ok and build({"image_level": {"alpha": ALPHA, "T": 20.0}, "fusion": {}}).K == 3
    if not ok:
        return False

    g = torch.Generator().manual_seed(8)
    text = _rand_unit((C, D), seed=9)
    P, active, usage = init_bank_v2(text, K)
    P = torch.randn(C, K, D, generator=g)
    active = torch.full((C,), 2, dtype=torch.int32)
    x = _norm(torch.randn(D, generator=g))
    maxcos = bank_v2_score(x.unsqueeze(0), text, P, active, ALPHA).squeeze(0)
    image_proto_logits = maxcos.half().unsqueeze(0)                 # [1, C] fp16
    clip = (100.0 * torch.randn(1, C, generator=g)).half()          # fp16
    final = adapter.fusion.forward(clip.clone(), image_proto_logits, None)
    expected = 1.0 * clip + 100.0 * image_proto_logits
    ok = ok and bool(torch.isfinite(final.float()).all())
    ok = ok and torch.allclose(final, expected, atol=1e-3)
    ok = ok and int(final.argmax().item()) == int(expected.argmax().item())
    return bool(ok)


def main():
    checks = [
        ("(a) K=1 reproduces base PTA exactly (anchor absorb-only + refine)",
         check_a_k1_matches_base_pta),
        ("(b) K>1 a new mode GROWS a slot (active_k increases, no collapse)",
         check_b_new_mode_grows),
        ("(c) K>1 bank full: new mode recycles specialized slot, anchor safe",
         check_c_anchor_never_recycled),
        ("(d) score is the max over ACTIVE slots of cos(x, refine[c,k])",
         check_d_max_of_active),
        ("(e) fused logits finite; prediction matches clip + 100*maxcos",
         check_e_fusion_path),
    ]
    failures = 0
    for name, fn in checks:
        try:
            ok = bool(fn())
        except Exception as exc:  # noqa: BLE001 - report any failure
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
