"""CPU unit tests for models/bank_pta.py (synthetic fp32 tensors, no GPU/data).

Run:
    /share_98/projects/brandon/envs/pta/bin/python scripts/test_bank_pta_unit.py

Checks:
  (a) with K=1 the method reproduces base PTA EXACTLY: zero-init prototype,
      EMA write without re-normalization, and scoring via
      refine = normalize(0.01*text + 0.99*proto).  State and logits must match
      a reference that mirrors models/image_level/pta_image.py bit-for-bit.
  (b) with K>1 a written feature updates ONLY the nearest prototype (nearest
      measured against the base-PTA-style refine), others unchanged.
  (c) the score is the max over K of cos(x, refine[c, k]).
  (d) prototypes start at ZERO (like base PTA) and stay finite after many
      updates; a never-written prototype scores exactly like the class text.
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
from models.bank_pta import (
    BankPTAAdapter,
    build,
    init_bank,
    bank_ema_write,
    bank_score,
    refine_bank,
)
from models.fusion import WeightedFusion

ALPHA = 0.01


def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)


def _rand_unit(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return _norm(torch.randn(shape, generator=g))


def _reference_base_pta(x, clip_logits, text, proto, T):
    """Reference: base PTA's exact update + score (models/image_level/pta_image.py).

    proto[c] = (1 - w_c) * proto[c] + w_c * x           (NO re-normalization)
    refined  = normalize(0.01 * text + 0.99 * proto)
    logit    = cos(x, refined)

    Args:
        x:           (D,) L2-normalized image feature.
        clip_logits: (1, C) zero-shot logits.
        text:        (C, D) L2-normalized class text embeddings.
        proto:       (C, D) prototype (mutated in-place, like base PTA).
        T:           EMA temperature.

    Returns:
        (C,) reference logits.
    """
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
    """(a) K=1 reproduces base PTA exactly (zero-init, refine scoring)."""
    C, D, K, T = 5, 16, 1, 20.0
    text = _rand_unit((C, D), seed=0)
    P = init_bank(text, K)                                  # [C, 1, D] zeros
    ref_proto = torch.zeros(C, D)                           # base PTA: zero-init

    # Zero-init must match base PTA's init_state.
    if not torch.equal(P[:, 0, :], ref_proto):
        return False

    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        x = _norm(torch.randn(D, generator=g))
        clip_logits = 2.0 * torch.randn(1, C, generator=g)
        bank_ema_write(x.unsqueeze(0), clip_logits, text, P, ALPHA, T)
        ref_logit = _reference_base_pta(x, clip_logits, text, ref_proto, T)

        # Score must match base PTA's cos(x, refined) for this sample.
        bank_logit = bank_score(x.unsqueeze(0), text, P, ALPHA).squeeze(0)
        if not torch.allclose(bank_logit, ref_logit, atol=1e-5):
            return False

    # State must match base PTA's prototype after the same 50 writes.
    return bool(torch.allclose(P[:, 0, :], ref_proto, atol=1e-5))


def check_b_nearest_only():
    """(b) K>1: a written feature updates ONLY the nearest prototype."""
    C, D, K, T = 2, 16, 3, 20.0
    e = torch.eye(D)
    text = _norm(torch.stack([e[0], e[1]]))
    P = init_bank(text, K)                                  # zeros
    # Seed two prototypes of class 0 so the nearest is unique and non-trivial.
    P[0, 1] = e[2].clone()
    P[0, 2] = _norm(e[0] + 0.1 * e[3])
    snap = P.clone()

    x = _norm(e[0] + 0.3 * e[5]).unsqueeze(0)               # (1, D)
    clip_logits = torch.tensor([[10.0, -10.0]])             # only class 0 passes 0.1
    writes = bank_ema_write(x, clip_logits, text, P, ALPHA, T)

    # Determine the expected nearest prototype via the base-PTA-style refine.
    refine_c = refine_bank(text[0:1], snap[0:1], ALPHA)[0]  # (K, D)
    expected_k = int((x.squeeze(0) @ refine_c.T).argmax())
    if [(c, k) for c, k, _w in writes] != [(0, expected_k)]:
        return False

    # Nearest prototype updated with the base PTA EMA weight (no re-normalize).
    p0 = float(F.softmax(clip_logits, dim=-1).float().squeeze(0)[0])
    w_c = 1.0 - math.exp(-p0 / T)
    expected = (1.0 - w_c) * snap[0, expected_k] + w_c * x.squeeze(0)
    if not torch.allclose(P[0, expected_k], expected, atol=1e-6):
        return False

    # All other prototypes (and the whole unwritten class) unchanged.
    for k in range(K):
        if k != expected_k and not torch.equal(P[0, k], snap[0, k]):
            return False
    if not torch.equal(P[1], snap[1]):
        return False
    return True


def check_c_max_of_k():
    """(c) the score is the max over K of cos(x, refine[c, k])."""
    C, D, K = 4, 24, 5
    g = torch.Generator().manual_seed(3)
    text = _rand_unit((C, D), seed=9)
    P = torch.randn(C, K, D, generator=g)                  # arbitrary (non-zero)
    x = _rand_unit((D,), seed=4).unsqueeze(0)

    refine = refine_bank(text, P, ALPHA)
    got = bank_score(x, text, P, ALPHA).squeeze(0)
    manual = (x.squeeze(0) @ refine.permute(1, 2, 0)).max(dim=0).values  # (C,)
    if not torch.allclose(got, manual, atol=1e-6):
        return False

    # Structured case: a prototype aligned with x -> refine ~ x -> logit ~ 1.
    P2 = P.clone()
    P2[1, 2] = x.squeeze(0).clone() / ALPHA * (1 - ALPHA)  # refine ~ x (alpha small)
    got2 = bank_score(x, text, P2, ALPHA).squeeze(0)
    if float(got2[1]) < 0.99:
        return False
    return True


def check_d_zero_init_and_stability():
    """(d) zero-init like base PTA; finite after updates; zero proto ~ text."""
    C, D, K, T = 3, 32, 4, 20.0
    text = _rand_unit((C, D), seed=5)
    P = init_bank(text, K)

    # Zero-init (base PTA's init_state returns zeros).
    if not torch.equal(P, torch.zeros(C, K, D)):
        return False

    # A never-written (zero) prototype scores exactly like the class text.
    x = _rand_unit((D,), seed=6).unsqueeze(0)
    logit_zero = bank_score(x, text, P, ALPHA).squeeze(0)
    logit_text = x.squeeze(0) @ text.T                      # cos(x, text)
    if not torch.allclose(logit_zero, logit_text, atol=1e-5):
        return False

    g = torch.Generator().manual_seed(7)
    for _ in range(200):
        x = _norm(torch.randn(D, generator=g)).unsqueeze(0)
        clip_logits = 2.0 * torch.randn(1, C, generator=g)
        bank_ema_write(x, clip_logits, text, P, ALPHA, T)

    return bool(torch.isfinite(P).all() and torch.isfinite(bank_score(x, text, P, ALPHA)).all())


def check_e_fusion_path():
    """(e) fused logits finite; prediction path matches clip + 100*maxcos."""
    C, D, K = 6, 16, 3
    cfg = {
        "image_level": {"alpha": ALPHA, "T": 20.0},
        "fusion": {"tau_image_proto": 100.0, "tau_patch_proto": 0.0},
        "K": K,
    }
    adapter = build(cfg)
    ok = isinstance(adapter, BaseAdapter) and isinstance(adapter, BankPTAAdapter)
    ok = ok and adapter.K == K
    ok = ok and isinstance(adapter.fusion, WeightedFusion)
    ok = ok and (
        adapter.fusion.tau_text == 1.0
        and adapter.fusion.tau_image_proto == 100.0
        and adapter.fusion.tau_patch_proto == 0.0
    )
    ok = ok and build({"image_level": {"alpha": ALPHA, "T": 20.0}, "fusion": {}}).K == 3
    if not ok:
        return False

    # Fusion mirrors clip + 100 * maxcos (fp16, as in the runner).
    g = torch.Generator().manual_seed(8)
    text = _rand_unit((C, D), seed=9)
    P = torch.randn(C, K, D, generator=g)
    x = _norm(torch.randn(D, generator=g))
    maxcos = bank_score(x.unsqueeze(0), text, P, ALPHA).squeeze(0)  # (C,) fp32
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
        ("(a) K=1 reproduces base PTA exactly (zero-init + refine scoring)",
         check_a_k1_matches_base_pta),
        ("(b) K>1 written feature updates ONLY the nearest prototype",
         check_b_nearest_only),
        ("(c) score is the max over K of cos(x, refine[c,k])",
         check_c_max_of_k),
        ("(d) zero-init like base PTA; finite; zero proto scores like text",
         check_d_zero_init_and_stability),
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
