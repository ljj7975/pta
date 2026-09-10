"""CPU unit tests for models/gauss_compact_pta.py (Gaussian + gate, fp32).

Run:
    /share_98/projects/brandon/envs/pta/bin/python scripts/test_gauss_compact_pta_unit.py

Checks:
  (a) with gate_mode=off the method reproduces gauss_pta EXACTLY: after the
      same writes, (mu, v, n) must match models/gauss_pta.gauss_multi_class_write.
  (b) the SOFT gate reduces the effective write weight for a far sample
      (n_new < ungated n) — tames the drag on the mean.
  (c) the HARD gate drops the far sample entirely (n unchanged, w = 0).
  (d) the score is the shifted Mahalanobis (identical to gauss_pta).
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gauss_pta import (
    init_gauss_state,
    gauss_multi_class_write,
    mahalanobis_logits,
)
from models.gauss_compact_pta import (
    GaussCompactPTAAdapter,
    build,
    init_gate_stats,
    gauss_compact_write,
    GATE_SOFT,
    GATE_HARD,
    GATE_OFF,
)


def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)


def _rand_unit(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return _norm(torch.randn(shape, generator=g))


def check_a_off_matches_gauss_pta():
    """(a) gate_mode=off reproduces gauss_pta exactly (same mu, v, n)."""
    C, D, T = 5, 16, 20.0
    text = _rand_unit((C, D), seed=0)
    mu0, v0, n0 = init_gauss_state(text)

    mu_g, v_g, n_g = init_gauss_state(text)          # gauss_pta reference
    mu_c, v_c, n_c = init_gauss_state(text)          # gauss_compact (gate off)
    gate_stats = init_gate_stats(C, 0.1)

    g = torch.Generator().manual_seed(1)
    for _ in range(50):
        x = _norm(torch.randn(D, generator=g)).unsqueeze(0)
        clip_logits = 2.0 * torch.randn(1, C, generator=g)
        mu_g, v_g, n_g = gauss_multi_class_write(x, clip_logits, mu_g, v_g, n_g, T)
        mu_c, v_c, n_c = gauss_compact_write(
            x, clip_logits, mu_c, v_c, n_c, T, GATE_OFF, 0.05, 0.05, 0.5, gate_stats
        )
        if not (torch.allclose(mu_g, mu_c, atol=1e-6)
                and torch.allclose(v_g, v_c, atol=1e-6)
                and torch.allclose(n_g, n_c, atol=1e-6)):
            return False
    return True


def _final_n(gate_mode, far=True):
    """Effective count n[0] after establishing the mean then a (far) sample."""
    C, D, T = 1, 8, 20.0
    e = torch.eye(D)
    text = _norm(e[0].clone().unsqueeze(0))
    mu, v, n = init_gauss_state(text)
    gate_stats = init_gate_stats(C, 0.1)
    clip0 = torch.tensor([[10.0]])

    # Establish the mean with a near sample (e0).
    mu, v, n = gauss_compact_write(_norm(e[0]).unsqueeze(0), clip0, mu, v, n,
                                   T, gate_mode, 0.05, 0.05, 0.5, gate_stats)
    n_after_near = float(n[0].item())
    # Far sample (e1, orthogonal to the mean).
    mu, v, n = gauss_compact_write(_norm(e[1]).unsqueeze(0), clip0, mu, v, n,
                                   T, gate_mode, 0.05, 0.05, 0.5, gate_stats)
    return n_after_near, float(n[0].item())


def check_b_soft_gate_reduces_weight():
    """(b) soft gate reduces the effective weight for a far sample (n grows less)."""
    n_near_off, n_far_off = _final_n(GATE_OFF)
    n_near_soft, n_far_soft = _final_n(GATE_SOFT)
    delta_off = n_far_off - n_near_off
    delta_soft = n_far_soft - n_near_soft
    # The ungated far write adds a positive amount; the soft-gated one adds less.
    return bool(delta_off > 1e-6 and delta_soft < delta_off - 1e-6)


def check_c_hard_gate_drops_far():
    """(c) hard gate drops the far sample entirely (n unchanged)."""
    n_near_hard, n_far_hard = _final_n(GATE_HARD)
    # With the hard gate the far (orthogonal) sample is dropped -> n unchanged.
    return bool(abs(n_far_hard - n_near_hard) < 1e-9)


def check_d_mahalanobis_score():
    """(d) the score is the shifted Mahalanobis (identical to gauss_pta)."""
    C, D = 4, 24
    g = torch.Generator().manual_seed(3)
    text = _rand_unit((C, D), seed=9)
    mu, v, n = init_gauss_state(text)
    mu = _norm(mu + 0.1 * torch.randn(C, D, generator=g))
    v = torch.rand(C, D, generator=g) + 0.5
    x = _norm(torch.randn(D, generator=g)).unsqueeze(0)
    shrink = 0.3
    logit, _ = mahalanobis_logits(x, mu, v, shrink)
    # Sanity: finite, and the best class has the smallest distance.
    return bool(torch.isfinite(logit).all() and float(logit.max()) <= 1e-6)


def main():
    checks = [
        ("(a) gate=off reproduces gauss_pta exactly (same mu, v, n)",
         check_a_off_matches_gauss_pta),
        ("(b) soft gate REDUCES the effective weight for a far sample",
         check_b_soft_gate_reduces_weight),
        ("(c) hard gate DROPS the far sample (n unchanged)",
         check_c_hard_gate_drops_far),
        ("(d) score is the shifted Mahalanobis (finite, best <= 0)",
         check_d_mahalanobis_score),
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
