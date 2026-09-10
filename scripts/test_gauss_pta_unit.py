"""CPU headless unit test for models/gauss_pta.py (per-class Gaussian TTA).

Verifies, on synthetic float32 tensors (CPU only, no CUDA, no data):
  (a) with all variances isotropic (no writes yet), the Mahalanobis ranking is
      identical to the cosine ranking (monotonic equivalence);
  (b) after simulated writes the per-dim variance becomes anisotropic and the
      Mahalanobis score down-weights high-variance directions relative to
      cosine;
  (c) the EMA update of mean/variance matches a hand-computed reference on a
      3-step sequence;
  (d) no NaN/inf for any state (including classes never written);
  (e) the fused logits are finite and the per-sample rescaling keeps the
      prototype term in [-1, 0] (cosine scale).

Run:
  /share_98/projects/brandon/envs/pta/bin/python scripts/test_gauss_pta_unit.py
Exits nonzero if any check fails.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gauss_pta import (  # noqa: E402
    fuse_mahalanobis,
    gauss_multi_class_write,
    init_gauss_state,
    mahalanobis_logits,
)

torch.manual_seed(0)


def _unit(vec):
    return vec / vec.norm()


def _softmax_py(logits):
    m = max(logits)
    ex = [math.exp(z - m) for z in logits]
    s = sum(ex)
    return [e / s for e in ex]


def check_a_isotropic_equivalence():
    """(a) Isotropic state: Mahalanobis ranking == cosine ranking."""
    C, D, shrink = 5, 64, 0.5
    text = torch.randn(C, D)
    text = text / text.norm(dim=-1, keepdim=True)
    mu, v, n = init_gauss_state(text)

    # No writes yet: state is exactly isotropic / zero-count.
    assert torch.equal(v, torch.ones(C, D)), "v must init isotropic at 1.0"
    assert torch.equal(n, torch.zeros(C)), "n must init at 0"
    assert torch.allclose(mu, text, atol=1e-6), "mu must init at text embedding"

    x = _unit(torch.randn(D))
    logit, vbar = mahalanobis_logits(x, mu, v, shrink)
    cos = x @ mu.t()

    # Identical ranking (argsort) and identical pairwise ordering.
    order_m = torch.argsort(logit, descending=True)
    order_c = torch.argsort(cos, descending=True)
    assert torch.equal(order_m, order_c), "Mahalanobis ranking != cosine ranking"
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            assert (logit[i] - logit[j]).item() * (cos[i] - cos[j]).item() >= 0.0, (
                f"pairwise order flipped for classes ({i}, {j})"
            )

    # Exact monotonic (affine) relation: with v=1, Vbar=1,
    # logit_c = (2*cos_c - 2) / (D * (1 + shrink)).
    expected = (2.0 * cos - 2.0) / (D * (1.0 + shrink))
    assert torch.allclose(logit, expected, rtol=1e-5, atol=1e-6), (
        "isotropic Mahalanobis logit is not an affine function of cosine"
    )
    return "argsort identical + logit == (2cos-2)/(D(1+shrink)) (affine, monotonic)"


def check_b_anisotropy_and_downweighting():
    """(b) After writes: v anisotropic; Mahalanobis down-weights high-var dir."""
    C, D, shrink, T = 2, 64, 0.1, 1.0
    t0 = _unit(torch.randn(D))
    # t1 orthogonal to t0 (clean cosine-indifference baseline).
    t1 = _unit(torch.randn(D) - (torch.randn(D) @ t0) * t0)
    text = torch.stack([t0, t1], dim=0)
    mu, v, n = init_gauss_state(text)

    # Simulated writes: class-0 samples spread along dim 5 (anisotropic class
    # spread), alternating +/- so the mean lags and the variance accumulates.
    e5 = torch.zeros(D)
    e5[5] = 1.0
    logits_c0 = torch.tensor([[20.0, 0.0]])  # softmax ~= [1, 0] -> w0 = 1-exp(-1/T)
    for sign in (+1.0, -1.0, +1.0, -1.0, +1.0, -1.0):
        xw = _unit(t0 + sign * 1.2 * e5)
        mu, v, n = gauss_multi_class_write(xw, logits_c0, mu, v, n, T)

    v0 = v[0]
    j_hi = int(v0.argmax())
    j_lo = int(v0.argmin())
    aniso = (v0.max() - v0.min()).item() / (v0.max().item() + 1e-8)
    assert j_hi == 5, f"max-variance dim is {j_hi}, expected written dim 5"
    assert aniso > 0.5, f"variance not anisotropic after writes (aniso={aniso:.4f})"
    assert n[0].item() > 0.0 and n[1].item() == 0.0, "write counts wrong"

    # Same-magnitude deviation, high-variance dir vs low-variance dir.
    a = 0.4
    e_hi = torch.zeros(D)
    e_hi[j_hi] = 1.0
    e_lo = torch.zeros(D)
    e_lo[j_lo] = 1.0
    x_hi = _unit(mu[0] + a * e_hi)
    x_lo = _unit(mu[0] + a * e_lo)
    # mahalanobis_logits returns logit = -d: a HIGHER logit means the
    # deviation was down-weighted less.
    logit_hi, _ = mahalanobis_logits(x_hi, mu, v, shrink)
    logit_lo, _ = mahalanobis_logits(x_lo, mu, v, shrink)
    assert logit_hi[0].item() > logit_lo[0].item(), (
        f"Mahalanobis did not down-weight high-variance direction "
        f"(logit_hi={logit_hi[0].item():.6f} <= logit_lo={logit_lo[0].item():.6f})"
    )

    # Cosine is direction-agnostic: both same-magnitude deviations are
    # penalized (nearly) equally — the contrast Mahalanobis breaks.
    cos_hi = (x_hi @ mu[0]).item()
    cos_lo = (x_lo @ mu[0]).item()
    assert abs(cos_hi - cos_lo) < 0.05, (
        f"cosine unexpectedly direction-sensitive ({cos_hi:.4f} vs {cos_lo:.4f})"
    )

    # Effective per-dim weight 1/(v + shrink*Vbar): high-var dir down-weighted
    # vs cosine's uniform weight-1.
    vbar = v.mean(dim=0)
    w_hi = 1.0 / (v0[j_hi].item() + shrink * vbar[j_hi].item())
    w_lo = 1.0 / (v0[j_lo].item() + shrink * vbar[j_lo].item())
    assert w_hi < w_lo, "Mahalanobis weight not lower in high-variance direction"
    return (
        f"aniso={aniso:.3f} (argmax dim={j_hi}), logit_hi={logit_hi[0].item():.5f} > "
        f"logit_lo={logit_lo[0].item():.5f} (high-var dir down-weighted), "
        f"cosine indifferent (|{cos_hi:.4f}-{cos_lo:.4f}|={abs(cos_hi - cos_lo):.5f})"
    )


def _ema_sequence_state():
    """Build the shared 3-step EMA state used by checks (c), (d), (e).

    C=4, D=16, T=5.0. Step 0 writes class 0 only (p1 ~= 0.047 < 0.1); step 1
    writes class 2 only; step 2 is a multi-class write (classes 0 and 3 both
    >= 0.1). Class 1 is never written.
    """
    C, D, T = 4, 16, 5.0
    text = torch.randn(C, D)
    text = text / text.norm(dim=-1, keepdim=True)
    mu, v, n = init_gauss_state(text)
    xs = [_unit(torch.randn(D)) for _ in range(3)]
    logits_steps = [
        [2.0, -1.0, -5.0, -5.0],
        [-5.0, -5.0, 2.5, -5.0],
        [0.5, -5.0, -5.0, 0.3],
    ]
    for s in range(3):
        mu, v, n = gauss_multi_class_write(
            xs[s], torch.tensor([logits_steps[s]]), mu, v, n, T
        )
    logit, _ = mahalanobis_logits(_unit(torch.randn(D)), mu, v, 0.1)
    return text, mu, v, n, logit, xs, logits_steps


def check_c_ema_matches_reference(text, mu, v, n, xs, logits_steps):
    """(c) 3-step EMA update of mu/v/n matches a hand-computed reference."""
    C, D, T = 4, 16, 5.0
    # Hand-computed reference state (pure Python float64), replaying the exact
    # same 3-step sequence (xs, logits_steps) as _ema_sequence_state.
    mu_ref = [[float(z) for z in row] for row in text.tolist()]
    v_ref = [[1.0] * D for _ in range(C)]
    n_ref = [0.0] * C

    for s in range(3):
        x = xs[s]
        p = _softmax_py(logits_steps[s])
        for c in range(C):
            if p[c] >= 1e-1:
                w = 1.0 - math.exp(-p[c] / T)
                mu_c = [(1.0 - w) * mu_ref[c][j] + w * x[j].item() for j in range(D)]
                norm = math.sqrt(sum(m * m for m in mu_c))
                mu_c = [m / norm for m in mu_c]
                v_ref[c] = [
                    (1.0 - w) * v_ref[c][j] + w * (x[j].item() - mu_c[j]) ** 2
                    for j in range(D)
                ]
                n_ref[c] += w
                mu_ref[c] = mu_c

    assert torch.allclose(mu, torch.tensor(mu_ref), rtol=1e-5, atol=1e-6), (
        "mu does not match hand-computed reference"
    )
    assert torch.allclose(v, torch.tensor(v_ref), rtol=1e-5, atol=1e-6), (
        "v does not match hand-computed reference"
    )
    assert torch.allclose(n, torch.tensor(n_ref), rtol=1e-5, atol=1e-6), (
        "n does not match hand-computed reference"
    )
    # Sanity: the sequence actually exercised the interesting paths.
    assert n_ref[0] > 0 and n_ref[2] > 0 and n_ref[3] > 0, "expected writes missing"
    assert n_ref[1] == 0.0, "class 1 should never have been written"
    return f"3-step mu/v/n match float64 reference (n={[round(x, 4) for x in n_ref]})"


def check_d_no_nan_inf(text, mu, v, n, logit):
    """(d) No NaN/inf anywhere, including classes never written."""
    for name, t in (("mu", mu), ("v", v), ("n", n), ("logit", logit)):
        assert torch.isfinite(t).all(), f"non-finite value in {name}"
    # Never-written class (index 1 in the (c) state): untouched init state.
    assert torch.allclose(mu[1], text[1], atol=1e-6), "never-written mu changed"
    assert torch.equal(v[1], torch.ones_like(v[1])), "never-written v changed"
    assert n[1].item() == 0.0, "never-written n changed"
    return "mu/v/n/logit all finite; never-written class state untouched"


def check_e_fusion_bounded(mu, v, n):
    """(e) Fused logits finite; rescaled proto term kept in [-1, 0]."""
    C, shrink = 4, 0.1
    x = _unit(torch.randn(16))
    logit, vbar = mahalanobis_logits(x, mu, v, shrink)
    assert torch.isfinite(logit).all(), "Mahalanobis logit not finite"

    spread = (logit.max() - logit.min()).item()
    proto_term = (logit - logit.max()) / (spread + 1e-8)
    assert bool((proto_term <= 1e-6).all()), "rescaled proto term exceeds 0"
    assert bool((proto_term >= -1.0 - 1e-6).all()), (
        f"rescaled proto term below -1 (min={proto_term.min().item():.6f})"
    )

    # Fusion with fp16 clip logits (100*cos scale), as on the GPU path.
    cos = torch.randn(1, C)
    clip16 = (100.0 * cos).half()
    final = fuse_mahalanobis(clip16, logit, tau_text=1.0, tau_image_proto=100.0)
    assert final.dtype == torch.float16, f"fused logits not fp16 ({final.dtype})"
    assert torch.isfinite(final.float()).all(), "fused logits not finite"

    # Cross-check against the manual formula (fp32 math, cast to fp16).
    expected = (1.0 * clip16.float() + 100.0 * proto_term.unsqueeze(0)).half()
    assert torch.allclose(final, expected, rtol=2e-3, atol=0.1), (
        "fused logits deviate from manual formula"
    )
    return (
        f"rescaled proto term in [{proto_term.min().item():.6f}, "
        f"{proto_term.max().item():.6f}] == [-1, 0] (raw spread={spread:.6f}); "
        f"fused fp16 finite, matches formula"
    )


def main():
    # Shared state for (c)/(d)/(e) — built once, deterministic (seeded).
    text, mu, v, n, logit_c, xs, logits_steps = _ema_sequence_state()

    checks = [
        ("a: isotropic Mahalanobis == cosine ranking", check_a_isotropic_equivalence),
        ("b: anisotropic v + Mahalanobis down-weights high-var dir",
         check_b_anisotropy_and_downweighting),
        ("c: 3-step EMA matches hand-computed reference",
         lambda: check_c_ema_matches_reference(text, mu, v, n, xs, logits_steps)),
        ("d: no NaN/inf in any state (incl. never-written class)",
         lambda: check_d_no_nan_inf(text, mu, v, n, logit_c)),
        ("e: fused logits finite, rescaled proto term in [-1, 0]",
         lambda: check_e_fusion_bounded(mu, v, n)),
    ]

    n_fail = 0
    for name, fn in checks:
        try:
            detail = fn()
        except AssertionError as e:
            n_fail += 1
            print(f"FAIL: {name}")
            print(f"      {e}")
        else:
            print(f"PASS: {name}")
            print(f"      {detail}")

    if n_fail:
        print(f"\n{n_fail}/{len(checks)} CHECKS FAILED")
        return 1
    print(f"\nALL {len(checks)} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
