"""CPU unit tests for models/aug_pta.py (TPT-style view ensembling, fp32, no GPU).

Run:
    /share_98/projects/brandon/envs/pta/bin/python scripts/test_aug_pta_unit.py

Checks (each drives the REAL ``_ensemble_evidence`` path with a content-agnostic
lookup encoder so per-view logits are fully controlled):
  (a) V=1: the ensemble evidence is the IDENTITY — returned feature/logits
      equal the single-view encode + L2-norm + 100*cos computation that base
      PTA performs via get_clip_logits, so AugPTA reduces to base PTA exactly
      when the ensemble is disabled.
  (b) V=3 K=2: among three views (two confident class-0 views + one uniform /
      high-entropy outlier view) the adapter keeps exactly the two LOWEST-
      entropy views, drops the outlier, and returns the mean of the kept
      views' logits — the TPT selection semantics.
  (c) V=3 K=3: selecting all views yields the plain mean over ALL view logits.
  (d) K_views > V clamps to V; V < 1 clamps to 1; flat --override path works.
  (e) build(cfg) returns an AugPTAAdapter instance of BaseAdapter with the
      PTA wiring (image_level + fusion taus).
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.aug_pta import AugPTAAdapter, build  # noqa: E402
from models.base import BaseAdapter  # noqa: E402


def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True)


class _LookupEncoder:
    """Content-agnostic encoder: view v maps to feats_list[v] (pre-normalized).

    Lets the test fully control per-view logits through any input pixels —
    the augmentation randomness is irrelevant to the result.
    """

    def __init__(self, feats_list):
        self.feats_list = [f.clone() for f in feats_list]

    def encode_image(self, views, **kwargs):
        n = views.shape[0]
        if n > len(self.feats_list):
            raise AssertionError(f"lookup has {len(self.feats_list)} feats, batch {n}")
        # Mimic the real CLIP encoder: (V, 1, D) -> (V, D).
        return torch.stack([_norm(self.feats_list[i].squeeze(0)) for i in range(n)])


def _ortho_text(D, C, seed=3):
    """(D, C) matrix with orthonormal columns (CLIP convention: D feats x C)."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(D, C, generator=g)
    cols = torch.linalg.qr(M).Q[:, :C]
    return cols  # columns already orthonormal


def _entropy_of(logits):
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p.clamp(min=1e-12))).sum(dim=-1)


def _expected_selection(logits, K):
    return torch.argsort(_entropy_of(logits), descending=False)[:K]


def check_a_v1_is_identity():
    """(a) V=1: ens evidence == single-view encode+norm+100cos (base PTA)."""
    torch.manual_seed(0)
    C, D = 5, 16
    text = _ortho_text(D, C)                           # (D, C)
    feats_list = [_norm(torch.randn(1, D))]            # single view
    enc = _LookupEncoder(feats_list)
    adapter = build({"V": 1, "K_views": 4})

    img = torch.randn(1, 3, 8, 8)
    f, l = adapter._ensemble_evidence(img, enc, text)

    # Base PTA reference (get_clip_logits): encode -> L2-norm -> 100*feat@text.
    ref_feats = _norm(feats_list[0])
    ref_logits = (100.0 * ref_feats @ text).float()

    if f.shape != (1, D) or l.shape != (1, C):
        return False
    if not torch.allclose(f, ref_feats, atol=1e-6):
        return False
    if not torch.allclose(l, ref_logits, atol=1e-5):
        return False
    return True


def check_b_selection_drops_outlier():
    """(b) V=3 K=2: lowest-entropy pair kept; uniform outlier dropped."""
    torch.manual_seed(0)
    C, D = 3, 16
    text = _ortho_text(D, C)                            # (D, C), orthonormal cols
    t0, t1, t2 = text[:, 0], text[:, 1], text[:, 2]

    # Views: 0 and 1 confident on class 0; view 2 near-uniform (high entropy).
    f0 = t0.clone().unsqueeze(0)
    f1 = t0.clone().unsqueeze(0)
    f2 = _norm((t0 + t1 + t2).unsqueeze(0))
    enc = _LookupEncoder([f0, f1, f2])
    adapter = build({"V": 3, "K_views": 2})

    img = torch.randn(1, 3, 8, 8)
    orig_feat, ens_logits = adapter._ensemble_evidence(img, enc, text)

    # Per-view logits (identical math to the adapter: 100 * f @ text).
    view_logits = torch.stack([100.0 * f @ text for f in [f0, f1, f2]]).float()
    sel = _expected_selection(view_logits, 2)
    expected_ens = view_logits[sel].mean(0).unsqueeze(0)
    expected_orig = _norm(f0)

    if not torch.allclose(ens_logits, expected_ens, atol=1e-4):
        return False
    # The outlier (view 2) must NOT be part of the average.
    mean_all = view_logits.mean(0).unsqueeze(0)
    if torch.allclose(ens_logits, mean_all, atol=1e-3):
        return False
    if not torch.allclose(orig_feat, expected_orig, atol=1e-6):
        return False
    # Sanity: view 2 is genuinely the high-entropy outlier in this construction.
    if not bool(_entropy_of(view_logits)[2] > _entropy_of(view_logits)[0] + 0.5):
        return False
    return True


def check_c_k_all_keep_all():
    """(c) K_views == V: ensemble is the plain mean over ALL view logits."""
    torch.manual_seed(0)
    C, D = 4, 16
    text = _ortho_text(D, C)
    g = torch.Generator().manual_seed(5)
    feats_list = [_norm(torch.randn(1, D, generator=g)) for _ in range(4)]
    enc = _LookupEncoder(feats_list)
    adapter = build({"V": 4, "K_views": 4})

    img = torch.randn(1, 3, 8, 8)
    _, ens_logits = adapter._ensemble_evidence(img, enc, text)

    view_logits = torch.stack([100.0 * f @ text for f in feats_list]).float()
    expected = view_logits.mean(0).unsqueeze(0)
    return bool(torch.allclose(ens_logits, expected, atol=1e-4))


def check_d_clamping():
    """(d) K_views > V clamps to V; V < 1 clamps to 1."""
    a0 = build({"V": 0, "K_views": 0})
    if a0.V != 1 or a0.K_views != 1:
        return False
    a2 = build({"V": 2, "K_views": 9})
    if a2.V != 2 or a2.K_views != 2:
        return False
    a3 = build({"V": 8, "K_views": 4})
    if a3.V != 8 or a3.K_views != 4:
        return False
    # Flat --override path (runner's interface: --override V=8 K_views=4).
    a_flat = build({"V": 4, "K_views": 2, "image_level": {}})
    return bool(a_flat.V == 4 and a_flat.K_views == 2)


def check_e_build_type():
    """(e) build() returns an AugPTAAdapter instance of BaseAdapter."""
    adapter = build({"V": 8, "K_views": 4, "image_level": {"alpha": 0.01, "T": 20.0},
                     "fusion": {}})
    if not isinstance(adapter, BaseAdapter) or not isinstance(adapter, AugPTAAdapter):
        return False
    if adapter.image_level is None:
        return False
    if adapter.fusion.tau_text != 1.0 or adapter.fusion.tau_image_proto != 100.0:
        return False
    return bool(adapter.fusion.tau_patch_proto == 0.0)


def main():
    checks = [
        ("(a) V=1: ensemble evidence == single-view base PTA computation",
         check_a_v1_is_identity),
        ("(b) V=3 K=2: lowest-entropy selection drops the uniform outlier",
         check_b_selection_drops_outlier),
        ("(c) K_views == V: ensemble == mean over all view logits",
         check_c_k_all_keep_all),
        ("(d) K_views/V clamping + nested aug_pta.V precedence",
         check_d_clamping),
        ("(e) build() returns valid AugPTAAdapter (BaseAdapter + PTA wiring)",
         check_e_build_type),
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