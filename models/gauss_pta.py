"""Prototype-Based TTA with per-class Gaussian prototypes (TCA-style upgrade).

Motivation: pre-validation on a one-off feature dump (favorable full-stream
proxy) showed that variance-aware (Mahalanobis) scoring beats single-mean
cosine scoring on all 3 datasets (dtd +4.2pp at shrink=0.5, flowers +1.7,
pets +2.0). CLIP features within a class are spread out anisotropically and a
single mean is a poor representative — this is the TCA-style per-class
Gaussian upgrade of base PTA.

Differences from base PTA (``models/pta.py``):

* **State** — per class c (all float32, even on GPU where the feature flow is
  fp16):
    - ``mu_c``: L2-normalized mean, INITIALIZED AT THE CLASS TEXT EMBEDDING
      (``text_embeddings.t()[c]``, normalized) — mirrors PTA's "start from
      text" behavior, NOT zero.
    - ``v_c``: per-dim variance, INITIALIZED ISOTROPIC at 1.0 (vector of
      ones, D-dim).
    - ``n_c``: effective write count, initialized 0.
* **Write rule** — base PTA's MULTI-CLASS write (logic of
  ``PTAImageLevel.update_prototypes``, models/image_level/pta_image.py):
  for every class c with ``softmax(clip_logits)[c] >= 0.1``,
  ``w_c = 1 - exp(-p_c / T)`` with ``T`` from config (default 20.0).
  (NOT the single top-1 write of ``models/reweight_pta.py``.)
* **Update** for each written class c with image feature x (1, D L2-norm)
  and weight w_c:
    - ``mu_c <- normalize((1 - w_c) * mu_c + w_c * x)``
    - ``v_c  <- (1 - w_c) * v_c + w_c * (x - mu_c) * (x - mu_c)``
      (deviation from the UPDATED mean)
    - ``n_c  <- n_c + w_c``
* **Score** for image x, per class c:
    - ``d_c = (1/D) * sum_d (x - mu_c)^2 / (v_c + shrink * Vbar_d)`` where
      ``Vbar_d`` is the per-dim mean of ``v_c`` across ALL classes
      (recomputed each step, O(C*D)); ``shrink`` is a config key (default
      0.1).
    - ``logit_c = -d_c`` (shifted Mahalanobis, NOT cosine).
* **Fusion** — ``final = 1.0 * clip_logits + 100.0 * (logit_c - max_c
  logit_c)``. All Mahalanobis math runs in float32; the fused logits are
  cast back to fp16 for the record/softmax path.

Image-level only — no patch vote, no patch bank, no patch-level signal.

Recording (env-gated, behavior-neutral) mirrors ``models/reweight_pta.py``:
``RECORD_DIR`` / ``RESULT_LABEL`` / ``SEED`` / ``RESULT_FILE`` / ``MAX_BATCHES``
env vars, ``write_record_header`` / ``write_record`` / ``write_summary`` /
``write_convergence`` calls, per-class accuracy accumulators, periodic
logging, and the ``{label}'s performance on {dataset}: Top1- {acc:.2f}.``
result-file append. Per-sample ``proto_stats`` carries per-class Gaussian
diagnostics (``n`` effective count, ``var_aniso`` anisotropy ratio) plus the
run's ``shrink`` value as a top-level key.
"""

import os
from typing import Any, Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from utils import get_clip_logits, cls_acc
from utils.records import (
    write_record_header,
    write_record,
    write_summary,
    write_convergence,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pure state functions (float32, CPU/GPU-agnostic — unit-tested on CPU)
# ─────────────────────────────────────────────────────────────────────────────


def init_gauss_state(text_features):
    """Initialize the per-class Gaussian state from text embeddings.

    Args:
        text_features: (C, D) class text embeddings (any dtype; typically
            ``text_embeddings.t()`` in fp16).

    Returns:
        (mu, v, n) — all float32, same device as ``text_features``:
            mu: (C, D) L2-normalized means, initialized at the text embeddings.
            v:  (C, D) per-dim variances, initialized isotropic at 1.0.
            n:  (C,)   effective write counts, initialized 0.
    """
    mu = text_features.float()
    mu = mu / mu.norm(dim=-1, keepdim=True)
    v = torch.ones_like(mu)
    n = torch.zeros(mu.shape[0], dtype=torch.float32, device=mu.device)
    return mu, v, n


def gauss_multi_class_write(image_feature, clip_logits, mu, v, n, T):
    """Base PTA multi-class EMA write onto the per-class Gaussian state.

    Write rule (identical to ``PTAImageLevel.update_prototypes``): every class
    c with ``softmax(clip_logits)[c] >= 0.1`` is written with
    ``w_c = 1 - exp(-p_c / T)``. For each written class:
        mu_c <- normalize((1 - w_c) * mu_c + w_c * x)
        v_c  <- (1 - w_c) * v_c + w_c * (x - mu_c)^2   (deviation from the
                                                        UPDATED mean)
        n_c  <- n_c + w_c
    All math in float32; inputs may be fp16 (the GPU feature flow).

    Args:
        image_feature: (1, D) L2-normalized CLIP image embedding.
        clip_logits:   (1, C) zero-shot logits (before softmax).
        mu, v:         (C, D) float32 state (means / per-dim variances).
        n:             (C,)   float32 effective counts.
        T:             float EMA temperature (config ``T``, default 20.0).

    Returns:
        (mu_new, v_new, n_new) — new float32 tensors (inputs not mutated).
    """
    x = image_feature.reshape(-1).float()                    # (D,) fp32
    probs = F.softmax(clip_logits.reshape(-1).float(), dim=0)  # (C,) fp32

    w = torch.zeros_like(probs)                              # (C,)
    mask = probs >= 1e-1                                     # (C,) bool
    w[mask] = 1.0 - torch.exp(-probs[mask] / T)              # (K,)

    mu_new = mu.clone()
    v_new = v.clone()
    n_new = n.clone()
    if bool(mask.any()):
        m = w[mask]                                          # (K,)
        mu_sel = (1.0 - m.unsqueeze(1)) * mu[mask] + m.unsqueeze(1) * x
        mu_sel = mu_sel / mu_sel.norm(dim=-1, keepdim=True)
        dev2 = (x - mu_sel) ** 2                             # (K, D)
        v_sel = (1.0 - m.unsqueeze(1)) * v[mask] + m.unsqueeze(1) * dev2
        mu_new[mask] = mu_sel
        v_new[mask] = v_sel
        n_new[mask] = n_new[mask] + m
    return mu_new, v_new, n_new


def mahalanobis_logits(image_feature, mu, v, shrink):
    """Variance-aware (Mahalanobis) per-class score.

        d_c = (1/D) * sum_d (x - mu_c)^2 / (v_c + shrink * Vbar_d)
        logit_c = -d_c

    where ``Vbar_d`` is the per-dim mean of ``v_c`` across ALL classes
    (recomputed each step, O(C*D)). All math in float32.

    Args:
        image_feature: (1, D) L2-normalized CLIP image embedding.
        mu, v:         (C, D) float32 state.
        shrink:        float variance-regularization coefficient (default 0.1).

    Returns:
        (logit, vbar) — logit: (C,) float32 Mahalanobis logits (-d_c);
        vbar: (D,) float32 cross-class per-dim variance mean.
    """
    x = image_feature.reshape(-1).float()                    # (D,) fp32
    vbar = v.mean(dim=0)                                     # (D,)
    denom = v + shrink * vbar.unsqueeze(0)                   # (C, D)
    d = ((x.unsqueeze(0) - mu) ** 2 / denom).mean(dim=-1)    # (C,)
    return -d, vbar


def fuse_mahalanobis(clip_logits, logit, tau_text=1.0, tau_image_proto=100.0):
    """Fused prediction: ``final = tau_text * clip + tau_proto * proto_term``.

    The prototype term is the Mahalanobis logit rescaled to cosine's scale so
    it is weighted comparably to base PTA's ``100 * cos`` term.  The raw
    Mahalanobis distance ``d_c`` is ~100x smaller than cosine (unit-norm
    features spread over D=512 dims), so at ``tau=100`` the unrescaled term is
    negligible and the method collapses to CLIP.  We therefore normalize the
    per-sample logit spread to [-1, 0]:

        proto_term = (logit - logit.max()) / (logit.max() - logit.min())

    This preserves the Mahalanobis RANKING (dividing by a positive constant)
    while putting its magnitude on the same [-1, 0] scale as cosine's
    [-1, 1], so ``tau_image_proto=100`` gives the prototype signal the same
    weight base PTA gives its cosine term.  Math in float32; the result is
    cast back to the dtype of ``clip_logits`` (fp16 on GPU) for the
    record/softmax path.

    Args:
        clip_logits:   (1, C) zero-shot logits (100*cos scale, fp16).
        logit:         (C,) float32 Mahalanobis logits.
        tau_text:      weight on zero-shot CLIP logits (default 1.0).
        tau_image_proto: weight on the rescaled prototype term (default 100.0).

    Returns:
        (1, C) fused logits in the dtype of ``clip_logits``.
    """
    spread = logit.max() - logit.min()                        # delta > 0
    proto_term = (logit - logit.max()) / (spread + 1e-8)      # (C,) in [-1, 0]
    final = tau_text * clip_logits.reshape(-1).float() + tau_image_proto * proto_term
    return final.reshape(clip_logits.shape).to(clip_logits.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class GaussPTAAdapter(BaseAdapter):
    """PTA with per-class Gaussian prototypes and a variance-aware score.

    Write rule = base PTA multi-class EMA (every class with softmax prob
    >= 0.1, weight ``1 - exp(-p_c / T)``). State per class = L2-normalized
    mean (init: text embedding) + per-dim variance (init: isotropic 1.0) +
    effective count (init: 0). Score = shifted Mahalanobis
    ``logit_c = -(1/D) * sum_d (x - mu_c)^2 / (v_c + shrink * Vbar_d)``.
    Fusion = ``clip + 100 * (logit - max_c logit)`` in fp32, cast to fp16.
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        # ── Backward-compatible nested config ──────────────────────────────
        # Flat configs (alpha, T at root level) are still used by some
        # callers. Propagate them into the nested "image_level" sub-dict
        # (mirrors models/pta.py). T is consumed by the multi-class write
        # rule; alpha is propagated for config consistency.
        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg

        # ── Fusion weights ─────────────────────────────────────────────────
        # PTA defaults: tau_text=1.0, tau_image_proto=100.0, tau_patch_proto=0.0
        fusion_cfg = cfg.get("fusion", {})
        fusion_cfg.setdefault("tau_text", 1.0)
        fusion_cfg.setdefault("tau_image_proto", 100.0)
        fusion_cfg.setdefault("tau_patch_proto", 0.0)
        cfg["fusion"] = fusion_cfg
        self.tau_text = float(fusion_cfg["tau_text"])
        self.tau_image_proto = float(fusion_cfg["tau_image_proto"])

        # ── Method hyper-parameters ────────────────────────────────────────
        self.shrink = float(cfg.get("shrink", 0.1))
        self.T = float(image_level_cfg.get("T", cfg.get("T", 20.0)))

    def run(
        self,
        loader,
        encoder,
        text_embeddings,
        dataset_name: str,
    ) -> float:
        """
        Run GaussPTA on test set. Per-dataset config is loaded from self.cfg.
        """
        os.makedirs("outputs", exist_ok=True)

        # ── ENV-GATED PER-SAMPLE RECORDING (behavior-neutral) ─────────────
        # When RECORD_DIR is set, write the JSONL header + per-sample records
        # + summary via utils.records. When unset/empty the writer is a strict
        # no-op; everything below is skipped entirely so no tensor, RNG, or
        # loop-order state is perturbed. RESULT_LABEL / SEED only affect the
        # recorded metadata, never the computation.
        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "GaussPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []                                     # Track per-sample accuracy

            # Per-class Gaussian state (float32). mu is initialized at the
            # class text embeddings (mirrors PTA's "start from text"), v is
            # initialized isotropic, n starts at 0.
            mu, v, n = init_gauss_state(text_embeddings.t())    # [C, D] / [C, D] / [C]

            # Per-class accuracy accumulators for the summary's per_class dict.
            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes

            # Support early termination via MAX_BATCHES env var
            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            # Main evaluation loop: process one test sample per iteration
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[GaussPTA] {dataset_name}")
            ):
                # Early termination check
                if max_batches is not None and i >= max_batches:
                    break

                # ── ZERO-SHOT PREDICTION ───────────────────────────────────
                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )

                target = target.cuda()

                # ── ONLINE GAUSSIAN UPDATE (base PTA multi-class write) ───
                n_before = n
                mu, v, n = gauss_multi_class_write(
                    image_features, clip_logits, mu, v, n, self.T
                )

                # ── VARIANCE-AWARE (MAHALANOBIS) SCORE ────────────────────
                logit, _ = mahalanobis_logits(image_features, mu, v, self.shrink)

                # ── FUSED PREDICTION ───────────────────────────────────────
                # final = clip + 100 * (logit - max_c logit), fp32 math,
                # cast back to fp16.
                final_logits = fuse_mahalanobis(
                    clip_logits, logit, self.tau_text, self.tau_image_proto
                )

                # ── MEASURE ACCURACY ───────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # Per-class accuracy accumulation (consumed by summary below).
                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                # ── RECORD PER-SAMPLE (env-gated, behavior-neutral) ───────
                if record_dir:
                    # Per-class Gaussian diagnostics. write_record
                    # (utils/records.py) has a fixed signature with no shrink
                    # kwarg, so the run's shrink value rides along as a
                    # top-level proto_stats key (same convention as
                    # bank_pta's "K").
                    proto_stats: Dict[str, Any] = {"shrink": self.shrink}
                    for c in range(num_classes):
                        n_c = float(n[c].item())
                        if n_c > 0.0:
                            v_max = float(v[c].max().item())
                            var_aniso = (
                                v_max - float(v[c].min().item())
                            ) / (v_max + 1e-8)
                        else:
                            var_aniso = 0.0
                        proto_stats["class_{}".format(c)] = {
                            "n": n_c,
                            "var_aniso": var_aniso,
                        }
                    write_record(
                        batch_idx=i,
                        target=int(target.item()),
                        pred=int(final_logits.argmax(dim=-1).item()),
                        correct=bool(acc),
                        conf=float(F.softmax(final_logits, dim=-1).max().item()),
                        quality_gate=None,
                        gate_mode=None,
                        proto_alpha=None,
                        logits={
                            "clip": clip_logits.squeeze(0).float().cpu().tolist(),
                            "image_proto": logit.float().cpu().tolist(),
                            "patch_proto": None,
                            "final": final_logits.squeeze(0).float().cpu().tolist(),
                        },
                        proto_stats=proto_stats,
                        write_gate=None,
                        write_occurred=bool((n > n_before).any().item()),
                    )

                # Periodic logging (every 1000 samples)
                if i % 1000 == 0:
                    print(
                        f"---- GaussPTA(shrink={self.shrink}) "
                        f"test accuracy: {sum(accuracies)/len(accuracies):.2f}. ----"
                    )

        # ── FINAL RESULTS ──────────────────────────────────────────────────
        final_acc = sum(accuracies) / len(accuracies)
        print(
            f"---- GaussPTA(shrink={self.shrink}) test accuracy: "
            f"{final_acc:.2f}. ----\n"
        )

        # ── SUMMARY (env-gated, behavior-neutral) ──────────────────────────
        if record_dir:
            per_class = {}
            for c in range(num_classes):
                total_c = cls_total[c]
                correct_c = cls_correct[c]
                per_class["class_{}".format(c)] = {
                    "total": total_c,
                    "correct": correct_c,
                    "acc": (100.0 * correct_c / total_c) if total_c else 0.0,
                }
            write_summary(
                method=os.environ.get("RESULT_LABEL", "GaussPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
            write_convergence(accuracies)

        # Append results to output file (append mode, multiple runs accumulate)
        label = os.environ.get("RESULT_LABEL", "GaussPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def build(cfg: dict) -> GaussPTAAdapter:
    """Factory: instantiate a GaussPTAAdapter with the given config."""
    return GaussPTAAdapter(cfg)
