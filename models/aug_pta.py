"""AugPTA — TPT-style input augmentation ensembling for base PTA.

Mechanism
---------
For every test sample the adapter builds ``V`` views — the original image plus
``V-1`` augmented copies (utils.augmentation._augment_image: rotation,
translation+scale affine, brightness, contrast, blur, grayscale/edge extra).
All views are encoded by the *frozen* CLIP image encoder and produce per-view
zero-shot logits ``100 * f_v^T t``. The ``K_views`` views with the LOWEST
softmax entropy (TPT's "most confident" selection, Shu et al. NeurIPS 2022)
are kept, and their logits are AVERAGED to form the ensemble evidence.

Two uses (and only two) — nothing else changes vs base PTA:

1. **Pseudo-label source for the prototype update.** The ensemble logits
   replace the single-view clip logits as the input to PTA's softmax write
   rule. The write *content* is still the ORIGINAL (real, unaugmented) image
   feature, and the write rule is PTA's exactly — multi-class softmax-EMA,
   no skips, no gates, no reweighting. The only lever is label quality:
   the ~50%-wrong single-view pseudo-label stream of base PTA is replaced by
   a K-view-majority signal that is more reliable (TPT's evidence), so wrong
   writes decrease WITHOUT starving correct ones (the failure mode that sank
   every write-time gate tested in this study).

2. **Zero-shot term of the fusion.** The ensemble logits replace the
   single-view clip logits in ``final = tau_text*clip + tau_image_proto*proto``.

The adaptation itself is exactly PTA's EMA prototype (PTAImageLevel) — the
updates are ungated and every sample still writes, exactly as in base PTA.

Why this is not a repetition of the tested "diagnostic not lever" failure:
every previously-tested lever (compactness distance, margin, patch-vote
agreement, soft reweight) judged the *current write* with statistics derived
from CLIP's own biased outputs and resolved by REMOVING/attenuating writes.
AugPTA never removes or attenuates anything: it changes the *label* the
write uses. The signal (majority of V independently-augmented frozen-encoder
views) is not an estimate from the biased stream. With ``V=1`` the adapter
reduces to base PTA EXACTLY (no augmentation, no selection, identity
ensemble): a CPU unit test asserts this, and the smoke stage re-checks it
end-to-end (same accuracy as PTA on the same seed).

Config keys (root level; selected per-run via ``--override``):
    V         — number of views (original + V-1 augs). Default 8.
    K_views   — number of lowest-entropy views to keep & average. Default 4.
                1 <= K_views <= V; when V == 1 selection is identity.

Records: same schema as base PTA, plus per-class write counters in
``proto_stats.class_<c>.n_updates`` (write-purity analytics) and
``write_gate`` = "aug-V..K..".
"""

import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.pta import PTAAdapter
from utils import cls_acc
from utils.augmentation import _augment_image
from utils.records import (
    write_convergence,
    write_record,
    write_record_header,
    write_summary,
)


def _encode_device() -> str:
    """CUDA when available (production), CPU fallback (CPU unit tests).

    Image tensors from the (batch=1) DataLoader arrive on CPU; base PTA moves
    them with ``.cuda()`` unconditionally. ``_encode_device()`` keeps that
    behaviour on any GPU node while letting the CPU-only unit tests exercise
    the exact same code path.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


class AugPTAAdapter(PTAAdapter):
    """PTA + TPT-style V-view entropy-selected logit ensembling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.V = int(cfg.get("V", cfg.get("aug_views", 8)))
        self.V = max(1, self.V)
        self.K_views = int(cfg.get("K_views", cfg.get("select_views", 4)))
        self.K_views = max(1, min(self.K_views, self.V))

    # ------------------------------------------------------------------
    # View ensemble (the only difference vs base PTA)
    # ------------------------------------------------------------------
    def _ensemble_evidence(self, images, encoder, text_embeddings):
        """Encode V views; return (orig_feature, ensemble_logits).

        Args:
            images:          ``[1, C, H, W]`` CLIP-preprocessed test image.
            encoder:         CLIP encoder wrapper (encode_image).
            text_embeddings: ``(D, C)`` class text matrix (fp16 CUDA — the
                             dtype CLIP actually emits; unit stubs use fp32).

        Returns:
            orig_feature:    ``[1, D]`` fp32 embedding of the ORIGINAL image
                             (write content + image_proto query — identical
                             role to base PTA's image_features).
            ens_logits:      ``[1, C]`` fp32 mean of the K_views lowest-entropy
                             view logits (pseudo-label + fusion clip term).
        """
        if self.V == 1:
            feats = encoder.encode_image(images.to(_encode_device()))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = (100.0 * feats @ text_embeddings).float()
            return feats, logits

        views = torch.cat(
            [images] + [_augment_image(images) for _ in range(self.V - 1)],
            dim=0,
        )
        feats = encoder.encode_image(views.to(_encode_device()))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        # Cast BOTH operands to fp32: real CLIP (encode_image / encode_text)
        # emits fp16 features, and fp32 @ fp16 raises RuntimeError in torch.
        # Base PTA's fp16 matmul is reproduced identically per-view (same
        # values), with fp32 accumulation only for the entropy ranking below.
        logits = (100.0 * feats.float() @ text_embeddings.float()).float()    # [V, C]

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)  # [V]
        sel = torch.argsort(entropy, descending=False)[: self.K_views]     # [K]

        ens_logits = logits[sel].mean(0).unsqueeze(0)                     # [1, C]
        return feats[0:1], ens_logits

    # ------------------------------------------------------------------
    # Run (base PTA loop with the ensemble evidence injected)
    # ------------------------------------------------------------------
    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        os.makedirs("outputs", exist_ok=True)

        record_dir = os.environ.get("RECORD_DIR", "").strip()
        if record_dir:
            write_record_header(
                method=os.environ.get("RESULT_LABEL", "AugPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                C=int(text_embeddings.shape[1]),
                classnames=[],
                resolved_config=self.cfg,
            )

        with torch.no_grad():
            accuracies = []

            refine_feature = text_embeddings.t()                 # [C, D] fp16
            target_prototype = self.image_level.init_state(refine_feature)

            num_classes = text_embeddings.shape[1]
            cls_total = [0] * num_classes
            cls_correct = [0] * num_classes
            n_updates = [0] * num_classes

            max_batches = os.environ.get("MAX_BATCHES")
            if max_batches is not None:
                max_batches = int(max_batches)

            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[AugPTA] {dataset_name}")
            ):
                if max_batches is not None and i >= max_batches:
                    break

                # ── ENSEMBLE EVIDENCE (label + fusion clip term) ─────────
                image_features, ens_logits = self._ensemble_evidence(
                    images, encoder, text_embeddings
                )
                target = target.cuda()

                # ── ONLINE PROTOTYPE UPDATE (base PTA rule, better label) ──
                refine_feature, target_prototype = (
                    self.image_level.update_prototypes(
                        image_features, ens_logits, refine_feature,
                        target_prototype,
                    )
                )

                # Per-class write counters (softmax(w) >= 0.1 = PTA's write mask)
                w = F.softmax(ens_logits, dim=-1).squeeze(0)
                for c in range(num_classes):
                    if float(w[c].item()) >= 1e-1:
                        n_updates[c] += 1

                # ── FUSED PREDICTION (unchanged math) ─────────────────────
                image_proto_logits = self.image_level.compute_logits(
                    image_features, refine_feature
                )
                final_logits = self.fusion.forward(
                    ens_logits.clone(), image_proto_logits, None
                )

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                cls_total[int(target.item())] += 1
                if acc:
                    cls_correct[int(target.item())] += 1

                if record_dir:
                    proto_stats = {}
                    proto_stats["V"] = self.V
                    proto_stats["K_views"] = self.K_views
                    for c in range(num_classes):
                        proto_stats["class_{}".format(c)] = {
                            "n_updates": int(n_updates[c]),
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
                            "clip": ens_logits.squeeze(0).float().cpu().tolist(),
                            "image_proto": image_proto_logits.squeeze(0).float().cpu().tolist(),
                            "patch_proto": None,
                            "final": final_logits.squeeze(0).float().cpu().tolist(),
                        },
                        proto_stats=proto_stats,
                        write_gate=(
                            f"aug-V{self.V}K{self.K_views}"
                            if self.V > 1 else None
                        ),
                        write_occurred=bool(int((w >= 1e-1).sum().item()) > 0),
                    )

                if i % 1000 == 0:
                    print(
                        f"---- AugPTA accuracy: "
                        f"{sum(accuracies) / len(accuracies):.2f}. ----"
                    )

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- AugPTA's test accuracy: {final_acc:.2f}. ----\n")

        if record_dir:
            per_class = {}
            for c in range(num_classes):
                per_class["class_{}".format(c)] = {
                    "total": cls_total[c],
                    "correct": cls_correct[c],
                    "acc": (100.0 * cls_correct[c] / cls_total[c]) if cls_total[c] else 0.0,
                }
            write_summary(
                method=os.environ.get("RESULT_LABEL", "AugPTA"),
                dataset=dataset_name,
                seed=int(os.environ.get("SEED", 1)),
                total=len(accuracies),
                acc=final_acc,
                per_class=per_class,
            )
        write_convergence(accuracies)

        label = os.environ.get("RESULT_LABEL", "AugPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(f"{label}'s performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> AugPTAAdapter:
    """Factory: instantiate AugPTAAdapter (module-level, required by runner)."""
    return AugPTAAdapter(cfg)