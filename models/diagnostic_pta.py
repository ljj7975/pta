"""
Diagnostic Adapter: Patch-Only / Image-Only Signal Isolation.

Isolates patch-level or image-level prototype scores as the sole classifier
to measure whether each signal carries useful class-discriminative information.

Modes (configurable via fusion.mode):
  - "patch_only": final logit = raw_proto (Gaussian patch prototype scores)
  - "image_only": final logit = image_proto_logits (image-level prototype dot product)
"""
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.base import BaseAdapter
from models.image_level import create as create_image_level
from models.patch_level import create as create_patch_level
from models.patch_level.base import _alpha_from_evidence
from utils.clip_inference import _safe_normalize
from utils import cls_acc, get_clip_logits


class DiagnosticPTAAdapter(BaseAdapter):

    def __init__(self, cfg):
        super().__init__(cfg)

        image_level_cfg = cfg.get("image_level", {})
        if "alpha" not in image_level_cfg and "alpha" in cfg:
            image_level_cfg["alpha"] = cfg["alpha"]
        if "T" not in image_level_cfg and "T" in cfg:
            image_level_cfg["T"] = cfg["T"]
        cfg["image_level"] = image_level_cfg

        self.image_level = create_image_level(cfg)
        self.patch_level = create_patch_level(cfg)

        fusion_cfg = cfg.get("fusion", {})
        self.tau_text = float(fusion_cfg.get("tau_text", 1.0))
        self.tau_image_proto = float(fusion_cfg.get("tau_image_proto", 80.0))
        self.fusion_mode = str(fusion_cfg.get("mode", "patch_only"))

    def run(self, loader, encoder, text_embeddings, dataset_name: str) -> float:
        if not hasattr(encoder.visual, "positional_embedding") and not hasattr(encoder.visual, "pos_embed"):
            raise ValueError(
                "DiagnosticPTA requires ViT backbone for patch extraction."
            )

        _il_cfg = self.cfg.get("image_level", {})
        alpha_pta = float(_il_cfg.get("alpha", self.cfg.get("alpha", 0.01)))
        T = float(_il_cfg.get("T", self.cfg.get("T", 20.0)))

        _pl_cfg = self.cfg.get("patch_level", {})
        conf_thresh = float(_pl_cfg.get("conf_threshold", 0.5))
        conf_margin_thresh = float(_pl_cfg.get("conf_margin_threshold", 0.05))
        n_half = float(self.cfg.get("n_half", 15.0))
        alpha_max = float(self.cfg.get("proto_alpha_max", 0.2))
        conf_source = str(self.cfg.get("conf_source", "text"))
        multi_gate = bool(self.cfg.get("multi_gate", False))

        os.makedirs("outputs", exist_ok=True)

        text_proto = _safe_normalize(text_embeddings.t().float())
        C, D = text_proto.shape
        device = text_proto.device

        refine_feature = text_embeddings.t().float()
        target_prototype = self.image_level.init_state(refine_feature)
        states = self.patch_level.init_state(refine_feature)

        _filter_mode = self.cfg.get("patch_level", {}).get("patch_filter_mode", "none")
        if _filter_mode != "none":
            self.patch_level.set_text_context(text_embeddings, encoder, device)

        max_batches = int(os.environ.get("MAX_BATCHES", "0"))
        accuracies = []
        qg_history = []

        with torch.no_grad():
            for i, (images, target) in enumerate(
                tqdm(loader, desc=f"[DiagnosticPTA] {dataset_name}")
            ):
                if max_batches > 0 and i >= max_batches:
                    break
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                image_features, clip_logits, _, _, _ = get_clip_logits(
                    images, encoder, text_embeddings
                )
                feat = image_features.squeeze(0).float()
                feat_norm = _safe_normalize(feat)

                patch_proto_logits, quality_gate = (
                    self.patch_level.compute_patch_logits(images, encoder, states)
                )
                qg_history.append(float(quality_gate))

                proto_alpha = torch.tensor(
                    [
                        min(alpha_max, _alpha_from_evidence(
                            states[c]["n_images"], n_half
                        ))
                        for c in range(C)
                    ],
                    device=device,
                )

                soft_logits = F.softmax(clip_logits, dim=-1)
                refine_feature, target_prototype = _update_text_features(
                    image_features, soft_logits.half(),
                    refine_feature, target_prototype,
                    alpha=alpha_pta, T=T,
                )

                image_proto_logits = (
                    image_features.half() @ refine_feature.half().T
                )

                if self.fusion_mode == "patch_only":
                    final_logits = patch_proto_logits.unsqueeze(0)
                elif self.fusion_mode == "image_only":
                    final_logits = image_proto_logits
                else:
                    raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                if conf_source == "text":
                    gate_logits = clip_logits
                elif conf_source == "image":
                    gate_logits = image_proto_logits
                else:
                    gate_logits = clip_logits

                pred_conf = F.softmax(gate_logits, dim=-1).squeeze(0)

                if multi_gate:
                    above_thresh = (pred_conf > conf_thresh).nonzero(as_tuple=True)[0]
                    for cls_idx in above_thresh:
                        cls = int(cls_idx.item())
                        states[cls] = self.patch_level.update_state(
                            states[cls], images, encoder, feat_norm,
                            target_class_idx=cls,
                        )
                else:
                    top2_vals, top2_idx = pred_conf.topk(min(2, C))
                    best_conf = float(top2_vals[0].item())
                    second_conf = float(top2_vals[1].item()) if C > 1 else 0.0
                    conf_margin = best_conf - second_conf
                    best_cls = int(top2_idx[0].item())

                    if best_conf > conf_thresh and conf_margin >= conf_margin_thresh:
                        states[best_cls] = self.patch_level.update_state(
                            states[best_cls], images, encoder, feat_norm,
                            target_class_idx=best_cls,
                        )

                if i % 500 == 0:
                    running = sum(accuracies) / len(accuracies)
                    avg_qg = sum(qg_history) / len(qg_history)
                    print(f"---- DiagnosticPTA {running:.2f}% (avg_qg={avg_qg:.4f}) ----")

        final_acc = sum(accuracies) / len(accuracies)
        avg_qg = sum(qg_history) / len(qg_history)
        print(f"---- DiagnosticPTA FINAL {final_acc:.2f}% (avg_qg={avg_qg:.4f}) ----")

        label = os.environ.get("RESULT_LABEL", "DiagnosticPTA")
        result_file = os.environ.get("RESULT_FILE", "outputs/result.txt")
        with open(result_file, "a") as f:
            f.write(
                f"{label}'s performance on {dataset_name}: "
                f"Top1- {final_acc:.2f}.\n"
            )

        return final_acc


def _update_text_features(
    image_feature, probs, text_features, target_prototype,
    alpha=0.01, T=20.0,
):
    w = probs.squeeze(0)
    w_new = torch.zeros_like(w)
    mask = w >= 1e-1
    w_new[mask] = 1 - torch.exp(-w[mask] / T)
    w_new = w_new.unsqueeze(1)
    target_prototype[mask] = (
        (1 - w_new[mask]) * target_prototype[mask]
        + w_new[mask] * image_feature.squeeze(0)
    )
    refined_text = alpha * text_features + (1 - alpha) * target_prototype
    refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)
    return refined_text, target_prototype


def build(cfg: dict) -> DiagnosticPTAAdapter:
    return DiagnosticPTAAdapter(cfg)
