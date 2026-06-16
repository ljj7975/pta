"""
Headless prototype visualization engine for MultiProtoPTA.

Designed to run with or without a display. In headless mode it serializes
snapshots to disk (JSON + PNG).  The GUI app (proto_viz_app.py) loads these
snapshots for interactive browsing, or calls the engine live with a real
clip_model + loader.

Public API
----------
ProtoVizEngine(cfg, classnames, clip_weights, clip_model)
    .run_sample(image_tensor, target_label)
        -> SampleRecord  (all data needed for one frame of the GUI)
    .get_bank_snapshot()
        -> BankSnapshot  (current state of every prototype bank)
    .save_snapshot(path)   / .load_snapshot(path)

Data structures (all plain Python + numpy, picklable)
------------------------------------------------------
ProtoInfo:
    class_id        int
    class_name      str
    proto_idx       int          index within class bank
    center          np.ndarray   [D] normalized embedding
    appearance      float        # images this proto appeared in
    evidence_weight float        0.5 + 0.5*(K-1)/4, clamped to [0,1]

MatchInfo:
    proto           ProtoInfo
    max_patch_sim   float        highest cosine sim to any patch in image
    raw_score       float        evidence_weight * max_patch_sim  (before centering)
    centered_score  float        raw_score - mean(raw_scores across classes)
    final_contrib   float        tau_proto * centered_score  (added to logit)
    top_patches     list[int]    indices of top-3 matching patches (by sim to this proto)

ClassRecord:
    class_id        int
    class_name      str
    text_score      float        100.0 * cos(text_proto, global_feat)
    raw_proto_score float        evidence_weighted bank score (before centering)
    centered_proto  float        after mean-subtraction
    final_logit     float        text_score + tau_proto * centered_proto
    softmax_prob    float        softmax(final_logits)[c]
    bank_size       int          K prototypes stored

SampleRecord:
    sample_idx      int
    target          int
    target_name     str
    predicted       int
    predicted_name  str
    correct         bool
    clip_pred       int          CLIP zero-shot prediction
    clip_conf       float        max softmax of CLIP logits
    global_feat     np.ndarray   [D]
    patch_embs      np.ndarray   [P, D]
    classes         list[ClassRecord]
    top_matches     list[MatchInfo]  top-10 proto matches across ALL classes
    running_acc     float

BankSnapshot:
    sample_idx      int
    class_protos    list[list[ProtoInfo]]  [C][K]
"""

from __future__ import annotations

import base64
import json
import math
import os
import pickle
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import sys
_PTA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PTA_ROOT not in sys.path:
    sys.path.insert(0, _PTA_ROOT)

from models.multi_proto_pta_base import (
    _safe_normalize,
    _extract_patch_embeddings,
    _incremental_kmeans_step,
)
from models.multi_proto_pta import _proto_score_for_class
from utils import get_clip_logits, cls_acc


@dataclass
class ProtoInfo:
    class_id: int
    class_name: str
    proto_idx: int
    center: np.ndarray
    appearance: float
    evidence_weight: float
    update_samples: int = 0
    rep_patch_data: Optional[str] = None
    rep_patch_idx: Optional[int] = None
    rep_patch_sim: Optional[float] = None


@dataclass
class MatchInfo:
    proto: ProtoInfo
    max_patch_sim: float
    raw_score: float
    centered_score: float
    final_contrib: float
    top_patch_indices: List[int]


@dataclass
class ClassRecord:
    class_id: int
    class_name: str
    text_score: float
    raw_proto_score: float
    delta_proto: float
    alpha: float
    quality_gate: float
    tau_proto: float
    centered_proto: float
    proto_term: float
    tau_eff: float
    class_penalty: float
    final_logit: float
    softmax_prob: float
    bank_size: int


@dataclass
class SampleRecord:
    sample_idx: int
    target: int
    target_name: str
    predicted: int
    predicted_name: str
    correct: bool
    clip_pred: int
    clip_conf: float
    global_feat: np.ndarray
    patch_embs: np.ndarray
    classes: List[ClassRecord]
    top_matches: List[MatchInfo]
    running_acc: float
    text_running_acc: float = 0.0
    quality_gate: float = 0.0
    tau_proto: float = 0.0
    all_matches: List[MatchInfo] = field(default_factory=list)
    image_data: Optional[str] = None  # base64 encoded PNG of the original image
    update_gate_passed: bool = False
    update_gate_class_id: Optional[int] = None
    update_gate_class_name: Optional[str] = None
    update_gate_best_conf: float = 0.0
    update_gate_second_conf: float = 0.0
    update_gate_margin: float = 0.0
    update_gate_conf_thresh: float = 0.0
    update_gate_margin_thresh: float = 0.0
    update_gate_reason: str = ""
    update_gate_bank_k: int = 0
    update_gate_max_k: int = 0
    foreground_appw_thresh: float = 0.5


@dataclass
class BankSnapshot:
    sample_idx: int
    class_protos: List[List[ProtoInfo]]


class ProtoVizEngine:
    """
    Drop-in replacement for MultiProtoPTABase.run() that records every
    intermediate quantity for visualization.

    Usage (live mode):
        engine = ProtoVizEngine(cfg, classnames, clip_weights, clip_model)
        for images, targets in loader:
            record = engine.run_sample(images, targets)
            # record has everything needed to render one GUI frame

    Usage (headless / batch mode):
        engine = ProtoVizEngine(cfg, classnames, clip_weights, clip_model)
        records = []
        for i, (images, targets) in enumerate(loader):
            records.append(engine.run_sample(images, targets))
            if i % 100 == 0:
                engine.save_snapshot(f"outputs/viz/snap_{i:05d}.pkl")
        engine.save_records(records, "outputs/viz/records.pkl")
    """

    def __init__(self, cfg: dict, classnames: List[str], clip_weights: torch.Tensor, clip_model):
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self._clip_weights_tensor = clip_weights

        # Keep defaults consistent with models/multi_proto_pta.py.
        self.max_K        = int(cfg.get("max_K", 100))
        self.match_thresh = float(cfg.get("match_threshold", 0.60))
        self.conf_thresh  = float(cfg.get("conf_threshold", 0.5))
        self.conf_margin_thresh = float(cfg.get("conf_margin_threshold", 0.05))
        self.n_half       = float(cfg.get("n_half", 15.0))
        self.alpha_max    = float(cfg.get("alpha_max", 0.2))
        self.top_m        = int(cfg.get("soft_nn_top_m", 4))
        self.patch_group_threshold = float(cfg.get("patch_group_threshold", 0.9))
        self.exclude_pos  = bool(cfg.get("exclude_pos", False))
        self.tau_proto    = float(cfg.get("tau_proto", cfg.get("T", 20.0)))
        self.quality_eps  = float(cfg.get("quality_eps", 1e-3))
        self.foreground_appw_thresh = float(cfg.get("foreground_appw_threshold", 0.5))

        text_proto = _safe_normalize(clip_weights.t().float())
        self.text_proto = text_proto
        self.C = text_proto.shape[0]
        self.device = text_proto.device

        D = text_proto.shape[1]
        self.states: List[Dict] = [
            {
                "centers":       torch.empty(0, D, device=self.device),
                "appearance":    torch.empty(0,    device=self.device),
                "n_images":      0,
                "rep_patch_data": [],
                "rep_patch_idx":  [],
                "rep_patch_sim":  [],
            }
            for c in range(self.C)
        ]

        self._sample_idx = 0
        self._accuracies: List[float] = []
        self._text_accuracies: List[float] = []

    def run_sample(self, images: torch.Tensor, target: torch.Tensor) -> SampleRecord:
        if isinstance(images, list):
            images = torch.cat(images, dim=0)

        images = images.to(self.device)
        target = target.to(self.device)

        image_features, clip_logits, _, _, _ = get_clip_logits(
            images, self.clip_model, self._clip_weights_tensor
        )
        feat = image_features.squeeze(0).float()
        feat_norm = _safe_normalize(feat)

        patch_embs = _extract_patch_embeddings(images, self.clip_model, exclude_pos=self.exclude_pos)
        patches_norm = _safe_normalize(patch_embs, dim=-1)

        clip_probs = F.softmax(clip_logits, dim=-1).squeeze(0)
        clip_pred = int(clip_probs.argmax().item())
        clip_conf = float(clip_probs.max().item())

        text_logits = 100.0 * (self.text_proto @ feat_norm)

        raw_proto_scores = torch.zeros(self.C, device=self.device)
        class_alpha = torch.zeros(self.C, device=self.device)
        per_proto_max_sims = []
        per_proto_app_weights = []
        per_proto_assigned_patch_idx = []
        per_proto_weighted_scores = []

        for c in range(self.C):
            centers = self.states[c]["centers"]
            appearance = self.states[c]["appearance"]
            K_c = centers.shape[0]

            if K_c == 0:
                raw_proto_scores[c] = 0.0
                class_alpha[c] = 0.0
                per_proto_max_sims.append(torch.empty(0))
                per_proto_app_weights.append(torch.empty(0))
                per_proto_assigned_patch_idx.append(torch.empty(0, dtype=torch.long))
                per_proto_weighted_scores.append(torch.empty(0))
                continue

            centers_norm = _safe_normalize(centers, dim=-1)
            raw_bank, max_sims, app_weights, assigned_patch_idx, weighted, _ = _proto_score_for_class(
                patches_norm,
                centers_norm,
                appearance,
                update_samples=int(self.states[c].get("n_images", 0)),
                top_m=self.top_m,
                patch_group_threshold=self.patch_group_threshold,
                return_details=True,
            )

            raw_proto_scores[c] = raw_bank
            class_alpha[c] = min(
                self.alpha_max,
                self._alpha_from_evidence(self.states[c]["n_images"], self.n_half),
            )
            per_proto_max_sims.append(max_sims.cpu())
            per_proto_app_weights.append(app_weights.cpu())
            per_proto_assigned_patch_idx.append(assigned_patch_idx.cpu())
            per_proto_weighted_scores.append(weighted.cpu())

        proto_var = raw_proto_scores.var()
        quality_gate = proto_var / (proto_var + self.quality_eps)

        proto_term = self.tau_proto * class_alpha * quality_gate * raw_proto_scores
        final_logits = (text_logits + proto_term).unsqueeze(0)
        text_scores = []
        for c in range(self.C):
            text_scores.append(float(text_logits[c].item()))

        softmax_probs = F.softmax(final_logits, dim=-1).squeeze(0)
        pred = int(final_logits.argmax(dim=1).item())
        target_val = int(target.item())
        correct = pred == target_val

        acc = cls_acc(final_logits, target)
        self._accuracies.append(acc)
        running_acc = float(np.mean(self._accuracies))

        text_logits_only = text_logits.unsqueeze(0)
        text_acc = cls_acc(text_logits_only, target)
        self._text_accuracies.append(text_acc)
        text_running_acc = float(np.mean(self._text_accuracies))

        class_records = []
        for c in range(self.C):
            class_records.append(ClassRecord(
                class_id=c,
                class_name=self.classnames[c],
                text_score=text_scores[c],
                raw_proto_score=float(raw_proto_scores[c].item()),
                delta_proto=float(raw_proto_scores[c].item()),
                alpha=float(class_alpha[c].item()),
                quality_gate=float(quality_gate.item()),
                tau_proto=float(self.tau_proto),
                centered_proto=float((class_alpha[c] * quality_gate * raw_proto_scores[c]).item()),
                proto_term=float(proto_term[c].item()),
                tau_eff=float((self.tau_proto * quality_gate).item()),
                class_penalty=float(class_alpha[c].item()),
                final_logit=float(final_logits[0, c].item()),
                softmax_prob=float(softmax_probs[c].item()),
                bank_size=self.states[c]["centers"].shape[0],
            ))

        all_matches: List[MatchInfo] = []
        for c in range(self.C):
            centers = self.states[c]["centers"]
            appearance = self.states[c]["appearance"]
            max_sims_per_proto = per_proto_max_sims[c]
            app_weights = per_proto_app_weights[c]
            assigned_patch_idx = per_proto_assigned_patch_idx[c]
            weighted_scores = per_proto_weighted_scores[c]
            ew = float(class_alpha[c].item())

            if centers.shape[0] == 0:
                continue

            for k in range(centers.shape[0]):
                assigned_idx = int(assigned_patch_idx[k].item()) if k < assigned_patch_idx.numel() else -1
                top_patch_idx = [assigned_idx] if assigned_idx >= 0 else []
                raw_s = float(weighted_scores[k].item()) if k < weighted_scores.numel() else 0.0
                ctr_s = float((ew * quality_gate * raw_s).item())

                pi = ProtoInfo(
                    class_id=c,
                    class_name=self.classnames[c],
                    proto_idx=k,
                    center=centers[k].cpu().numpy(),
                    appearance=float(appearance[k].item()),
                    evidence_weight=ew,
                    update_samples=int(self.states[c].get("n_images", 0)),
                    rep_patch_data=self.states[c]["rep_patch_data"][k] if k < len(self.states[c]["rep_patch_data"]) else None,
                    rep_patch_idx=self.states[c]["rep_patch_idx"][k] if k < len(self.states[c]["rep_patch_idx"]) else None,
                    rep_patch_sim=self.states[c]["rep_patch_sim"][k] if k < len(self.states[c]["rep_patch_sim"]) else None,
                )
                all_matches.append(MatchInfo(
                    proto=pi,
                    max_patch_sim=float(max_sims_per_proto[k].item()),
                    raw_score=raw_s,
                    centered_score=ctr_s,
                    final_contrib=float((self.tau_proto * ctr_s)),
                    top_patch_indices=top_patch_idx,
                ))

        all_matches.sort(key=lambda m: m.max_patch_sim, reverse=True)
        top_matches = all_matches[:20]

        pred_conf = F.softmax(clip_logits, dim=-1).squeeze(0)
        top2_vals, top2_idx = pred_conf.topk(min(2, self.C))

        best_conf = float(top2_vals[0].item())
        second_conf = float(top2_vals[1].item()) if self.C > 1 else 0.0
        conf_margin = best_conf - second_conf
        best_cls = int(top2_idx[0].item())
        gate_passed = False
        gate_reason = ""
        bank_k = int(self.states[best_cls]["centers"].shape[0])

        conf_ok = best_conf > self.conf_thresh
        margin_ok = conf_margin >= self.conf_margin_thresh

        if conf_ok and margin_ok:
            self.states[best_cls] = self._update_state(
                best_cls,
                patches_norm,
                feat_norm,
                images.squeeze(0),
            )
            gate_passed = True
            gate_reason = "Update executed: confidence/margin passed; bank is pruned by app_w if needed."
        elif (not conf_ok) and (not margin_ok):
            gate_reason = (
                "Skipped: confidence and margin are both below thresholds "
                f"(need top1>{100.0 * self.conf_thresh:.1f}% and margin>={100.0 * self.conf_margin_thresh:.1f}%)."
            )
        elif not conf_ok:
            gate_reason = (
                "Skipped: confidence is below threshold "
                f"(top1={100.0 * best_conf:.1f}%, need>{100.0 * self.conf_thresh:.1f}%)."
            )
        else:
            gate_reason = (
                "Skipped: top1-top2 margin is below threshold "
                f"(margin={100.0 * conf_margin:.1f}%, need>={100.0 * self.conf_margin_thresh:.1f}%)."
            )

        record = SampleRecord(
            sample_idx=self._sample_idx,
            target=target_val,
            target_name=self.classnames[target_val],
            predicted=pred,
            predicted_name=self.classnames[pred],
            correct=correct,
            clip_pred=clip_pred,
            clip_conf=clip_conf,
            global_feat=feat.cpu().numpy(),
            patch_embs=patch_embs.cpu().numpy(),
            classes=class_records,
            top_matches=top_matches,
            all_matches=all_matches,
            running_acc=running_acc,
            text_running_acc=text_running_acc,
            quality_gate=float(quality_gate.item()),
            tau_proto=float(self.tau_proto),
            image_data=self._tensor_to_base64_png(images.squeeze(0)) if images.shape[0] == 1 else None,
            update_gate_passed=gate_passed,
            update_gate_class_id=best_cls,
            update_gate_class_name=self.classnames[best_cls],
            update_gate_best_conf=best_conf,
            update_gate_second_conf=second_conf,
            update_gate_margin=conf_margin,
            update_gate_conf_thresh=self.conf_thresh,
            update_gate_margin_thresh=self.conf_margin_thresh,
            update_gate_reason=gate_reason,
            update_gate_bank_k=bank_k,
            update_gate_max_k=self.max_K,
            foreground_appw_thresh=self.foreground_appw_thresh,
        )
        self._sample_idx += 1
        return record

    def _update_state_impl(
        self,
        class_idx: int,
        patches_norm: torch.Tensor,
        global_feat_norm: Optional[torch.Tensor],
        image_tensor: torch.Tensor,
    ) -> dict:
        state = self.states[class_idx]
        centers = state["centers"]
        apps = state["appearance"]
        n_images_next = int(state.get("n_images", 0) + 1)
        grow_cap = max(self.max_K + int(patches_norm.shape[0]), self.max_K)
        rep_patch_data = list(state.get("rep_patch_data", [None] * centers.shape[0]))
        rep_patch_idx = list(state.get("rep_patch_idx", [-1] * centers.shape[0]))
        rep_patch_sim = list(state.get("rep_patch_sim", [-1.0] * centers.shape[0]))

        if centers.shape[0] == 0:
            if global_feat_norm is None:
                raise ValueError("global_feat_norm is required when initializing an empty class bank")
            init = global_feat_norm.unsqueeze(0)
            updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
                init, patches_norm, self.match_thresh, grow_cap
            )

            # First center is seeded from global feature; it starts with one appearance.
            n_new = updated_centers.shape[0] - 1
            updated_apps = torch.ones(1 + max(n_new, 0), device=self.device)

            # Seed metadata for the global-feature prototype.
            if len(rep_patch_data) == 0:
                rep_patch_data.append(None)
                rep_patch_idx.append(-1)
                rep_patch_sim.append(-1.0)

            old_K = 1
        else:
            old_K = centers.shape[0]
            updated_centers, appeared, matched, best_clusters, new_groups = _incremental_kmeans_step(
                centers, patches_norm, self.match_thresh, grow_cap
            )

            updated_apps = apps.clone()
            updated_apps[appeared] += 1
            n_new = updated_centers.shape[0] - old_K
            if n_new > 0:
                updated_apps = torch.cat([updated_apps, torch.ones(n_new, device=self.device)], dim=0)

        # Update representative patch for existing prototypes if a better match is seen.
        sims_updated = patches_norm @ _safe_normalize(updated_centers, dim=-1).t()
        for k in range(old_K):
            mask = matched & (best_clusters == k)
            if not mask.any():
                continue
            candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            candidate_sims = sims_updated[candidate_indices, k]
            best_local = int(candidate_indices[int(candidate_sims.argmax().item())].item())
            best_local_sim = float(sims_updated[best_local, k].item())
            if best_local_sim > rep_patch_sim[k]:
                rep_patch_data[k] = self._extract_patch_crop_base64(image_tensor, best_local, patches_norm.shape[0])
                rep_patch_idx[k] = best_local
                rep_patch_sim[k] = best_local_sim

        # Initialize representative patches for newly created prototypes.
        for i, member_idx in enumerate(new_groups):
            new_k = old_K + i
            if new_k >= updated_centers.shape[0] or member_idx.numel() == 0:
                rep_patch_data.append(None)
                rep_patch_idx.append(-1)
                rep_patch_sim.append(-1.0)
                continue
            sims_new = sims_updated[member_idx, new_k]
            best_m = int(member_idx[int(sims_new.argmax().item())].item())
            rep_patch_data.append(self._extract_patch_crop_base64(image_tensor, best_m, patches_norm.shape[0]))
            rep_patch_idx.append(best_m)
            rep_patch_sim.append(float(sims_updated[best_m, new_k].item()))

        # Ensure metadata lists match center count in edge cases.
        while len(rep_patch_data) < updated_centers.shape[0]:
            rep_patch_data.append(None)
            rep_patch_idx.append(-1)
            rep_patch_sim.append(-1.0)

        if updated_centers.shape[0] > self.max_K:
            app_w = updated_apps / float(max(n_images_next, 1))
            keep = torch.argsort(app_w, descending=True)[:self.max_K]
            keep = torch.sort(keep).values
            keep_list = [int(k) for k in keep.tolist()]

            updated_centers = updated_centers[keep]
            updated_apps = updated_apps[keep]
            rep_patch_data = [rep_patch_data[k] for k in keep_list]
            rep_patch_idx = [rep_patch_idx[k] for k in keep_list]
            rep_patch_sim = [rep_patch_sim[k] for k in keep_list]

        return {
            "centers": updated_centers,
            "appearance": updated_apps,
            "n_images": n_images_next,
            "rep_patch_data": rep_patch_data[:updated_centers.shape[0]],
            "rep_patch_idx": rep_patch_idx[:updated_centers.shape[0]],
            "rep_patch_sim": rep_patch_sim[:updated_centers.shape[0]],
        }

    def _update_state(
        self,
        class_idx: int,
        patches_norm: torch.Tensor,
        global_feat_norm: torch.Tensor,
        image_tensor: torch.Tensor,
    ) -> dict:
        return self._update_state_impl(class_idx, patches_norm, global_feat_norm, image_tensor)

    @staticmethod
    def _alpha_from_evidence(n_images: float, n_half: float = 15.0) -> float:
        if n_images <= 0:
            return 0.0
        return min(1.0, math.log(1.0 + n_images) / math.log(1.0 + n_half))

    def _extract_patch_crop_base64(self, image_tensor: torch.Tensor, patch_idx: int, num_patches: int) -> Optional[str]:
        """Extract a single patch crop from the image tensor and encode as base64 PNG."""
        if patch_idx < 0 or num_patches <= 0:
            return None
        grid = int(round(float(np.sqrt(num_patches))))
        if grid <= 0 or grid * grid != num_patches:
            return None

        img = image_tensor.detach().cpu().float()
        if img.dim() != 3:
            return None

        # Revert CLIP normalization for RGB tensors when needed.
        if img.shape[0] == 3:
            t_min = float(img.min().item())
            t_max = float(img.max().item())
            if t_min < 0.0 or t_max > 1.5:
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=img.dtype).view(3, 1, 1)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=img.dtype).view(3, 1, 1)
                img = img * clip_std + clip_mean

        img = img.clamp(0.0, 1.0)
        h, w = int(img.shape[1]), int(img.shape[2])
        ph = h // grid
        pw = w // grid
        if ph <= 0 or pw <= 0:
            return None

        px = patch_idx % grid
        py = patch_idx // grid
        if px >= grid or py >= grid:
            return None

        y0 = py * ph
        y1 = (py + 1) * ph if py < grid - 1 else h
        x0 = px * pw
        x1 = (px + 1) * pw if px < grid - 1 else w
        crop = img[:, y0:y1, x0:x1]
        return self._tensor_to_base64_png(crop)

    def _tensor_to_base64_png(self, tensor: torch.Tensor) -> str:
        """Convert a CHW or HW tensor to base64 PNG string.
        
        Args:
            tensor: torch.Tensor of shape [C, H, W] or [H, W] with values in [0, 1] or [0, 255]
        
        Returns:
            Base64 encoded PNG string (data:image/png;base64,...)
        """
        try:
            from PIL import Image

            tensor = tensor.detach().cpu().float()

            # Revert CLIP normalization for RGB tensors when needed.
            # Input images are often normalized with CLIP mean/std before model forward.
            if tensor.dim() == 3 and tensor.shape[0] == 3:
                t_min = float(tensor.min().item())
                t_max = float(tensor.max().item())
                if t_min < 0.0 or t_max > 1.5:
                    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=tensor.dtype).view(3, 1, 1)
                    clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=tensor.dtype).view(3, 1, 1)
                    tensor = tensor * clip_std + clip_mean

            # Clamp safely before uint8 conversion (prevents negative wrap-around color artifacts).
            tensor = tensor.clamp(0.0, 1.0)

            # Handle CHW format
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)  # HWC

            # Convert to uint8 numpy
            arr = (tensor * 255.0).round().to(torch.uint8).numpy()

            # Convert to PIL Image
            if arr.ndim == 3 and arr.shape[-1] == 3:
                img = Image.fromarray(arr, mode="RGB")
            elif arr.ndim == 3 and arr.shape[-1] == 4:
                img = Image.fromarray(arr, mode="RGBA")
            elif arr.ndim == 2:
                img = Image.fromarray(arr, mode="L").convert("RGB")
            else:
                # Fallback for unexpected shapes
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    img = Image.fromarray(arr, mode="L").convert("RGB")
                else:
                    return None
            
            # Encode to PNG
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            png_bytes = buffer.getvalue()

            # Encode to base64
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None

    def get_bank_snapshot(self) -> BankSnapshot:
        protos_per_class = []
        for c in range(self.C):
            centers = self.states[c]["centers"]
            appearance = self.states[c]["appearance"]
            K_c = centers.shape[0]
            ew = min(
                self.alpha_max,
                self._alpha_from_evidence(self.states[c].get("n_images", 0), self.n_half),
            ) if K_c >= 1 else 0.0
            class_protos = [
                ProtoInfo(
                    class_id=c,
                    class_name=self.classnames[c],
                    proto_idx=k,
                    center=centers[k].cpu().numpy(),
                    appearance=float(appearance[k].item()),
                    evidence_weight=ew,
                )
                for k in range(K_c)
            ]
            protos_per_class.append(class_protos)
        return BankSnapshot(sample_idx=self._sample_idx, class_protos=protos_per_class)

    def save_snapshot(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        snap = self.get_bank_snapshot()
        with open(path, "wb") as f:
            pickle.dump(snap, f)

    def save_records(self, records: List[SampleRecord], path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(records, f)

    @staticmethod
    def load_records(path: str) -> List[SampleRecord]:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_snapshot(path: str) -> BankSnapshot:
        with open(path, "rb") as f:
            return pickle.load(f)


