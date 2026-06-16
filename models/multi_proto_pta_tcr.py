"""
MultiProtoPTA-TCR: Multi-prototype TTA with adaptive Term-Class Relevance weighting.

Extends MultiProtoPTABase with per-prototype cross-class specificity weights that
accumulate evidence over time via two running counters per prototype:

  appearance[k]  — how many images from THIS class activated prototype k  (already in base)
  cc_hits[k]     — how many times prototype k was found geometrically close
                   to some other class's prototype AFTER an EMA update

The weight for prototype k is a Beta-posterior specificity estimate gated by a
confidence term that starts flat (weight=1.0, no effect) and sharpens as evidence
accumulates:

  spec_k      = (appearance[k] + m) / (appearance[k] + cc_hits[k] + 2m)   # Beta mean
  confidence  = 1 - exp(-N_k / n_half)                                      # evidence gate
  weight_k    = (1 - confidence) * 1.0  +  confidence * (2 * spec_k)

Properties:
  - New prototype (0 evidence): weight = 1.0 exactly → no distortion early on
  - High appearance, zero cc_hits: weight → 2.0 → class-specific protos amplified
  - High cc_hits relative to appearance: weight → 0.0 → cross-class protos suppressed
  - cc_hits are DECAYED each time the prototype moves (EMA update), so a proto that
    drifts away from the cross-class region gradually recovers its specificity score

Key differences from MultiProtoPTABase:
  1. _make_class_state:    adds cc_hits field
  2. _update_class_state:  after EMA update, scans other classes and increments cc_hits;
                           decays cc_hits for moved prototypes to track drift
  3. _score_prototype_bank: multiplies per-proto TCR weight into app_weights before
                            aggregation (not a post-hoc class-level penalty)
  4. run():                disables penalize_common by default (superseded by TCR);
                           passes full states + class_id into _update_class_state
"""

import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.multi_proto_pta_base import (
    MultiProtoPTABase,
    _safe_normalize,
    _incremental_kmeans_step,
    _extract_patch_embeddings,
)
from utils import get_clip_logits, cls_acc


class MultiProtoPTATCR(MultiProtoPTABase):
    """
    MultiProtoPTA with adaptive TCR (Term-Class Relevance) prototype weighting.

    All clustering, patch extraction, and main-loop logic is inherited from
    MultiProtoPTABase. Only state initialisation, state update, and scoring
    are overridden.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────────────────────────────────

    def _make_class_state(self, prototype_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        D      = prototype_vec.shape[0]
        device = prototype_vec.device
        return {
            "centers":    torch.empty(0, D, device=device),
            "appearance": torch.empty(0,    device=device),
            "cc_hits":    torch.empty(0,    device=device),
        }

    def _update_class_state(
        self,
        state: Dict[str, torch.Tensor],
        patch_embeddings: torch.Tensor,   # [P, D]
        match_threshold: float,
        max_K: int,
        # TCR-specific (keyword-only so base callers still work if they don't pass them)
        states: List[Dict[str, torch.Tensor]] = None,
        class_id: int = None,
        cc_sim_threshold: float = 0.92,
        cc_decay_rate: float = 0.05,
    ) -> Dict[str, torch.Tensor]:
        """
                Run the standard incremental k-means step, then:
          1. Decay cc_hits for prototypes that moved (their cross-class neighbourhood
             may have changed — old evidence fades at the same rate as the EMA drift).
          2. Scan updated prototype positions against other classes that have been
             updated at least once (appearance.sum() > 0). This skips classes that
             still hold their CLIP text-embedding initialization, which would cause
             spurious cc_hits spikes: CLIP text embeddings cluster tightly in cosine
             space (sim > 0.85 between semantically related classes is common), so
             scanning against them contaminates cc_hits before any visual evidence
             accumulates.

        cc_sim_threshold (default 0.92): cosine similarity above which another class
            is counted as "sharing" this prototype. Set higher than 0.85 because
            ViT patch embeddings are denser on the unit sphere than natural-image
            features, and we only want genuine visual overlap, not text-space proximity.
        cc_decay_rate (default 0.05): fraction to decay cc_hits on each update.
            Higher than 0.01 so a prototype that drifts away from a cross-class
            region recovers its specificity score within ~20 updates rather than ~100.
        """
        centers = state["centers"]    # [K, D]
        apps    = state["appearance"] # [K]
        cc_hits = state["cc_hits"]    # [K]
        old_K   = centers.shape[0]

        updated_centers, appeared, _, _, _ = _incremental_kmeans_step(
            centers, patch_embeddings, match_threshold, max_K
        )

        # ── appearance counts ─────────────────────────────────────────────────
        updated_apps = apps.clone()
        updated_apps[appeared] += 1

        # ── cc_hits: decay for prototypes that just moved ─────────────────────
        # Only prototypes that were matched (appeared) got nudged by EMA.
        # Their old cc evidence is slightly stale — decay it proportionally.
        updated_cc = cc_hits.clone()
        updated_cc[appeared] *= (1.0 - cc_decay_rate)

        # ── extend tensors for newborn prototypes ─────────────────────────────
        num_new = updated_centers.shape[0] - old_K
        if num_new > 0:
            new_apps = torch.ones(num_new, device=apps.device)
            updated_apps = torch.cat([updated_apps, new_apps], dim=0)
            # Newborns start with cc_hits=0 (neutral prior; formula handles pseudo-counts)
            updated_cc = torch.cat([updated_cc, torch.zeros(num_new, device=cc_hits.device)], dim=0)

        # ── cc_hits: geometric scan against visually-updated classes only ────────
        # CRITICAL: skip classes whose appearance.sum() == 0, meaning they still
        # hold their CLIP text embedding initialisation. Text embeddings cluster
        # tightly in cosine space (sim > 0.85 is common between related classes),
        # so scanning against them would spike cc_hits on the first update and
        # instantly suppress every prototype via the confidence gate.
        # Only classes that have seen at least one real image update are meaningful
        # comparators for visual cross-class overlap.
        if states is not None and class_id is not None:
            protos_norm = _safe_normalize(updated_centers, dim=-1)  # [K', D]

            for c_other in range(len(states)):
                if c_other == class_id:
                    continue
                # Skip text-initialised classes — no visual evidence yet
                if states[c_other]["appearance"].sum() == 0:
                    continue
                other_norm = _safe_normalize(states[c_other]["centers"], dim=-1)  # [K_other, D]
                # [K', K_other] — cosine sim of THIS class's protos vs other class's protos
                cross_sims = protos_norm @ other_norm.t()
                # Class c_other "shares" proto k if ANY of its protos is close enough
                is_shared = cross_sims.max(dim=1).values >= cc_sim_threshold  # [K'] bool
                updated_cc += is_shared.float()

        state = dict(state)
        state["centers"]    = updated_centers
        state["appearance"] = updated_apps
        state["cc_hits"]    = updated_cc
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # TCR weight computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_tcr_weights(
        self,
        state: Dict[str, torch.Tensor],
        prior_strength: float = 5.0,
        n_half: float = 20.0,
    ) -> torch.Tensor:
        """
        Per-prototype TCR weight ∈ (0, 2).

        Stage 1 — Beta posterior mean (specificity):
            spec_k = (appearance[k] + m) / (appearance[k] + cc_hits[k] + 2m)
            Prior Beta(m, m) has mean 0.5 (neutral).  As evidence grows the
            posterior mean moves toward the empirical ratio.

        Stage 2 — Confidence gate:
            confidence = 1 - exp(-N_k / n_half)
            At N=0 → 0.0 (flat).  At N=n_half → ~0.63.  At N>>n_half → 1.0.

        Stage 3 — Interpolation:
            weight = (1 - confidence) * 1.0  +  confidence * (2 * spec_k)
            Neutral point: spec=0.5 → weight=1.0 (no distortion at max uncertainty).
            Class-specific (spec→1): weight→2.0 (amplified).
            Cross-class (spec→0):   weight→0.0 (suppressed).
        """
        appearance = state["appearance"]   # [K]
        cc_hits    = state["cc_hits"]      # [K]
        m = prior_strength

        # Stage 1: specificity (Beta posterior mean)
        spec = (appearance + m) / (appearance + cc_hits + 2.0 * m)   # [K]

        # Stage 2: confidence gate
        N = appearance + cc_hits                                        # [K]
        confidence = 1.0 - torch.exp(-N / n_half)                      # [K]

        # Stage 3: interpolate flat ↔ sharp
        weight = (1.0 - confidence) * 1.0 + confidence * (2.0 * spec)  # [K]
        return weight   # [K] ∈ (0, 2)

    # ─────────────────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _score_prototype_bank(
        self,
        patch_embeddings: torch.Tensor,
        feat: torch.Tensor,
        state: Dict[str, torch.Tensor],
        top_n: int = 5,
        global_gate: bool = False,
        softmax_vote: bool = False,
        softmax_vote_temp: float = 0.05,
        # TCR-specific — pass prior_strength=None to skip TCR and use TF only
        prior_strength=None,
        n_half: float = 20.0,
    ) -> torch.Tensor:
        centers    = state["centers"]     # [K, D]
        appearance = state["appearance"]  # [K]

        if centers.shape[0] == 0:
            return torch.tensor(0.0, device=centers.device)

        patches_norm = _safe_normalize(patch_embeddings, dim=-1)  # [P, D]

        sims     = patches_norm @ _safe_normalize(centers, dim=-1).t()  # [P, K]
        max_sims = sims.max(dim=0)[0]                                    # [K]

        # TF weight: intra-class appearance frequency
        app_weights = appearance / appearance.sum().clamp_min(1e-6)      # [K]

        if prior_strength is not None:
            # TCR weight: accumulating cross-class specificity
            tcr_weights  = self._compute_tcr_weights(state, prior_strength, n_half)
            per_proto_weight = app_weights * tcr_weights                 # [K]
        else:
            per_proto_weight = app_weights                               # [K] — TF only

        weighted_scores = max_sims * per_proto_weight                    # [K]

        if global_gate:
            gate = (feat @ _safe_normalize(centers[0], dim=-1)).clamp(min=0.0)
            weighted_scores = weighted_scores * (0.5 + 0.5 * gate)

        k = min(top_n, weighted_scores.numel())
        if k == 1:
            return weighted_scores.max()

        if softmax_vote:
            top_vals = weighted_scores.topk(k).values
            w = F.softmax(top_vals / softmax_vote_temp, dim=0)
            return (top_vals * w).sum()

        return weighted_scores.topk(k).values.mean()

    # ─────────────────────────────────────────────────────────────────────────
    # Main evaluation loop (override to wire TCR params and pass states to update)
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        loader,
        clip_model,
        clip_weights,
        dataset_name: str,
    ) -> float:
        # ── config ────────────────────────────────────────────────────────────
        max_K             = int(self.cfg.get("max_K", 30))
        match_thresh      = float(self.cfg.get("match_threshold", 0.8))
        conf_thresh       = float(self.cfg.get("conf_threshold", 0.1))
        tau_proto         = float(self.cfg.get("tau_proto", self.cfg.get("T", 20.0)))
        proto_top_n       = int(self.cfg.get("proto_top_n", 5))
        disable_centering = bool(self.cfg.get("disable_centering", False))
        topk_update       = int(self.cfg.get("topk_update", 1))
        adaptive_tau      = bool(self.cfg.get("adaptive_tau", False))
        use_global_gate   = bool(self.cfg.get("global_gate", False))
        use_softmax_vote  = bool(self.cfg.get("softmax_vote", False))
        softmax_vote_temp = float(self.cfg.get("softmax_vote_temp", 0.05))
        # TCR-specific
        cc_sim_threshold  = float(self.cfg.get("cc_sim_threshold", 0.92))
        cc_decay_rate     = float(self.cfg.get("cc_decay_rate", 0.05))
        cc_disable        = bool(self.cfg.get("cc_disable", False))
        tcr_prior_m       = float(self.cfg.get("tcr_prior_m", 5.0))
        tcr_n_half        = float(self.cfg.get("tcr_n_half", 20.0))

        os.makedirs("outputs", exist_ok=True)

        # ── setup ─────────────────────────────────────────────────────────────
        text_proto = _safe_normalize(clip_weights.t().float())   # [C, D]
        C, _       = text_proto.shape
        device     = text_proto.device

        states     = [self._make_class_state(text_proto[c]) for c in range(C)]
        accuracies = []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader, desc=f"[MultiProtoPTA-TCR] {dataset_name}")):
                # Some OOD loaders return list[tensor]. Normalize to one tensor first.
                if isinstance(images, list):
                    images = torch.cat(images, dim=0).to(device)
                else:
                    images = images.to(device)
                target = target.to(device)

                # ── CLIP zero-shot branch ─────────────────────────────────────
                image_features, clip_logits, _, _, _ = get_clip_logits(images, clip_model, clip_weights)
                feat = image_features.squeeze(0).float()   # [D]

                # ── patch embeddings ──────────────────────────────────────────
                patch_embs = _extract_patch_embeddings(images, clip_model)  # [P, D]

                # ── per-class proto scores ────────────────────────────────────
                raw_proto_scores = torch.zeros(C, device=device)
                for c in range(C):
                    raw_score = self._score_prototype_bank(
                        patch_embeddings=patch_embs,
                        feat=feat,
                        state=states[c],
                        top_n=proto_top_n,
                        global_gate=use_global_gate,
                        softmax_vote=use_softmax_vote,
                        softmax_vote_temp=softmax_vote_temp,
                        prior_strength=None if cc_disable else tcr_prior_m,
                        n_half=tcr_n_half,
                    )
                    K_c = states[c]["centers"].shape[0]
                    evidence_weight = min(1.0, 0.5 + 0.5 * (K_c - 1) / 4.0) if K_c >= 1 else 0.0
                    raw_proto_scores[c] = evidence_weight * raw_score

                # ── center + adaptive tau ─────────────────────────────────────
                centered_proto = (
                    raw_proto_scores if disable_centering
                    else raw_proto_scores - raw_proto_scores.mean()
                )

                eff_tau_proto = tau_proto
                if adaptive_tau:
                    proto_var = centered_proto.var().item()
                    eff_tau_proto = tau_proto * (1.0 + proto_var / (proto_var + 0.01))

                # ── fuse text + proto logits ──────────────────────────────────
                # penalize_common is DISABLED — TCR handles cross-class specificity
                # at the per-prototype level before aggregation (more principled).
                final_logits = torch.zeros(1, C, device=device)
                for c in range(C):
                    score_text = 100.0 * (text_proto[c] @ feat)
                    final_logits[0, c] = score_text + eff_tau_proto * centered_proto[c]

                # ── accuracy ──────────────────────────────────────────────────
                acc = cls_acc(final_logits, target)
                accuracies.append(acc)

                # ── online update ─────────────────────────────────────────────
                pred_conf  = F.softmax(clip_logits, dim=-1).squeeze(0)  # [C]
                scan_states = None if cc_disable else states

                if topk_update == 1:
                    pred_class = pred_conf.argmax()
                    if pred_conf[pred_class] > conf_thresh:
                        states[pred_class] = self._update_class_state(
                            states[pred_class], patch_embs, match_thresh, max_K,
                            states=scan_states,
                            class_id=int(pred_class),
                            cc_sim_threshold=cc_sim_threshold,
                            cc_decay_rate=cc_decay_rate,
                        )
                else:
                    topk_vals, topk_idx = pred_conf.topk(min(topk_update, C))
                    for rank, (conf_val, cls_idx) in enumerate(zip(topk_vals, topk_idx)):
                        if conf_val > conf_thresh * (0.6 ** rank):
                            states[cls_idx] = self._update_class_state(
                                states[cls_idx], patch_embs, match_thresh, max_K,
                                states=scan_states,
                                class_id=int(cls_idx),
                                cc_sim_threshold=cc_sim_threshold,
                                cc_decay_rate=cc_decay_rate,
                            )

                if i % 1000 == 0:
                    running = sum(accuracies) / len(accuracies)
                    print(f"---- MultiProtoPTA-TCR test accuracy: {running:.2f}. ----")

        final_acc = sum(accuracies) / len(accuracies)
        print(f"---- MultiProtoPTA-TCR test accuracy: {final_acc:.2f}. ----\n")

        with open("outputs/result.txt", "a") as f:
            f.write(f"MultiProtoPTA-TCR-v3's performance on {dataset_name}: Top1- {final_acc:.2f}.\n")

        return final_acc


def build(cfg: dict) -> MultiProtoPTATCR:
    return MultiProtoPTATCR(cfg)
