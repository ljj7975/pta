import torch
import torch.nn.functional as F

from models.image_level.base import BaseImageLevel


class PTAImageLevel(BaseImageLevel):
    """Image-level PTA adapter.

    Maintains a per-class running prototype (EMA) that is updated from
    high-confidence image features.  The refined text features are a convex
    combination of the original CLIP text embedding and the prototype bank.

    Config keys (read from ``self._cfg``):
        alpha (float): Weight on original text features (default 0.01).
        T     (float): Temperature controlling EMA update rate (default 20.0).
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def init_state(self, text_features):
        """Return zero-initialised prototype bank matching ``text_features`` shape."""
        return torch.zeros_like(text_features)

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def update_prototypes(
        self,
        image_feature,
        clip_logits,
        text_features,
        prototype_state,
        **kwargs,
    ):
        """Online prototype update via EMA.

        Args:
            image_feature:   (1, D) L2-normalized CLIP image embedding.
            clip_logits:     (1, C) zero-shot logits (before softmax).
            text_features:   (C, D) current (possibly already refined) text features.
            prototype_state: (C, D) running prototype bank (mutated in-place).
            **kwargs:        Unused, accepts extra keyword arguments for compatibility.

        Returns:
            (refined_text, prototype_state), where both are (C, D) tensors.
        """
        # Hyper-parameters ──────────────────────────────────────────────
        alpha = self._cfg.get("alpha", 0.01)
        T = float(self._cfg.get("T", 20.0))

        # Softmax probabilities ─────────────────────────────────────────
        probs = F.softmax(clip_logits, dim=-1)          # (1, C)

        # Squeeze batch dimension for per-class processing
        w = probs.squeeze(0)                             # (C)

        # Compute update weights: w_new = 1 - exp(-w / T)  for w >= 0.1  ─
        w_new = torch.zeros_like(w)                      # (C)
        mask = w >= 1e-1                                 # (C) bool
        w_new[mask] = 1 - torch.exp(-w[mask] / T)        # (C)
        w_new = w_new.unsqueeze(1)                       # (C, 1)

        # EMA update on prototype bank ──────────────────────────────────
        # prototype_state[c] = (1 - w_new[c]) * old + w_new[c] * img_feat
        prototype_state[mask] = (
            (1 - w_new[mask]) * prototype_state[mask]
            + w_new[mask] * image_feature.squeeze(0)
        )

        # Blend original text with updated prototype ────────────────────
        refined_text = alpha * text_features + (1 - alpha) * prototype_state

        # L2-normalise ──────────────────────────────────────────────────
        refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)

        return refined_text, prototype_state

    # ------------------------------------------------------------------
    # Logit computation (inherits default from BaseImageLevel)
    # ------------------------------------------------------------------


def build(cfg: dict) -> PTAImageLevel:
    """Factory function for :class:`PTAImageLevel`.

    Called by the module dispatch mechanism (analogous to ``models.pta.build``).
    """
    return PTAImageLevel(cfg)
