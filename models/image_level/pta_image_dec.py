import torch
import torch.nn.functional as F

from models.image_level.base import BaseImageLevel


class PTAImageLevelDEC(BaseImageLevel):
    """Image-level PTA adapter with DEC Certainty Regularizer.

    Extends PTAImageLevel with entropy + logit-norm aware confidence scoring
    to handle noisy pseudo-labels under distribution shift.

    The key innovation: combine entropy (prediction uncertainty) with logit norm
    (feature magnitude) to detect unreliable samples and apply adaptive smoothing.

    Config keys (read from ``self._cfg``):
        alpha (float): Weight on original text features (default 0.01).
        T     (float): Temperature controlling EMA update rate (default 20.0).
        
        # DEC-specific hyperparameters
        use_dec (bool): Enable DEC certainty regularizer (default True).
        entropy_baseline (float): Baseline entropy h_0 (default 0.5).
        entropy_max (float): Max entropy h_max (default log(C), computed dynamically).
        temp_min (float): Min temperature t_min (default 0.5).
        temp_max (float): Max temperature t_max (default 1.0).
        logit_norm_weight (float): Weight for logit norm in confidence (default 0.1).
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
    # DEC Certainty Regularizer
    # ------------------------------------------------------------------

    def compute_certainty_temperature(self, clip_logits, num_classes):
        """Compute adaptive temperature τ_i based on entropy + logit norm.

        Args:
            clip_logits: (1, C) raw logits from CLIP.
            num_classes: C (number of classes).

        Returns:
            tau: (C,) adaptive temperature per class.
        """
        # Get hyperparameters
        entropy_baseline = self._cfg.get("entropy_baseline", 0.5)
        entropy_max = self._cfg.get("entropy_max", None)
        temp_min = self._cfg.get("temp_min", 0.5)
        temp_max = self._cfg.get("temp_max", 1.0)
        logit_norm_weight = self._cfg.get("logit_norm_weight", 0.1)

        # Compute entropy from softmax probabilities
        probs = F.softmax(clip_logits, dim=-1)  # (1, C)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # (1,)
        entropy = entropy.squeeze(0)  # scalar

        # Compute logit norm (proxy for feature magnitude / confidence)
        logit_norm = torch.norm(clip_logits, p=2, dim=-1)  # (1,)
        logit_norm = logit_norm.squeeze(0)  # scalar

        # Normalize logit norm to [0, 1] range
        # Typical CLIP logits are in [-1, 1] range, so norm is in [0, sqrt(C)]
        logit_norm_normalized = torch.clamp(logit_norm / (num_classes ** 0.5), 0, 1)

        # Set entropy_max if not provided (log of number of classes)
        if entropy_max is None:
            entropy_max = torch.tensor(
                torch.log(torch.tensor(float(num_classes))),
                device=entropy.device,
                dtype=entropy.dtype,
            )

        # Compute entropy contrast: (H(x_i) - h_0) / h_max
        entropy_contrast = (entropy - entropy_baseline) / (entropy_max + 1e-8)
        entropy_contrast = torch.clamp(entropy_contrast, -1, 1)  # Normalize to [-1, 1]

        # Combine entropy contrast with logit norm
        # High entropy (uncertain) → entropy_contrast > 0 → τ_i ≈ temp_max (weaker smoothing)
        # Low entropy (confident) → entropy_contrast < 0 → τ_i ≈ temp_min (stronger smoothing)
        combined_signal = entropy_contrast - logit_norm_weight * logit_norm_normalized

        # Apply sigmoid to smooth transition
        tau = torch.sigmoid(combined_signal) * (temp_max - temp_min) + temp_min  # (scalar)

        # Expand to per-class (all classes get same tau for now; can be per-class later)
        tau = tau.expand(num_classes)  # (C,)

        return tau

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
        """Online prototype update via EMA with DEC certainty regularizer.

        Args:
            image_feature:   (1, D) L2-normalized CLIP image embedding.
            clip_logits:     (1, C) zero-shot logits (before softmax).
            text_features:   (C, D) current (possibly already refined) text features.
            prototype_state: (C, D) running prototype bank (mutated in-place).
            **kwargs:        Unused, accepts extra keyword arguments for compatibility.

        Returns:
            (refined_text, prototype_state), where both are (C, D) tensors.
        """
        # Hyper-parameters
        alpha = self._cfg.get("alpha", 0.01)
        T = float(self._cfg.get("T", 20.0))
        use_dec = self._cfg.get("use_dec", True)

        num_classes = clip_logits.shape[-1]
        probs = F.softmax(clip_logits, dim=-1)  # (1, C)

        # Squeeze batch dimension for per-class processing
        w = probs.squeeze(0)  # (C)

        # Compute update weights: w_new = 1 - exp(-w / T)  for w >= 0.1
        w_new = torch.zeros_like(w)  # (C)
        mask = w >= 1e-1  # (C) bool

        if use_dec:
            # Apply DEC certainty regularizer
            tau = self.compute_certainty_temperature(clip_logits, num_classes)  # (C,)
            w_new[mask] = tau[mask] * (1 - torch.exp(-w[mask] / T))  # (C,)
        else:
            # Original PTA update (baseline)
            w_new[mask] = 1 - torch.exp(-w[mask] / T)  # (C,)

        w_new = w_new.unsqueeze(1)  # (C, 1)

        # EMA update on prototype bank
        # prototype_state[c] = (1 - w_new[c]) * old + w_new[c] * img_feat
        prototype_state[mask] = (
            (1 - w_new[mask]) * prototype_state[mask]
            + w_new[mask] * image_feature.squeeze(0)
        )

        # Blend original text with updated prototype
        refined_text = alpha * text_features + (1 - alpha) * prototype_state

        # L2-normalise
        refined_text = refined_text / refined_text.norm(dim=-1, keepdim=True)

        return refined_text, prototype_state

    # ------------------------------------------------------------------
    # Logit computation (inherits default from BaseImageLevel)
    # ------------------------------------------------------------------


def build(cfg: dict) -> PTAImageLevelDEC:
    """Factory function for :class:`PTAImageLevelDEC`.

    Called by the module dispatch mechanism (analogous to ``models.pta.build``).
    """
    return PTAImageLevelDEC(cfg)
