"""Abstract base class for patch-level TTA methods and shared evidence helper."""
import math
from abc import ABC, abstractmethod

import torch


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base class for patch-level TTA methods
# ═══════════════════════════════════════════════════════════════════════════════


class BasePatchLevel(ABC):
    def __init__(self, cfg: dict):
        self._cfg = cfg.get("patch_level", {})

    @abstractmethod
    def init_state(self, text_features) -> list:
        """Return per-class state dicts list."""
        ...

    @abstractmethod
    def compute_patch_logits(self, images, encoder, states):
        """Return (raw_proto, quality_gate)."""
        ...

    @abstractmethod
    def update_state(self, state, images, encoder, global_feat,
                     *, filter_scores=None, target_class_idx=None):
        """Update state dict."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence-based alpha scheduling (used by multiple adapter classes)
# ═══════════════════════════════════════════════════════════════════════════════


def _alpha_from_evidence(n_images: float, n_half: float = 15.0) -> float:
    """Compute an evidence-weighted alpha from the number of seen images.

    Uses a logarithmic schedule that saturates at 1.0 after ``n_half`` images.
    """
    if n_images <= 0:
        return 0.0
    return min(1.0, math.log(1.0 + n_images) / math.log(1.0 + n_half))
