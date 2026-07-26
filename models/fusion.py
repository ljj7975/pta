"""
Fusion module: Decoupled logit fusion strategies.

Provides:
    - BaseFusion              Abstract base for all fusion strategies.
    - WeightedFusion          Fixed-weight fusion (no proto_alpha).
    - ProtoAlphaFusion        Adds proto_alpha modulation to patch term.
    - QualityGatedFusion      Adds quality_gate on top of proto_alpha.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class BaseFusion(ABC):
    """Abstract base class for logit fusion strategies.

    Each subclass implements ``forward(clip_logits, image_proto_logits,
    patch_proto_logits, **kwargs) -> Tensor``.

    Args:
        cfg: Full configuration dictionary.  The ``fusion`` sub-dict is
             extracted automatically and stored as ``self._cfg``.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg.get("fusion", {})

    @abstractmethod
    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor],
        **kwargs,
    ) -> Tensor:
        """Fuse multiple logit sources into a single prediction.

        Args:
            clip_logits:         Zero-shot CLIP logits  ``[1, C]``.
            image_proto_logits:  Image-level prototype logits  ``[1, C]``.
            patch_proto_logits:  Patch-level prototype logits  ``[1, C]``
                                 or ``None``.
            **kwargs:            Additional per-fusion arguments.

        Returns:
            Fused logits  ``[1, C]``.
        """
        ...


class WeightedFusion(BaseFusion):
    """Fixed-weight fusion — no proto_alpha, no quality_gate.

    Reads three scalar weights from ``cfg["fusion"]``:

        tau_text          — weight on zero-shot CLIP logits  (default 1.0)
        tau_image_proto   — weight on image-level prototype (default 100.0)
        tau_patch_proto   — weight on patch-level prototype (default 0.0)

    The forward pass computes::

        result = tau_text * clip_logits.clone()
        result += tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            result += tau_patch_proto * patch_proto_logits
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.tau_text = float(self._cfg.get("tau_text", 1.0))
        self.tau_image_proto = float(self._cfg.get("tau_image_proto", 100.0))
        self.tau_patch_proto = float(self._cfg.get("tau_patch_proto", 0.0))
        self.patch_squash = str(self._cfg.get("patch_squash", "none"))
        self.patch_squash_scale = float(self._cfg.get("patch_squash_scale", 3.0))

    def _squash_patch(self, patch_proto_logits: Tensor) -> Tensor:
        """Optionally bound the patch term before it enters the linear fusion.

        Raw Gaussian prototype scores live in [0, 1], but z-score aggregations
        are unbounded (roughly [-3, +10]), which puts them on a completely
        different scale from ``tau_patch_proto``'s tuning. ``tanh`` maps them
        back into [-1, 1] so the existing tau stays roughly valid.

        Monotonic, so it cannot change the prediction of a patch-only argmax.
        """
        if self.patch_squash == "none":
            return patch_proto_logits
        if self.patch_squash == "tanh":
            return torch.tanh(patch_proto_logits / self.patch_squash_scale)
        raise ValueError(f"Unknown patch_squash: {self.patch_squash}")

    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        result = self.tau_text * clip_logits.clone()
        result += self.tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            result += self.tau_patch_proto * self._squash_patch(patch_proto_logits)
        return result


class ProtoAlphaFusion(WeightedFusion):
    """Fixed-weight fusion with proto_alpha modulation on patch term.

    The patch-level term is scaled by ``proto_alpha`` from ``**kwargs``,
    which grows from 0 toward ``alpha_max`` as prototypes accumulate::

        result = tau_text * clip_logits.clone()
        result += tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            proto_alpha = kwargs.get('proto_alpha', 1.0)
            result += tau_patch_proto * proto_alpha * patch_proto_logits
    """

    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        result = self.tau_text * clip_logits.clone()
        result += self.tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            proto_alpha = kwargs.get("proto_alpha", 1.0)
            result += (
                self.tau_patch_proto * proto_alpha
                * self._squash_patch(patch_proto_logits)
            )
        return result


class QualityGatedFusion(ProtoAlphaFusion):
    """Fixed-weight fusion with proto_alpha AND quality_gate on patch term.

    The patch-level term is scaled by both ``proto_alpha`` and
    ``quality_gate`` from ``**kwargs``::

        result = tau_text * clip_logits.clone()
        result += tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            quality_gate = kwargs.get('quality_gate', 1.0)
            proto_alpha  = kwargs.get('proto_alpha', 1.0)
            result += tau_patch_proto * proto_alpha * quality_gate * patch_proto_logits
    """

    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        result = self.tau_text * clip_logits.clone()
        result += self.tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            quality_gate = kwargs.get("quality_gate", 1.0)
            proto_alpha = kwargs.get("proto_alpha", 1.0)
            result += (
                self.tau_patch_proto * proto_alpha * quality_gate
                * self._squash_patch(patch_proto_logits)
            )
        return result
