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
            result += self.tau_patch_proto * patch_proto_logits
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
            result += self.tau_patch_proto * proto_alpha * patch_proto_logits
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
            result += self.tau_patch_proto * proto_alpha * quality_gate * patch_proto_logits
        return result


class AdaptiveImageWeightFusion(QualityGatedFusion):
    """Fusion with per-class adaptive image-level weighting.

    Instead of a single ``tau_image_proto`` scalar applied uniformly to
    all classes, this variant computes a per-class weight:

        tau_image[c] = tau_image_max - (tau_image_max - tau_image_min) * proto_alpha[c]

    Config keys (read from ``cfg["fusion"]``):

        tau_image_max   — upper bound for per-class image weight (default 100.0)
        tau_image_min   — lower bound for per-class image weight (default 50.0)
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.tau_image_max = float(self._cfg.get("tau_image_max", 100.0))
        self.tau_image_min = float(self._cfg.get("tau_image_min", 50.0))

    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        proto_alpha = kwargs.get("proto_alpha", None)

        if proto_alpha is not None and proto_alpha.numel() > 0:
            tau_image_per_class = (
                self.tau_image_max
                - (self.tau_image_max - self.tau_image_min) * proto_alpha
            )
            image_weight = tau_image_per_class.unsqueeze(0)
        else:
            image_weight = self.tau_image_proto

        result = self.tau_text * clip_logits.clone()
        result += image_weight * image_proto_logits
        if patch_proto_logits is not None:
            quality_gate = kwargs.get("quality_gate", 1.0)
            if proto_alpha is None:
                proto_alpha = torch.ones(1, device=clip_logits.device)
            result += self.tau_patch_proto * proto_alpha * quality_gate * patch_proto_logits
        return result
