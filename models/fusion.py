"""
Fusion module: Decoupled logit fusion strategies.

Provides:
    - BaseFusion              Abstract base for all fusion strategies.
    - WeightedFusion          Fixed-weight fusion matching PTA's original formula.
    - QualityGatedFusion      Extended fusion with quality-gated patch modulation.
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
    """Fixed-weight fusion matching PTA's original formula.

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
        # NOTE: clone() is essential — never mutate clip_logits in-place.
        result = self.tau_text * clip_logits.clone()
        result += self.tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            result += self.tau_patch_proto * patch_proto_logits
        return result


class QualityGatedFusion(WeightedFusion):
    """Extended fusion with quality-gated patch-level modulation.

    In addition to the three tau weights inherited from
    :class:`WeightedFusion`, this variant reads:

        quality_modulation   — scales the quality-gate effect (default 1.0)

    The forward pass applies ``quality_gate`` and ``proto_alpha`` from
    ``**kwargs`` to the patch-level term::

        result = tau_text * clip_logits.clone()
        result += tau_image_proto * image_proto_logits
        if patch_proto_logits is not None:
            quality_gate = kwargs.get('quality_gate', 1.0)
            proto_alpha  = kwargs.get('proto_alpha', 1.0)
            result += tau_patch_proto * proto_alpha * quality_gate * patch_proto_logits
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.quality_modulation = float(self._cfg.get("quality_modulation", 1.0))

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
