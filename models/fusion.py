"""
Fusion module: Decoupled logit fusion strategies.

Provides:
    - BaseFusion              Abstract base for all fusion strategies.
    - WeightedFusion          Fixed-weight fusion (no proto_alpha).
    - ProtoAlphaFusion        Adds proto_alpha modulation to patch term.
    - QualityGatedFusion      Adds quality_gate on top of proto_alpha.
    - MajorityVoteFusion      2-of-3 majority vote over source argmaxes.
    - AgreementGateFusion     Mutes patch term on CLIP disagreement.
    - TrustAdaptiveFusion     Scales tau_image_proto per-sample by a
                              confidence x patch-agreement trust signal.
"""

from abc import ABC, abstractmethod
from collections import Counter
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


class MajorityVoteFusion(WeightedFusion):
    """2-of-3 majority vote among clip, image_proto, and patch_proto.

    Each source casts one vote for its argmax class. If any class receives
    at least two votes it becomes the prediction; otherwise the patch vote is
    discarded and the fused logits fall back to ``clip + image`` (no patch
    term). A patch branch that produces all-zero logits abstains.

    The weighted fusion is still computed first, so logits stay calibrated;
    when the winner differs from the weighted argmax, the winner's logit is
    bumped to ``max + 1.0`` to make the vote decisive without distorting the
    remaining logits (and thus the confidence gate).

    Validated offline against recorded logits (scripts/counterfactual_gate_analysis.py):
    +1.48 DTD, +2.43 flowers over the always-on baseline.
    """

    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        clip_i = int(clip_logits.argmax(dim=-1).item())
        img_i = int(image_proto_logits.argmax(dim=-1).item())

        votes = [clip_i, img_i]
        if (
            patch_proto_logits is not None
            and not bool(torch.all(torch.abs(patch_proto_logits) < 1e-9))
        ):
            votes.append(int(patch_proto_logits.argmax(dim=-1).item()))

        winner, n = Counter(votes).most_common(1)[0]
        if n >= 2:
            result = super().forward(
                clip_logits, image_proto_logits, patch_proto_logits, **kwargs
            )
            if int(result.argmax(dim=-1).item()) != winner:
                result = result.clone()
                result[0, winner] = result.max() + 1.0
            return result
        # No majority (3-way split, or clip/image disagreement when the patch
        # abstains): the patch vote is discarded, fall back to clip + image.
        return self.tau_text * clip_logits.clone() + self.tau_image_proto * image_proto_logits


class AgreementGateFusion(WeightedFusion):
    """Mute the patch term whenever its vote disagrees with zero-shot CLIP.

    The patch branch only contributes when ``patch.argmax == clip.argmax``::

        result = tau_text * clip_logits.clone()
        result += tau_image_proto * image_proto_logits
        if patch.argmax == clip.argmax:
            result += tau_patch_proto * squash(patch_proto_logits)

    Validated offline against recorded logits (scripts/counterfactual_gate_analysis.py):
    +1.30 DTD, +2.19 flowers over the always-on baseline.
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
            clip_i = int(clip_logits.argmax(dim=-1).item())
            patch_i = int(patch_proto_logits.argmax(dim=-1).item())
            gate = 1.0 if patch_i == clip_i else 0.0
            result += (
                self.tau_patch_proto * gate * self._squash_patch(patch_proto_logits)
            )
        return result


class TrustAdaptiveFusion(WeightedFusion):
    """Scale ``tau_image_proto`` per-sample by a causal confidence x
    patch-agreement trust signal, instead of changing what gets written
    into the image prototype (Part 4a/4b already showed that write-side
    levers on this signal don't help — see
    ``experimental_results/PatchModPTA_Purity_Separability_Trust_Analysis.md``).

    The write rule is untouched; only how much the *existing* prototype is
    trusted at prediction time changes::

        trusted = (clip_margin >= conf_margin_thresh) and (patch_vote_pred == clip_top1)
        tau_eff = tau_image_proto * (tau_scale_trusted if trusted else tau_scale_untrusted)
        result  = tau_text * clip_logits + tau_eff * image_proto_logits

    ``trusted`` is passed in via ``**kwargs`` (computed causally by the
    adapter from frozen clip logits + a stateless patch vote, exactly as in
    ``models/write_gate_pta.py``); this class does not compute the vote
    itself. ``tau_scale_trusted = tau_scale_untrusted = 1.0`` reproduces
    base PTA's ``WeightedFusion`` exactly (in-grid control).

    No patch-level logit term is used here (``tau_patch_proto`` stays 0);
    this fusion mode is orthogonal to the patch-content fusion mechanisms
    above.
    """

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.tau_scale_trusted = float(self._cfg.get("tau_scale_trusted", 1.0))
        self.tau_scale_untrusted = float(self._cfg.get("tau_scale_untrusted", 1.0))

    def forward(
        self,
        clip_logits: Tensor,
        image_proto_logits: Tensor,
        patch_proto_logits: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        trusted = bool(kwargs.get("trusted", False))
        scale = self.tau_scale_trusted if trusted else self.tau_scale_untrusted
        tau_eff = self.tau_image_proto * scale

        result = self.tau_text * clip_logits.clone()
        result += tau_eff * image_proto_logits
        if patch_proto_logits is not None:
            result += self.tau_patch_proto * self._squash_patch(patch_proto_logits)
        return result
