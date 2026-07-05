from abc import ABC


class BaseImageLevel(ABC):
    """Abstract base for image-level prototype computation.

    Each subclass implements a different style of updating text features
    from image-level information (e.g., EMA prototype update, quality-gated
    modulation).
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg.get("image_level", {})

    def update_prototypes(
        self,
        image_feature,
        clip_logits,
        text_features,
        prototype_state,
        **kwargs,
    ):
        """Update prototype state and return (refined_text, prototype_state).

        Args:
            image_feature: [1, D] L2-normalized CLIP image embedding.
            clip_logits: [1, C] zero-shot logits.
            text_features: [C, D] current refined text features.
            prototype_state: Mutable state tensor updated in-place.
            **kwargs: May contain ``quality_gate``, ``quality_modulation``
                for quality-modulated variants.
        """
        ...

    def compute_logits(self, image_features, refine_feature):
        """Compute logits from image features and refined text features.

        ``refine_feature`` is the output of ``update_prototypes``.
        Both inputs are cast to fp16 before the matmul to match the
        original PTA precision (CLIP image features are fp16).
        """
        return image_features.half() @ refine_feature.half().T

    def init_state(self, text_features):
        """Return initial prototype state (zero vector matching text_features shape)."""
        import torch
        return torch.zeros_like(text_features)
