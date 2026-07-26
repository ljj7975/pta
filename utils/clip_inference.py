"""CLIP text classifier construction and forward-inference helpers."""

import torch

from typing import Optional


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize vectors so that dot-product equals cosine similarity.

    The clamp(min=eps) prevents dividing by zero for all-zero vectors.
    """
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def _identify_relevant_patches(
    heatmap: torch.Tensor,
    threshold: float = 0.5,
    min_patches: int = 1,
) -> torch.Tensor:
    """
    Identify relevant patches from a similarity heatmap using thresholding.

    Takes a heatmap of similarity scores (one score per patch), applies min-max
    normalization to [0, 1], and returns a boolean mask.

    Args:
        heatmap: [P] tensor of similarity scores (higher = more relevant).
                 Can be any scale (cosine similarity, surgery scores, etc.).
        threshold: Absolute threshold on the **normalised** [0, 1] score.
                   Patches with normalised score >= this value are kept.
        min_patches: Minimum number of patches to always keep (default: 1).
                     Ensures at least this many patches survive filtering.

    Returns:
        mask: [P] boolean tensor, True for patches to keep.

    Example:
        >>> scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])  # 5 patches
        >>> mask = _identify_relevant_patches(scores, threshold=0.7)
        >>> mask
        tensor([False,  True, False,  True, False])  # 2 patches above threshold
    """
    P = heatmap.shape[0]

    # Min-max normalize scores to [0, 1]
    s_min, s_max = heatmap.min(), heatmap.max()
    if s_max > s_min:
        scores_norm = (heatmap - s_min) / (s_max - s_min)
    else:
        # Degenerate case: all scores identical → keep all
        return torch.ones(P, dtype=torch.bool, device=heatmap.device)

    # Absolute threshold on normalised scores
    mask = scores_norm >= threshold

    # Safety: always keep at least min_patches (the highest scoring one)
    mask[scores_norm.argmax()] = True

    return mask


def clip_classifier(classnames: list, template: list, encoder) -> torch.Tensor:
    """Build a text-embedding matrix for zero-shot CLIP classification.

    For each classname the function:
    1. Formats the name into all template strings.
    2. Encodes with the CLIP text encoder.
    3. L2-normalises each embedding.
    4. Averages across templates to get a single per-class vector.

    Returns a ``(num_classes, D)`` tensor on CUDA.
    """
    with torch.no_grad():
        text_embeddings = []

        for classname in classnames:
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in template]

            class_embeddings = encoder.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_embeddings.append(class_embedding)

        text_embeddings = torch.stack(text_embeddings, dim=1).cuda()
    return text_embeddings


def get_clip_logits(images, encoder, text_embeddings: torch.Tensor):
    """Run CLIP inference and return features, logits, entropy, prob_map, and prediction.

    Args:
        images: single image tensor or list of tensors (e.g. AugMix views).
        encoder: CLIP encoder with ``encode_image()``.
        text_embeddings: ``(D, C)`` text-embedding matrix from ``clip_classifier()``.

    Returns:
        image_features, clip_logits, loss (entropy), prob_map, pred (int).
    """
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features = encoder.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        clip_logits = 100.0 * image_features @ text_embeddings

        if image_features.size(0) > 1:
            from .metrics import softmax_entropy, avg_entropy

            batch_entropy = softmax_entropy(clip_logits)
            selected_idx = torch.argsort(batch_entropy, descending=False)[
                : int(batch_entropy.size()[0] * 0.1)
            ]
            output = clip_logits[selected_idx]
            image_features = image_features[selected_idx].mean(0).unsqueeze(0)
            clip_logits = output.mean(0).unsqueeze(0)

            loss = avg_entropy(output)
            prob_map = output.softmax(1).mean(0).unsqueeze(0)
            pred = int(output.mean(0).unsqueeze(0).topk(1, 1, True, True)[1].t())
        else:
            from .metrics import softmax_entropy

            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred


def compute_surgery_scores(
    images: torch.Tensor,
    encoder,
    text_features: torch.Tensor,
    empty_text_feat: torch.Tensor,
    filter_mode: str,
) -> Optional[torch.Tensor]:
    """Compute per-patch CLIP Surgery relevance scores for all classes.

    Runs the CLIP Surgery forward pass on *images* and returns patch-level
    scores as a ``[P, C]`` tensor (CLS token stripped, batch dim squeezed),
    ready for per-class indexing via ``scores[:, class_idx]``.

    Args:
        images:          ``[1, C_img, H, W]`` input image tensor.
        encoder:         encoder wrapper (provides encode_image).
        text_features:   ``[C, D]`` L2-normalised class text embeddings.
        empty_text_feat: ``[1, D]`` L2-normalised empty-string embedding;
                         used in *surgery_no_labels* mode to subtract the
                         label-agnostic visual signal.
        filter_mode:     ``"surgery_with_labels"`` or ``"surgery_no_labels"``.
                         Returns ``None`` for any other value.

    Returns:
        ``[P, C]`` float tensor of surgery relevance scores, or ``None`` if
        *filter_mode* is not a surgery variant.
    """
    if filter_mode not in ("surgery_with_labels", "surgery_no_labels"):
        return None

    from third_party.CLIP_Surgery.clip_surgery.clip import clip_feature_surgery

    all_tokens = encoder.encode_image(images, CLS_token_only=False, preprocess=False)  # [1, 1+P, D]

    if filter_mode == "surgery_with_labels":
        scores = clip_feature_surgery(
            all_tokens.float(), text_features.float()
        )  # [1, 1+P, C]
    else:  # surgery_no_labels
        scores = clip_feature_surgery(
            all_tokens.float(),
            text_features.float(),
            redundant_feats=empty_text_feat.float(),
        )  # [1, 1+P, C]

    scores_out = scores[0, 1:]  # [P, C] — strip CLS token and batch dim
    # Standard CLIP (non-surgery) patch features produce inverted scores when passed
    # through clip_feature_surgery, which is designed for CS-ViT-B/16 features.
    # Negating corrects the direction so higher scores always mean more class-relevant.
    if not getattr(encoder, 'is_surgery_encoder', True):
        scores_out = -scores_out
    return scores_out


def filter_patches_by_text_alignment(
    patches_norm: torch.Tensor,
    target_class_idx: int,
    *,
    text_features: Optional[torch.Tensor] = None,
    empty_text_feat: Optional[torch.Tensor] = None,
    filter_mode: str = "none",
    filter_threshold: float = 0.5,
    aug_copies: int = 0,
    precomputed_scores: Optional[torch.Tensor] = None,
    encoder: object = None,
    return_scores: bool = False,
) -> torch.Tensor:
    """Compute per-patch text-alignment scores and return a boolean keep-mask.

    Supports four filter modes:
    - **none**: keeps all patches.
    - **cosine_with_labels**: relative specificity (target cosine minus mean-other-class cosine).
    - **cosine_no_labels**: cosine against label-adjusted text feature (target minus empty-string baseline).
    - **surgery_with_labels** / **surgery_no_labels**: uses precomputed CLIP Surgery scores.

    Args:
        patches_norm:        ``[P, D]`` L2-normalised patch embeddings.
        target_class_idx:    int, index of the target class.
        text_features:       ``[C, D]`` L2-normalised class text embeddings (required for cosine modes).
        empty_text_feat:     ``[1, D]`` L2-normalised empty-string embedding (required for cosine_no_labels).
        filter_mode:         One of ``"none"``, ``"cosine_with_labels"``, ``"cosine_no_labels"``,
                             ``"surgery_with_labels"``, ``"surgery_no_labels"``.
        filter_threshold:    Absolute threshold on the **normalised** [0, 1] score.
                             Patches with normalised score >= this value are kept.
        aug_copies:          Number of augmented views concatenated with the original.
                             Used to repeat surgery scores across views when precomputed.
        precomputed_scores:  ``[P]`` per-patch scores for *target_class_idx* (optional);
                             bypasses internal score computation for any mode.
        encoder:             Optional encoder object. If ``encoder.is_surgery_encoder``
                             is False (standard CLIP), patches are negated before cosine
                             scoring to correct the inverted similarity direction.
        return_scores:       If True, return ``(mask, scores)`` tuple instead of just ``mask``.

    Returns:
        mask: ``[P]`` bool tensor, True for patches to keep.
        scores: ``[P]`` float tensor of raw scores (only when *return_scores* is True).
    """
    P = patches_norm.shape[0]

    if text_features is None or filter_mode == "none":
        mask = torch.ones(P, dtype=torch.bool, device=patches_norm.device)
        return (mask, None) if return_scores else mask

    # Standard CLIP patch cosine similarity is inverted relative to CS-ViT features.
    # Negate patches_norm so cosine scores point in the correct direction.
    # Only applied when precomputed_scores is absent (cosine path would be used).
    if encoder is not None and not getattr(encoder, 'is_surgery_encoder', True) and precomputed_scores is None:
        patches_norm = -patches_norm

    # --- Precomputed scores fast-path (any mode) ---
    if precomputed_scores is not None:
        scores = precomputed_scores.float()  # [P_original]
        n_views = 1 + aug_copies
        if scores.shape[0] < patches_norm.shape[0]:
            scores = scores.repeat(n_views)[:patches_norm.shape[0]]
    # --- Mode: cosine_with_labels ---
    elif filter_mode == "cosine_with_labels":
        sim = patches_norm @ text_features.T  # [P, C]
        target_score = sim[:, target_class_idx]  # [P]
        other_mean = (sim.sum(1) - target_score) / max(sim.shape[1] - 1, 1)  # [P]
        scores = target_score - other_mean  # [P]
    # --- Mode: cosine_no_labels ---
    elif filter_mode == "cosine_no_labels":
        target_feat = text_features[target_class_idx]  # [D]
        adjusted = target_feat - empty_text_feat.squeeze(0)  # [D]
        adjusted = adjusted / adjusted.norm().clamp(min=1e-8)
        scores = patches_norm @ adjusted  # [P]
    else:
        # Unknown mode — return all patches (safe fallback)
        mask = torch.ones(P, dtype=torch.bool, device=patches_norm.device)
        return (mask, None) if return_scores else mask

    # Scores are similarity-like for ALL modes (higher = more relevant):
    # - cosine_with_labels: target_score - other_mean (higher = more target-specific)
    # - cosine_no_labels: patches @ adjusted_text (higher = more aligned with target)
    # - surgery modes: clip_feature_surgery output (higher = more class-specific)
    # No negation needed — _identify_relevant_patches expects higher = more relevant.

    mask = _identify_relevant_patches(
        scores,
        threshold=filter_threshold,
        min_patches=1,
    )
    return (mask, scores) if return_scores else mask
