"""CLIP text classifier construction and forward-inference helpers."""

import torch

from typing import Optional


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize vectors so that dot-product equals cosine similarity.

    The clamp(min=eps) prevents dividing by zero for all-zero vectors.
    """
    return x / x.norm(dim=dim, keepdim=True).clamp(min=eps)


def identify_relevant_patches(
    heatmap: torch.Tensor,
    top_k_ratio: float = 0.5,
    absolute_threshold: Optional[float] = None,
    min_patches: int = 1,
) -> torch.Tensor:
    """
    Identify relevant patches from a similarity heatmap using thresholding.

    Takes a heatmap of similarity scores (one score per patch), applies min-max
    normalization to [0, 1], and returns a boolean mask.

    Two thresholding strategies (use one):
    - **top_k_ratio**: keeps the top *ratio* fraction of patches (relative).
    - **absolute_threshold**: keeps patches whose normalised score >= threshold
      (absolute). The number of kept patches is dynamic.

    Args:
        heatmap: [P] tensor of similarity scores (higher = more relevant).
                 Can be any scale (cosine similarity, surgery scores, etc.).
        top_k_ratio: Fraction of patches to keep (0.0 to 1.0). Ignored if
                     *absolute_threshold* is set.
        absolute_threshold: Absolute threshold on the **normalised** [0, 1]
                            score. Patches with normalised score >= this value
                            are kept. Set to ``None`` (default) to use
                            *top_k_ratio* instead.
        min_patches: Minimum number of patches to always keep (default: 1).
                     Ensures at least this many patches survive filtering.

    Returns:
        mask: [P] boolean tensor, True for patches to keep.

    Example:
        >>> scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])  # 5 patches
        >>> mask = identify_relevant_patches(scores, top_k_ratio=0.6)
        >>> mask
        tensor([False,  True,  True,  True, False])  # top 3 (60%) kept
        >>> mask = identify_relevant_patches(scores, absolute_threshold=0.7)
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

    if absolute_threshold is not None:
        # Absolute threshold on normalised scores
        mask = scores_norm >= absolute_threshold
    else:
        # Top-k ratio (relative)
        top_k = max(min_patches, int(P * top_k_ratio))
        topk_vals, _ = scores_norm.topk(top_k)
        threshold = topk_vals[-1]
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

    return scores[0, 1:]  # [P, C] — strip CLS token and batch dim


def filter_patches_by_text_alignment(
    patches_norm: torch.Tensor,
    target_class_idx: int,
    *,
    text_features: Optional[torch.Tensor] = None,
    empty_text_feat: Optional[torch.Tensor] = None,
    filter_mode: str = "none",
    filter_top_k_ratio: float = 0.5,
    filter_absolute_threshold: Optional[float] = None,
    aug_copies: int = 0,
    precomputed_scores: Optional[torch.Tensor] = None,
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
        filter_top_k_ratio:  Fraction of patches to keep (passed to ``identify_relevant_patches``).
                             Ignored if *filter_absolute_threshold* is set.
        filter_absolute_threshold: Absolute threshold on the **normalised** [0, 1]
                                   score. Patches with normalised score >= this value
                                   are kept. Set to ``None`` (default) to use
                                   *filter_top_k_ratio* instead.
        aug_copies:          Number of augmented views concatenated with the original.
                             Used to repeat surgery scores across views when precomputed.
        precomputed_scores:  ``[P]`` per-patch surgery scores for *target_class_idx* (optional);
                             required for surgery modes.

    Returns:
        mask: ``[P]`` bool tensor, True for patches to keep.
    """
    P = patches_norm.shape[0]

    if text_features is None or filter_mode == "none":
        return torch.ones(P, dtype=torch.bool, device=patches_norm.device)

    # --- Precomputed scores fast-path (surgery modes) ---
    if precomputed_scores is not None and filter_mode in ("surgery_with_labels", "surgery_no_labels"):
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
        return torch.ones(P, dtype=torch.bool, device=patches_norm.device)

    # Negate scores: raw scores are "distance-like" (lower = more object-relevant),
    # so we invert them so that higher = more relevant for top-k/threshold filtering.
    scores = -scores

    return identify_relevant_patches(
        scores,
        top_k_ratio=filter_top_k_ratio,
        absolute_threshold=filter_absolute_threshold,
        min_patches=1,
    )
