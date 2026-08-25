"""Patch-to-text aggregation pooling for the CLS-vote-vs-patch-vote
corroboration signal (Experiments H/I/J/K in
outputs/patch_fix_experiments_plan.md).

Shared by both the offline analysis scripts
(scripts/analyze_patch_vote_aggregation.py, scripts/validate_patch_vote_signal.py)
and the live write-time corroboration gate (models/patch_modulated_pta.py,
Experiment K) so the live gate can never silently drift from what
Experiments I/J actually validated -- one implementation, not a
re-derivation.
"""
import torch


def topk_pool(sims, k):
    k_eff = min(k, sims.shape[0])
    vals, _ = sims.topk(k_eff, dim=0)
    return vals.mean(dim=0)


def _otsu_threshold_1d(values, n_bins=64):
    """Otsu's method: the threshold that maximizes between-group variance of
    a 1D distribution split into two groups -- the classic zero-hyperparameter
    algorithm for splitting a distribution into "high" and "low" clusters
    (from image binarization; here, foreground- vs background-like patch
    similarities). No k, no z-score cutoff to pick -- given the data there is
    a unique optimum. `n_bins` only controls histogram resolution (a
    numerical-precision knob, not a qualitative behavior knob like k).

    torch.histc has no deterministic CUDA kernel, and the live TTA loop
    (runner.py) runs under torch.use_deterministic_algorithms(True) --
    offloaded to CPU, same fix already used for the cpm write-rule's
    torch.cumsum in models/patch_modulated_pta.py. n_bins is tiny (64) so
    the offload is negligible."""
    values = values.detach().cpu()
    vmin, vmax = values.min(), values.max()
    if vmax <= vmin:
        return float(vmin)  # degenerate: all patches identical -- no real split
    hist = torch.histc(values, bins=n_bins, min=float(vmin), max=float(vmax))
    edges = torch.linspace(float(vmin), float(vmax), n_bins + 1, device=values.device)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    cum_n = torch.cumsum(hist, dim=0)
    cum_sum = torch.cumsum(hist * centers, dim=0)
    global_sum = cum_sum[-1]
    w0 = cum_n / total
    w1 = 1.0 - w0
    mean0 = cum_sum / cum_n.clamp(min=1e-8)
    mean1 = (global_sum - cum_sum) / (total - cum_n).clamp(min=1e-8)
    between_var = w0 * w1 * (mean0 - mean1).pow(2)
    # Exclude the last bin (w1=0 there, always degenerate) from the search.
    best_bin = int(between_var[:-1].argmax().item())
    return float(centers[best_bin].item())


def otsu_mean(sims):
    """Per class: find the Otsu threshold on that class's patch similarities,
    keep patches at/above it, mean-pool raw cosine similarity over just
    those. A data-driven "adaptive top-k" -- the cutoff count emerges from
    each image/class's own similarity distribution instead of a fixed k."""
    C = sims.shape[1]
    out = torch.zeros(C, device=sims.device)
    for c in range(C):
        col = sims[:, c]
        t = _otsu_threshold_1d(col)
        mask = col >= t
        if not mask.any():
            mask = col >= col.max()  # always keep at least the single best patch
        out[c] = col[mask].mean()
    return out


MARGIN_LOGIT_SCALE = 100.0  # matches CLIP's own zero-shot temperature convention


def compute_patch_vote(patches_norm, text_embeddings, aggregation="topk20"):
    """One image's patch-vote prediction + confidence margin.

    Args:
        patches_norm:    [P, D] L2-normalized patch embeddings (one CLIP
                          Surgery forward pass, no augmentation).
        text_embeddings: [D, C] text class embeddings (clip_classifier's
                          convention).
        aggregation:      "topk20" | "otsu_mean" | "mean" -- the two
                          Experiment-K candidates plus mean for reference/
                          debugging.

    Returns:
        (pred: int, margin: float) -- argmax class and its top1-top2 softmax
        margin (temperature-scaled the same way CLIP's own zero-shot logits
        are, so margin is directly usable as a confidence-style weight).
    """
    sims = patches_norm @ text_embeddings  # [P, C]
    if aggregation == "topk20":
        scores = topk_pool(sims, 20)
    elif aggregation == "otsu_mean":
        scores = otsu_mean(sims)
    elif aggregation == "mean":
        scores = sims.mean(dim=0)
    else:
        raise ValueError(f"Unknown patch_vote aggregation: {aggregation!r}")

    probs = torch.softmax(scores * MARGIN_LOGIT_SCALE, dim=-1)
    top2 = probs.topk(min(2, probs.shape[0]))
    margin = float((top2.values[0] - top2.values[-1]).item())
    pred = int(scores.argmax().item())
    return pred, margin
