"""Per-prototype running score statistics for z-score normalization.

Prototypes are built independently per class, so each one has its own natural
score distribution: a Gaussian score of 0.8 may be routine for a permissive
prototype and exceptional for a picky one. These helpers maintain a reference
distribution per prototype so raw scores can be converted to prototype-specific
z-scores before class-level aggregation.

The reference distribution is built **online** from the test stream. There is no
train split in this codebase (every dataset exposes only `test`), so the caller
scores an image against the statistics accumulated from *previously seen* images
and only afterwards folds that image's scores in. The current query image is
therefore never part of its own mu/sigma.

State is three parallel ``[K]`` tensors held alongside ``appearance`` in each
class's prototype bank. They must be grown and permuted in lockstep with
``centers`` / ``variance`` / ``appearance`` — see :func:`grow` and
:func:`permute`.

Used exclusively by GaussianPatchLevel and _gaussian_score_for_class.
"""
from typing import Dict, Tuple

import torch


# Keys used in the per-class state dict. Kept together so callers can iterate.
STAT_KEYS = ("score_count", "score_sum", "score_sqsum")


def init_stats(K: int, device) -> Dict[str, torch.Tensor]:
    """Create empty running-statistic accumulators for a bank of K prototypes."""
    return {key: torch.zeros(K, device=device) for key in STAT_KEYS}


def has_stats(state: dict) -> bool:
    """True if *state* already carries accumulators (banks created before this
    feature existed, or restored from an old checkpoint, will not)."""
    return all(key in state for key in STAT_KEYS)


def accumulate(
    state: dict,
    best_per_proto: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Fold one image's raw prototype scores into the running statistics.

    Call this **after** the image has been scored, never before — that ordering
    is what keeps the query image out of its own reference distribution.

    Args:
        state:          per-class bank dict holding the three accumulators.
        best_per_proto: ``[K]`` raw score of every prototype against this image.
                        Prototypes that won no patch group score exactly 0.0 and
                        are included deliberately: a prototype that usually fires
                        but is absent now is evidence against the class.

    Returns:
        Dict of the three updated ``[K]`` tensors (new tensors, not in-place).
    """
    scores = best_per_proto.detach().float()
    K = scores.shape[0]
    device = scores.device

    count = state.get("score_count")
    if count is None or count.shape[0] != K:
        # Bank was resized without the accumulators following along, or this is
        # a legacy state dict. Restart from zero rather than mis-attributing
        # statistics to the wrong prototypes.
        stats = init_stats(K, device)
        count = stats["score_count"]
        total = stats["score_sum"]
        sq_total = stats["score_sqsum"]
    else:
        total = state["score_sum"]
        sq_total = state["score_sqsum"]

    return {
        "score_count": count + 1.0,
        "score_sum": total + scores,
        "score_sqsum": sq_total + scores.pow(2),
    }


def compute_mu_sigma(
    state: dict,
    min_count: int,
    sigma_eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive per-prototype mean, standard deviation and validity mask.

    Args:
        state:      per-class bank dict holding the three accumulators.
        min_count:  a prototype needs this many reference observations before its
                    z-score is trusted. Below it the prototype abstains.
        sigma_eps:  floor applied to sigma by :func:`zscore`; used here only to
                    keep the returned sigma strictly positive.

    Returns:
        ``(mu, sigma, valid)`` — each ``[K]``; *valid* is bool.
    """
    count = state["score_count"]
    total = state["score_sum"]
    sq_total = state["score_sqsum"]

    safe_count = count.clamp(min=1.0)
    mu = total / safe_count

    # Population variance, then Bessel-corrected to the sample variance wherever
    # we have at least two observations. clamp(min=0) guards the catastrophic
    # cancellation that E[x²] - E[x]² suffers when the scores are near-constant.
    var_pop = (sq_total / safe_count - mu.pow(2)).clamp(min=0.0)
    bessel = torch.where(count > 1.0, count / (count - 1.0).clamp(min=1.0),
                         torch.ones_like(count))
    sigma = (var_pop * bessel).sqrt().clamp(min=sigma_eps)

    valid = count >= float(min_count)
    return mu, sigma, valid


def zscore(
    raw: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    valid: torch.Tensor,
    sigma_eps: float = 1e-6,
) -> torch.Tensor:
    """Convert raw prototype scores to prototype-specific z-scores.

    ``z_i = (raw_i - mu_i) / max(sigma_i, eps)``

    Prototypes without enough reference observations abstain at 0.0 (the
    "typical" value on the z scale) rather than contributing noise.
    """
    z = (raw - mu) / sigma.clamp(min=sigma_eps)
    return torch.where(valid, z, torch.zeros_like(z))


def grow(state: dict, new_K: int, device) -> Dict[str, torch.Tensor]:
    """Resize accumulators to a bank that has grown to *new_K* prototypes.

    Existing entries keep their history; freshly created prototypes start with
    zero observations, so they abstain until they have accumulated ``min_count``
    reference scores.
    """
    if not has_stats(state):
        # Legacy/uninitialised bank: start every prototype from zero.
        return init_stats(new_K, device)

    current_K = state[STAT_KEYS[0]].shape[0]
    if current_K == new_K:
        return {key: state[key] for key in STAT_KEYS}
    if current_K > new_K:
        # Bank shrank without going through permute(); we cannot tell which
        # prototypes survived, so drop the history rather than misattribute it.
        return init_stats(new_K, device)

    pad = torch.zeros(new_K - current_K, device=device)
    return {key: torch.cat([state[key], pad], dim=0) for key in STAT_KEYS}


def permute(stats: Dict[str, torch.Tensor], keep_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Reindex accumulators to follow a prune/reorder of the prototype bank.

    Must be applied wherever ``centers`` / ``variance`` / ``appearance`` are
    reindexed; skipping it silently attaches every prototype's statistics to a
    different prototype.
    """
    return {key: stats[key][keep_idx] for key in STAT_KEYS}


def low_variance_count(
    sigma: torch.Tensor,
    valid: torch.Tensor,
    sigma_warn: float,
) -> int:
    """Number of prototypes whose reference distribution is near-degenerate.

    A high count means either too few reference samples or a prototype whose
    score is nearly constant — in both cases its z-scores are unreliable and
    dominated by the sigma floor.
    """
    return int((valid & (sigma < sigma_warn)).sum().item())
