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

Three accumulation modes (controlled by ``proto_stats_mode`` in the adapter
config):

* ``"ema"`` (default without center-awareness) — exponential moving average.
  Each call to :func:`accumulate` first multiplies the existing counts/sums by
  a decay factor (``proto_stats_ema_decay``, default 0.99) before adding the
  new observation.  Recent scores receive more weight, so the reference
  distribution tracks the current center position more closely.

* ``"center_aware"`` (default) — drift-gated validity.  :func:`accumulate`
  maintains an EMA of the center vector at the time of each observation
  (``proto_stats_center_decay``, default 0.95).  :func:`compute_mu_sigma` then
  scales the effective observation count by the cosine similarity between the
  stored center EMA and the current center; prototypes whose center has drifted
  far abstain until their effective count re-crosses ``min_count``.

* ``"ema_center_aware"`` — applies both corrections simultaneously.

Used exclusively by GaussianPatchLevel and _gaussian_score_for_class.
"""
from typing import Dict, Optional, Tuple

import torch


# Scalar accumulator keys (each [K]) — kept together for lockstep operations.
STAT_KEYS = ("score_count", "score_sum", "score_sqsum")
# 2-D key ([K, D]) storing the EMA of center vectors at accumulation time.
# Only present when proto_stats_mode is "center_aware" or "ema_center_aware".
CENTER_KEY = "score_center"


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
    *,
    decay: float = 1.0,
    centers: Optional[torch.Tensor] = None,
    center_decay: float = 0.95,
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
        decay:          EMA decay applied to existing counts/sums *before* adding
                        the new observation (``proto_stats_mode="ema"`` or
                        ``"ema_center_aware"``).  1.0 = no decay.
        centers:        ``[K, D]`` current prototype center vectors.  When
                        provided, a per-prototype center EMA (``CENTER_KEY``) is
                        maintained so :func:`compute_mu_sigma` can detect drift
                        (``proto_stats_mode="center_aware"`` or
                        ``"ema_center_aware"``).
        center_decay:   EMA coefficient for the stored center snapshot.

    Returns:
        Dict of updated tensors (new tensors, not in-place).
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

    if decay < 1.0:
        # EMA: down-weight stale observations before folding in the new one.
        count   = count   * decay + 1.0
        total   = total   * decay + scores
        sq_total = sq_total * decay + scores.pow(2)
    else:
        count    = count    + 1.0
        total    = total    + scores
        sq_total = sq_total + scores.pow(2)

    result: Dict[str, torch.Tensor] = {
        "score_count": count,
        "score_sum":   total,
        "score_sqsum": sq_total,
    }

    if centers is not None:
        centers_f = centers.detach().float()
        norms = centers_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        centers_n = centers_f / norms  # [K, D] unit vectors
        old_snap = state.get(CENTER_KEY)
        if old_snap is None or old_snap.shape != centers_n.shape:
            # First observation or shape mismatch after bank resize: seed with
            # the current center so similarity starts at 1.0.
            result[CENTER_KEY] = centers_n
        else:
            blended = center_decay * old_snap + (1.0 - center_decay) * centers_n
            result[CENTER_KEY] = blended / blended.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    elif CENTER_KEY in state:
        # Pass the existing snapshot through unchanged.
        result[CENTER_KEY] = state[CENTER_KEY]

    return result


def compute_mu_sigma(
    state: dict,
    min_count: int,
    sigma_eps: float = 1e-6,
    *,
    current_centers: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive per-prototype mean, standard deviation and validity mask.

    Args:
        state:           per-class bank dict holding the three accumulators.
        min_count:       a prototype needs this many reference observations before
                         its z-score is trusted. Below it the prototype abstains.
        sigma_eps:       floor applied to sigma by :func:`zscore`; used here only
                         to keep the returned sigma strictly positive.
        current_centers: ``[K, D]`` current center vectors.  When provided *and*
                         the state holds a ``CENTER_KEY`` snapshot, the effective
                         observation count is scaled by the cosine similarity
                         between each prototype's stored center EMA and its
                         current position.  mu and sigma are computed from the
                         raw (unscaled) totals — they remain unaffected as ratios
                         — but validity is gated by the drift-adjusted count.
                         Prototypes whose center has moved far abstain until
                         enough drift-weighted observations re-cross ``min_count``.

    Returns:
        ``(mu, sigma, valid)`` — each ``[K]``; *valid* is bool.
    """
    count    = state["score_count"]
    total    = state["score_sum"]
    sq_total = state["score_sqsum"]

    # ── Drift-aware effective count (center_aware / ema_center_aware) ─────
    if current_centers is not None and CENTER_KEY in state:
        stored = state[CENTER_KEY].float()                       # [K, D]
        curr_f = current_centers.detach().float()                # [K, D]
        curr_n   = curr_f   / curr_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        stored_n = stored   / stored.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sim = (curr_n * stored_n).sum(dim=-1).clamp(min=0.0)    # [K] in [0, 1]
        # Scale count only; mu and sigma are ratios and are unaffected.
        effective_count = count * sim
    else:
        effective_count = count

    safe_count = count.clamp(min=1.0)  # for mu/sigma — use raw count
    mu = total / safe_count

    # Population variance, then Bessel-corrected to the sample variance wherever
    # we have at least two observations. clamp(min=0) guards the catastrophic
    # cancellation that E[x²] - E[x]² suffers when the scores are near-constant.
    var_pop = (sq_total / safe_count - mu.pow(2)).clamp(min=0.0)
    bessel = torch.where(count > 1.0, count / (count - 1.0).clamp(min=1.0),
                         torch.ones_like(count))
    sigma = (var_pop * bessel).sqrt().clamp(min=sigma_eps)

    valid = effective_count >= float(min_count)
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
    reference scores.  The ``CENTER_KEY`` snapshot (if present) is grown in
    lockstep; new entries start as zero vectors so their cosine similarity to
    any real center is 0, forcing the prototype to abstain immediately.
    """
    if not has_stats(state):
        # Legacy/uninitialised bank: start every prototype from zero.
        return init_stats(new_K, device)

    current_K = state[STAT_KEYS[0]].shape[0]
    if current_K == new_K:
        result = {key: state[key] for key in STAT_KEYS}
    elif current_K > new_K:
        # Bank shrank without going through permute(); we cannot tell which
        # prototypes survived, so drop the history rather than misattribute it.
        result = init_stats(new_K, device)
    else:
        pad = torch.zeros(new_K - current_K, device=device)
        result = {key: torch.cat([state[key], pad], dim=0) for key in STAT_KEYS}

    # Propagate the center snapshot when present.
    if CENTER_KEY in state and current_K <= new_K:
        if current_K == new_K:
            result[CENTER_KEY] = state[CENTER_KEY]
        else:
            D = state[CENTER_KEY].shape[1]
            pad_c = torch.zeros(new_K - current_K, D, device=device)
            result[CENTER_KEY] = torch.cat([state[CENTER_KEY], pad_c], dim=0)
    # If current_K > new_K the bank was reset above — no center to propagate.

    return result


def permute(stats: Dict[str, torch.Tensor], keep_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Reindex accumulators to follow a prune/reorder of the prototype bank.

    Must be applied wherever ``centers`` / ``variance`` / ``appearance`` are
    reindexed; skipping it silently attaches every prototype's statistics to a
    different prototype.  The ``CENTER_KEY`` snapshot is reindexed too when
    present.
    """
    result = {key: stats[key][keep_idx] for key in STAT_KEYS if key in stats}
    if CENTER_KEY in stats:
        result[CENTER_KEY] = stats[CENTER_KEY][keep_idx]
    return result


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
