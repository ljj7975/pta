"""Classification metrics and entropy utilities."""

import torch
import numpy as np


def cls_acc(output: torch.Tensor, target: torch.Tensor, topk: int = 1) -> float:
    """Compute classification accuracy (percentage, 0–100)."""
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of a softmax distribution."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def avg_entropy(outputs: torch.Tensor) -> torch.Tensor:
    """Log-average of per-sample log-probabilities (used for AugMix selection)."""
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)
