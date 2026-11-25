"""Single-view supervised contrastive loss on pooled mention vectors.

This module does not load a model or change batching. The caller supplies one
vector and one TRAIN label per mention. The classifier may keep using the
original vectors; normalization here is only for the contrastive objective.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def supervised_contrastive_loss(
    features: Tensor, labels: Tensor, *, temperature: float,
) -> Tensor:
    """Return mean SupCon loss over anchors having a same-label partner.

    ``features`` has shape [batch, dimension]; ``labels`` has shape [batch]
    and contains integer 0/1 labels. A positive is another mention with the
    same label. The denominator includes all other mentions, never the anchor
    itself. Average log probabilities over positives, then over valid anchors.

    An anchor with no positive is omitted from this average, but can still
    receive gradients as a comparison for other anchors. Its CE loss is the
    caller's responsibility. If no anchor has a positive, return a differentiable
    zero. No extra temperature multiplier or projection head is used.

    Temperature is explicit: this function does not choose an experimental
    hyperparameter. Half-precision inputs are promoted to float32 for stability.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and strictly positive")
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must have nonempty shape [batch, dimension]")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("features must contain finite floating-point values")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must have shape [batch], matching features")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("labels must be integer binary labels (0 or 1)")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("labels must contain only 0 (LITERAL) or 1 (METONYMIC)")
    if labels.device != features.device:
        raise ValueError("features and labels must be on the same device")

    # 1. Find same-label partners, excluding each mention's self-comparison.
    self_pairs = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_pairs = (labels[:, None] == labels[None, :]) & ~self_pairs
    positive_counts = positive_pairs.sum(dim=1)
    valid_anchors = positive_counts > 0

    values = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
    if not valid_anchors.any():
        return values.sum() * 0.0

    # 2. Unit-length vectors make dot products cosine similarities.
    normalized = F.normalize(values, p=2, dim=1)
    similarities = (normalized @ normalized.T) / temperature

    # 3. Normalize over every OTHER mention using stable log probabilities.
    logits = similarities.masked_fill(self_pairs, float("-inf"))
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # 4. Average positive log probabilities. masked_fill avoids 0 * -inf.
    positive_log_probabilities = log_probabilities.masked_fill(~positive_pairs, 0.0)
    per_anchor_loss = -positive_log_probabilities.sum(dim=1) / positive_counts.clamp_min(1)
    return per_anchor_loss[valid_anchors].mean()
