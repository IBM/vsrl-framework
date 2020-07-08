#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from typing import Optional

import torch


def focal_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-15,
) -> torch.Tensor:
    """
    If y == 1 the loss is
    """
    label1_idx = labels == 1
    not_label1_idx = ~label1_idx
    n_objects = label1_idx.sum()
    not_label1_loss = (
        (1 - labels[not_label1_idx]).pow(beta)
        * scores[not_label1_idx].pow(alpha)
        * torch.log(1 - scores[not_label1_idx] + eps)
    ).sum()
    if n_objects:
        label1_loss = (
            (1 - scores[label1_idx]).pow(alpha) * torch.log(scores[label1_idx] + eps)
        ).sum()
        return -(label1_loss + not_label1_loss) / n_objects
    return -not_label1_loss


def fast_focal_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    is_center: Optional[torch.Tensor] = None,
    n_objects: Optional[torch.Tensor] = None,
    alpha: float = 2,
    beta: float = 4,
    eps: float = 1e-15,
) -> torch.Tensor:
    """
    Compute focal loss faster but with more memory.
    :param scores: NCHW
    :param labels: NCHW
    """
    if is_center is None:
        is_center = labels == 1
    if n_objects is None:
        n_objects = is_center.sum()

    not_center = ~is_center

    one_minus_scores = 1 - scores

    not_label1_loss = (
        not_center
        * scores.pow(alpha)
        * torch.log(one_minus_scores + eps)
        * (1 - labels).pow(beta)
    ).sum()

    if n_objects:
        label1_loss = (
            is_center * one_minus_scores.pow(alpha) * torch.log(scores + eps)
        ).sum()
        return -(label1_loss + not_label1_loss) / n_objects
    return -not_label1_loss


def offset_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    is_center: torch.Tensor,
    n_objects: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    :param preds: N2HW (float32)
    :param labels: N2HW (float32)
    :param is_center: NKHW (torch) for K-way classification
    """
    if n_objects is None:
        n_objects = is_center.sum()
    if not n_objects:
        return 0

    is_center = is_center.any(1).unsqueeze(1)
    # we could omit `* is_center` for labels if we assume they're 0 except for centers
    l1_loss = torch.nn.functional.l1_loss(
        preds * is_center, labels * is_center, reduction="sum"
    )
    return l1_loss / n_objects
