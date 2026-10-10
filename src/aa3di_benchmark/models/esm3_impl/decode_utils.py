# SPDX-License-Identifier: Apache-2.0
"""Shared decoding helpers for training validation and inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def constrained_argmax(
    logits: torch.Tensor,
    valid_token_ids: Sequence[int],
) -> torch.Tensor:
    """Argmax over a restricted token set, returning global token IDs."""
    if not valid_token_ids:
        raise ValueError("valid_token_ids must not be empty")
    valid_ids = torch.as_tensor(
        list(valid_token_ids), dtype=torch.long, device=logits.device
    )
    local_indices = logits.index_select(-1, valid_ids).argmax(dim=-1)
    return valid_ids[local_indices]


def masked_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_token_ids: Sequence[int] | None = None,
    ignore_index: int = -100,
) -> float:
    """Token accuracy on labeled positions, optionally with constrained argmax."""
    mask = labels != ignore_index
    if not mask.any():
        return 0.0
    if valid_token_ids is None:
        preds = logits.argmax(dim=-1)
    else:
        preds = constrained_argmax(logits, valid_token_ids)
    return (preds[mask] == labels[mask]).float().mean().item()
