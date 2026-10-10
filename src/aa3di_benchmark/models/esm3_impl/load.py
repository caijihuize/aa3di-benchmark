# SPDX-License-Identifier: Apache-2.0
"""Model loading helpers shared by CLI and notebook."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from . import ESM3LoRA, ThreeDiTokenizer
from .constants import DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK
from esm.tokenization import EsmSequenceTokenizer

logger = logging.getLogger(__name__)


def peek_adapter_meta(adapter_path: str | Path) -> dict[str, Any]:
    """Read adapter metadata without constructing the model."""
    blob = torch.load(adapter_path, map_location="cpu", weights_only=False)
    meta: dict[str, Any] = {}
    if not isinstance(blob, dict):
        return meta
    meta = dict(blob.get("meta") or {})
    args = blob.get("args") or {}
    if isinstance(args, dict):
        for key in (
            "lora_rank",
            "lora_alpha",
            "lora_qkv",
            "translate_mode",
            "max_len",
            "resume_checkpoint",
        ):
            if key in args and key not in meta:
                meta[key] = args[key]
    return meta


def _load_base_state(model: ESM3LoRA, base_checkpoint: str | Path) -> None:
    logger.info("Loading base checkpoint: %s", base_checkpoint)
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base_state = ckpt.get("model_state") or ckpt.get("model") or ckpt
    base_state = {k.removeprefix("_orig_mod."): v for k, v in base_state.items()}
    base_state = {k.replace("module.", ""): v for k, v in base_state.items()}
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if unexpected:
        logger.warning("Base unexpected keys (sample): %s", list(unexpected)[:10])
    unsupported = [k for k in missing if "lora_A" not in k and "lora_B" not in k]
    if unsupported:
        raise RuntimeError(
            "Base checkpoint omitted required tensors; "
            "check base_checkpoint path"
        )


def load_lora_model(
    base_checkpoint: str | Path | None,
    adapter_path: str | Path | None = None,
    mode: str = "aa2di",
    device: str | torch.device | None = None,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: float = DEFAULT_LORA_ALPHA,
    warn_cli_mismatch: bool = True,
) -> tuple[ESM3LoRA, EsmSequenceTokenizer, ThreeDiTokenizer, dict[str, Any]]:
    """
    Load a Stage 1 base (+ optional LoRA adapter), or the original ESM3 backbone
    for an adapter explicitly trained with ``--skip_stage1``.

    For translation / aa2di / di2aa embeddings, pass the matching adapter.
    For Stage1-only aadi embeddings, set mode="aadi" and adapter_path=None.
    """
    if mode not in ("aa2di", "di2aa", "aadi"):
        raise ValueError(f"Unsupported mode: {mode!r}")
    if adapter_path is None and mode != "aadi":
        raise ValueError(
            f"mode={mode!r} requires an adapter; only mode='aadi' may omit adapter_path"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    meta: dict[str, Any] = {}
    resolved_rank = int(lora_rank)
    resolved_alpha = float(lora_alpha)

    if adapter_path is not None:
        meta = peek_adapter_meta(adapter_path)
        resolved_rank = int(meta.get("lora_rank", lora_rank))
        resolved_alpha = float(meta.get("lora_alpha", lora_alpha))

        if "translate_mode" in meta and meta["translate_mode"] != mode:
            raise ValueError(
                f"Adapter mode={meta['translate_mode']!r} does not match "
                f"requested mode={mode!r}"
            )
        if warn_cli_mismatch:
            if meta.get("lora_rank") is not None and int(meta["lora_rank"]) != int(lora_rank):
                logger.warning(
                    "Requested lora_rank=%s differs from checkpoint=%s; using %s",
                    lora_rank,
                    meta["lora_rank"],
                    resolved_rank,
                )
            if (
                meta.get("lora_alpha") is not None
                and float(meta["lora_alpha"]) != float(lora_alpha)
            ):
                logger.warning(
                    "Requested lora_alpha=%s differs from checkpoint=%s; using %s",
                    lora_alpha,
                    meta["lora_alpha"],
                    resolved_alpha,
                )

    stage1_initialized = meta.get("stage1_initialized")
    if stage1_initialized is False and base_checkpoint is not None:
        raise ValueError(
            "This adapter was trained without Stage 1; do not load a Stage 1 base checkpoint."
        )
    if stage1_initialized is not False and base_checkpoint is None:
        raise ValueError(
            "A Stage 1 base checkpoint is required unless adapter metadata says "
            "stage1_initialized=false."
        )

    logger.info(
        "Loading ESM3LoRA (rank=%s, alpha=%s, mode=%s, adapter=%s)",
        resolved_rank,
        resolved_alpha,
        mode,
        adapter_path is not None,
    )
    t0 = time.time()
    model = ESM3LoRA(
        device=str(device),
        lora_rank=resolved_rank,
        lora_alpha=resolved_alpha,
    )
    if base_checkpoint is None:
        logger.info("Using original ESM3 backbone (no Stage 1 checkpoint)")
    else:
        _load_base_state(model, base_checkpoint)

    if adapter_path is not None:
        logger.info("Loading LoRA adapter: %s", adapter_path)
        model.load_lora_adapter(
            str(adapter_path),
            map_location="cpu",
            expected_mode=mode if mode != "aadi" else None,
            expected_rank=resolved_rank,
            expected_alpha=resolved_alpha,
        )

    model = model.to(device)
    model.eval()
    gate_mode = mode if mode in ("aa2di", "di2aa", "aadi") else "aadi"
    gate_stats = model.fusion.gate_stats(fusion_mode=gate_mode)
    logger.info(
        "Fusion gate [%s]: mean=%.3f, aa_dominant=%.1f%%",
        gate_mode,
        gate_stats["gate_mean"],
        100 * gate_stats["gate_aa_dominant_ratio"],
    )
    logger.info(
        "Loaded in %.2fs | params=%.1fM",
        time.time() - t0,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )
    return model, EsmSequenceTokenizer(), ThreeDiTokenizer(), meta
