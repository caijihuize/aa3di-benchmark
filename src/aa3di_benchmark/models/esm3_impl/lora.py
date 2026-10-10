# SPDX-License-Identifier: Apache-2.0
"""
LoRA (Low-Rank Adaptation) 模块。

在预训练 Linear 层旁插入低秩矩阵 B·A，
仅训练 B 和 A，冻结原始权重。

  x → W_orig(x) + α/r · B(A(x))
        冻结       ↓
                可训练 (rank=r)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
#  LoRA Linear
# ══════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    对原始 Linear 层添加低秩适配。

    Args:
        original: 预训练的 Linear 层（将被冻结）
        rank: LoRA 秩
        alpha: 缩放系数 (实际缩放 = alpha / rank)
        dropout: dropout 概率
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)  # 冻结原始权重
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features
        device = original.weight.device
        dtype = original.weight.dtype

        # LoRA 低秩矩阵: A (down), B (up), 与原始权重同设备同精度
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化: A ~ kaiming uniform, B = 0 (初始等效于原始权重)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        output = W_orig(x) + scaling * B · A(x)
        """
        result = self.original(x)
        # LoRA 分支: x → dropout → A^T → B^T (支持 3D 输入)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling
        return result

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, "
            f"scaling={self.scaling:.3f}, "
            f"orig_weight={list(self.original.weight.shape)}"
        )


# ══════════════════════════════════════════════════════════════════
#  LoRA Sequential (方便批量替换)
# ══════════════════════════════════════════════════════════════════

def replace_linear_with_lora(
    module: nn.Module,
    target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
) -> None:
    """
    递归替换 module 中所有匹配 target_modules 的 Linear 为 LoRALinear。
    原地修改（in-place）。

    Args:
        module: 要替换的模块（如 transformer）
        target_modules: 要应用 LoRA 的 Linear 层名称后缀
        rank: LoRA 秩
        alpha: 缩放系数
        dropout: dropout 概率
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(name.endswith(t) for t in target_modules):
            lora_layer = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, name, lora_layer)
        else:
            replace_linear_with_lora(child, target_modules, rank, alpha, dropout)


def count_lora_params(module: nn.Module) -> int:
    """统计 module 中所有 LoRA 参数（lora_A + lora_B）."""
    return sum(
        p.numel() for name, p in module.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )


def inject_esm3_block_lora(
    block: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    apply_ffn: bool = True,
) -> None:
    """Inject LoRA into one ESM3 UnifiedTransformerBlock.

    ESM3 fused QKV is ``attn.layernorm_qkv = Sequential(LayerNorm, Linear)``.
    The Linear is named ``\"1\"``, so matching ``layernorm_qkv`` on ``attn``
    never hits it. This helper wraps that Linear plus ``out_proj`` and FFN.
    """
    attn = getattr(block, "attn", None)
    if attn is not None:
        replace_linear_with_lora(
            attn, target_modules=("out_proj",), rank=rank, alpha=alpha, dropout=dropout
        )
        qkv = getattr(attn, "layernorm_qkv", None)
        if qkv is not None:
            replace_linear_with_lora(
                qkv, target_modules=("1",), rank=rank, alpha=alpha, dropout=dropout
            )
    if apply_ffn and hasattr(block, "ffn"):
        replace_linear_with_lora(
            block.ffn, target_modules=("1", "3"), rank=rank, alpha=alpha, dropout=dropout
        )
