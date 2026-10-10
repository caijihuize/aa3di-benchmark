# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible ESM3 AA–3Di backbone, fusion gates and LoRA."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .constants import DEFAULT_LORA_ALPHA, DEFAULT_LORA_RANK
from .lora import count_lora_params, inject_esm3_block_lora
from .tokenizer import ThreeDiTokenizer

from esm.layers.regression_head import RegressionHead
from esm.utils.structure.affine3d import Affine3D, RotationMatrix


# ============================================================
#  复用 MultiTaskFusion（与原始项目一致）
# ============================================================

class MultiTaskFusion(nn.Module):
    """
    多任务按位加权融合模块。
    为 aadi / aa2di / di2aa 三个任务分别维护独立的 gate。
    """

    MODES = ["aadi", "aa2di", "di2aa"]

    def __init__(self, d_model: int, init_bias: float = 0.0):
        super().__init__()
        for mode in self.MODES:
            setattr(self, f"raw_gate_{mode}",
                    nn.Parameter(torch.full((d_model,), init_bias)))

    def forward(self, seq_embed, di_embed, fusion_mode="aadi"):
        if fusion_mode not in self.MODES:
            raise ValueError(
                f"Unknown fusion_mode: {fusion_mode!r}; expected one of {self.MODES}"
            )
        raw_gate = getattr(self, f"raw_gate_{fusion_mode}")
        gate = torch.sigmoid(raw_gate)
        return gate * seq_embed + (1.0 - gate) * di_embed

    def gate_values(self, fusion_mode=None):
        if fusion_mode is not None:
            return torch.sigmoid(getattr(self, f"raw_gate_{fusion_mode}")).detach()
        return {m: torch.sigmoid(getattr(self, f"raw_gate_{m}")).detach() for m in self.MODES}

    def gate_stats(self, fusion_mode=None):
        def _stats(g):
            return {
                "gate_mean": g.mean().item(),
                "gate_std": g.std().item(),
                "gate_min": g.min().item(),
                "gate_max": g.max().item(),
                "gate_aa_dominant_ratio": (g > 0.5).float().mean().item(),
            }
        if fusion_mode is not None:
            return _stats(self.gate_values(fusion_mode))
        return {m: _stats(g) for m, g in self.gate_values().items()}


# ============================================================
#  主模型 (LoRA)
# ============================================================

class ESM3LoRA(nn.Module):
    """
    ESM3-SM backbone + LoRA 适配的 AA↔3Di 模型。

    训练策略：
      - transformer backbone: 冻结
      - LoRA adapter (A/B matrices): 可训练
      - three_di_embed, three_di_head, sequence_head: 可训练
      - MultiTaskFusion: 可训练

    Args:
        lora_rank: LoRA 秩 (默认 16)
        lora_alpha: LoRA 缩放系数 (默认 32)
        lora_dropout: LoRA dropout (默认 0.0)
        lora_target_modules: 保留参数；实际注入见 inject_esm3_block_lora
        lora_apply_ffn: 是否对 FFN 也应用 LoRA (默认 True)
        gradient_checkpointing: 启用梯度检查点
        fusion_init_bias: MultiTaskFusion 初始偏置
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_alpha: float = DEFAULT_LORA_ALPHA,
        lora_dropout: float = 0.0,
        lora_target_modules: Sequence[str] = ("layernorm_qkv", "out_proj"),
        lora_apply_ffn: bool = True,
        gradient_checkpointing: bool = False,
        fusion_init_bias: float = 0.0,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_apply_ffn = lora_apply_ffn

        # Reconstruct the ESM3-SM AA backbone locally; all base tensors must
        # be provided by the selected Stage 1 checkpoint (no network download).
        from esm.layers.transformer_stack import TransformerStack
        d_model = 1536
        self.sequence_embed = nn.Embedding(64, d_model).to(device)
        self.transformer = TransformerStack(
            d_model, 24, 256, 48, mask_and_zero_frameless=True).to(device)
        self.sequence_head = RegressionHead(d_model, 64).to(device)

        backbone_device = next(self.sequence_embed.parameters()).device
        backbone_dtype = self.sequence_embed.weight.dtype

        # ── 2. 新增 3Di 组件 ──
        self.three_di_embed = nn.Embedding(64, d_model).to(
            device=backbone_device, dtype=backbone_dtype)
        self.three_di_head = RegressionHead(d_model, 64).to(
            device=backbone_device, dtype=backbone_dtype)

        # ── 3. 多任务融合门 ──
        self.fusion = MultiTaskFusion(d_model, init_bias=fusion_init_bias).to(
            device=backbone_device, dtype=backbone_dtype)

        # 初始化 3Di embed
        with torch.no_grad():
            w = self.sequence_embed.weight
            if w.device.type != "meta":
                self.three_di_embed.weight.normal_(w.mean().item(), w.std().item())

        # ── 4. 冻结 backbone + 注入 LoRA ──
        self._freeze_backbone()
        self._inject_lora(
            rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        # ── 5. 统计 ──
        lora_params = count_lora_params(self)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[ESM3LoRA] LoRA rank={lora_rank}, alpha={lora_alpha}")
        print(f"  Total params:  {total:>12,}")
        print(f"  LoRA + new:    {trainable:>12,} ({trainable/total*100:.2f}%)")
        print(f"  LoRA only:     {lora_params:>12,} ({lora_params/total*100:.2f}%)")

    # ──────────── 冻结 ────────────

    def _freeze_backbone(self):
        """冻结 transformer backbone + embeddings (默认策略)."""
        for p in self.transformer.parameters():
            p.requires_grad = False
        # sequence_embed 也冻结（LoRA 不应用于 embedding）
        for p in self.sequence_embed.parameters():
            p.requires_grad = False

    def configure_for_mode(self, mode: str):
        """
        按翻译方向冻结无关 head / fusion gate。

        aa2di: 训练 three_di_head + raw_gate_aa2di
        di2aa: 训练 sequence_head + raw_gate_di2aa
        aadi:  两个 head 与三个 gate 均可训练
        """
        if mode not in MultiTaskFusion.MODES:
            raise ValueError(f"Unknown mode: {mode!r}; expected one of {MultiTaskFusion.MODES}")

        train_seq_head = mode in ("aadi", "di2aa")
        train_di_head = mode in ("aadi", "aa2di")
        for p in self.sequence_head.parameters():
            p.requires_grad = train_seq_head
        for p in self.three_di_head.parameters():
            p.requires_grad = train_di_head
        for p in self.three_di_embed.parameters():
            p.requires_grad = True

        for gate_mode in MultiTaskFusion.MODES:
            getattr(self.fusion, f"raw_gate_{gate_mode}").requires_grad = (
                gate_mode == mode or mode == "aadi"
            )

    # ──────────── LoRA 注入 ────────────

    def _inject_lora(self, rank=8, alpha=16, dropout=0.0, target_modules=None):
        """
        对 transformer 中所有匹配 target_modules 的 Linear 层注入 LoRA。
        """
        for name, child in self.transformer.named_children():
            if "blocks" in name or name == "blocks":
                for block in child:
                    self._inject_lora_to_block(block, rank, alpha, dropout, target_modules)

    def _inject_lora_to_block(self, block, rank, alpha, dropout, target_modules):
        """对单个 UnifiedTransformerBlock 注入 LoRA（含融合 QKV Linear）。"""
        del target_modules  # QKV 必须打进 Sequential 的 Linear(\"1\"), 不按此列表匹配
        inject_esm3_block_lora(
            block,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            apply_ffn=getattr(self, "lora_apply_ffn", True),
        )

    # ──────────── 前向 ────────────

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """兼容旧版单 gate checkpoint + LoRA key 映射."""
        # 1) 旧版单 gate → 3 gate
        if "fusion.raw_gate" in state_dict and not any(
            k.startswith("fusion.raw_gate_") for k in state_dict
        ):
            old_gate = state_dict.pop("fusion.raw_gate")
            for mode in MultiTaskFusion.MODES:
                state_dict[f"fusion.raw_gate_{mode}"] = old_gate.clone()

        # 2) LoRA key 映射: checkpoint 中 key = "blocks.0.attn.out_proj.weight"
        #    模型实际 key = "transformer.blocks.0.attn.out_proj.original.weight"
        #    需要将 checkpoint key 映射到带 .original 的 key
        model_keys = set(dict(self.named_parameters()).keys())
        mapping = {}
        for ckpt_key in list(state_dict.keys()):
            # 检查是否因 LoRA 包装导致 key 不匹配
            # 例: 模型有 "transformer.blocks.0.attn.out_proj.original.weight"
            #     checkpoint 有 "transformer.blocks.0.attn.out_proj.weight"
            #     插入 .original → "transformer.blocks.0.attn.out_proj.original.weight"
            if ckpt_key not in model_keys:
                # 尝试在倒数第二个 '.' 前插入 '.original'
                parts = ckpt_key.rsplit(".", 1)
                if len(parts) == 2:
                    alt_key = f"{parts[0]}.original.{parts[1]}"
                    if alt_key in model_keys:
                        mapping[ckpt_key] = alt_key

        for old_k, new_k in mapping.items():
            state_dict[new_k] = state_dict.pop(old_k)

        if mapping:
            print(f"[LoRA key map] 映射了 {len(mapping)} 个 checkpoint key → LoRA-wrapped keys")

        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        sequence_tokens: torch.Tensor,
        three_di_tokens: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        fusion_mode: str = "aadi",
        active_heads: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            sequence_tokens: (B, L) AA token IDs
            three_di_tokens:  (B, L) 3Di token IDs
            sequence_id:      (B, L) attention mask
            fusion_mode:      aadi / aa2di / di2aa
            active_heads:     optional subset of {"sequence", "three_di"};
                              default depends on fusion_mode

        Returns:
            dict with requested logits and embedding
        """
        if fusion_mode not in MultiTaskFusion.MODES:
            raise ValueError(
                f"Unknown fusion_mode: {fusion_mode!r}; "
                f"expected one of {MultiTaskFusion.MODES}"
            )
        if active_heads is None:
            if fusion_mode == "aa2di":
                active_heads = ("three_di",)
            elif fusion_mode == "di2aa":
                active_heads = ("sequence",)
            else:
                active_heads = ("sequence", "three_di")

        # 1. Embedding
        seq_embed = self.sequence_embed(sequence_tokens)
        di_embed = self.three_di_embed(three_di_tokens)

        # 2. Fusion (按模式选择对应 gate)
        x = self.fusion(seq_embed, di_embed, fusion_mode=fusion_mode)

        # 3. Transformer (identity affine → 跳过 geometric attention)
        B, L = sequence_tokens.shape
        affine_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else x.dtype
        )
        identity_affine = Affine3D.identity(
            (B, L), rotation_type=RotationMatrix,
            dtype=affine_dtype, device=x.device,
        )
        identity_mask = torch.zeros((B, L), dtype=torch.bool, device=x.device)

        if self.gradient_checkpointing and self.training:
            transformer_output = torch.utils.checkpoint.checkpoint(
                self.transformer, x, sequence_id, identity_affine, identity_mask, None,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            transformer_output = self.transformer(
                x, sequence_id,
                affine=identity_affine, affine_mask=identity_mask, chain_id=None,
            )

        x, embedding = transformer_output[:2]

        # 4. Prediction heads (only requested ones)
        output = {"embedding": embedding}
        if "sequence" in active_heads:
            output["sequence_logits"] = self.sequence_head(x)
        if "three_di" in active_heads:
            output["three_di_logits"] = self.three_di_head(x)
        return output

    # ──────────── 加载 adapter ────────────

    def load_lora_adapter(
        self,
        path: str,
        map_location="cpu",
        expected_mode: str | None = None,
        expected_rank: int | None = None,
        expected_alpha: float | None = None,
        require_trainable_keys: bool = True,
    ):
        """加载 LoRA 权重，并尽量校验元数据与关键 key。

        兼容两种格式:
          - 直接权重 dict (无 meta 包裹的旧格式)
          - 含 model_state / args / meta 的 checkpoint (save_lora_checkpoint 产物)
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        meta = {}
        if isinstance(ckpt, dict):
            meta = dict(ckpt.get("meta") or {})
            args = ckpt.get("args") or {}
            if isinstance(args, dict):
                for key in (
                    "lora_rank", "lora_alpha", "translate_mode",
                    "max_len", "resume_checkpoint",
                ):
                    if key in args and key not in meta:
                        meta[key] = args[key]
            state = ckpt.get("model_state") or ckpt
        else:
            state = ckpt

        if expected_mode is not None:
            ckpt_mode = meta.get("translate_mode")
            if ckpt_mode is not None and ckpt_mode != expected_mode:
                raise ValueError(
                    f"Adapter mode mismatch: checkpoint={ckpt_mode!r}, "
                    f"expected={expected_mode!r}"
                )
        if expected_rank is not None and meta.get("lora_rank") is not None:
            if int(meta["lora_rank"]) != int(expected_rank):
                raise ValueError(
                    f"Adapter lora_rank mismatch: checkpoint={meta['lora_rank']}, "
                    f"expected={expected_rank}"
                )
        if expected_alpha is not None and meta.get("lora_alpha") is not None:
            if float(meta["lora_alpha"]) != float(expected_alpha):
                raise ValueError(
                    f"Adapter lora_alpha mismatch: checkpoint={meta['lora_alpha']}, "
                    f"expected={expected_alpha}"
                )

        missing, unexpected = self.load_state_dict(state, strict=False)
        n_lora = sum(1 for key in state if "lora_A" in key or "lora_B" in key)
        if require_trainable_keys and n_lora == 0:
            raise RuntimeError(f"Adapter contains no LoRA parameters: {path}")
        if unexpected:
            print(f"  [INFO] Unexpected keys (ignored): {unexpected}")
        # Missing backbone / frozen-head keys are expected for adapter-only files.
        print(f"[LoRA] Loaded adapter ({len(state)} keys, {n_lora} LoRA matrices) from {path}")
        if meta:
            print(f"  meta: {meta}")
        return meta

    # ──────────── 信息 ────────────

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_info(self):
        lora_n = count_lora_params(self)
        total = self.num_total_params
        trainable = self.num_trainable_params
        print(f"[ESM3LoRA]")
        print(f"  Total params:        {total:>12,}")
        print(f"  Trainable (LoRA+new): {trainable:>12,} ({trainable/total*100:.2f}%)")
        print(f"  LoRA only:           {lora_n:>12,} ({lora_n/total*100:.2f}%)")
        all_stats = self.fusion.gate_stats()
        for mode, stats in all_stats.items():
            print(f"  Fusion gate [{mode:>5}]: mean={stats['gate_mean']:.3f}, "
                  f"aa_dominant={stats['gate_aa_dominant_ratio']:.1%}")
