# SPDX-License-Identifier: Apache-2.0
"""Stage 1/2 inference using bundled AA3Di components and the esm dependency."""
from pathlib import Path
import torch
from ..data import Record


class ESM3Runner:
    nondeterministic = False

    def __init__(self, *, base_checkpoint, adapter, direction, precision, device, stage=2):
        self.base_checkpoint = Path(base_checkpoint)
        self.adapter = Path(adapter) if adapter else None
        self.mode = direction
        self.stage = stage
        if stage == 2 and self.adapter is None:
            raise ValueError('Stage 2 requires the matching direction adapter')
        self.precision = precision
        self.device = torch.device(device)
        self.valid_characters = 'ALGVSREDTIPKFQNYMHWC' if direction == 'aa2di' else 'ACDEFGHIKLMNPQRSTVWY'
        self.fusion_mode = direction if stage == 2 else 'aadi'

    def load(self):
        from .esm3_impl.load import load_lora_model
        from .esm3_impl.collate import collate_for_inference
        from .esm3_impl.constants import AA_STANDARD_TOKEN_IDS, THREE_DI_STANDARD_TOKEN_IDS
        from .esm3_impl.decode_utils import constrained_argmax
        self.collate = collate_for_inference
        self.argmax = constrained_argmax
        self.valid_ids = THREE_DI_STANDARD_TOKEN_IDS if self.mode == 'aa2di' else AA_STANDARD_TOKEN_IDS
        self.model, self.aa_tokenizer, self.di_tokenizer, self.meta = load_lora_model(
            self.base_checkpoint, self.adapter, mode=self.fusion_mode, device=self.device)
        self.dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[self.precision]
        self.model.to(dtype=self.dtype)
        self.cost = {'loaded_total_parameters': sum(p.numel() for p in self.model.parameters())}
        if self.adapter:
            blob = torch.load(self.adapter, map_location='cpu', weights_only=False)
            state = blob.get('model_state') or blob
            tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
            lora = sum(v.numel() for k, v in tensors.items() if 'lora_A' in k or 'lora_B' in k)
            saved = sum(v.numel() for v in tensors.values())
            self.cost.update(adapter_saved_parameters=saved, adapter_lora_parameters=lora,
                             adapter_other_parameters=saved-lora,
                             adapter_tensor_bytes=sum(v.numel()*v.element_size() for v in tensors.values()))

    def prepare(self, batch):
        source = [r.sequence for r in batch]
        aa = source if self.mode == 'aa2di' else ['A'*len(s) for s in source]
        di = ['a'*len(s) for s in source] if self.mode == 'aa2di' else source
        return {k: v.to(self.device) for k, v in self.collate(aa, di, self.aa_tokenizer, self.di_tokenizer,
                aa_mask=self.mode == 'di2aa', di_mask=self.mode == 'aa2di').items()}

    @torch.inference_mode()
    def forward(self, prepared):
        with torch.autocast(self.device.type, dtype=self.dtype,
                            enabled=self.device.type == 'cuda' and self.precision != 'fp32'):
            out = self.model(**prepared, fusion_mode=self.fusion_mode)
        return out['three_di_logits' if self.mode == 'aa2di' else 'sequence_logits']

    def decode(self, logits, batch):
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError('Nonfinite logits; change the frozen precision protocol before retrying')
        records = []
        for i, record in enumerate(batch):
            ids = self.argmax(logits[i, 1:len(record.sequence)+1], self.valid_ids).tolist()
            text = self.di_tokenizer.decode(ids, skip_special_tokens=True).upper() if self.mode == 'aa2di' else ''.join(self.aa_tokenizer.convert_ids_to_tokens(ids))
            records.append(Record(record.header, text))
        return records

    def metadata(self):
        return {'base_checkpoint': str(self.base_checkpoint), 'adapter': str(self.adapter) if self.adapter else None,
                'stage': self.stage, 'direction': self.mode, 'fusion_mode': self.fusion_mode,
                'precision': self.precision, 'checkpoint_meta': self.meta, 'parameter_cost': self.cost}
