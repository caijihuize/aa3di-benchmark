# SPDX-License-Identifier: Apache-2.0
"""Official bidirectional ProstT5 encoder-decoder benchmark adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

from ..data import Record


class ProstT5DecoderRunner:
    method_id = "prostt5_decode_fp16_bs"
    model_name = "ProstT5"
    def __init__(
        self,
        *,
        model_path: str,
        direction: str,
        precision: str,
        device: str,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.mode = direction
        self.valid_characters = "ALGVSREDTIPKFQNYMHWC" if direction == "aa2di" else "ACDEFGHIKLMNPQRSTVWY"
        self.generation = "paper"
        self.precision = precision
        self.device = torch.device(device)
        self.variant = f"decode-paper-{direction}-{precision}-bs1"
        self.nondeterministic = True
        self.model: Any = None
        self.tokenizer: Any = None
        self.bad_words_ids: list[list[int]] = []

    def load(self) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        if self.precision == "fp16":
            self.model.half()
        elif self.precision == "bf16":
            self.model.bfloat16()
        else:
            self.model.float()
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        forbidden = (
            "ARNDBCEQZGHILKMFPSTWYVXOU"
            if self.mode == "aa2di"
            else "acdefghiklmnpqrstvwyXBZ"
        )
        self.bad_words_ids = self.tokenizer(
            [" ".join(forbidden)],
            add_special_tokens=False,
        ).input_ids

    def prepare(self, batch: Sequence[Record]) -> dict[str, Any]:
        if self.mode == "aa2di":
            normalized = [
                record.sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                for record in batch
            ]
            prefix = "<AA2fold>"
        else:
            normalized = [record.sequence.lower() for record in batch]
            prefix = "<fold2AA>"
        prefixed = [f"{prefix} {' '.join(sequence)}" for sequence in normalized]
        tokenized = self.tokenizer.batch_encode_plus(
            prefixed,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized.input_ids.to(self.device),
            "attention_mask": tokenized.attention_mask.to(self.device),
            "min_length": min(len(record.sequence) for record in batch) + 1,
            "max_length": max(len(record.sequence) for record in batch) + 1,
        }

    @torch.inference_mode()
    def forward(self, prepared: dict[str, Any]) -> torch.Tensor:
        generation_kwargs: dict[str, Any] = (
            {
                "do_sample": True,
                "num_beams": 3,
                "top_p": 0.95,
                "temperature": 1.2,
                "top_k": 6,
                "repetition_penalty": 1.2,
                "early_stopping": True,
                "length_penalty": 1.0,
            }
            if self.mode == "aa2di"
            else {
                "do_sample": True,
                "top_p": 0.85,
                "temperature": 1.0,
                "top_k": 3,
                "repetition_penalty": 1.2,
            }
        )
        return self.model.generate(
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            min_length=prepared["min_length"],
            max_length=prepared["max_length"],
            bad_words_ids=self.bad_words_ids,
            num_return_sequences=1,
            **generation_kwargs,
        )

    def decode(self, generated: torch.Tensor, batch: Sequence[Record]) -> list[Record]:
        sequences = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        predictions: list[Record] = []
        for record, sequence in zip(batch, sequences):
            decoded = sequence.replace(" ", "").upper()
            predictions.append(Record(record.header, decoded))
        return predictions

    def metadata(self) -> dict[str, Any]:
        return {
            "loaded_total_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_path": str(self.model_path),
            "generation": self.generation,
            "precision": self.precision,
            "direction": self.mode,
        }
