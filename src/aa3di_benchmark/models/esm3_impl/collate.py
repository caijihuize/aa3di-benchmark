# SPDX-License-Identifier: Apache-2.0
import torch
from esm.tokenization import EsmSequenceTokenizer
from .tokenizer import ThreeDiTokenizer, THREE_DI_MASK_TOKEN_ID

def collate_for_inference(
    aa_strs: list[str],
    di_strs: list[str] | None,
    aa_tokenizer: EsmSequenceTokenizer,
    di_tokenizer: ThreeDiTokenizer,
    aa_mask: bool = False,
    di_mask: bool = False,
) -> dict[str, torch.Tensor]:
    batch_size = len(aa_strs)
    batch_seq_ids: list[list[int]] = []
    batch_di_ids: list[list[int]] = []
    max_seq_len = 0

    for index in range(batch_size):
        aa_ids = aa_tokenizer(
            aa_strs[index],
            add_special_tokens=True,
            padding=False,
            return_tensors=None,
        )["input_ids"]
        di_str = (di_strs[index] if di_strs else "").lower()
        di_ids = di_tokenizer.encode(di_str, add_special_tokens=True)
        batch_seq_ids.append(aa_ids)
        batch_di_ids.append(di_ids)
        max_seq_len = max(max_seq_len, len(aa_ids), len(di_ids))

    seq_tokens = torch.full(
        (batch_size, max_seq_len), aa_tokenizer.pad_token_id, dtype=torch.long
    )
    di_tokens = torch.full(
        (batch_size, max_seq_len), di_tokenizer.pad_token_id, dtype=torch.long
    )
    seq_id = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    for index in range(batch_size):
        length_s = len(batch_seq_ids[index])
        length_d = len(batch_di_ids[index])
        seq_tokens[index, :length_s] = torch.tensor(
            batch_seq_ids[index], dtype=torch.long
        )
        di_tokens[index, :length_d] = torch.tensor(
            batch_di_ids[index], dtype=torch.long
        )
        seq_id[index, : max(length_s, length_d)] = 1

    if aa_mask:
        mask_id = aa_tokenizer.mask_token_id
        for index in range(batch_size):
            length = int(seq_id[index].sum().item())
            for position in range(1, length - 1):
                seq_tokens[index, position] = mask_id

    if di_mask:
        for index in range(batch_size):
            length = int(seq_id[index].sum().item())
            for position in range(1, length - 1):
                di_tokens[index, position] = THREE_DI_MASK_TOKEN_ID

    return {
        "sequence_tokens": seq_tokens,
        "three_di_tokens": di_tokens,
        "sequence_id": seq_id,
    }
