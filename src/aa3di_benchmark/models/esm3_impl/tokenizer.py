# SPDX-License-Identifier: Apache-2.0
"""
3Di Tokenizer: 字符级分词器，与 ESM3 SequenceTokenizer 设计一致。

词汇表 (vocab_size=64):
    ID 0: <pad>
    ID 1: <eos>
    ID 2: <unk>
    ID 3-22: a l g v s r e d t i p k f q n y m h w c  (20 种 3Di 状态)
    ID 23: <mask>
    ID 24-63: (预留)
"""

from typing import Sequence

import torch

# 3Di 词汇表 (20 种)
THREE_DI_VOCAB = [
    "a", "l", "g", "v", "s", "r", "e", "d", "t", "i",
    "p", "k", "f", "q", "n", "y", "m", "h", "w", "c",
]
THREE_DI_STANDARD_TOKEN_IDS = tuple(range(3, 3 + len(THREE_DI_VOCAB)))

# 特殊 Token
THREE_DI_PAD_TOKEN = "<pad>"
THREE_DI_EOS_TOKEN = "<eos>"
THREE_DI_UNK_TOKEN = "<unk>"
THREE_DI_MASK_TOKEN = "<mask>"

THREE_DI_PAD_TOKEN_ID = 0
THREE_DI_EOS_TOKEN_ID = 1
THREE_DI_UNK_TOKEN_ID = 2
THREE_DI_MASK_TOKEN_ID = 23


class ThreeDiTokenizer:
    """
    字符级 3Di 分词器。
    将 3Di 字符串与 token IDs 互相转换。
    """

    def __init__(self):
        self._token_to_id: dict[str, int] = {
            THREE_DI_PAD_TOKEN: THREE_DI_PAD_TOKEN_ID,
            THREE_DI_EOS_TOKEN: THREE_DI_EOS_TOKEN_ID,
            THREE_DI_UNK_TOKEN: THREE_DI_UNK_TOKEN_ID,
        }
        for i, token in enumerate(THREE_DI_VOCAB):
            self._token_to_id[token] = i + 3
        self._token_to_id[THREE_DI_MASK_TOKEN] = THREE_DI_MASK_TOKEN_ID

        self._id_to_token: dict[int, str] = {
            v: k for k, v in self._token_to_id.items()
        }
        for i in range(24, 64):
            self._id_to_token[i] = f"<reserved_{i}>"

    @property
    def vocab_size(self) -> int:
        return 64

    @property
    def pad_token_id(self) -> int:
        return THREE_DI_PAD_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        return THREE_DI_EOS_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return THREE_DI_UNK_TOKEN_ID

    @property
    def mask_token_id(self) -> int:
        return THREE_DI_MASK_TOKEN_ID

    @property
    def mask_token(self) -> str:
        return THREE_DI_MASK_TOKEN

    @property
    def special_token_ids(self) -> list[int]:
        return [
            THREE_DI_PAD_TOKEN_ID,
            THREE_DI_EOS_TOKEN_ID,
            THREE_DI_UNK_TOKEN_ID,
            THREE_DI_MASK_TOKEN_ID,
        ]

    @property
    def standard_token_ids(self) -> list[int]:
        return list(THREE_DI_STANDARD_TOKEN_IDS)

    def encode(
        self, sequence: str, add_special_tokens: bool = True
    ) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(THREE_DI_PAD_TOKEN_ID)

        for char in sequence:
            if char in self._token_to_id:
                ids.append(self._token_to_id[char])
            else:
                ids.append(THREE_DI_UNK_TOKEN_ID)

        if add_special_tokens:
            ids.append(THREE_DI_EOS_TOKEN_ID)

        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special_ids = {
            THREE_DI_PAD_TOKEN_ID,
            THREE_DI_EOS_TOKEN_ID,
            THREE_DI_UNK_TOKEN_ID,
            THREE_DI_MASK_TOKEN_ID,
        }
        chars = []
        for tid in ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self._id_to_token:
                token = self._id_to_token[tid]
                if token.startswith("<reserved_"):
                    chars.append("?")
                else:
                    chars.append(token)
            else:
                chars.append("?")
        return "".join(chars)
