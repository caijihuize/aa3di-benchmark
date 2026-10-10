# SPDX-License-Identifier: Apache-2.0
"""Strict FASTA IO shared by preparation and evaluation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

@dataclass(frozen=True)
class Record:
    header: str
    sequence: str

    @property
    def identifier(self) -> str:
        return self.header.removeprefix(">").split()[0]


def read_fasta(path: str | Path) -> list[Record]:
    records: list[Record] = []
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(Record(header, "".join(chunks)))
                header, chunks = line, []
            else:
                if header is None:
                    raise ValueError(f"Sequence before FASTA header in {path}")
                chunks.append(line)
    if header is not None:
        records.append(Record(header, "".join(chunks)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    identifiers = [record.identifier for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate FASTA identifiers in {path}")
    return records


def write_fasta(records: Sequence[Record], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            header = record.header if record.header.startswith(">") else f">{record.header}"
            handle.write(f"{header}\n{record.sequence}\n")
