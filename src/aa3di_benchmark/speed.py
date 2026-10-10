# SPDX-License-Identifier: Apache-2.0
"""Shared, model-agnostic utilities for inference speed benchmarks."""

from __future__ import annotations

import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

try:
    import torch
except ModuleNotFoundError:  # Dataset preparation also runs in torch-free base envs.
    torch = None  # type: ignore[assignment]


from .data import Record, read_fasta, write_fasta


def make_batches(
    records: Sequence[Record],
    *,
    batch_size: int,
    max_residues: int = 0,
    sort_by_length: bool = True,
) -> list[list[Record]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ordered = list(records)
    if sort_by_length:
        ordered.sort(key=lambda record: (len(record.sequence), record.identifier))
    batches: list[list[Record]] = []
    current: list[Record] = []
    current_residues = 0
    for record in ordered:
        length = len(record.sequence)
        exceeds_tokens = max_residues > 0 and current and current_residues + length > max_residues
        if current and (len(current) >= batch_size or exceeds_tokens):
            batches.append(current)
            current, current_residues = [], 0
        current.append(record)
        current_residues += length
    if current:
        batches.append(current)
    return batches


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "")),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
        )
    return metadata


def evaluate_predictions(
    predictions: Sequence[Record],
    references: Sequence[Record],
    *,
    valid_characters: str,
) -> dict[str, Any]:
    pred_by_id = {record.identifier: record.sequence.upper() for record in predictions}
    ref_by_id = {record.identifier: record.sequence.upper() for record in references}
    common = sorted(pred_by_id.keys() & ref_by_id.keys())
    equal_length = [identifier for identifier in common if len(pred_by_id[identifier]) == len(ref_by_id[identifier])]
    correct = 0
    total = 0
    exact = 0
    for identifier in equal_length:
        prediction = pred_by_id[identifier]
        reference = ref_by_id[identifier]
        correct += sum(a == b for a, b in zip(prediction, reference))
        total += len(reference)
        exact += int(prediction == reference)
    valid = set(valid_characters.upper())
    invalid_tokens = sum(character not in valid for identifier in common for character in pred_by_id[identifier])
    return {
        "n_predictions": len(pred_by_id),
        "n_references": len(ref_by_id),
        "n_common": len(common),
        "n_missing": len(ref_by_id.keys() - pred_by_id.keys()),
        "n_extra": len(pred_by_id.keys() - ref_by_id.keys()),
        "n_length_mismatch": len(common) - len(equal_length),
        "n_equal_length": len(equal_length),
        "micro_accuracy_equal_length": correct / total if total else None,
        "exact_match_equal_length": exact / len(equal_length) if equal_length else None,
        "evaluated_residues": total,
        "invalid_tokens": invalid_tokens,
    }


def run_benchmark(
    *,
    runner: Any,
    records: Sequence[Record],
    output_dir: str | Path,
    batch_size: int,
    max_residues: int,
    warmup_batches: int,
    repeats: int,
    load_seconds: float,
    references: Sequence[Record] | None = None,
    valid_characters: str = "",
    sort_by_length: bool = True,
    shard_rank: int | None = None,
    shard_world_size: int | None = None,
) -> dict[str, Any]:
    """Benchmark runner.prepare/forward/decode without timed file I/O."""
    if torch is None:
        raise RuntimeError("PyTorch is required to run an inference benchmark")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(runner.device)
    batches = make_batches(
        records,
        batch_size=batch_size,
        max_residues=max_residues,
        sort_by_length=sort_by_length,
    )

    for batch in batches[: max(0, warmup_batches)]:
        prepared = runner.prepare(batch)
        raw = runner.forward(prepared)
        runner.decode(raw, batch)
    _sync(device)

    batch_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    canonical_predictions: list[Record] | None = None
    for repeat in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        repeat_start = time.perf_counter()
        repeat_predictions: list[Record] = []
        pre_total = gpu_total = post_total = 0.0
        for batch_index, batch in enumerate(batches):
            pre_start = time.perf_counter()
            prepared = runner.prepare(batch)
            _sync(device)
            pre_seconds = time.perf_counter() - pre_start

            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                raw = runner.forward(prepared)
                end_event.record()
                end_event.synchronize()
                gpu_seconds = start_event.elapsed_time(end_event) / 1000.0
            else:
                gpu_start = time.perf_counter()
                raw = runner.forward(prepared)
                gpu_seconds = time.perf_counter() - gpu_start

            post_start = time.perf_counter()
            decoded = runner.decode(raw, batch)
            _sync(device)
            post_seconds = time.perf_counter() - post_start
            repeat_predictions.extend(decoded)
            pre_total += pre_seconds
            gpu_total += gpu_seconds
            post_total += post_seconds
            n_residues = sum(len(record.sequence) for record in batch)
            batch_rows.append(
                {
                    "repeat": repeat,
                    "batch_index": batch_index,
                    "n_sequences": len(batch),
                    "n_residues": n_residues,
                    "min_length": min(len(record.sequence) for record in batch),
                    "max_length": max(len(record.sequence) for record in batch),
                    "preprocess_seconds": pre_seconds,
                    "gpu_seconds": gpu_seconds,
                    "postprocess_seconds": post_seconds,
                    "compute_seconds": pre_seconds + gpu_seconds + post_seconds,
                    "shard_rank": shard_rank,
                }
            )
            del prepared, raw, decoded

        _sync(device)
        wall_seconds = time.perf_counter() - repeat_start
        if canonical_predictions is None:
            canonical_predictions = repeat_predictions
        elif [(r.identifier, r.sequence) for r in canonical_predictions] != [
            (r.identifier, r.sequence) for r in repeat_predictions
        ]:
            runner.nondeterministic = True
        repeat_rows.append(
            {
                "repeat": repeat,
                "wall_seconds": wall_seconds,
                "preprocess_seconds": pre_total,
                "gpu_seconds": gpu_total,
                "postprocess_seconds": post_total,
                "compute_seconds": pre_total + gpu_total + post_total,
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                ),
                "peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
                ),
                "shard_rank": shard_rank,
            }
        )

    assert canonical_predictions is not None
    total_sequences = len(records)
    total_residues = sum(len(record.sequence) for record in records)
    wall_values = [row["wall_seconds"] for row in repeat_rows]
    gpu_values = [row["gpu_seconds"] for row in repeat_rows]
    median_wall = statistics.median(wall_values)
    batch_latencies = [
        1000.0 * row["compute_seconds"] / row["n_sequences"] for row in batch_rows
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "method_id": runner.method_id,
        "model": runner.model_name,
        "mode": runner.mode,
        "variant": runner.variant,
        "n_sequences": total_sequences,
        "n_residues": total_residues,
        "n_batches": len(batches),
        "batch_size": batch_size,
        "max_residues": max_residues,
        "warmup_batches": warmup_batches,
        "repeats": repeats,
        "sort_by_length": sort_by_length,
        "shard_rank": shard_rank,
        "shard_world_size": shard_world_size,
        "valid_characters": valid_characters,
        "load_seconds": load_seconds,
        "median_wall_seconds": median_wall,
        "median_gpu_seconds": statistics.median(gpu_values),
        "sequences_per_second": total_sequences / median_wall,
        "residues_per_second": total_residues / median_wall,
        "milliseconds_per_protein": 1000.0 * median_wall / total_sequences,
        "batch_latency_ms_p50": _percentile(batch_latencies, 0.50),
        "batch_latency_ms_p95": _percentile(batch_latencies, 0.95),
        "peak_allocated_bytes": max(
            row["peak_allocated_bytes"] or 0 for row in repeat_rows
        ),
        "peak_reserved_bytes": max(
            row["peak_reserved_bytes"] or 0 for row in repeat_rows
        ),
        "nondeterministic": bool(getattr(runner, "nondeterministic", False)),
        "runner_metadata": runner.metadata(),
        "environment": environment_metadata(device),
    }
    if references is not None:
        summary["quality"] = evaluate_predictions(
            canonical_predictions,
            references,
            valid_characters=valid_characters,
        )

    write_fasta(canonical_predictions, output / "predictions.fasta")
    with (output / "batches.jsonl").open("w", encoding="utf-8") as handle:
        for row in batch_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "repeats.jsonl").open("w", encoding="utf-8") as handle:
        for row in repeat_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary
