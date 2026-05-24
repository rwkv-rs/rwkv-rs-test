#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_KIND = "core_forward_sample_throughput"
TASKS = ("decode", "prefill", "batch_decode", "batch_prefill")

PREFILL_T_DEFAULT = (16, 64, 256, 1024, 4096)
BATCH_DECODE_B_DEFAULT = (2, 4, 8, 16, 32, 64, 128, 256, 512, 960, 1024)
BATCH_PREFILL_PAIRS_DEFAULT = ((2, 2), (4, 4), (8, 8), (16, 16), (32, 32))

CSV_FIELDS = [
    "run_id",
    "repo",
    "backend",
    "runner",
    "benchmark_kind",
    "task",
    "model_size",
    "model_path",
    "model_format",
    "device",
    "gpu_name",
    "gpu_uuid",
    "dtype",
    "quantization",
    "B",
    "T",
    "warmup",
    "repeat",
    "seed",
    "status",
    "error",
    "input_tokens",
    "measured_tokens",
    "total_time_s",
    "forward_time_s",
    "sample_time_s",
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "forward_sample_tps",
    "entrypoint",
    "measurement_boundary",
    "command",
    "binary_path",
    "binary_build_id",
    "model_bytes",
    "model_sha256",
    "started_at",
    "ended_at",
]


def task_shape(task: str, B: int, T: int) -> tuple[str, int, int]:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    if B <= 0 or T <= 0:
        raise ValueError(f"{task} requires positive B and T, got B={B} T={T}")
    if task == "decode" and (B, T) != (1, 1):
        raise ValueError("decode must be B=1,T=1")
    if task == "prefill" and B != 1:
        raise ValueError("prefill must be single stream B=1,T=n")
    if task == "batch_decode" and (B <= 1 or T != 1):
        raise ValueError("batch_decode must be B>1,T=1")
    if task == "batch_prefill" and (B <= 1 or T <= 1):
        raise ValueError("batch_prefill must be B>1,T>1")
    return task, B, T


def measured_tokens(task: str, B: int, T: int) -> int:
    task_shape(task, B, T)
    if task in ("decode", "batch_decode"):
        return B
    return B * T


def unsupported_row(
    *,
    task: str,
    B: int,
    T: int,
    entrypoint: str,
    error: str,
    **metadata: Any,
) -> dict[str, Any]:
    task_shape(task, B, T)
    if task in {"decode", "prefill"} and not metadata.get("allow_required_task_unsupported"):
        raise ValueError(f"{task} is a required Task5 core workload; use ok or failed, not unsupported")
    row = _base_row(metadata, task=task, B=B, T=T)
    row.update(
        {
            "status": "unsupported",
            "error": error,
            "entrypoint": entrypoint,
            "measurement_boundary": metadata.get(
                "measurement_boundary",
                "unsupported; no forward+sampler measurement was run",
            ),
        }
    )
    return row


def ok_row(
    *,
    task: str,
    B: int,
    T: int,
    total_time_s: float,
    p50_ms: float,
    entrypoint: str,
    measurement_boundary: str,
    forward_time_s: float | None = None,
    sample_time_s: float | None = None,
    p10_ms: float | None = None,
    p90_ms: float | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    if total_time_s <= 0:
        raise ValueError("total_time_s must be positive for status=ok")
    tokens = measured_tokens(task, B, T)
    row = _base_row(metadata, task=task, B=B, T=T)
    row.update(
        {
            "status": "ok",
            "input_tokens": B * T,
            "measured_tokens": tokens,
            "total_time_s": total_time_s,
            "forward_time_s": _optional_float(forward_time_s),
            "sample_time_s": _optional_float(sample_time_s),
            "p10_ms": _optional_float(p10_ms),
            "p50_ms": p50_ms,
            "p90_ms": _optional_float(p90_ms),
            "forward_sample_tps": tokens / total_time_s,
            "entrypoint": entrypoint,
            "measurement_boundary": measurement_boundary,
        }
    )
    return row


def failed_row(
    *,
    task: str,
    B: int,
    T: int,
    entrypoint: str,
    error: str,
    **metadata: Any,
) -> dict[str, Any]:
    task_shape(task, B, T)
    row = _base_row(metadata, task=task, B=B, T=T)
    row.update(
        {
            "status": "failed",
            "error": error,
            "entrypoint": entrypoint,
            "measurement_boundary": metadata.get(
                "measurement_boundary",
                "forward+sampler attempted; no successful timing",
            ),
        }
    )
    return row


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _base_row(metadata: dict[str, Any], *, task: str, B: int, T: int) -> dict[str, Any]:
    task_shape(task, B, T)
    row = {field: "" for field in CSV_FIELDS}
    row.update({field: metadata.get(field, "") for field in CSV_FIELDS})
    row.update(
        {
            "benchmark_kind": BENCHMARK_KIND,
            "task": task,
            "B": B,
            "T": T,
            "input_tokens": B * T,
        }
    )
    return row


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else value
