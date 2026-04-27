#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


BSZ_DEFAULT = [1, 16, 64, 128, 256, 512, 1024]
PROMPT_LEN_DEFAULT = [16, 256, 512, 1024, 4096]
DECODE_LEN_DEFAULT = 16
CSV_FIELDS = [
    "repo",
    "backend",
    "model_path",
    "model_format",
    "device",
    "dtype",
    "quantization",
    "bsz",
    "prompt_len",
    "decode_len",
    "warmup",
    "repeat",
    "seed",
    "status",
    "error",
    "prefill_tokens",
    "output_tokens",
    "prefill_time_s",
    "ttft_s",
    "e2el_s",
    "token_generation_time_s",
    "prefill_tps",
    "decode_tps",
    "e2e_tps",
    "time_per_output_token_ms",
    "itl_mean_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p95_ms",
    "itl_p99_ms",
]


def parse_int_list(value: str) -> list[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if part:
            items.append(int(part))
    if not items:
        raise ValueError("expected at least one integer")
    return items


def infer_quantization(model_path: str | Path) -> str:
    name = Path(model_path).stem.lower()
    for quant in ("q8_0", "q6_k", "q5_k_m", "q4_k_m"):
        if quant in name:
            return quant
    if "fp16" in name or "f16" in name:
        return "fp16"
    return "unknown"


def infer_dtype(quantization: str) -> str:
    return "fp16" if quantization == "fp16" else quantization


def workspace_root_from_llama_root(llama_root: Path) -> Path:
    return llama_root.parent.parent


def ensure_results_tree(results_dir: Path) -> None:
    (results_dir / "raw" / "batched_bench").mkdir(parents=True, exist_ok=True)
    (results_dir / "raw" / "latency").mkdir(parents=True, exist_ok=True)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def append_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in CSV_FIELDS}
            writer.writerow(normalized)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in CSV_FIELDS}
            writer.writerow(normalized)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
    return rows


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def base_row(
    *,
    model_path: Path,
    backend: str,
    device: str,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    quantization = infer_quantization(model_path)
    return {
        "repo": "llama.cpp",
        "backend": backend,
        "model_path": str(model_path),
        "model_format": "gguf",
        "device": device,
        "dtype": infer_dtype(quantization),
        "quantization": quantization,
        "bsz": bsz,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "status": status,
        "error": error,
    }
