#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


BSZ_DEFAULT = [1, 16, 64, 128, 256, 320, 512, 960, 1024]
PROMPT_LEN_DEFAULT = [16, 256, 512, 1024, 4096]
DECODE_LEN_DEFAULT = 16
CSV_FIELDS = [
    "run_id",
    "repo",
    "backend",
    "runner",
    "benchmark_kind",
    "model_size",
    "model_path",
    "model_format",
    "device",
    "gpu_name",
    "gpu_uuid",
    "dtype",
    "quantization",
    "bsz",
    "concurrency",
    "request_index",
    "sample_index",
    "prompt_len",
    "decode_len",
    "warmup",
    "repeat",
    "seed",
    "status",
    "error",
    "prompt_source",
    "prompt_count",
    "prompt_tokens",
    "prefill_tokens",
    "output_tokens",
    "prefill_time_s",
    "ttft_s",
    "ttft_p95_s",
    "e2el_s",
    "e2el_p95_s",
    "token_generation_time_s",
    "prefill_tps",
    "decode_tps",
    "e2e_tps",
    "time_per_output_token_ms",
    "requests_per_s",
    "itl_mean_ms",
    "itl_p50_ms",
    "itl_p90_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "command",
    "binary_path",
    "binary_build_id",
    "model_bytes",
    "model_sha256",
    "started_at",
    "ended_at",
]

TELEMETRY_FIELDS = [
    "timestamp",
    "run_id",
    "gpu_uuid",
    "gpu_util",
    "mem_used",
    "mem_total",
    "power_w",
    "sm_clock",
    "mem_clock",
    "pstate",
    "process_name",
]

_MODEL_HASH_CACHE: dict[Path, str] = {}


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
    (results_dir / "raw" / "throughput").mkdir(parents=True, exist_ok=True)
    (results_dir / "raw" / "latency_synthetic").mkdir(parents=True, exist_ok=True)
    (results_dir / "raw" / "latency_gsm8k").mkdir(parents=True, exist_ok=True)


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


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        handle.write("\n")


def append_gpu_telemetry(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TELEMETRY_FIELDS})


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


def iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_run_id(kind: str) -> str:
    return f"task5-{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def command_to_string(command: str | Iterable[str]) -> str:
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


def sha256_file(path: Path) -> str:
    resolved = path.resolve()
    if resolved in _MODEL_HASH_CACHE:
        return _MODEL_HASH_CACHE[resolved]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _MODEL_HASH_CACHE[resolved] = value
    return value


def infer_model_size(model_path: str | Path) -> str:
    name = Path(model_path).stem.lower()
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)b", name)
    return f"{match.group(1)}B" if match else "unknown"


def binary_build_id(binary: Path) -> str:
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    text = " ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())
    return text[:500]


def query_gpu_info() -> dict[str, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", maxsplit=1)[0].strip() or "0"
    command = [
        "nvidia-smi",
        "-i",
        visible,
        "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"nvidia-smi preflight failed: {message}")
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"unexpected nvidia-smi output: {first}")
    return {"gpu_name": parts[0], "gpu_uuid": parts[1], "driver_version": parts[2], "device": f"cuda{visible}"}


def sample_gpu_telemetry(run_id: str, gpu_uuid: str) -> dict[str, str]:
    query = (
        "timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,"
        "clocks.sm,clocks.mem,pstate"
    )
    selector = ["-i", gpu_uuid] if gpu_uuid else []
    command = ["nvidia-smi", *selector, f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    process_name = query_gpu_process_names()
    return {
        "timestamp": iso_now(),
        "run_id": run_id,
        "gpu_uuid": parts[1] if len(parts) > 1 else gpu_uuid,
        "gpu_util": parts[2] if len(parts) > 2 else "",
        "mem_used": parts[3] if len(parts) > 3 else "",
        "mem_total": parts[4] if len(parts) > 4 else "",
        "power_w": parts[5] if len(parts) > 5 else "",
        "sm_clock": parts[6] if len(parts) > 6 else "",
        "mem_clock": parts[7] if len(parts) > 7 else "",
        "pstate": parts[8] if len(parts) > 8 else "",
        "process_name": process_name,
    }


class GpuTelemetrySampler:
    def __init__(
        self,
        *,
        path: Path,
        run_id: str,
        gpu_uuid: str,
        process_name: str = "",
        interval_s: float = 1.0,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.gpu_uuid = gpu_uuid
        self.process_name = process_name
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._rows: list[dict[str, str]] = []
        self._thread = threading.Thread(target=self._run, name=f"gpu-telemetry-{run_id}", daemon=True)

    def __enter__(self) -> "GpuTelemetrySampler":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self.interval_s * 2.0, 2.0))
        if not self._rows:
            try:
                self._rows.append(sample_gpu_telemetry(self.run_id, self.gpu_uuid))
            except RuntimeError as exc:
                self._rows.append(
                    {
                        "timestamp": iso_now(),
                        "run_id": self.run_id,
                        "gpu_uuid": self.gpu_uuid,
                        "process_name": f"telemetry_error:{exc}",
                    }
                )
        append_gpu_telemetry(self.path, self._rows)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = sample_gpu_telemetry(self.run_id, self.gpu_uuid)
                if not row.get("process_name"):
                    row["process_name"] = self.process_name
                self._rows.append(row)
            except RuntimeError as exc:
                self._rows.append(
                    {
                        "timestamp": iso_now(),
                        "run_id": self.run_id,
                        "gpu_uuid": self.gpu_uuid,
                        "process_name": f"telemetry_error:{exc}",
                    }
                )
            self._stop.wait(self.interval_s)


def query_gpu_process_names() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
    )
    names: list[str] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[-1] != "-":
            names.append(parts[-1])
    return ";".join(sorted(set(names)))


def base_row(
    *,
    model_path: Path,
    backend: str,
    device: str,
    runner: str | None = None,
    benchmark_kind: str | None = None,
    gpu_name: str = "",
    gpu_uuid: str = "",
    bsz: int,
    prompt_len: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    status: str,
    error: str = "",
    run_id: str | None = None,
    command: str | Iterable[str] = "",
    binary_path: Path | str = "",
    binary_build_id: str = "",
    started_at: str = "",
    ended_at: str = "",
    prompt_source: str = "",
    prompt_count: int | str = "",
) -> dict[str, Any]:
    quantization = infer_quantization(model_path)
    model_bytes = model_path.stat().st_size if model_path.exists() else ""
    model_sha256 = sha256_file(model_path) if model_path.exists() else ""
    kind = benchmark_kind or "unknown"
    return {
        "run_id": run_id or make_run_id(kind),
        "repo": "llama.cpp",
        "backend": backend,
        "runner": runner or backend,
        "benchmark_kind": kind,
        "model_size": infer_model_size(model_path),
        "model_path": str(model_path),
        "model_format": "gguf",
        "device": device,
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
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
        "prompt_source": prompt_source,
        "prompt_count": prompt_count,
        "prompt_tokens": "",
        "command": command_to_string(command),
        "binary_path": str(binary_path),
        "binary_build_id": binary_build_id,
        "model_bytes": model_bytes,
        "model_sha256": model_sha256,
        "started_at": started_at,
        "ended_at": ended_at,
    }
