#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from scripts.task5_core_schema import (
    BATCH_DECODE_B_DEFAULT,
    BATCH_PREFILL_PAIRS_DEFAULT,
    PREFILL_T_DEFAULT,
)


MEASUREMENT_BOUNDARY = "forward+sampler; no tokenizer decode; no scheduler; no server"


def add_repo_root_to_path(file: str) -> Path:
    root = Path(file).resolve()
    while root != root.parent:
        if (root / "scripts" / "task5_core_schema.py").exists():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
        root = root.parent
    raise RuntimeError("could not find rwkv-rs-test repo root")


def iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_run_id(backend: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"task5-core-{backend}-{stamp}-{uuid.uuid4().hex[:8]}"


def parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return items


def parse_pair_list(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        left, sep, right = part.partition("x")
        if sep != "x":
            raise argparse.ArgumentTypeError(f"expected BxT pair, got {part!r}")
        pairs.append((int(left), int(right)))
    if not pairs:
        raise argparse.ArgumentTypeError("expected at least one BxT pair")
    return pairs


def task_cases(
    tasks: Iterable[str],
    prefill_t: Iterable[int] = PREFILL_T_DEFAULT,
    batch_decode_b: Iterable[int] = BATCH_DECODE_B_DEFAULT,
    batch_prefill_pairs: Iterable[tuple[int, int]] = BATCH_PREFILL_PAIRS_DEFAULT,
) -> list[tuple[str, int, int]]:
    cases: list[tuple[str, int, int]] = []
    requested = list(tasks)
    if "decode" in requested:
        cases.append(("decode", 1, 1))
    if "prefill" in requested:
        cases.extend(("prefill", 1, int(T)) for T in prefill_t)
    if "batch_decode" in requested:
        cases.extend(("batch_decode", int(B), 1) for B in batch_decode_b)
    if "batch_prefill" in requested:
        cases.extend(("batch_prefill", int(B), int(T)) for B, T in batch_prefill_pairs)
    return cases


def split_tasks(value: str) -> list[str]:
    tasks = [part.strip() for part in value.split(",") if part.strip()]
    if not tasks:
        raise argparse.ArgumentTypeError("expected at least one task")
    return tasks


def infer_model_size(path: str | Path) -> str:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)b", Path(path).stem.lower())
    return f"{match.group(1)}B" if match else "unknown"


def infer_model_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def infer_quantization(path: str | Path) -> str:
    stem = Path(path).stem.upper()
    for name in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16", "F16"):
        if name in stem:
            return "FP16" if name == "F16" else name
    return "fp16" if Path(path).suffix.lower() in {".pth", ".st", ".safetensors"} else "unknown"


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_gpu_info() -> dict[str, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"gpu_name": "", "gpu_uuid": ""}
    parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    return {"gpu_name": parts[0] if parts else "", "gpu_uuid": parts[1] if len(parts) > 1 else ""}


def common_metadata(
    *,
    repo: str,
    backend: str,
    runner: str,
    model_path: str | Path,
    warmup: int,
    repeat: int,
    seed: int,
    command: str | None = None,
) -> dict[str, object]:
    model = Path(model_path)
    gpu = query_gpu_info()
    return {
        "run_id": make_run_id(backend),
        "repo": repo,
        "backend": backend,
        "runner": runner,
        "model_size": infer_model_size(model),
        "model_path": str(model),
        "model_format": infer_model_format(model),
        "device": os.environ.get("CUDA_VISIBLE_DEVICES", "cuda0"),
        "gpu_name": gpu["gpu_name"],
        "gpu_uuid": gpu["gpu_uuid"],
        "dtype": infer_quantization(model).lower(),
        "quantization": infer_quantization(model),
        "warmup": warmup,
        "repeat": repeat,
        "seed": seed,
        "command": command or " ".join(sys.argv),
        "binary_path": sys.executable,
        "binary_build_id": "",
        "model_bytes": model.stat().st_size if model.exists() else "",
        "model_sha256": sha256_file(model),
        "started_at": iso_now(),
    }
