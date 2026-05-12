#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BSZ_DEFAULT = [1, 16, 64, 128, 256, 320, 512, 960, 1024]
PROMPT_LEN_DEFAULT = [16, 256, 512, 1024, 4096]
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


def iso_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"task5-rwkv-mobile-throughput-{stamp}-{uuid.uuid4().hex[:8]}"


def parse_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def infer_model_size(path: Path) -> str:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)b", path.stem.lower())
    return f"{match.group(1)}B" if match else "unknown"


def infer_quantization(path: Path) -> str:
    stem = path.stem.upper()
    for name in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"):
        if name in stem:
            return name
    return "unknown"


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_gpu_info() -> dict[str, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"nvidia-smi preflight failed: {message}")
    parts = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
    return {"gpu_name": parts[0], "gpu_uuid": parts[1], "driver_version": parts[2]}


def sample_gpu_telemetry(run_id: str, gpu_uuid: str) -> dict[str, str]:
    query = "timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,pstate"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"timestamp": iso_now(), "run_id": run_id, "gpu_uuid": gpu_uuid, "process_name": "telemetry_error"}
    parts = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
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
        "process_name": "rwkv-mobile",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def append_gpu_telemetry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in TELEMETRY_FIELDS})


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Write README Task 5 rwkv-mobile unsupported contract rows.")
    parser.add_argument("--model", type=Path, default=Path("../../weights/rwkv7-g1d-0.1b-20260129-ctx8192-Q4_K_M.gguf"))
    parser.add_argument("--backend", default="llama.cpp")
    parser.add_argument("--output", type=Path, default=Path("results/task5_rwkv_mobile.csv"))
    parser.add_argument("--telemetry-output", type=Path, default=Path("results/gpu_telemetry.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("results/manifest.jsonl"))
    parser.add_argument("--bsz", default=",".join(map(str, BSZ_DEFAULT)))
    parser.add_argument("--prompt-len", default=",".join(map(str, PROMPT_LEN_DEFAULT)))
    parser.add_argument("--decode-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gpu = query_gpu_info()
    if "RTX 5090" not in gpu["gpu_name"]:
        raise SystemExit(f"Task 5 requires RTX 5090, got {gpu['gpu_name']}")
    command = ["python3", *sys.argv]
    binary_path = str(Path("build-linux/examples/simple_benchmark").resolve())
    model_sha256 = sha256_file(args.model)
    quantization = infer_quantization(args.model)
    error = (
        "current rwkv-mobile simple_benchmark loads the llama.cpp backend with n_gpu_layers=0 "
        "and exposes no RTX 5090 core-throughput runner args; CPU fallback is disallowed"
    )

    append_manifest(
        args.manifest_output,
        {
            "run_id": make_run_id(),
            "repo": "rwkv-mobile",
            "gpu_name": gpu["gpu_name"],
            "gpu_uuid": gpu["gpu_uuid"],
            "driver_version": gpu["driver_version"],
            "model_path": str(args.model),
            "model_format": args.model.suffix.removeprefix("."),
            "model_size": infer_model_size(args.model),
            "model_bytes": args.model.stat().st_size if args.model.exists() else "",
            "model_sha256": model_sha256,
            "quantization": quantization,
            "command": " ".join(command),
            "binary_path": binary_path,
            "binary_build_id": f"driver={gpu['driver_version']}",
            "created_at": iso_now(),
            "unsupported_reason": error,
        },
    )

    rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, str]] = []
    for bsz in parse_list(args.bsz):
        for prompt_len in parse_list(args.prompt_len):
            run_id = make_run_id()
            now = iso_now()
            rows.append(
                {
                    "run_id": run_id,
                    "repo": "rwkv-mobile",
                    "backend": args.backend,
                    "runner": "simple_benchmark",
                    "benchmark_kind": "synthetic_throughput",
                    "model_size": infer_model_size(args.model),
                    "model_path": str(args.model),
                    "model_format": args.model.suffix.removeprefix("."),
                    "device": "unsupported",
                    "gpu_name": gpu["gpu_name"],
                    "gpu_uuid": gpu["gpu_uuid"],
                    "dtype": "mixed",
                    "quantization": quantization,
                    "bsz": bsz,
                    "prompt_len": prompt_len,
                    "decode_len": args.decode_len,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "seed": args.seed,
                    "status": "unsupported",
                    "error": error,
                    "prompt_source": "synthetic_rng",
                    "prompt_count": bsz,
                    "prompt_tokens": bsz * prompt_len,
                    "prefill_tokens": bsz * prompt_len,
                    "output_tokens": bsz * args.decode_len,
                    "command": " ".join(command),
                    "binary_path": binary_path,
                    "binary_build_id": f"driver={gpu['driver_version']}",
                    "model_bytes": args.model.stat().st_size if args.model.exists() else "",
                    "model_sha256": model_sha256,
                    "started_at": now,
                    "ended_at": now,
                }
            )
            telemetry_rows.append(sample_gpu_telemetry(run_id, gpu["gpu_uuid"]))
    write_csv(args.output, rows)
    append_gpu_telemetry(args.telemetry_output, telemetry_rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
