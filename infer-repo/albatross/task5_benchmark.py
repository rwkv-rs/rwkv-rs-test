#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


BSZ_DEFAULT = [1, 16, 64, 128, 256, 320, 512, 960, 1024]
PROMPT_LEN_DEFAULT = [16, 256, 512, 1024, 4096]
DECODE_LEN_DEFAULT = 16
_SHA256_CACHE: dict[Path, str] = {}
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
    return f"task5-albatross-throughput-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def infer_model_size(model_path: str | Path) -> str:
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)b", Path(model_path).stem.lower())
    return f"{match.group(1)}B" if match else "unknown"


def sha256_file(path: Path) -> str:
    resolved = path.resolve()
    if resolved in _SHA256_CACHE:
        return _SHA256_CACHE[resolved]
    if not resolved.exists():
        return ""
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _SHA256_CACHE[resolved] = value
    return value


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
    return {"gpu_name": parts[0], "gpu_uuid": parts[1], "driver_version": parts[2], "device": "cuda0"}


def sample_gpu_telemetry(run_id: str, gpu_uuid: str) -> dict[str, str]:
    query = "timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,pstate"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
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
        "process_name": "albatross-direct",
    }


def parse_int_list(value: str) -> list[int]:
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError("expected at least one integer")
    return items


def benchmark_cases(bsz_values: list[int], prompt_lens: list[int]) -> list[tuple[int, int]]:
    return [(bsz, prompt_len) for bsz in bsz_values for prompt_len in prompt_lens]


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
    return float(np.mean(values)) if values else 0.0


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def make_row(
    *,
    model_path: str,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    status: str,
    error: str = "",
    metrics: dict[str, float] | None = None,
    run_id: str | None = None,
    gpu_name: str = "",
    gpu_uuid: str = "",
    command: list[str] | str = "",
    binary_path: str = "",
    binary_build_id: str = "",
    started_at: str = "",
    ended_at: str = "",
    prompt_source: str = "",
    prompt_count: int | str = "",
) -> dict[str, Any]:
    metrics = metrics or {}
    model = Path(model_path)
    command_text = command if isinstance(command, str) else " ".join(command)
    row = {
        "run_id": run_id or make_run_id(),
        "repo": "albatross",
        "backend": "albatross-direct",
        "runner": "albatross-direct",
        "benchmark_kind": "synthetic_throughput",
        "model_size": infer_model_size(model_path),
        "model_path": model_path,
        "model_format": "pth",
        "device": metrics.get("device", "cuda0" if gpu_uuid else ""),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "dtype": "fp16",
        "quantization": "fp16",
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
        "prompt_tokens": bsz * prompt_len,
        "prefill_tokens": bsz * prompt_len,
        "output_tokens": bsz * decode_len,
        "prefill_time_s": metrics.get("prefill_time_s"),
        "ttft_s": metrics.get("ttft_s"),
        "ttft_p95_s": metrics.get("ttft_s"),
        "e2el_s": metrics.get("e2el_s"),
        "e2el_p95_s": metrics.get("e2el_s"),
        "token_generation_time_s": metrics.get("token_generation_time_s"),
        "prefill_tps": metrics.get("prefill_tps"),
        "decode_tps": metrics.get("decode_tps"),
        "e2e_tps": metrics.get("e2e_tps"),
        "time_per_output_token_ms": metrics.get("time_per_output_token_ms"),
        "requests_per_s": "",
        "itl_mean_ms": metrics.get("itl_mean_ms"),
        "itl_p50_ms": metrics.get("itl_p50_ms"),
        "itl_p90_ms": metrics.get("itl_p90_ms"),
        "itl_p95_ms": metrics.get("itl_p95_ms"),
        "itl_p99_ms": metrics.get("itl_p99_ms"),
        "command": command_text,
        "binary_path": binary_path,
        "binary_build_id": binary_build_id,
        "model_bytes": model.stat().st_size if model.exists() else "",
        "model_sha256": sha256_file(model) if model.exists() else "",
        "started_at": started_at,
        "ended_at": ended_at,
    }
    return {field: fmt(row.get(field, "")) for field in CSV_FIELDS}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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


def load_gsm8k_questions(path: Path) -> list[str]:
    questions = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question = str(record.get("question") or "").strip()
            if question:
                questions.append(question)
    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions


def build_prompt_tokens(tokenizer, questions: list[str], prompt_len: int, bsz: int, seed: int) -> list[list[int]]:
    prompts = []
    cursor = seed % len(questions)
    separator = tokenizer.encode("\n\nUser: ")
    for batch_index in range(bsz):
        token_ids: list[int] = []
        while len(token_ids) < prompt_len:
            question = questions[(cursor + batch_index + len(token_ids)) % len(questions)]
            text = f"User: {question}\n\nAssistant:"
            token_ids.extend(int(token_id) for token_id in tokenizer.encode(text))
            if len(token_ids) < prompt_len:
                token_ids.extend(int(token_id) for token_id in separator)
        prompts.append(token_ids[:prompt_len])
    return prompts


def chunk_prompt_tokens(tokens: list[list[int]], chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("prefill chunk size must be greater than zero")
    if not tokens:
        return
    prompt_len = len(tokens[0])
    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        yield [sequence[start:end] for sequence in tokens]


def chunk_prompt_tensor(token_tensor, prompt_len: int, chunk_size: int):
    if chunk_size <= 0:
        raise ValueError("prefill chunk size must be greater than zero")
    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        yield token_tensor[:, start:end]


def synchronize(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def run_single_batch_once(
    model,
    sampler_simple_batch,
    torch_module,
    tokens: list[list[int]],
    decode_len: int,
    prefill_chunk_size: int,
) -> dict[str, Any]:
    bsz = len(tokens)
    prompt_len = len(tokens[0])
    state = model.generate_zero_state(bsz)
    use_tensor_forward = hasattr(model, "forward_seq_batch_1")
    token_tensor = None
    if use_tensor_forward:
        token_tensor = torch_module.tensor(tokens, dtype=torch_module.long, device="cuda")
    synchronize(torch_module)
    start = time.perf_counter()
    logits = None
    if use_tensor_forward:
        for chunk in chunk_prompt_tensor(token_tensor, prompt_len, prefill_chunk_size):
            logits = model.forward_seq_batch_1(chunk, state)
    else:
        for chunk in chunk_prompt_tokens(tokens, prefill_chunk_size):
            logits = model.forward_batch(chunk, state)
    if logits is None:
        raise ValueError("prompt tokens must not be empty")
    synchronize(torch_module)
    prefill_time_s = time.perf_counter() - start

    token_ids = sampler_simple_batch(logits.float(), noise=0).reshape(-1)
    itl_s: list[float] = []
    for _ in range(max(0, decode_len - 1)):
        step_tokens = token_ids.reshape(bsz, 1) if use_tensor_forward else [[int(token)] for token in token_ids.tolist()]
        synchronize(torch_module)
        step_start = time.perf_counter()
        logits = model.forward_seq_batch_1(step_tokens, state) if use_tensor_forward else model.forward_batch(step_tokens, state)
        synchronize(torch_module)
        step_time_s = time.perf_counter() - step_start
        itl_s.append(step_time_s)
        token_ids = sampler_simple_batch(logits.float(), noise=0).reshape(-1)

    token_generation_time_s = sum(itl_s)
    e2el_s = prefill_time_s + token_generation_time_s
    return {
        "prefill_time_s": prefill_time_s,
        "ttft_s": prefill_time_s,
        "e2el_s": e2el_s,
        "token_generation_time_s": token_generation_time_s,
        "prefill_tps": bsz * prompt_len / prefill_time_s if prefill_time_s > 0 else 0.0,
        "decode_tps": bsz * max(0, decode_len - 1) / token_generation_time_s
        if token_generation_time_s > 0
        else 0.0,
        "e2e_tps": bsz * (prompt_len + decode_len) / e2el_s if e2el_s > 0 else 0.0,
        "time_per_output_token_ms": (token_generation_time_s / max(1, decode_len - 1)) * 1000.0,
        "itl_ms": [value * 1000.0 for value in itl_s],
    }


def run_decode_only_once(
    model,
    sampler_simple_batch,
    torch_module,
    bsz: int,
    decode_len: int,
) -> dict[str, Any]:
    state = model.generate_zero_state(bsz)
    use_tensor_forward = hasattr(model, "forward_seq_batch_1")
    token_ids = torch_module.zeros((bsz,), dtype=torch_module.long, device="cuda")
    itl_s: list[float] = []
    for _ in range(max(1, decode_len)):
        step_tokens = token_ids.reshape(bsz, 1) if use_tensor_forward else [[int(token)] for token in token_ids.tolist()]
        synchronize(torch_module)
        step_start = time.perf_counter()
        logits = model.forward_seq_batch_1(step_tokens, state) if use_tensor_forward else model.forward_batch(step_tokens, state)
        synchronize(torch_module)
        step_time_s = time.perf_counter() - step_start
        itl_s.append(step_time_s)
        token_ids = sampler_simple_batch(logits.float(), noise=0).reshape(-1)

    token_generation_time_s = sum(itl_s)
    return {
        "prefill_time_s": None,
        "ttft_s": None,
        "e2el_s": token_generation_time_s,
        "token_generation_time_s": token_generation_time_s,
        "prefill_tps": None,
        "decode_tps": bsz * max(1, decode_len) / token_generation_time_s if token_generation_time_s > 0 else 0.0,
        "e2e_tps": None,
        "time_per_output_token_ms": (token_generation_time_s / max(1, decode_len)) * 1000.0,
        "itl_ms": [value * 1000.0 for value in itl_s],
    }


def run_once(
    model,
    sampler_simple_batch,
    torch_module,
    tokens: list[list[int]],
    decode_len: int,
    prefill_chunk_size: int,
    micro_batch_size: int | None = None,
) -> dict[str, Any]:
    bsz = len(tokens)
    if micro_batch_size and micro_batch_size < bsz:
        raise ValueError(
            f"micro-batch-size {micro_batch_size} would split real bsz {bsz}; "
            "Task 5 requires the measured batch dimension to stay literal"
        )
    return run_single_batch_once(model, sampler_simple_batch, torch_module, tokens, decode_len, prefill_chunk_size)


def aggregate(results: list[dict[str, Any]], device: str) -> dict[str, float | str]:
    itl_ms = [value for result in results for value in result["itl_ms"]]
    metrics: dict[str, float | str] = {"device": device}
    for key in (
        "prefill_time_s",
        "ttft_s",
        "e2el_s",
        "token_generation_time_s",
        "prefill_tps",
        "decode_tps",
        "e2e_tps",
        "time_per_output_token_ms",
    ):
        values = [float(result[key]) for result in results if result.get(key) is not None]
        metrics[key] = mean(values) if values else None
    metrics["itl_mean_ms"] = mean(itl_ms)
    metrics["itl_p50_ms"] = percentile(itl_ms, 0.50)
    metrics["itl_p90_ms"] = percentile(itl_ms, 0.90)
    metrics["itl_p95_ms"] = percentile(itl_ms, 0.95)
    metrics["itl_p99_ms"] = percentile(itl_ms, 0.99)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run README Task 5 throughput for albatross direct inference.")
    parser.add_argument("--model", type=Path, default=Path("../../weights/rwkv7-g1f-1.5b-20260419-ctx8192.pth"))
    parser.add_argument("--dataset", type=Path, default=Path("../../data/gsm8k.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/task5_albatross_fp16.csv"))
    parser.add_argument("--telemetry-output", type=Path, default=Path("results/gpu_telemetry.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("results/manifest.jsonl"))
    parser.add_argument("--bsz", default=",".join(map(str, BSZ_DEFAULT)))
    parser.add_argument("--prompt-len", default=",".join(map(str, PROMPT_LEN_DEFAULT)))
    parser.add_argument("--decode-len", type=int, default=DECODE_LEN_DEFAULT)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    args = parser.parse_args()
    command = ["python3", "task5_benchmark.py"] + [
        item for pair in (
            ("--model", str(args.model)),
            ("--dataset", str(args.dataset)),
            ("--output", str(args.output)),
            ("--telemetry-output", str(args.telemetry_output)),
            ("--manifest-output", str(args.manifest_output)),
            ("--bsz", args.bsz),
            ("--prompt-len", args.prompt_len),
            ("--decode-len", str(args.decode_len)),
            ("--warmup", str(args.warmup)),
            ("--repeat", str(args.repeat)),
            ("--seed", str(args.seed)),
            ("--prefill-chunk-size", str(args.prefill_chunk_size)),
            ("--micro-batch-size", str(args.micro_batch_size)),
        ) for item in pair
    ]
    gpu = query_gpu_info()
    if "RTX 5090" not in gpu["gpu_name"]:
        raise SystemExit(f"Task 5 requires RTX 5090, got {gpu['gpu_name']}")
    append_manifest(
        args.manifest_output,
        {
            "run_id": make_run_id(),
            "repo": "albatross",
            "gpu_name": gpu["gpu_name"],
            "gpu_uuid": gpu["gpu_uuid"],
            "driver_version": gpu["driver_version"],
            "model_path": str(args.model),
            "model_format": "pth",
            "model_size": infer_model_size(args.model),
            "model_bytes": args.model.stat().st_size if args.model.exists() else "",
            "model_sha256": sha256_file(args.model),
            "quantization": "fp16",
            "command": " ".join(command),
            "binary_path": "python3",
            "binary_build_id": f"driver={gpu['driver_version']}",
            "created_at": iso_now(),
        },
    )

    bsz_values = parse_int_list(args.bsz)
    prompt_lens = parse_int_list(args.prompt_len)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, str]] = []
    runnable_cases = benchmark_cases(bsz_values, prompt_lens)

    if runnable_cases:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

        try:
            import torch
            import types

            from reference.rwkv7 import RWKV_x070
            from reference.utils import TRIE_TOKENIZER, sampler_simple_batch

            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.set_grad_enabled(False)
            tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")
            questions = load_gsm8k_questions(args.dataset)
            model_args = types.SimpleNamespace(vocab_size=65536, head_size=64, MODEL_NAME=str(args.model).removesuffix(".pth"))
            model = RWKV_x070(model_args)
            torch_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            if "RTX 5090" not in torch_device_name:
                raise SystemExit(f"Task 5 requires RTX 5090 torch device, got {torch_device_name}")
            device = "cuda0"
        except Exception as exc:
            setup_error = f"setup={type(exc).__name__}: {exc}"
            for bsz, prompt_len in runnable_cases:
                run_id = make_run_id()
                now = iso_now()
                rows.append(
                    make_row(
                        run_id=run_id,
                        model_path=str(args.model),
                        bsz=bsz,
                        prompt_len=prompt_len,
                        decode_len=args.decode_len,
                        warmup=args.warmup,
                        repeat=args.repeat,
                        seed=args.seed,
                        status="failed",
                        error=setup_error,
                        gpu_name=gpu["gpu_name"],
                        gpu_uuid=gpu["gpu_uuid"],
                        command=command,
                        binary_path="python3",
                        binary_build_id=f"driver={gpu['driver_version']}",
                        started_at=now,
                        ended_at=now,
                        prompt_source=str(args.dataset),
                        prompt_count=bsz,
                    )
                )
            rows.sort(key=lambda row: (int(row["bsz"]), int(row["prompt_len"]), row["backend"]))
            write_csv(args.output, rows)
            append_gpu_telemetry(args.telemetry_output, [sample_gpu_telemetry(str(rows[-1]["run_id"]), gpu["gpu_uuid"])])
            print(f"wrote {args.output}")
            return

        for bsz, prompt_len in runnable_cases:
            print(f"albatross task5 bsz={bsz} prompt_len={prompt_len}", flush=True)
            try:
                tokens = build_prompt_tokens(tokenizer, questions, prompt_len, bsz, args.seed)
                for _ in range(args.warmup):
                    run_once(
                        model,
                        sampler_simple_batch,
                        torch,
                        tokens,
                        args.decode_len,
                        args.prefill_chunk_size,
                        args.micro_batch_size,
                    )
                results = [
                    run_once(
                        model,
                        sampler_simple_batch,
                        torch,
                        tokens,
                        args.decode_len,
                        args.prefill_chunk_size,
                        args.micro_batch_size,
                    )
                    for _ in range(args.repeat)
                ]
                row = make_row(
                    model_path=str(args.model),
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=args.decode_len,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    seed=args.seed,
                    status="ok",
                    metrics=aggregate(results, device),
                    gpu_name=gpu["gpu_name"],
                    gpu_uuid=gpu["gpu_uuid"],
                    command=command,
                    binary_path="python3",
                    binary_build_id=f"driver={gpu['driver_version']}",
                    started_at=iso_now(),
                    ended_at=iso_now(),
                    prompt_source=str(args.dataset),
                    prompt_count=bsz,
                )
                rows.append(row)
                telemetry_row = sample_gpu_telemetry(str(row["run_id"]), gpu["gpu_uuid"])
                telemetry_rows.append(telemetry_row)
                rows.sort(key=lambda row: (int(row["bsz"]), int(row["prompt_len"]), row["backend"]))
                write_csv(args.output, rows)
                append_gpu_telemetry(args.telemetry_output, [telemetry_row])
            except Exception as exc:
                prefill_error = f"prefill={type(exc).__name__}: {exc}"
                exc.__traceback__ = None
                del exc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                decode_error = ""
                decode_metrics: dict[str, Any] | None = None
                decode_status = "failed"
                try:
                    decode_results = [
                        run_decode_only_once(
                            model,
                            sampler_simple_batch,
                            torch,
                            bsz,
                            args.decode_len,
                        )
                        for _ in range(args.repeat)
                    ]
                    decode_metrics = aggregate(decode_results, device)
                    decode_status = "decode_only"
                except Exception as decode_exc:
                    decode_error = f"; decode={type(decode_exc).__name__}: {decode_exc}"
                    decode_exc.__traceback__ = None
                    del decode_exc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                row = make_row(
                    model_path=str(args.model),
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=args.decode_len,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    seed=args.seed,
                    status=decode_status,
                    error=f"{prefill_error}{decode_error}",
                    metrics=decode_metrics,
                    gpu_name=gpu["gpu_name"],
                    gpu_uuid=gpu["gpu_uuid"],
                    command=command,
                    binary_path="python3",
                    binary_build_id=f"driver={gpu['driver_version']}",
                    started_at=iso_now(),
                    ended_at=iso_now(),
                    prompt_source=str(args.dataset),
                    prompt_count=bsz,
                )
                rows.append(row)
                telemetry_row = sample_gpu_telemetry(str(row["run_id"]), gpu["gpu_uuid"])
                telemetry_rows.append(telemetry_row)
                rows.sort(key=lambda row: (int(row["bsz"]), int(row["prompt_len"]), row["backend"]))
                write_csv(args.output, rows)
                append_gpu_telemetry(args.telemetry_output, [telemetry_row])
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
