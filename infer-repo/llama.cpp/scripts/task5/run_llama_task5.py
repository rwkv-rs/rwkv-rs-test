#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib import request
from urllib.error import URLError

from convert_batched_bench import convert_rows
from stream_latency_bench import (
    build_latency_rows,
    collect_stream_metrics,
    make_gsm8k_prompt_tokens,
    make_synthetic_prompt_tokens,
    stream_completion,
)
from task5_common import (
    BSZ_DEFAULT,
    DECODE_LEN_DEFAULT,
    PROMPT_LEN_DEFAULT,
    GpuTelemetrySampler,
    append_csv,
    append_manifest,
    base_row,
    binary_build_id,
    ensure_results_tree,
    infer_model_size,
    infer_quantization,
    iso_now,
    make_run_id,
    parse_int_list,
    query_gpu_info,
    sanitize_filename,
    sha256_file,
    workspace_root_from_llama_root,
)


TASK5_MODEL_MARKERS = (
    "rwkv7-g1d-0.1b-20260129-ctx8192",
    "rwkv7-g1d-0.4b-20260210-ctx8192",
    "rwkv7-g1f-2.9b-20260420-ctx8192",
    "rwkv7-g1f-7.2b-20260414-ctx8192",
    "rwkv7-g1f-13.3b-20260415-ctx8192",
)


def wait_for_server(url: str, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{url.rstrip('/')}/health"
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {process.returncode}")
        try:
            with request.urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy: {last_error}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def is_valid_gguf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"GGUF"


def default_models(weights_dir: Path) -> list[Path]:
    models = []
    for path in sorted(weights_dir.glob("*.gguf")):
        stem = path.stem.lower()
        if any(marker in stem for marker in TASK5_MODEL_MARKERS):
            models.append(path)
    return models


def write_preflight(path: Path, *, gpu: dict[str, str], models: list[Path], binaries: dict[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_models: dict[str, dict[str, object]] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            for item in previous.get("models", []):
                if isinstance(item, dict) and item.get("path"):
                    existing_models[str(item["path"])] = item
        except json.JSONDecodeError:
            existing_models = {}
    for model in models:
        existing_models[str(model)] = {
            "path": str(model),
            "model_size": infer_model_size(model),
            "quantization": infer_quantization(model),
            "bytes": model.stat().st_size,
            "sha256": existing_models.get(str(model), {}).get("sha256", ""),
        }
    payload = {
        "started_at": iso_now(),
        "gpu": gpu,
        "binaries": {name: {"path": str(binary), "build_id": binary_build_id(binary)} for name, binary in binaries.items()},
        "models": sorted(existing_models.values(), key=lambda item: str(item["path"])),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def row_for_skipped_case(
    *,
    model: Path,
    backend: str,
    runner: str,
    benchmark_kind: str,
    device: str,
    gpu_name: str,
    gpu_uuid: str,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    status: str,
    error: str,
    run_id: str,
    command: list[str],
    binary_path: Path,
    binary_build_id_value: str,
    started_at: str,
    ended_at: str,
    prompt_source: str,
    prompt_count: int,
) -> dict[str, object]:
    row = base_row(
        model_path=model,
        backend=backend,
        runner=runner,
        benchmark_kind=benchmark_kind,
        device=device,
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        bsz=bsz,
        prompt_len=prompt_len,
        decode_len=decode_len,
        warmup=warmup,
        repeat=repeat,
        seed=seed,
        status=status,
        error=error,
        run_id=run_id,
        command=command,
        binary_path=binary_path,
        binary_build_id=binary_build_id_value,
        started_at=started_at,
        ended_at=ended_at,
        prompt_source=prompt_source,
        prompt_count=prompt_count,
    )
    row.update({"concurrency": bsz, "prompt_tokens": bsz * prompt_len, "output_tokens": 0})
    return row


def append_run_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    benchmark_kind: str,
    model: Path,
    command: list[str],
    gpu: dict[str, str],
    status: str,
    started_at: str,
    ended_at: str,
    output: Path,
    error: str = "",
) -> None:
    append_manifest(
        manifest_path,
        {
            "run_id": run_id,
            "repo": "llama.cpp",
            "benchmark_kind": benchmark_kind,
            "model_path": str(model),
            "model_size": infer_model_size(model),
            "quantization": infer_quantization(model),
            "model_bytes": model.stat().st_size if model.exists() else "",
            "model_sha256": sha256_file(model) if model.exists() else "",
            "gpu_name": gpu["gpu_name"],
            "gpu_uuid": gpu["gpu_uuid"],
            "driver_version": gpu["driver_version"],
            "command": " ".join(str(part) for part in command),
            "status": status,
            "error": error,
            "started_at": started_at,
            "ended_at": ended_at,
            "output": str(output),
        },
    )


def runnable_or_error(*, bsz: int, prompt_len: int, decode_len: int, max_parallel: int, max_ctx_size: int) -> str:
    if bsz > max_parallel:
        return f"bsz {bsz} exceeds llama.cpp RWKV n_seq_max limit {max_parallel}"
    required_ctx = bsz * (prompt_len + decode_len)
    if required_ctx > max_ctx_size:
        return f"required ctx {required_ctx} exceeds max ctx {max_ctx_size}"
    return ""


def run_throughput(
    *,
    binary: Path,
    build_id: str,
    model: Path,
    results_dir: Path,
    manifest_path: Path,
    telemetry_path: Path,
    gpu: dict[str, str],
    bsz_values: list[int],
    prompt_lens: list[int],
    decode_len: int,
    max_ctx_size: int,
    n_batch: int,
    n_ubatch: int,
    warmup: int,
    repeat: int,
    seed: int,
    max_parallel: int,
    extra_args: list[str],
    env: dict[str, str],
    timeout_s: float | None,
) -> None:
    csv_output = results_dir / "task5_throughput.csv"
    raw_dir = results_dir / "raw" / "throughput"
    for prompt_len in prompt_lens:
        runnable_bsz = [
            bsz
            for bsz in bsz_values
            if not runnable_or_error(
                bsz=bsz,
                prompt_len=prompt_len,
                decode_len=decode_len,
                max_parallel=max_parallel,
                max_ctx_size=max_ctx_size,
            )
        ]
        run_id = make_run_id("throughput")
        raw_output = raw_dir / f"{sanitize_filename(model.stem)}_pp{prompt_len}.jsonl"
        command = [
            str(binary),
            "-m",
            str(model),
            "-c",
            str(max_ctx_size),
            "-b",
            str(n_batch),
            "-ub",
            str(n_ubatch),
            "-npp",
            str(prompt_len),
            "-ntg",
            str(decode_len),
            "-npl",
            ",".join(map(str, runnable_bsz or [1])),
            "-ngl",
            "999",
            "--output-format",
            "jsonl",
            *extra_args,
        ]
        started_at = iso_now()
        rows = []
        status = "unsupported" if not runnable_bsz else "failed"
        error = ""
        if runnable_bsz:
            with raw_output.open("w", encoding="utf-8") as handle:
                with GpuTelemetrySampler(
                    path=telemetry_path,
                    run_id=run_id,
                    gpu_uuid=gpu["gpu_uuid"],
                    process_name="llama-batched-bench",
                ):
                    try:
                        completed = subprocess.run(
                            command,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            check=False,
                            env=env,
                            timeout=timeout_s,
                        )
                        status = "ok" if completed.returncode == 0 else "failed"
                        error = "" if completed.returncode == 0 else f"llama-batched-bench exited {completed.returncode}"
                    except subprocess.TimeoutExpired:
                        status = "failed"
                        error = f"llama-batched-bench timed out after {timeout_s:.0f}s"
            if raw_output.exists():
                rows.extend(
                    convert_rows(
                        raw_output,
                        model_path=model,
                        backend="llama.cpp",
                        runner="llama-batched-bench",
                        benchmark_kind="synthetic_throughput",
                        device=gpu["device"],
                        gpu_name=gpu["gpu_name"],
                        gpu_uuid=gpu["gpu_uuid"],
                        warmup=warmup,
                        repeat=repeat,
                        seed=seed,
                        run_id=run_id,
                        command=command,
                        binary_path=binary,
                        binary_build_id=build_id,
                        started_at=started_at,
                        ended_at=iso_now(),
                    )
                )
        else:
            raw_output.write_text("", encoding="utf-8")
            error = "all bsz values are unsupported for this prompt_len"

        ended_at = iso_now()
        seen = {(int(row["bsz"]), int(row["prompt_len"])) for row in rows}
        for bsz in bsz_values:
            if (bsz, prompt_len) in seen:
                continue
            case_error = runnable_or_error(
                bsz=bsz,
                prompt_len=prompt_len,
                decode_len=decode_len,
                max_parallel=max_parallel,
                max_ctx_size=max_ctx_size,
            )
            rows.append(
                row_for_skipped_case(
                    model=model,
                    backend="llama.cpp",
                    runner="llama-batched-bench",
                    benchmark_kind="synthetic_throughput",
                    device=gpu["device"],
                    gpu_name=gpu["gpu_name"],
                    gpu_uuid=gpu["gpu_uuid"],
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=decode_len,
                    warmup=warmup,
                    repeat=repeat,
                    seed=seed,
                    status="unsupported" if case_error else status,
                    error=case_error or error or "runner did not return this case",
                    run_id=run_id,
                    command=command,
                    binary_path=binary,
                    binary_build_id_value=build_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    prompt_source="synthetic_runner_tokens",
                    prompt_count=bsz,
                )
            )
        append_csv(csv_output, rows)
        append_run_manifest(
            manifest_path,
            run_id=run_id,
            benchmark_kind="synthetic_throughput",
            model=model,
            command=command,
            gpu=gpu,
            status=status,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            output=raw_output,
        )


def start_server(
    *,
    binary: Path,
    model: Path,
    log_path: Path,
    port: int,
    ctx_size: int,
    n_batch: int,
    n_ubatch: int,
    parallel: int,
    extra_args: list[str],
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], list[str], str]:
    server_url = f"http://127.0.0.1:{port}"
    command = [
        str(binary),
        "-m",
        str(model),
        "-c",
        str(ctx_size),
        "-b",
        str(n_batch),
        "-ub",
        str(n_ubatch),
        "-np",
        str(max(parallel, 1)),
        "-ngl",
        "999",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-webui",
        "-lv",
        "1",
        *extra_args,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    process._task5_log_handle = log  # type: ignore[attr-defined]
    return process, command, server_url


def stop_server(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log = getattr(process, "_task5_log_handle", None)
    if log is not None:
        log.close()


def run_synthetic_latency(
    *,
    binary: Path,
    build_id: str,
    model: Path,
    results_dir: Path,
    manifest_path: Path,
    telemetry_path: Path,
    gpu: dict[str, str],
    bsz_values: list[int],
    prompt_lens: list[int],
    decode_len: int,
    max_ctx_size: int,
    n_batch: int,
    n_ubatch: int,
    warmup: int,
    repeat: int,
    seed: int,
    max_parallel: int,
    startup_timeout: float,
    request_timeout: float,
    extra_args: list[str],
    env: dict[str, str],
) -> None:
    csv_output = results_dir / "task5_latency_synthetic.csv"
    raw_dir = results_dir / "raw" / "latency_synthetic"
    for bsz in bsz_values:
        for prompt_len in prompt_lens:
            run_id = make_run_id("latency-synthetic")
            required_ctx = bsz * (prompt_len + decode_len)
            command_placeholder = [str(binary), "-m", str(model)]
            started_at = iso_now()
            case_error = runnable_or_error(
                bsz=bsz,
                prompt_len=prompt_len,
                decode_len=decode_len,
                max_parallel=max_parallel,
                max_ctx_size=max_ctx_size,
            )
            if case_error:
                row = row_for_skipped_case(
                    model=model,
                    backend="llama.cpp",
                    runner="llama-server",
                    benchmark_kind="synthetic_latency",
                    device=gpu["device"],
                    gpu_name=gpu["gpu_name"],
                    gpu_uuid=gpu["gpu_uuid"],
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=decode_len,
                    warmup=warmup,
                    repeat=repeat,
                    seed=seed,
                    status="unsupported",
                    error=case_error,
                    run_id=run_id,
                    command=command_placeholder,
                    binary_path=binary,
                    binary_build_id_value=build_id,
                    started_at=started_at,
                    ended_at=iso_now(),
                    prompt_source="synthetic_fixed_tokens",
                    prompt_count=bsz * repeat,
                )
                append_csv(csv_output, [row])
                continue

            port = find_free_port()
            log_path = raw_dir / f"{sanitize_filename(model.stem)}_bsz{bsz}_pp{prompt_len}.server.log"
            process, command, server_url = start_server(
                binary=binary,
                model=model,
                log_path=log_path,
                port=port,
                ctx_size=max(required_ctx, prompt_len + decode_len),
                n_batch=max(n_batch, bsz),
                n_ubatch=n_ubatch,
                parallel=bsz,
                extra_args=extra_args,
                env=env,
            )
            status = "failed"
            error = ""
            try:
                with GpuTelemetrySampler(
                    path=telemetry_path,
                    run_id=run_id,
                    gpu_uuid=gpu["gpu_uuid"],
                    process_name="llama-server",
                ):
                    wait_for_server(server_url, process, startup_timeout)
                    prompts = [(i, prompt) for i, prompt in enumerate(make_synthetic_prompt_tokens(prompt_len=prompt_len, bsz=bsz, repeat=repeat, seed=seed))]
                    if warmup and prompts:
                        stream_completion(server_url, prompts[0][1], decode_len=decode_len, seed=seed, timeout=request_timeout)
                    results = collect_stream_metrics(
                        server_url=server_url,
                        prompts=prompts,
                        decode_len=decode_len,
                        seed=seed,
                        timeout=request_timeout,
                        concurrency=bsz,
                    )
                ended_at = iso_now()
                summary, request_rows = build_latency_rows(
                    results,
                    model_path=model,
                    backend="llama.cpp",
                    runner="llama-server",
                    benchmark_kind="synthetic_latency",
                    device=gpu["device"],
                    gpu_name=gpu["gpu_name"],
                    gpu_uuid=gpu["gpu_uuid"],
                    bsz=bsz,
                    decode_len=decode_len,
                    warmup=warmup,
                    repeat=repeat,
                    seed=seed,
                    run_id=run_id,
                    command=command,
                    binary_path=binary,
                    binary_build_id=build_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    prompt_source="synthetic_fixed_tokens",
                )
                with (raw_dir / f"{run_id}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in request_rows:
                        handle.write(json.dumps(row, ensure_ascii=True))
                        handle.write("\n")
                append_csv(csv_output, [summary])
                status = str(summary["status"])
                error = str(summary["error"])
            except Exception as exc:
                ended_at = iso_now()
                error = str(exc)
                row = row_for_skipped_case(
                    model=model,
                    backend="llama.cpp",
                    runner="llama-server",
                    benchmark_kind="synthetic_latency",
                    device=gpu["device"],
                    gpu_name=gpu["gpu_name"],
                    gpu_uuid=gpu["gpu_uuid"],
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=decode_len,
                    warmup=warmup,
                    repeat=repeat,
                    seed=seed,
                    status="failed",
                    error=error,
                    run_id=run_id,
                    command=command,
                    binary_path=binary,
                    binary_build_id_value=build_id,
                    started_at=started_at,
                    ended_at=ended_at,
                    prompt_source="synthetic_fixed_tokens",
                    prompt_count=bsz * repeat,
                )
                append_csv(csv_output, [row])
            finally:
                stop_server(process)
            append_run_manifest(
                manifest_path,
                run_id=run_id,
                benchmark_kind="synthetic_latency",
                model=model,
                command=command,
                gpu=gpu,
                status=status,
                error=error,
                started_at=started_at,
                ended_at=ended_at,
                output=log_path,
            )


def run_gsm8k_latency(
    *,
    binary: Path,
    build_id: str,
    model: Path,
    dataset: Path,
    results_dir: Path,
    manifest_path: Path,
    telemetry_path: Path,
    gpu: dict[str, str],
    concurrency_values: list[int],
    decode_len: int,
    max_ctx_size: int,
    n_batch: int,
    n_ubatch: int,
    warmup: int,
    seed: int,
    max_parallel: int,
    startup_timeout: float,
    request_timeout: float,
    extra_args: list[str],
    env: dict[str, str],
    gsm8k_limit: int | None,
) -> None:
    request_csv = results_dir / "task5_latency_gsm8k_requests.csv"
    summary_csv = results_dir / "task5_latency_gsm8k_summary.csv"
    raw_dir = results_dir / "raw" / "latency_gsm8k"
    for concurrency in concurrency_values:
        run_id = make_run_id("latency-gsm8k")
        started_at = iso_now()
        if concurrency > max_parallel:
            row = row_for_skipped_case(
                model=model,
                backend="llama.cpp",
                runner="llama-server",
                benchmark_kind="gsm8k_latency",
                device=gpu["device"],
                gpu_name=gpu["gpu_name"],
                gpu_uuid=gpu["gpu_uuid"],
                bsz=concurrency,
                prompt_len=0,
                decode_len=decode_len,
                warmup=warmup,
                repeat=1,
                seed=seed,
                status="unsupported",
                error=f"concurrency {concurrency} exceeds llama.cpp RWKV n_seq_max limit {max_parallel}",
                run_id=run_id,
                command=[str(binary), "-m", str(model)],
                binary_path=binary,
                binary_build_id_value=build_id,
                started_at=started_at,
                ended_at=iso_now(),
                prompt_source=str(dataset),
                prompt_count=0,
            )
            append_csv(summary_csv, [row])
            continue

        port = find_free_port()
        log_path = raw_dir / f"{sanitize_filename(model.stem)}_c{concurrency}.server.log"
        process, command, server_url = start_server(
            binary=binary,
            model=model,
            log_path=log_path,
            port=port,
            ctx_size=max_ctx_size,
            n_batch=max(n_batch, concurrency),
            n_ubatch=n_ubatch,
            parallel=concurrency,
            extra_args=extra_args,
            env=env,
        )
        status = "failed"
        error = ""
        try:
            with GpuTelemetrySampler(
                path=telemetry_path,
                run_id=run_id,
                gpu_uuid=gpu["gpu_uuid"],
                process_name="llama-server",
            ):
                wait_for_server(server_url, process, startup_timeout)
                prompts = make_gsm8k_prompt_tokens(server_url, dataset, limit=gsm8k_limit)
                if warmup and prompts:
                    stream_completion(server_url, prompts[0][1], decode_len=decode_len, seed=seed, timeout=request_timeout)
                results = collect_stream_metrics(
                    server_url=server_url,
                    prompts=prompts,
                    decode_len=decode_len,
                    seed=seed,
                    timeout=request_timeout,
                    concurrency=concurrency,
                )
            ended_at = iso_now()
            summary, request_rows = build_latency_rows(
                results,
                model_path=model,
                backend="llama.cpp",
                runner="llama-server",
                benchmark_kind="gsm8k_latency",
                device=gpu["device"],
                gpu_name=gpu["gpu_name"],
                gpu_uuid=gpu["gpu_uuid"],
                bsz=concurrency,
                decode_len=decode_len,
                warmup=warmup,
                repeat=1,
                seed=seed,
                run_id=run_id,
                command=command,
                binary_path=binary,
                binary_build_id=build_id,
                started_at=started_at,
                ended_at=ended_at,
                prompt_source=str(dataset),
            )
            append_csv(request_csv, request_rows)
            append_csv(summary_csv, [summary])
            with (raw_dir / f"{run_id}.jsonl").open("w", encoding="utf-8") as handle:
                for row in request_rows:
                    handle.write(json.dumps(row, ensure_ascii=True))
                    handle.write("\n")
            status = str(summary["status"])
            error = str(summary["error"])
        except Exception as exc:
            ended_at = iso_now()
            error = str(exc)
            row = row_for_skipped_case(
                model=model,
                backend="llama.cpp",
                runner="llama-server",
                benchmark_kind="gsm8k_latency",
                device=gpu["device"],
                gpu_name=gpu["gpu_name"],
                gpu_uuid=gpu["gpu_uuid"],
                bsz=concurrency,
                prompt_len=0,
                decode_len=decode_len,
                warmup=warmup,
                repeat=1,
                seed=seed,
                status="failed",
                error=error,
                run_id=run_id,
                command=command,
                binary_path=binary,
                binary_build_id_value=build_id,
                started_at=started_at,
                ended_at=ended_at,
                prompt_source=str(dataset),
                prompt_count=0,
            )
            append_csv(summary_csv, [row])
        finally:
            stop_server(process)
        append_run_manifest(
            manifest_path,
            run_id=run_id,
            benchmark_kind="gsm8k_latency",
            model=model,
            command=command,
            gpu=gpu,
            status=status,
            error=error,
            started_at=started_at,
            ended_at=ended_at,
            output=log_path,
        )


def main() -> None:
    llama_root_default = Path(__file__).resolve().parents[2]
    workspace_default = workspace_root_from_llama_root(llama_root_default)
    parser = argparse.ArgumentParser(description="Collect README Task 5 benchmark data for llama.cpp.")
    parser.add_argument("--llama-root", type=Path, default=llama_root_default)
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument("--binary-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--kind", choices=["all", "throughput", "latency-synthetic", "latency-gsm8k"], default="all")
    parser.add_argument("--bsz", default=",".join(map(str, BSZ_DEFAULT)))
    parser.add_argument("--prompt-lens", default=",".join(map(str, PROMPT_LEN_DEFAULT)))
    parser.add_argument("--gsm8k-concurrency", default="1,16,64,128")
    parser.add_argument("--gsm8k-limit", type=int, default=None)
    parser.add_argument("--decode-len", type=int, default=DECODE_LEN_DEFAULT)
    parser.add_argument("--max-ctx-size", type=int, default=131072)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--max-parallel", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--runner-timeout", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extra-batched-arg", action="append", default=[])
    parser.add_argument("--extra-server-arg", action="append", default=[])
    args = parser.parse_args()

    llama_root = args.llama_root.resolve()
    workspace_root = args.workspace_root.resolve()
    binary_dir = (args.binary_dir or llama_root / "build-cuda" / "bin").resolve()
    results_dir = (args.results_dir or llama_root / "results").resolve()
    dataset = (args.dataset or workspace_root / "data" / "gsm8k.jsonl").resolve()
    models = [model.resolve() for model in (args.models or default_models(workspace_root / "weights"))]
    bsz_values = parse_int_list(args.bsz)
    prompt_lens = parse_int_list(args.prompt_lens)
    concurrency_values = parse_int_list(args.gsm8k_concurrency)
    ensure_results_tree(results_dir)

    if args.overwrite:
        for name in (
            "task5_throughput.csv",
            "task5_latency_synthetic.csv",
            "task5_latency_gsm8k_requests.csv",
            "task5_latency_gsm8k_summary.csv",
            "gpu_telemetry.csv",
            "manifest.jsonl",
            "preflight.json",
        ):
            path = results_dir / name
            if path.exists():
                path.unlink()
        for name in ("throughput", "latency_synthetic", "latency_gsm8k"):
            raw_path = results_dir / "raw" / name
            if raw_path.exists():
                shutil.rmtree(raw_path)
        ensure_results_tree(results_dir)

    if not models:
        raise SystemExit(f"no Task 5 GGUF models found under {workspace_root / 'weights'}")
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    batched_binary = binary_dir / "llama-batched-bench"
    server_binary = binary_dir / "llama-server"
    if args.kind in ("all", "throughput") and not batched_binary.exists():
        raise SystemExit(f"missing binary: {batched_binary}")
    if args.kind in ("all", "latency-synthetic", "latency-gsm8k") and not server_binary.exists():
        raise SystemExit(f"missing binary: {server_binary}")

    gpu = query_gpu_info()
    if "RTX 5090" not in gpu["gpu_name"]:
        raise SystemExit(f"Task 5 requires RTX 5090, got {gpu['gpu_name']}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    existing_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        str(binary_dir) if not existing_library_path else f"{binary_dir}:{existing_library_path}"
    )
    os.environ["LD_LIBRARY_PATH"] = env["LD_LIBRARY_PATH"]
    manifest_path = results_dir / "manifest.jsonl"
    telemetry_path = results_dir / "gpu_telemetry.csv"
    build_ids = {
        "llama-batched-bench": binary_build_id(batched_binary) if batched_binary.exists() else "",
        "llama-server": binary_build_id(server_binary) if server_binary.exists() else "",
    }
    write_preflight(
        results_dir / "preflight.json",
        gpu=gpu,
        models=models,
        binaries={"llama-batched-bench": batched_binary, "llama-server": server_binary},
    )

    for model in models:
        if not model.exists():
            raise SystemExit(f"model not found: {model}")
        if not is_valid_gguf(model):
            raise SystemExit(f"invalid GGUF magic: {model}")
        print(f"[task5] model={model.name}", flush=True)
        if args.kind in ("all", "throughput"):
            print("[task5] synthetic throughput", flush=True)
            run_throughput(
                binary=batched_binary,
                build_id=build_ids["llama-batched-bench"],
                model=model,
                results_dir=results_dir,
                manifest_path=manifest_path,
                telemetry_path=telemetry_path,
                gpu=gpu,
                bsz_values=bsz_values,
                prompt_lens=prompt_lens,
                decode_len=args.decode_len,
                max_ctx_size=args.max_ctx_size,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                max_parallel=args.max_parallel,
                extra_args=args.extra_batched_arg,
                env=env,
                timeout_s=args.runner_timeout or None,
            )
        if args.kind in ("all", "latency-synthetic"):
            print("[task5] synthetic latency", flush=True)
            run_synthetic_latency(
                binary=server_binary,
                build_id=build_ids["llama-server"],
                model=model,
                results_dir=results_dir,
                manifest_path=manifest_path,
                telemetry_path=telemetry_path,
                gpu=gpu,
                bsz_values=bsz_values,
                prompt_lens=prompt_lens,
                decode_len=args.decode_len,
                max_ctx_size=args.max_ctx_size,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                max_parallel=args.max_parallel,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
                extra_args=args.extra_server_arg,
                env=env,
            )
        if args.kind in ("all", "latency-gsm8k"):
            print("[task5] GSM8K latency", flush=True)
            run_gsm8k_latency(
                binary=server_binary,
                build_id=build_ids["llama-server"],
                model=model,
                dataset=dataset,
                results_dir=results_dir,
                manifest_path=manifest_path,
                telemetry_path=telemetry_path,
                gpu=gpu,
                concurrency_values=concurrency_values,
                decode_len=args.decode_len,
                max_ctx_size=args.max_ctx_size,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                warmup=args.warmup,
                seed=args.seed,
                max_parallel=args.max_parallel,
                startup_timeout=args.startup_timeout,
                request_timeout=args.request_timeout,
                extra_args=args.extra_server_arg,
                env=env,
                gsm8k_limit=args.gsm8k_limit,
            )

    print(f"[task5] wrote results under {results_dir}", flush=True)


if __name__ == "__main__":
    main()
