#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import socket
import subprocess
import time
from pathlib import Path
from urllib import request
from urllib.error import URLError

from convert_batched_bench import convert_rows
from stream_latency_bench import run_latency_case
from task5_common import (
    BSZ_DEFAULT,
    DECODE_LEN_DEFAULT,
    PROMPT_LEN_DEFAULT,
    append_csv,
    base_row,
    ensure_results_tree,
    parse_int_list,
    sanitize_filename,
    workspace_root_from_llama_root,
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
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        return 18080


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def is_valid_gguf(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(4) == b"GGUF"


def append_model_failed_rows(
    *,
    csv_output: Path,
    model: Path,
    device: str,
    bsz_values: list[int],
    prompt_lens: list[int],
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    error: str,
) -> None:
    rows = []
    for backend in ("llama-batched-bench", "llama-server-stream"):
        for bsz in bsz_values:
            for prompt_len in prompt_lens:
                row = base_row(
                    model_path=model,
                    backend=backend,
                    device=device,
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=decode_len,
                    warmup=warmup,
                    repeat=repeat,
                    seed=seed,
                    status="failed",
                    error=error,
                )
                row.update({"prefill_tokens": bsz * prompt_len, "output_tokens": bsz * decode_len})
                rows.append(row)
    append_csv(csv_output, rows)


def run_batched_bench(
    *,
    binary: Path,
    model: Path,
    raw_output: Path,
    csv_output: Path,
    device: str,
    bsz_values: list[int],
    prompt_lens: list[int],
    decode_len: int,
    max_ctx_size: int,
    n_batch: int,
    n_ubatch: int,
    warmup: int,
    repeat: int,
    seed: int,
    extra_args: list[str],
    max_parallel: int,
) -> None:
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    runnable_bsz = [bsz for bsz in bsz_values if bsz <= max_parallel]
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
        ",".join(map(str, prompt_lens)),
        "-ntg",
        str(decode_len),
        "-npl",
        ",".join(map(str, runnable_bsz or [1])),
        "--output-format",
        "jsonl",
        *extra_args,
    ]
    rows = []
    completed = None
    if runnable_bsz:
        with raw_output.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    else:
        raw_output.write_text("", encoding="utf-8")

    if completed is not None and completed.returncode == 0:
        rows.extend(
            convert_rows(
                raw_output,
                model_path=model,
                backend="llama-batched-bench",
                device=device,
                warmup=warmup,
                repeat=repeat,
                seed=seed,
            )
        )

    seen = {(int(row["bsz"]), int(row["prompt_len"])) for row in rows}
    for bsz in bsz_values:
        for prompt_len in prompt_lens:
            if (bsz, prompt_len) in seen:
                continue
            required_ctx = bsz * (prompt_len + decode_len)
            if bsz > max_parallel:
                status = "unsupported"
                error = f"bsz {bsz} exceeds llama.cpp RWKV n_seq_max limit {max_parallel}"
            elif required_ctx > max_ctx_size:
                status = "unsupported"
                error = f"required ctx {required_ctx} exceeds max ctx {max_ctx_size}"
            else:
                status = "failed"
                exit_code = completed.returncode if completed is not None else "not-run"
                error = f"llama-batched-bench did not return this case; exit_code={exit_code}"
            row = base_row(
                model_path=model,
                backend="llama-batched-bench",
                device=device,
                bsz=bsz,
                prompt_len=prompt_len,
                decode_len=decode_len,
                warmup=warmup,
                repeat=repeat,
                seed=seed,
                status=status,
                error=error,
            )
            row.update({"prefill_tokens": bsz * prompt_len, "output_tokens": bsz * decode_len})
            rows.append(row)

    append_csv(csv_output, rows)


def run_latency_matrix(
    *,
    server_binary: Path,
    model: Path,
    dataset: Path,
    csv_output: Path,
    raw_dir: Path,
    device: str,
    bsz_values: list[int],
    prompt_lens: list[int],
    decode_len: int,
    max_ctx_size: int,
    n_batch: int,
    n_ubatch: int,
    warmup: int,
    repeat: int,
    seed: int,
    port: int,
    timeout: float,
    startup_timeout: float,
    extra_args: list[str],
    max_parallel: int,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for bsz in bsz_values:
        for prompt_len in prompt_lens:
            required_ctx = bsz * (prompt_len + decode_len)
            if bsz > max_parallel or required_ctx > max_ctx_size:
                error = (
                    f"bsz {bsz} exceeds llama.cpp RWKV n_seq_max limit {max_parallel}"
                    if bsz > max_parallel
                    else f"required ctx {required_ctx} exceeds max ctx {max_ctx_size}"
                )
                row = base_row(
                    model_path=model,
                    backend="llama-server-stream",
                    device=device,
                    bsz=bsz,
                    prompt_len=prompt_len,
                    decode_len=decode_len,
                    warmup=warmup,
                    repeat=repeat,
                    seed=seed,
                    status="unsupported",
                    error=error,
                )
                row.update({"prefill_tokens": bsz * prompt_len, "output_tokens": 0})
                append_csv(csv_output, [row])
                continue

            log_path = raw_dir / f"{sanitize_filename(model.stem)}_bsz{bsz}_pp{prompt_len}.server.log"
            case_port = find_free_port() if port == 0 else port
            server_url = f"http://127.0.0.1:{case_port}"
            command = [
                str(server_binary),
                "-m",
                str(model),
                "-c",
                str(required_ctx),
                "-b",
                str(max(n_batch, bsz)),
                "-ub",
                str(n_ubatch),
                "-np",
                str(max(bsz, 1)),
                "--host",
                "127.0.0.1",
                "--port",
                str(case_port),
                "--no-webui",
                "-lv",
                "1",
                *extra_args,
            ]
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    wait_for_server(server_url, process, startup_timeout)
                    row = run_latency_case(
                        server_url=server_url,
                        dataset=dataset,
                        model_path=model,
                        device=device,
                        bsz=bsz,
                        prompt_len=prompt_len,
                        decode_len=decode_len,
                        warmup=warmup,
                        repeat=repeat,
                        seed=seed,
                        timeout=timeout,
                    )
                except Exception as exc:
                    row = base_row(
                        model_path=model,
                        backend="llama-server-stream",
                        device=device,
                        bsz=bsz,
                        prompt_len=prompt_len,
                        decode_len=decode_len,
                        warmup=warmup,
                        repeat=repeat,
                        seed=seed,
                        status="failed",
                        error=str(exc),
                    )
                    row.update({"prefill_tokens": bsz * prompt_len, "output_tokens": 0})
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            append_csv(csv_output, [row])


def main() -> None:
    llama_root_default = Path(__file__).resolve().parents[2]
    workspace_default = workspace_root_from_llama_root(llama_root_default)
    parser = argparse.ArgumentParser(description="Collect README task 5 benchmark data for llama.cpp.")
    parser.add_argument("--llama-root", type=Path, default=llama_root_default)
    parser.add_argument("--workspace-root", type=Path, default=workspace_default)
    parser.add_argument("--binary-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--bsz", default=",".join(map(str, BSZ_DEFAULT)))
    parser.add_argument("--prompt-lens", default=",".join(map(str, PROMPT_LEN_DEFAULT)))
    parser.add_argument("--decode-len", type=int, default=DECODE_LEN_DEFAULT)
    parser.add_argument("--max-ctx-size", type=int, default=131072)
    parser.add_argument("--n-batch", type=int, default=2048)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--max-parallel", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=0, help="llama-server port; 0 chooses a free local port")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extra-batched-arg", action="append", default=[])
    parser.add_argument("--extra-server-arg", action="append", default=[])
    args = parser.parse_args()

    llama_root = args.llama_root.resolve()
    workspace_root = args.workspace_root.resolve()
    binary_dir = (args.binary_dir or llama_root / "build-cpu" / "bin").resolve()
    models = args.models or sorted((workspace_root / "weights").glob("*.gguf"))
    dataset = (args.dataset or workspace_root / "data" / "gsm8k.jsonl").resolve()
    results_dir = (args.results_dir or llama_root / "results" / "task5").resolve()
    csv_output = results_dir / "llama_cpp_task5.csv"
    bsz_values = parse_int_list(args.bsz)
    prompt_lens = parse_int_list(args.prompt_lens)

    ensure_results_tree(results_dir)
    if args.overwrite and csv_output.exists():
        csv_output.unlink()
    if not models:
        raise SystemExit(f"no GGUF models found under {workspace_root / 'weights'}")
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    batched_binary = binary_dir / "llama-batched-bench"
    server_binary = binary_dir / "llama-server"
    if not args.skip_throughput and not batched_binary.exists():
        raise SystemExit(f"missing binary: {batched_binary}")
    if not args.skip_latency and not server_binary.exists():
        raise SystemExit(f"missing binary: {server_binary}")

    for model in models:
        model = model.resolve()
        if not model.exists():
            raise SystemExit(f"model not found: {model}")
        model_tag = sanitize_filename(model.stem)
        print(f"[task5] model={model.name}", flush=True)
        if not is_valid_gguf(model):
            print(f"[task5] invalid GGUF -> {model.name}", flush=True)
            append_model_failed_rows(
                csv_output=csv_output,
                model=model,
                device=args.device,
                bsz_values=bsz_values,
                prompt_lens=prompt_lens,
                decode_len=args.decode_len,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                error="invalid GGUF magic; replace this weight file and rerun",
            )
            continue
        if not args.skip_throughput:
            raw_path = results_dir / "raw" / "batched_bench" / f"{model_tag}.jsonl"
            print(f"[task5] throughput -> {raw_path}", flush=True)
            run_batched_bench(
                binary=batched_binary,
                model=model,
                raw_output=raw_path,
                csv_output=csv_output,
                device=args.device,
                bsz_values=bsz_values,
                prompt_lens=prompt_lens,
                decode_len=args.decode_len,
                max_ctx_size=args.max_ctx_size,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                extra_args=args.extra_batched_arg,
                max_parallel=args.max_parallel,
            )
        if not args.skip_latency:
            raw_dir = results_dir / "raw" / "latency"
            print(f"[task5] latency -> {raw_dir}", flush=True)
            run_latency_matrix(
                server_binary=server_binary,
                model=model,
                dataset=dataset,
                csv_output=csv_output,
                raw_dir=raw_dir,
                device=args.device,
                bsz_values=bsz_values,
                prompt_lens=prompt_lens,
                decode_len=args.decode_len,
                max_ctx_size=args.max_ctx_size,
                n_batch=args.n_batch,
                n_ubatch=args.n_ubatch,
                warmup=args.warmup,
                repeat=args.repeat,
                seed=args.seed,
                port=args.port,
                timeout=args.timeout,
                startup_timeout=args.startup_timeout,
                extra_args=args.extra_server_arg,
                max_parallel=args.max_parallel,
            )

    print(f"[task5] wrote {csv_output}", flush=True)


if __name__ == "__main__":
    main()
