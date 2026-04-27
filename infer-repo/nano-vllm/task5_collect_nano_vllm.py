#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import benchmark_openai_api as api_bench


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
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "e2el_p50_ms",
    "e2el_p90_ms",
    "e2el_p95_ms",
    "e2el_p99_ms",
]


@dataclass
class DirectResult:
    prefill_tokens: int | None = None
    output_tokens: int | None = None
    prefill_time_s: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    error: str | None = None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    weight = pos - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def first_dist(summary: dict[str, float] | None, key: str) -> float | None:
    if not summary:
        return None
    value = summary.get(key)
    return None if value is None else float(value)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def load_gsm8k_questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            question = str(record.get("question") or "").strip()
            if question:
                questions.append(question)
    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions


def build_prompt_token_ids(
    *,
    questions: list[str],
    tokenizer,
    prompt_len: int,
    count: int,
    seed: int,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    cursor = seed % len(questions)
    for index in range(count):
        token_ids: list[int] = []
        while len(token_ids) < prompt_len:
            question = questions[(cursor + index + len(token_ids)) % len(questions)]
            text = f"User: {question}\n\nAssistant:"
            token_ids.extend(int(token_id) for token_id in tokenizer.encode(text))
            if len(token_ids) < prompt_len:
                token_ids.extend(int(token_id) for token_id in tokenizer.encode("\n\nUser: "))
        prompts.append(token_ids[:prompt_len])
    return prompts


def write_prompt_token_jsonl(path: Path, prompt_token_ids: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for token_ids in prompt_token_ids:
            f.write(json.dumps({"prompt_token_ids": token_ids}, separators=(",", ":")) + "\n")


def direct_once(
    *,
    model_pth: Path,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    prompt_token_ids: list[int],
    args,
) -> DirectResult:
    import benchmark_rwkv as direct_bench

    (
        _actual_n,
        _resident_blocks,
        prefill_tokens,
        steps,
        prefill_time_s,
        prefill_tps,
        decode_tps,
        _steady_decode_tps,
    ) = direct_bench.run_benchmark(
        str(model_pth),
        bsz,
        prompt_len,
        max(0, decode_len - 1),
        args.gpu_memory_utilization,
        args.max_state_slots,
        args.rwkv_state_cache_safety_reserve_slots,
        args.rwkv_prefill_token_budget,
        args.rwkv_prefill_max_batch_size,
        args.rwkv_prefill_chunk_size,
        args.rwkv_state_cache_enable,
        False,
        False,
        args.enforce_eager,
        args.seed,
        prompt_tokens=prompt_token_ids,
    )
    return DirectResult(
        prefill_tokens=prefill_tokens,
        output_tokens=bsz * steps,
        prefill_time_s=prefill_time_s,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
    )


def run_direct(
    *,
    model_pth: Path,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    prompt_token_ids: list[int],
    args,
) -> DirectResult:
    try:
        for _ in range(args.warmup):
            direct_once(
                model_pth=model_pth,
                bsz=bsz,
                prompt_len=prompt_len,
                decode_len=decode_len,
                prompt_token_ids=prompt_token_ids,
                args=args,
            )
        results = [
            direct_once(
                model_pth=model_pth,
                bsz=bsz,
                prompt_len=prompt_len,
                decode_len=decode_len,
                prompt_token_ids=prompt_token_ids,
                args=args,
            )
            for _ in range(args.repeat)
        ]
    except Exception as exc:
        return DirectResult(error=f"{type(exc).__name__}: {exc}")

    return DirectResult(
        prefill_tokens=max(result.prefill_tokens or 0 for result in results),
        output_tokens=max(result.output_tokens or 0 for result in results),
        prefill_time_s=mean([result.prefill_time_s for result in results if result.prefill_time_s is not None]),
        prefill_tps=mean([result.prefill_tps for result in results if result.prefill_tps is not None]),
        decode_tps=mean([result.decode_tps for result in results if result.decode_tps is not None]),
    )


async def run_api_once(
    *,
    prompt_file: Path,
    bsz: int,
    decode_len: int,
    args,
) -> tuple[api_bench.BenchmarkSummary, list[api_bench.RequestMetrics]]:
    run_args = SimpleNamespace(
        base_url=args.base_url,
        model=args.served_model_name,
        endpoint="completions",
        load_mode="closed-loop",
        users=bsz,
        arrival_rate=0.0,
        max_in_flight=0,
        max_connections=max(args.max_connections, bsz),
        max_keepalive_connections=max(args.max_connections, bsz),
        total_requests=bsz,
        duration=None,
        stream=True,
        max_tokens=decode_len,
        temperature=0.0,
        system_prompt=None,
        prompt=[],
        prompt_file=str(prompt_file),
        prompt_repeat=1,
        api_key=args.api_key,
        timeout=args.timeout,
        connect_timeout=args.connect_timeout,
        ramp_seconds=0.0,
        progress_interval=0.0,
        seed=args.seed,
        run_label=f"bsz={bsz}",
    )
    return await api_bench.run_single_benchmark(run_args)


def run_api(
    *,
    prompt_file: Path,
    bsz: int,
    decode_len: int,
    args,
) -> tuple[api_bench.BenchmarkSummary | None, list[api_bench.RequestMetrics], str | None]:
    try:
        for _ in range(args.warmup):
            asyncio.run(run_api_once(prompt_file=prompt_file, bsz=bsz, decode_len=decode_len, args=args))
        summaries: list[api_bench.BenchmarkSummary] = []
        metrics: list[api_bench.RequestMetrics] = []
        for _ in range(args.repeat):
            summary, run_metrics = asyncio.run(
                run_api_once(prompt_file=prompt_file, bsz=bsz, decode_len=decode_len, args=args)
            )
            summaries.append(summary)
            metrics.extend(run_metrics)
        if not summaries:
            return None, [], "repeat must be >= 1"
        return summaries[-1], metrics, None
    except Exception as exc:
        return None, [], f"{type(exc).__name__}: {exc}"


def build_row(
    *,
    model_pth: Path,
    device: str,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    args,
    direct: DirectResult,
    api_summary: api_bench.BenchmarkSummary | None,
    metrics: list[api_bench.RequestMetrics],
    api_error: str | None,
) -> dict[str, Any]:
    ok_metrics = [metric for metric in metrics if metric.ok]
    ttft_values = [metric.ttft_s for metric in ok_metrics if metric.ttft_s is not None]
    e2el_values = [metric.latency_s for metric in ok_metrics]
    tgt_values = [
        metric.latency_s - metric.ttft_s
        for metric in ok_metrics
        if metric.ttft_s is not None and metric.latency_s >= metric.ttft_s
    ]
    output_tokens = sum(metric.completion_tokens or 0 for metric in ok_metrics) or direct.output_tokens
    e2el_s = mean(e2el_values)
    tgt_s = mean(tgt_values)
    ttft_s = mean(ttft_values)
    e2e_tps = (output_tokens / sum(e2el_values)) if output_tokens and sum(e2el_values) > 0 else None
    api_decode_tps = (bsz * max(0, decode_len - 1) / tgt_s) if tgt_s and tgt_s > 0 else None
    status = "ok"
    errors = []
    if direct.error:
        status = "failed"
        errors.append(f"direct={direct.error}")
    if api_error:
        status = "failed"
        errors.append(f"api={api_error}")
    elif api_summary is not None and api_summary.error_requests:
        status = "failed"
        errors.extend(api_summary.sample_errors)
    elif metrics and len(ok_metrics) != len(metrics):
        status = "failed"
        errors.extend(metric.error or "request failed" for metric in metrics if not metric.ok)

    row = {
        "repo": "nano-vllm",
        "backend": "direct_engine+openai_api",
        "model_path": str(model_pth),
        "model_format": "pth",
        "device": device,
        "dtype": "fp16",
        "quantization": "fp16",
        "bsz": bsz,
        "prompt_len": prompt_len,
        "decode_len": decode_len,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "status": status,
        "error": "; ".join(errors),
        "prefill_tokens": direct.prefill_tokens,
        "output_tokens": output_tokens,
        "prefill_time_s": direct.prefill_time_s,
        "ttft_s": ttft_s,
        "e2el_s": e2el_s,
        "token_generation_time_s": tgt_s,
        "prefill_tps": direct.prefill_tps,
        "decode_tps": api_decode_tps or direct.decode_tps,
        "e2e_tps": e2e_tps,
        "time_per_output_token_ms": (tgt_s / max(1, decode_len - 1) * 1000.0) if tgt_s else None,
        "itl_mean_ms": mean([metric.itl_mean_ms for metric in ok_metrics if metric.itl_mean_ms is not None]),
        "itl_p50_ms": mean([metric.itl_p50_ms for metric in ok_metrics if metric.itl_p50_ms is not None]),
        "itl_p90_ms": mean([metric.itl_p90_ms for metric in ok_metrics if metric.itl_p90_ms is not None]),
        "itl_p95_ms": mean([metric.itl_p95_ms for metric in ok_metrics if metric.itl_p95_ms is not None]),
        "itl_p99_ms": mean([metric.itl_p99_ms for metric in ok_metrics if metric.itl_p99_ms is not None]),
        "ttft_p50_ms": percentile(ttft_values, 0.50) * 1000.0 if ttft_values else None,
        "ttft_p90_ms": percentile(ttft_values, 0.90) * 1000.0 if ttft_values else None,
        "ttft_p95_ms": percentile(ttft_values, 0.95) * 1000.0 if ttft_values else None,
        "ttft_p99_ms": percentile(ttft_values, 0.99) * 1000.0 if ttft_values else None,
        "e2el_p50_ms": percentile(e2el_values, 0.50) * 1000.0 if e2el_values else None,
        "e2el_p90_ms": percentile(e2el_values, 0.90) * 1000.0 if e2el_values else None,
        "e2el_p95_ms": percentile(e2el_values, 0.95) * 1000.0 if e2el_values else None,
        "e2el_p99_ms": percentile(e2el_values, 0.99) * 1000.0 if e2el_values else None,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in CSV_FIELDS})


def load_existing_csv(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    existing: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (int(row["bsz"]), int(row["prompt_len"]))
        except (KeyError, ValueError):
            continue
        existing[key] = dict(row)
    return existing


def merge_direct_into_existing(existing: dict[str, Any], direct: DirectResult) -> dict[str, Any]:
    row = dict(existing)
    if direct.error:
        previous_error = row.get("error", "")
        direct_error = f"direct={direct.error}"
        row["error"] = "; ".join(part for part in [previous_error, direct_error] if part)
        row["status"] = "failed"
        return row
    row["prefill_tokens"] = direct.prefill_tokens
    row["prefill_time_s"] = direct.prefill_time_s
    row["prefill_tps"] = direct.prefill_tps
    if direct.decode_tps is not None and not row.get("decode_tps"):
        row["decode_tps"] = direct.decode_tps
    error = str(row.get("error", ""))
    error_parts = [
        part.strip()
        for part in error.split(";")
        if part.strip() and not part.strip().startswith("direct=")
    ]
    row["error"] = "; ".join(error_parts)
    if not row["error"]:
        row["status"] = "ok"
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect README task 5 nano-vllm fp16 benchmark CSV.")
    parser.add_argument("--model-pth", default="../../weights/rwkv7-g1f-1.5b-20260419-ctx8192.pth")
    parser.add_argument("--gsm8k-path", default="../../data/gsm8k.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model-name", default="rwkv7-g1f-1.5b-fp16")
    parser.add_argument("--bsz", type=int, nargs="+", default=[1, 16, 64, 128, 256, 512, 1024])
    parser.add_argument("--prompt-len", type=int, nargs="+", default=[16, 256, 512, 1024, 4096])
    parser.add_argument("--decode-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/task5_nano_vllm_fp16.csv")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--max-connections", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-state-slots", type=int, default=-1)
    parser.add_argument("--rwkv-state-cache-safety-reserve-slots", type=int, default=0)
    parser.add_argument("--rwkv-prefill-token-budget", type=int, default=4096)
    parser.add_argument("--rwkv-prefill-max-batch-size", type=int, default=128)
    parser.add_argument("--rwkv-prefill-chunk-size", type=int, default=256)
    parser.add_argument("--rwkv-state-cache-enable", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="When --skip-api is used, preserve existing CSV API metrics and update direct TPS fields.",
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from nanovllm.tokenizers import get_rwkv_tokenizer

    model_pth = Path(args.model_pth).resolve()
    gsm8k_path = Path(args.gsm8k_path).resolve()
    out_path = Path(args.out)
    existing_rows = load_existing_csv(out_path) if args.merge_existing else {}
    questions = load_gsm8k_questions(gsm8k_path)
    tokenizer = get_rwkv_tokenizer()
    rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="task5_nano_vllm_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for prompt_len in args.prompt_len:
            prompt_count = max(args.bsz) * max(1, args.repeat)
            prompt_token_ids = build_prompt_token_ids(
                questions=questions,
                tokenizer=tokenizer,
                prompt_len=prompt_len,
                count=prompt_count,
                seed=args.seed,
            )
            prompt_file = tmpdir_path / f"prompts_pl{prompt_len}.jsonl"
            write_prompt_token_jsonl(prompt_file, prompt_token_ids)
            for bsz in args.bsz:
                started = time.perf_counter()
                print(f"[task5] bsz={bsz} prompt_len={prompt_len} decode_len={args.decode_len}", flush=True)
                direct = DirectResult()
                if not args.skip_direct:
                    direct = run_direct(
                        model_pth=model_pth,
                        bsz=bsz,
                        prompt_len=prompt_len,
                        decode_len=args.decode_len,
                        prompt_token_ids=prompt_token_ids[0],
                        args=args,
                    )
                api_summary = None
                metrics: list[api_bench.RequestMetrics] = []
                api_error = None
                if not args.skip_api:
                    api_summary, metrics, api_error = run_api(
                        prompt_file=prompt_file,
                        bsz=bsz,
                        decode_len=args.decode_len,
                        args=args,
                    )
                key = (bsz, prompt_len)
                if args.merge_existing and args.skip_api and key in existing_rows:
                    row = merge_direct_into_existing(existing_rows[key], direct)
                else:
                    row = build_row(
                            model_pth=model_pth,
                            device=args.device,
                            bsz=bsz,
                            prompt_len=prompt_len,
                            decode_len=args.decode_len,
                            args=args,
                            direct=direct,
                            api_summary=api_summary,
                            metrics=metrics,
                            api_error=api_error,
                        )
                rows.append(row)
                write_csv(out_path, rows)
                elapsed = time.perf_counter() - started
                print(f"[task5] wrote {out_path} elapsed_s={elapsed:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
