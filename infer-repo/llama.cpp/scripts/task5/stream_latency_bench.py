#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from task5_common import base_row, mean, percentile, write_csv


def post_json(url: str, payload: dict[str, Any], *, timeout: float = 600.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return json.loads(text)


def load_gsm8k_questions(path: Path, limit: int | None = None) -> list[str]:
    questions = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(questions) >= limit:
                break
            item = json.loads(line)
            question = str(item.get("question", "")).strip()
            if question:
                questions.append(question)
    if not questions:
        raise RuntimeError(f"no questions found in {path}")
    return questions


def tokenize(server_url: str, text: str) -> list[int]:
    result = post_json(f"{server_url}/tokenize", {"content": text, "add_special": True})
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError("tokenize endpoint returned no tokens")
    return [int(token) for token in tokens]


def make_synthetic_prompt_tokens(*, prompt_len: int, bsz: int, repeat: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    prompts: list[list[int]] = []
    for i in range(bsz * repeat):
        local = random.Random(rng.randint(0, 2**31 - 1) + i)
        prompts.append([local.randint(1, 50000) for _ in range(prompt_len)])
    return prompts


def make_prompt_tokens(server_url: str, dataset: Path, prompt_len: int, bsz: int) -> list[list[int]]:
    questions = load_gsm8k_questions(dataset, limit=max(bsz, 64))
    prompts = []
    token_cache: dict[str, list[int]] = {}
    for i in range(bsz):
        text = questions[i % len(questions)]
        if text not in token_cache:
            token_cache[text] = tokenize(server_url, text)
        tokens = token_cache[text]
        repeated = []
        while len(repeated) < prompt_len:
            repeated.extend(tokens)
        prompts.append(repeated[:prompt_len])
    return prompts


def make_gsm8k_prompt_tokens(server_url: str, dataset: Path, limit: int | None = None) -> list[tuple[int, list[int]]]:
    prompts: list[tuple[int, list[int]]] = []
    for index, question in enumerate(load_gsm8k_questions(dataset, limit=limit)):
        prompts.append((index, tokenize(server_url, question)))
    return prompts


def stream_completion(
    server_url: str,
    prompt_tokens: list[int],
    *,
    decode_len: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "prompt": prompt_tokens,
        "n_predict": decode_len,
        "seed": seed,
        "stream": True,
        "ignore_eos": True,
        "cache_prompt": False,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{server_url}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    submit_s = time.perf_counter()
    arrivals: list[float] = []
    chunks: list[dict[str, Any]] = []
    with request.urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.strip()
            if not line.startswith(b"data: "):
                continue
            payload_bytes = line[6:]
            if payload_bytes == b"[DONE]":
                break
            now_s = time.perf_counter()
            try:
                event = json.loads(payload_bytes.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            if event.get("stop") is True:
                continue
            arrivals.append(now_s)
            chunks.append(event)
            if len(arrivals) >= decode_len:
                break
    end_s = time.perf_counter()

    return {
        "submit_s": submit_s,
        "end_s": end_s,
        "arrivals_s": arrivals,
        "chunks": chunks,
    }


def collect_stream_metrics(
    *,
    server_url: str,
    prompts: list[tuple[int, list[int]]],
    decode_len: int,
    seed: int,
    timeout: float,
    concurrency: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(concurrency, 1)) as executor:
        futures = {
            executor.submit(
                stream_completion,
                server_url,
                prompt,
                decode_len=decode_len,
                seed=seed + request_index,
                timeout=timeout,
            ): (request_index, sample_index, len(prompt))
            for request_index, (sample_index, prompt) in enumerate(prompts)
        }
        for future in concurrent.futures.as_completed(futures):
            request_index, sample_index, prompt_len = futures[future]
            result = future.result()
            result.update(
                {
                    "request_index": request_index,
                    "sample_index": sample_index,
                    "prompt_len": prompt_len,
                    "output_tokens": len(result["arrivals_s"]),
                }
            )
            results.append(result)
    return sorted(results, key=lambda item: int(item["request_index"]))


def build_latency_rows(
    results: list[dict[str, Any]],
    *,
    model_path: Path,
    backend: str,
    runner: str,
    benchmark_kind: str,
    device: str,
    gpu_name: str,
    gpu_uuid: str,
    bsz: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    run_id: str,
    command: str | list[str],
    binary_path: Path | str,
    binary_build_id: str,
    started_at: str,
    ended_at: str,
    prompt_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    request_rows: list[dict[str, Any]] = []
    ttft: list[float] = []
    e2el: list[float] = []
    generation: list[float] = []
    itls: list[float] = []
    prompt_tokens = 0
    output_tokens = 0
    first_submit = min((float(result["submit_s"]) for result in results), default=0.0)
    last_end = max((float(result["end_s"]) for result in results), default=0.0)
    wall_s = max(last_end - first_submit, 0.0)

    for result in results:
        arrivals = result["arrivals_s"]
        prompt_len = int(result["prompt_len"])
        prompt_tokens += prompt_len
        output_tokens += int(result["output_tokens"])
        request_ttft = arrivals[0] - result["submit_s"] if arrivals else 0.0
        request_e2el = result["end_s"] - result["submit_s"]
        request_generation = max(request_e2el - request_ttft, 0.0) if arrivals else 0.0
        request_itls = [(cur - prev) * 1000.0 for prev, cur in zip(arrivals, arrivals[1:])]
        if arrivals:
            ttft.append(request_ttft)
            e2el.append(request_e2el)
            generation.append(request_generation)
            itls.extend(request_itls)

        row = base_row(
            model_path=model_path,
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
            status="ok" if arrivals else "failed",
            error="" if arrivals else "no streamed output tokens",
            run_id=run_id,
            command=command,
            binary_path=binary_path,
            binary_build_id=binary_build_id,
            started_at=started_at,
            ended_at=ended_at,
            prompt_source=prompt_source,
            prompt_count=len(results),
        )
        row.update(
            {
                "concurrency": bsz,
                "request_index": result["request_index"],
                "sample_index": result["sample_index"],
                "prompt_tokens": prompt_len,
                "output_tokens": result["output_tokens"],
                "ttft_s": request_ttft if arrivals else "",
                "e2el_s": request_e2el,
                "token_generation_time_s": request_generation if arrivals else "",
                "decode_tps": (len(arrivals) / request_generation) if request_generation > 0 else "",
                "e2e_tps": ((prompt_len + len(arrivals)) / request_e2el) if request_e2el > 0 else "",
                "time_per_output_token_ms": 1000.0 * request_generation / max(len(arrivals) - 1, 1)
                if request_generation > 0
                else "",
                "itl_mean_ms": mean(request_itls),
                "itl_p50_ms": percentile(request_itls, 0.50),
                "itl_p90_ms": percentile(request_itls, 0.90),
                "itl_p95_ms": percentile(request_itls, 0.95),
                "itl_p99_ms": percentile(request_itls, 0.99),
            }
        )
        request_rows.append(row)

    token_generation_time_s = mean(generation)
    total_e2el_s = max(e2el) if e2el else 0.0
    summary = base_row(
        model_path=model_path,
        backend=backend,
        runner=runner,
        benchmark_kind=benchmark_kind,
        device=device,
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        bsz=bsz,
        prompt_len=0 if benchmark_kind == "gsm8k_latency" else (results[0]["prompt_len"] if results else 0),
        decode_len=decode_len,
        warmup=warmup,
        repeat=repeat,
        seed=seed,
        status="ok" if len(ttft) == len(results) else "failed",
        error="" if len(ttft) == len(results) else f"completed {len(ttft)} of {len(results)} requests",
        run_id=run_id,
        command=command,
        binary_path=binary_path,
        binary_build_id=binary_build_id,
        started_at=started_at,
        ended_at=ended_at,
        prompt_source=prompt_source,
        prompt_count=len(results),
    )
    summary.update(
        {
            "concurrency": bsz,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_s": mean(ttft),
            "ttft_p95_s": percentile(ttft, 0.95),
            "e2el_s": mean(e2el),
            "e2el_p95_s": percentile(e2el, 0.95),
            "token_generation_time_s": token_generation_time_s,
            "decode_tps": output_tokens / token_generation_time_s if token_generation_time_s > 0 else "",
            "e2e_tps": (prompt_tokens + output_tokens) / total_e2el_s if total_e2el_s > 0 else "",
            "requests_per_s": len(results) / wall_s if wall_s > 0 else "",
            "time_per_output_token_ms": 1000.0 * token_generation_time_s / max(decode_len - 1, 1)
            if token_generation_time_s > 0
            else "",
            "itl_mean_ms": mean(itls),
            "itl_p50_ms": percentile(itls, 0.50),
            "itl_p90_ms": percentile(itls, 0.90),
            "itl_p95_ms": percentile(itls, 0.95),
            "itl_p99_ms": percentile(itls, 0.99),
        }
    )
    return summary, request_rows


def summarize_requests(
    results: list[dict[str, Any]],
    *,
    model_path: Path,
    device: str,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
) -> dict[str, Any]:
    ttft = []
    e2el = []
    generation = []
    itls = []
    output_tokens = 0
    for result in results:
        arrivals = result["arrivals_s"]
        if not arrivals:
            continue
        output_tokens += len(arrivals)
        request_ttft = arrivals[0] - result["submit_s"]
        request_e2el = result["end_s"] - result["submit_s"]
        ttft.append(request_ttft)
        e2el.append(request_e2el)
        generation.append(max(request_e2el - request_ttft, 0.0))
        for prev, cur in zip(arrivals, arrivals[1:]):
            itls.append((cur - prev) * 1000.0)

    row = base_row(
        model_path=model_path,
        backend="llama-server-stream",
        device=device,
        bsz=bsz,
        prompt_len=prompt_len,
        decode_len=decode_len,
        warmup=warmup,
        repeat=repeat,
        seed=seed,
        status="ok" if len(ttft) == bsz else "failed",
        error="" if len(ttft) == bsz else f"completed {len(ttft)} of {bsz} requests",
    )
    token_generation_time_s = mean(generation)
    total_e2el_s = max(e2el) if e2el else 0.0
    row.update(
        {
            "prefill_tokens": bsz * prompt_len,
            "output_tokens": output_tokens,
            "prefill_time_s": "",
            "ttft_s": mean(ttft),
            "e2el_s": mean(e2el),
            "token_generation_time_s": token_generation_time_s,
            "prefill_tps": "",
            "decode_tps": bsz * max(decode_len - 1, 1) / token_generation_time_s if token_generation_time_s > 0 else "",
            "e2e_tps": (bsz * (prompt_len + decode_len)) / total_e2el_s if total_e2el_s > 0 else "",
            "time_per_output_token_ms": 1000.0 * token_generation_time_s / max(decode_len - 1, 1)
            if token_generation_time_s > 0
            else "",
            "itl_mean_ms": mean(itls),
            "itl_p50_ms": percentile(itls, 0.50),
            "itl_p90_ms": percentile(itls, 0.90),
            "itl_p95_ms": percentile(itls, 0.95),
            "itl_p99_ms": percentile(itls, 0.99),
        }
    )
    return row


def run_latency_case(
    *,
    server_url: str,
    dataset: Path,
    model_path: Path,
    device: str,
    bsz: int,
    prompt_len: int,
    decode_len: int,
    warmup: int,
    repeat: int,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    try:
        prompts = make_prompt_tokens(server_url, dataset, prompt_len, bsz)
        if warmup:
            stream_completion(server_url, prompts[0], decode_len=decode_len, seed=seed, timeout=timeout)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=bsz) as executor:
            futures = [
                executor.submit(
                    stream_completion,
                    server_url,
                    prompt,
                    decode_len=decode_len,
                    seed=seed + i,
                    timeout=timeout,
                )
                for i, prompt in enumerate(prompts)
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        return summarize_requests(
            results,
            model_path=model_path,
            device=device,
            bsz=bsz,
            prompt_len=prompt_len,
            decode_len=decode_len,
            warmup=warmup,
            repeat=repeat,
            seed=seed,
        )
    except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
        row = base_row(
            model_path=model_path,
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
        return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect llama-server streaming latency metrics for README task 5.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bsz", type=int, required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    row = run_latency_case(
        server_url=args.server_url.rstrip("/"),
        dataset=args.dataset,
        model_path=args.model,
        device=args.device,
        bsz=args.bsz,
        prompt_len=args.prompt_len,
        decode_len=args.decode_len,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
        timeout=args.timeout,
    )
    write_csv(args.output, [row])


if __name__ == "__main__":
    main()
