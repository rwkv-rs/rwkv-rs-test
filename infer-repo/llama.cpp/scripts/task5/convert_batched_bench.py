#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from task5_common import base_row, read_jsonl, write_csv


def convert_rows(
    input_path: Path,
    *,
    model_path: Path,
    backend: str,
    device: str,
    warmup: int,
    repeat: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in read_jsonl(input_path):
        bsz = int(item["pl"])
        prompt_len = int(item["pp"])
        decode_len = int(item["tg"])
        prefill_time_s = float(item["t_pp"])
        token_generation_time_s = float(item["t_tg"])
        prefill_tokens = prompt_len if item.get("is_pp_shared") else bsz * prompt_len
        output_tokens = bsz * decode_len
        e2el_s = prefill_time_s + token_generation_time_s
        row = base_row(
            model_path=model_path,
            backend=backend,
            device=device,
            bsz=bsz,
            prompt_len=prompt_len,
            decode_len=decode_len,
            warmup=warmup,
            repeat=repeat,
            seed=seed,
            status="ok",
        )
        row.update(
            {
                "prefill_tokens": prefill_tokens,
                "output_tokens": output_tokens,
                "prefill_time_s": prefill_time_s,
                "ttft_s": "",
                "e2el_s": e2el_s,
                "token_generation_time_s": token_generation_time_s,
                "prefill_tps": float(item["speed_pp"]),
                "decode_tps": float(item["speed_tg"]),
                "e2e_tps": float(item["speed"]),
                "time_per_output_token_ms": 1000.0 * token_generation_time_s / max(decode_len - 1, 1),
                "itl_mean_ms": "",
                "itl_p50_ms": "",
                "itl_p90_ms": "",
                "itl_p95_ms": "",
                "itl_p99_ms": "",
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert llama-batched-bench JSONL into the README task 5 CSV schema.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", default="llama-batched-bench")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = convert_rows(
        args.input,
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )
    write_csv(args.output, rows)


if __name__ == "__main__":
    main()
