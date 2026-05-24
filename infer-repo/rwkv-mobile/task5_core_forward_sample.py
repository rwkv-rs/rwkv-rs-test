#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent:
    if (ROOT / "scripts" / "task5_core_schema.py").exists():
        sys.path.insert(0, str(ROOT))
        break
    ROOT = ROOT.parent

from scripts.task5_core_runner_utils import (  # noqa: E402
    MEASUREMENT_BOUNDARY,
    common_metadata,
    iso_now,
    parse_int_list,
    parse_pair_list,
    split_tasks,
    task_cases,
)
from scripts.task5_core_schema import (  # noqa: E402
    BATCH_DECODE_B_DEFAULT,
    BATCH_PREFILL_PAIRS_DEFAULT,
    PREFILL_T_DEFAULT,
    failed_row,
    ok_row,
    unsupported_row,
    write_csv,
)


ENTRYPOINTS = {
    "decode": "rwkv-mobile Runtime::eval_logits(int)+NucleusSampler",
    "prefill": "rwkv-mobile Runtime::eval_logits(vector<int>)+NucleusSampler",
    "batch_decode": "rwkv-mobile Runtime::eval_logits_batch_decode+NucleusSampler::sample_batch",
    "batch_prefill": "rwkv-mobile direct batch prefill",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task5 core forward+sample runner for rwkv-mobile.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="llama.cpp")
    parser.add_argument(
        "--binary",
        default=str(Path(__file__).resolve().parent / "build-linux" / "examples" / "task5_core_forward_sample"),
    )
    parser.add_argument("--tasks", type=split_tasks, default=split_tasks("decode,prefill,batch_decode,batch_prefill"))
    parser.add_argument("--prefill-t", type=parse_int_list, default=list(PREFILL_T_DEFAULT))
    parser.add_argument("--batch-decode-b", type=parse_int_list, default=list(BATCH_DECODE_B_DEFAULT))
    parser.add_argument("--batch-prefill-pairs", type=parse_pair_list, default=list(BATCH_PREFILL_PAIRS_DEFAULT))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--case-timeout-s",
        type=float,
        default=0.0,
        help="Maximum seconds for one native benchmark case; 0 disables the timeout.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="infer-repo/rwkv-mobile/results/task5_core_forward_sample.csv")
    args = parser.parse_args()

    metadata = common_metadata(
        repo="rwkv-mobile",
        backend=f"rwkv-mobile-{args.backend}",
        runner="task5_core_forward_sample.py",
        model_path=args.model,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )

    rows = []
    binary = Path(args.binary)
    metadata = {**metadata, "binary_path": str(binary)}
    for task, B, T in task_cases(args.tasks, args.prefill_t, args.batch_decode_b, args.batch_prefill_pairs):
        if not binary.exists():
            rows.append(
                failed_row(
                    **metadata,
                    task=task,
                    B=B,
                    T=T,
                    entrypoint=ENTRYPOINTS[task],
                    error=f"missing rwkv-mobile task5 binary: {binary}; build examples/task5_core_forward_sample first",
                    measurement_boundary=MEASUREMENT_BOUNDARY,
                )
            )
            continue
        command = [
            str(binary),
            args.model,
            args.backend,
            task,
            str(B),
            str(T),
            str(args.warmup),
            str(args.repeat),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.case_timeout_s or None,
            )
        except subprocess.TimeoutExpired as exc:
            rows.append(row_from_timeout(metadata, exc, task, B, T))
            continue
        rows.append(row_from_completed(metadata, completed, task, B, T, binary))

    write_csv(args.output, rows)


def row_from_completed(metadata, completed: subprocess.CompletedProcess[str], task: str, B: int, T: int, binary: Path):
    stdout = completed.stdout.strip().splitlines()
    parsed = []
    for line in reversed(stdout):
        candidate = next(csv.reader([line]))
        if candidate and candidate[0] in {"ok", "failed", "unsupported"}:
            parsed = candidate
            break
    status = parsed[0] if parsed else "failed"
    error = parsed[1] if len(parsed) > 1 else ""
    if completed.returncode != 0 and not error:
        error = (completed.stderr or completed.stdout).strip()
    if status == "ok":
        p10_ms = float(parsed[2])
        p50_ms = float(parsed[3])
        p90_ms = float(parsed[4])
        return ok_row(
            **metadata,
            task=task,
            B=B,
            T=T,
            total_time_s=p50_ms / 1000.0,
            p10_ms=p10_ms,
            p50_ms=p50_ms,
            p90_ms=p90_ms,
            entrypoint=ENTRYPOINTS[task],
            measurement_boundary=MEASUREMENT_BOUNDARY,
            ended_at=iso_now(),
        )
    if status == "unsupported" and task in {"batch_decode", "batch_prefill"}:
        return unsupported_row(
            **metadata,
            task=task,
            B=B,
            T=T,
            entrypoint=ENTRYPOINTS[task],
            error=error,
            measurement_boundary=MEASUREMENT_BOUNDARY,
        )
    return failed_row(
        **metadata,
        task=task,
        B=B,
        T=T,
        entrypoint=ENTRYPOINTS[task],
        error=error or f"task5 binary failed with exit code {completed.returncode}",
        measurement_boundary=MEASUREMENT_BOUNDARY,
    )


def row_from_timeout(metadata, exc: subprocess.TimeoutExpired, task: str, B: int, T: int):
    return failed_row(
        **metadata,
        task=task,
        B=B,
        T=T,
        entrypoint=ENTRYPOINTS[task],
        error=f"task5 binary timed out after {exc.timeout:.1f}s",
        measurement_boundary=MEASUREMENT_BOUNDARY,
        ended_at=iso_now(),
    )


if __name__ == "__main__":
    main()
