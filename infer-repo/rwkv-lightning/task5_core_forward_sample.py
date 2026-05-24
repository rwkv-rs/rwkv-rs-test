#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent:
    if (ROOT / "scripts" / "task5_core_schema.py").exists():
        sys.path.insert(0, str(ROOT))
        break
    ROOT = ROOT.parent

RWKV_LIGHTNING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RWKV_LIGHTNING_ROOT))

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
    write_csv,
)


ENTRYPOINTS = {
    "decode": "RWKV_x070.forward_one+sampler_simple",
    "prefill": "RWKV_x070.forward_seq+sampler_simple",
    "batch_decode": "RWKV_x070.forward_seq_batch+sampler_simple_batch",
    "batch_prefill": "RWKV_x070.forward_seq_batch+sampler_simple_batch",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task5 core forward+sample runner for rwkv-lightning.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=split_tasks, default=split_tasks("decode,prefill,batch_decode,batch_prefill"))
    parser.add_argument("--prefill-t", type=parse_int_list, default=list(PREFILL_T_DEFAULT))
    parser.add_argument("--batch-decode-b", type=parse_int_list, default=list(BATCH_DECODE_B_DEFAULT))
    parser.add_argument("--batch-prefill-pairs", type=parse_pair_list, default=list(BATCH_PREFILL_PAIRS_DEFAULT))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="infer-repo/rwkv-lightning/results/task5_core_forward_sample.csv")
    args = parser.parse_args()

    metadata = common_metadata(
        repo="rwkv-lightning",
        backend="rwkv-lightning-direct",
        runner="task5_core_forward_sample.py",
        model_path=args.model,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )

    rows = []
    cases = task_cases(args.tasks, args.prefill_t, args.batch_decode_b, args.batch_prefill_pairs)
    try:
        bench = LightningBench(Path(args.model))
    except Exception as exc:
        metadata["ended_at"] = iso_now()
        for task, B, T in cases:
            rows.append(
                failed_row(
                    **metadata,
                    task=task,
                    B=B,
                    T=T,
                    entrypoint=ENTRYPOINTS[task],
                    error=f"model setup failed: {type(exc).__name__}: {exc}",
                    measurement_boundary=MEASUREMENT_BOUNDARY,
                )
            )
        write_csv(args.out, rows)
        return

    for task, B, T in cases:
        try:
            rows.append(bench.run_case(metadata, task, B, T, args.warmup, args.repeat, args.seed))
        except Exception as exc:
            rows.append(
                failed_row(
                    **metadata,
                    task=task,
                    B=B,
                    T=T,
                    entrypoint=ENTRYPOINTS[task],
                    error=f"{type(exc).__name__}: {exc}",
                    measurement_boundary=MEASUREMENT_BOUNDARY,
                )
            )

    write_csv(args.out, rows)


class LightningBench:
    def __init__(self, model_path: Path) -> None:
        import torch
        from infer.rwkv_batch.rwkv7 import RWKV_x070
        from infer.rwkv_batch.utils import sampler_simple, sampler_simple_batch

        torch.set_grad_enabled(False)
        self.torch = torch
        self.sampler_simple = sampler_simple
        self.sampler_simple_batch = sampler_simple_batch
        model_name = str(model_path)
        if model_name.endswith(".pth"):
            model_name = model_name[:-4]
        args = types.SimpleNamespace(MODEL_NAME=model_name, vocab_size=65536, head_size=64)
        self.model = RWKV_x070(args)
        self.vocab_size = int(args.vocab_size)

    def run_case(
        self,
        metadata: dict[str, object],
        task: str,
        B: int,
        T: int,
        warmup: int,
        repeat: int,
        seed: int,
    ) -> dict[str, object]:
        torch = self.torch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + B * 1000003 + T * 9176)
        tokens = torch.randint(0, self.vocab_size, (B, T), generator=generator, dtype=torch.long).tolist()

        for _ in range(warmup):
            state = self._state(task, B)
            logits = self._forward(task, tokens, state)
            self._sample(logits)
        torch.cuda.synchronize()

        total_ms: list[float] = []
        forward_ms: list[float] = []
        sample_ms: list[float] = []
        for _ in range(repeat):
            state = self._state(task, B)
            start = torch.cuda.Event(enable_timing=True)
            mid = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = self._forward(task, tokens, state)
            mid.record()
            self._sample(logits)
            end.record()
            torch.cuda.synchronize()
            forward_ms.append(float(start.elapsed_time(mid)))
            sample_ms.append(float(mid.elapsed_time(end)))
            total_ms.append(float(start.elapsed_time(end)))

        p50 = percentile(total_ms, 0.50)
        return ok_row(
            **metadata,
            task=task,
            B=B,
            T=T,
            total_time_s=p50 / 1000.0,
            forward_time_s=percentile(forward_ms, 0.50) / 1000.0,
            sample_time_s=percentile(sample_ms, 0.50) / 1000.0,
            p10_ms=percentile(total_ms, 0.10),
            p50_ms=p50,
            p90_ms=percentile(total_ms, 0.90),
            entrypoint=ENTRYPOINTS[task],
            measurement_boundary=MEASUREMENT_BOUNDARY,
            ended_at=iso_now(),
        )

    def _state(self, task: str, B: int):
        return self.model.generate_zero_state(0 if task in {"decode", "prefill"} else B)

    def _forward(self, task: str, tokens: list[list[int]], state):
        if task == "decode":
            token = int(tokens[0][0])
            x = self.model.z["emb.weight"][token]
            return self.model.forward_one(x, state)
        if task == "prefill":
            return self.model.forward_seq([int(token) for token in tokens[0]], state, full_output=False)
        if task == "batch_decode":
            return self.model.forward_seq_batch([[int(row[0])] for row in tokens], state, full_output=False)
        if task == "batch_prefill":
            return self.model.forward_seq_batch([[int(token) for token in row] for row in tokens], state, full_output=False)
        raise ValueError(f"unknown task: {task}")

    def _sample(self, logits):
        if logits.dim() == 1:
            return self.sampler_simple(logits, noise=0)
        return self.sampler_simple_batch(logits, noise=0)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty list")
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[index])


if __name__ == "__main__":
    main()
