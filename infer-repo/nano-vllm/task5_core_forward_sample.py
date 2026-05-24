#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent:
    if (ROOT / "scripts" / "task5_core_schema.py").exists():
        sys.path.insert(0, str(ROOT))
        break
    ROOT = ROOT.parent

NANO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(NANO_ROOT))

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
    "decode": "model_runner.run_logits(seqs, False)+sampler",
    "prefill": "model_runner.run_logits(seqs, True)+sampler",
    "batch_decode": "model_runner.run_logits(seqs, False)+sampler",
    "batch_prefill": "model_runner.run_logits(seqs, True)+sampler",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task5 core forward+sample runner for nano-vllm.")
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--tasks", type=split_tasks, default=split_tasks("decode,prefill,batch_decode,batch_prefill"))
    parser.add_argument("--prefill-t", type=parse_int_list, default=list(PREFILL_T_DEFAULT))
    parser.add_argument("--batch-decode-b", type=parse_int_list, default=list(BATCH_DECODE_B_DEFAULT))
    parser.add_argument("--batch-prefill-pairs", type=parse_pair_list, default=list(BATCH_PREFILL_PAIRS_DEFAULT))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="infer-repo/nano-vllm/results/task5_core_forward_sample.csv")
    args = parser.parse_args()

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("NANOVLLM_DIST_PORT", str(29500 + os.getpid() % 10000))

    metadata = common_metadata(
        repo="nano-vllm",
        backend="nano-vllm-direct-model-runner",
        runner="task5_core_forward_sample.py",
        model_path=args.model_pth,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )
    cases = task_cases(args.tasks, args.prefill_t, args.batch_decode_b, args.batch_prefill_pairs)
    rows = []
    try:
        bench = NanoBench(Path(args.model_pth), max(B for _, B, _ in cases), max(T for _, _, T in cases))
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

    try:
        for task, B, T in cases:
            try:
                rows.append(bench.run_case(metadata, task, B, T, args.warmup, args.repeat, args.seed))
            except NotImplementedError as exc:
                row_fn = unsupported_row if task in {"batch_decode", "batch_prefill"} else failed_row
                rows.append(
                    row_fn(
                        **metadata,
                        task=task,
                        B=B,
                        T=T,
                        entrypoint=ENTRYPOINTS[task],
                        error=str(exc),
                        measurement_boundary=MEASUREMENT_BOUNDARY,
                    )
                )
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
    finally:
        bench.close()

    write_csv(args.out, rows)


class NanoBench:
    def __init__(self, model_pth: Path, max_b: int, max_t: int) -> None:
        import torch
        from nanovllm import LLM, SamplingParams
        from benchmark_rwkv import ensure_model_dir

        self.torch = torch
        self.SamplingParams = SamplingParams
        model_dir = ensure_model_dir(str(model_pth))
        max_batched_tokens = max(4096, max_b, max_t)
        self.llm = LLM(
            model_dir,
            enforce_eager=False,
            tensor_parallel_size=1,
            max_num_seqs=max(max_b, 1),
            max_num_batched_tokens=max_batched_tokens,
            max_model_len=max(4096, max_t + 1),
            gpu_memory_utilization=0.95,
            rwkv_prefill_token_budget=max_batched_tokens,
            rwkv_prefill_max_batch_size=max(max_b, 1),
            rwkv_prefill_chunk_size=-1,
            rwkv_state_cache_enable=False,
        )
        self.vocab_size = int(self.llm.model_runner.config.model_config.vocab_size)
        self.max_num_seqs = int(self.llm.model_runner.config.max_num_seqs)

    def close(self) -> None:
        if hasattr(self, "llm"):
            self.llm.exit()

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
        total_s: list[float] = []
        for _ in range(warmup):
            seqs = self._prepare(task, B, T, seed)
            try:
                token_ids = self._run_forward_sample(task, seqs)
                self._postprocess(seqs, token_ids)
            finally:
                self._release(seqs)
        self.torch.cuda.synchronize()

        for i in range(repeat):
            seqs = self._prepare(task, B, T, seed + i + 1)
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                token_ids = self._run_forward_sample(task, seqs)
                self.torch.cuda.synchronize()
                total_s.append(time.perf_counter() - started)
            finally:
                self._release(seqs)

        p50_s = percentile(total_s, 0.50)
        return ok_row(
            **metadata,
            task=task,
            B=B,
            T=T,
            total_time_s=p50_s,
            p10_ms=percentile(total_s, 0.10) * 1000.0,
            p50_ms=p50_s * 1000.0,
            p90_ms=percentile(total_s, 0.90) * 1000.0,
            entrypoint=ENTRYPOINTS[task],
            measurement_boundary=MEASUREMENT_BOUNDARY,
            ended_at=iso_now(),
        )

    def _prepare(self, task: str, B: int, T: int, seed: int):
        if task in {"decode", "batch_decode"}:
            return self._make_prefilled_seqs(B, 1, seed)
        if task in {"prefill", "batch_prefill"}:
            return self._make_waiting_seqs(B, T, seed)
        raise ValueError(f"unknown task: {task}")

    def _run_forward_sample(self, task: str, seqs):
        is_prefill = task in {"prefill", "batch_prefill"}
        logits = self.llm.model_runner.call("run_logits", seqs, is_prefill)
        if self.llm.model_runner.rank != 0:
            return None
        return self.llm.model_runner.sampler(logits, seqs).tolist()

    def _postprocess(self, seqs, token_ids) -> None:
        self.llm.model_runner.call("prepare_postprocess", seqs, token_ids)
        if token_ids is None:
            return
        for seq, token_id in zip(seqs, token_ids):
            if token_id is not None:
                seq.append_token(int(token_id))

    def _make_waiting_seqs(self, B: int, T: int, seed: int):
        from nanovllm.engine.sequence import Sequence, SequenceStatus
        from nanovllm.sampling_params import SamplingParams

        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(seed + B * 1000003 + T * 9176)
        params = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=2)
        seqs = []
        if B > self.max_num_seqs:
            raise RuntimeError(f"direct model_runner path needs B <= max_num_seqs ({self.max_num_seqs}), got B={B}")
        for slot in range(B):
            tokens = self.torch.randint(0, self.vocab_size, (T,), generator=generator, dtype=self.torch.int64).tolist()
            seq = Sequence(tokens, params)
            seq.block_table = [slot]
            seqs.append(seq)
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING
        return seqs

    def _make_prefilled_seqs(self, B: int, T: int, seed: int):
        seqs = self._make_waiting_seqs(B, T, seed)
        token_ids = self._run_forward_sample("prefill", seqs)
        self._postprocess(seqs, token_ids)
        return seqs

    def _release(self, seqs) -> None:
        for seq in seqs:
            seq.num_cached_tokens = 0
            seq.block_table.clear()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty list")
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[index])


if __name__ == "__main__":
    main()
