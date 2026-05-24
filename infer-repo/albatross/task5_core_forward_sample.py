#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path


ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent:
    if (ROOT / "scripts" / "task5_core_schema.py").exists():
        sys.path.insert(0, str(ROOT))
        break
    ROOT = ROOT.parent

from scripts.task5_core_runner_utils import (  # noqa: E402
    common_metadata,
    parse_int_list,
    parse_pair_list,
    split_tasks,
    task_cases,
)
from scripts.task5_core_runner_utils import MEASUREMENT_BOUNDARY, iso_now  # noqa: E402
from scripts.task5_core_schema import BATCH_DECODE_B_DEFAULT, BATCH_PREFILL_PAIRS_DEFAULT, PREFILL_T_DEFAULT, failed_row, ok_row, write_csv, unsupported_row  # noqa: E402


ENTRYPOINTS = {
    "decode": "RWKV7.forward_from_x+sampler_simple",
    "prefill": "RWKV7.forward_from_x+sampler_simple",
    "batch_decode": "RWKV7.forward_from_x+sampler_simple_batch",
    "batch_prefill": "RWKV7.forward_from_x+sampler_simple_batch",
}

TASK_NATIVE_BATCH_PREFILL_VERSIONS = {"faster3_2605", "faster3a_2605", "faster4_2605_cpp"}


def supports_task(version: str, task: str) -> bool:
    if task == "batch_prefill":
        return version in TASK_NATIVE_BATCH_PREFILL_VERSIONS
    return task in ENTRYPOINTS


def unsupported_task_message(version: str, task: str) -> str:
    if task == "batch_prefill":
        return (
            f"albatross {version} does not expose a task-native BnTn batch prefill path; "
            "legacy forward_batch(list[list[int]]) is not counted as Task5 batch_prefill support"
        )
    return f"albatross {version} does not support Task5 task {task}"


def partition_supported_cases(version: str, cases: list[tuple[str, int, int]]) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    supported: list[tuple[str, int, int]] = []
    unsupported: list[tuple[str, int, int]] = []
    for case in cases:
        task, _, _ = case
        if supports_task(version, task):
            supported.append(case)
        else:
            unsupported.append(case)
    return supported, unsupported


def unsupported_task_row(metadata: dict[str, object], version: str, task: str, B: int, T: int) -> dict[str, object]:
    return unsupported_row(
        **metadata,
        task=task,
        B=B,
        T=T,
        entrypoint=ENTRYPOINTS[task],
        error=unsupported_task_message(version, task),
        measurement_boundary=MEASUREMENT_BOUNDARY,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Task5 core forward+sample runner for Albatross.")
    parser.add_argument("--version", default="faster3a_2605")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", type=split_tasks, default=split_tasks("decode,prefill,batch_decode,batch_prefill"))
    parser.add_argument("--prefill-t", type=parse_int_list, default=list(PREFILL_T_DEFAULT))
    parser.add_argument("--batch-decode-b", type=parse_int_list, default=list(BATCH_DECODE_B_DEFAULT))
    parser.add_argument("--batch-prefill-pairs", type=parse_pair_list, default=list(BATCH_PREFILL_PAIRS_DEFAULT))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="infer-repo/albatross/results/task5_core_forward_sample.csv")
    args = parser.parse_args()

    metadata = common_metadata(
        repo="albatross",
        backend=f"albatross-{args.version}",
        runner="task5_core_forward_sample.py",
        model_path=args.model,
        warmup=args.warmup,
        repeat=args.repeat,
        seed=args.seed,
    )
    rows = []
    cases = task_cases(args.tasks, args.prefill_t, args.batch_decode_b, args.batch_prefill_pairs)
    supported_cases, unsupported_cases = partition_supported_cases(args.version, cases)
    if not supported_cases:
        write_csv(args.out, [unsupported_task_row(metadata, args.version, task, B, T) for task, B, T in cases])
        return

    try:
        configure_torch_extensions_dir(args.version)
        if args.version in {"faster3_2605", "faster3a_2605"}:
            bench = Faster3Bench(Path(args.model), args.version)
        elif args.version == "faster4_2605_cpp":
            bench = Faster4CppBench(Path(args.model), args.version)
        elif args.version in {"_ref_slower_", "faster_251101", "faster2_251201"}:
            bench = ReferenceBench(Path(args.model), args.version)
        else:
            raise RuntimeError(f"unknown or unsupported albatross version: {args.version}")
    except Exception as exc:
        metadata["ended_at"] = iso_now()
        for task, B, T in cases:
            if not supports_task(args.version, task):
                rows.append(unsupported_task_row(metadata, args.version, task, B, T))
            else:
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

    if isinstance(bench, Faster4CppBench):
        supported_rows = bench.run_cases(metadata, supported_cases, args.warmup, args.repeat)
        supported_by_case = {(row["task"], row["B"], row["T"]): row for row in supported_rows}
        rows = [
            supported_by_case[(task, B, T)] if supports_task(args.version, task) else unsupported_task_row(metadata, args.version, task, B, T)
            for task, B, T in cases
        ]
        write_csv(args.out, rows)
        return

    for task, B, T in cases:
        if not supports_task(args.version, task):
            rows.append(unsupported_task_row(metadata, args.version, task, B, T))
            continue
        try:
            rows.append(bench.run_case(metadata, task, B, T, args.warmup, args.repeat, args.seed))
        except RuntimeError as exc:
            row_fn = unsupported_row if task in {"batch_decode", "batch_prefill"} and "supports only" in str(exc) else failed_row
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

    write_csv(args.out, rows)


def configure_torch_extensions_dir(version: str) -> None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ.get("ALBATROSS_TASK5_TORCH_CUDA_ARCH_LIST", "12.0")
    safe_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", version)
    configured = os.environ.get("TORCH_EXTENSIONS_DIR")
    if configured:
        base = Path(configured)
    else:
        base = Path.cwd() / ".cache" / "torch_extensions" / "albatross-task5"
    os.environ["TORCH_EXTENSIONS_DIR"] = str(base / safe_version)


class Faster3Bench:
    def __init__(self, model_path: Path, version: str) -> None:
        import torch

        self.torch = torch
        version_dir = Path(__file__).resolve().parent / version
        sys.path.insert(0, str(version_dir))
        module_name = "rwkv7_fast_v3a" if version == "faster3a_2605" else "rwkv7_fast_v3"
        fast = importlib.import_module(module_name)
        fast.MODEL_PATH = str(model_path)
        fast.WKV_MODE = "fp16"
        fast.EMB_DEVICE = "cpu"
        fast.RKV_MODE = "off"
        fast.CMIX_SPARSE = "no-fc"
        if hasattr(fast, "LOWRANK_WEIGHT"):
            fast.LOWRANK_WEIGHT = "transpose"
        if hasattr(fast, "ORIG_LINEAR_GROUPS"):
            fast.ORIG_LINEAR_GROUPS = {"att_c2c", "ffn_key", "head"}
        fast.load_extensions(fast.WKV_MODE)
        self.fast = fast
        ref_dir = Path(__file__).resolve().parent / "_ref_slower_"
        sys.path.insert(0, str(ref_dir))
        sampler_utils = importlib.import_module("reference.utils")
        self.sampler_simple = sampler_utils.sampler_simple
        self.sampler_simple_batch = sampler_utils.sampler_simple_batch
        self._install_torch_load_mmap_fallback(torch)
        self.model = fast.RWKV7()

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
        if task == "batch_prefill" and B != T:
            raise RuntimeError("faster3a_2605 direct batch prefill runner currently supports only diagonal BxT cases")
        torch = self.torch
        token_device = "cpu" if self.model.emb_cpu else "cuda"
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + B * 1000003 + T * 9176)

        for i in range(warmup):
            state, x, path = self._prepare_case(task, B, T, token_device, generator)
            logits = self.model.forward_from_x(x, state, path)
            self.sample(logits)
        torch.cuda.synchronize()

        total_ms: list[float] = []
        forward_ms: list[float] = []
        sample_ms: list[float] = []
        for _ in range(repeat):
            state, x, path = self._prepare_case(task, B, T, token_device, generator)
            start = torch.cuda.Event(enable_timing=True)
            mid = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = self.model.forward_from_x(x, state, path)
            mid.record()
            self.sample(logits)
            end.record()
            torch.cuda.synchronize()
            forward_ms.append(float(start.elapsed_time(mid)))
            sample_ms.append(float(mid.elapsed_time(end)))
            total_ms.append(float(start.elapsed_time(end)))

        p10 = percentile(total_ms, 0.10)
        p50 = percentile(total_ms, 0.50)
        p90 = percentile(total_ms, 0.90)
        row = ok_row(
            **metadata,
            task=task,
            B=B,
            T=T,
            total_time_s=p50 / 1000.0,
            forward_time_s=percentile(forward_ms, 0.50) / 1000.0,
            sample_time_s=percentile(sample_ms, 0.50) / 1000.0,
            p10_ms=p10,
            p50_ms=p50,
            p90_ms=p90,
            entrypoint=ENTRYPOINTS[task],
            measurement_boundary=MEASUREMENT_BOUNDARY,
            ended_at=iso_now(),
        )
        return row

    def _prepare_case(self, task: str, B: int, T: int, token_device: str, generator):
        torch = self.torch
        state = self.model.zero_state(B)
        if task in {"decode", "batch_decode"}:
            prefix = torch.randint(0, self.fast.V, (B, 1), generator=generator, dtype=torch.long, device="cpu").to(token_device)
            prefix_x = self.model.embed(prefix)
            prefix_path = self.fast.select_path(B, 1)
            prefix_logits = self.model.forward_from_x(prefix_x, state, prefix_path)
            next_tokens = self._sample_tokens(prefix_logits).reshape(B, 1).to(token_device)
            return state, self.model.embed(next_tokens), prefix_path

        tokens = torch.randint(0, self.fast.V, (B, T), generator=generator, dtype=torch.long, device="cpu").to(token_device)
        return state, self.model.embed(tokens), self.fast.select_path(B, T)

    @staticmethod
    def _install_torch_load_mmap_fallback(torch_module) -> None:
        original_load = torch_module.load

        def load_with_fallback(*args, **kwargs):
            try:
                return original_load(*args, **kwargs)
            except RuntimeError as exc:
                if kwargs.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("mmap", None)
                    return original_load(*args, **retry_kwargs)
                raise

        torch_module.load = load_with_fallback

    def sample(self, logits):
        if logits.dim() == 1:
            return self.sampler_simple(logits, noise=0)
        if logits.shape[0] == 1:
            return self.sampler_simple(logits.view(-1), noise=0)
        return self.sampler_simple_batch(logits, noise=0)

    def _sample_tokens(self, logits):
        token = self.sample(logits)
        if hasattr(token, "detach"):
            token = token.detach().cpu()
        return self.torch.as_tensor(token, dtype=self.torch.long, device="cpu")


class ReferenceBench:
    def __init__(self, model_path: Path, version: str) -> None:
        import torch

        self.torch = torch
        self.version = version
        version_dir = Path(__file__).resolve().parent / version
        sys.path.insert(0, str(version_dir))
        rwkv7 = importlib.import_module("reference.rwkv7")
        sampler_utils = importlib.import_module("reference.utils")
        args = types.SimpleNamespace(
            MODEL_NAME=self._model_name(model_path),
            vocab_size=65536,
            head_size=64,
        )
        self._install_torch_load_mmap_fallback(torch)
        self.model = rwkv7.RWKV_x070(args)
        self.sampler_simple = sampler_utils.sampler_simple
        self.sampler_simple_batch = sampler_utils.sampler_simple_batch
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
        if not supports_task(self.version, task):
            raise RuntimeError(unsupported_task_message(self.version, task))
        torch = self.torch
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + B * 1000003 + T * 9176)
        tokens = torch.randint(0, self.vocab_size, (B, T), generator=generator, dtype=torch.long).tolist()

        for i in range(warmup):
            state, prepared_tokens = self._prepare_case(task, tokens)
            logits = self._run_once(task, prepared_tokens, state)
            self.sample(logits)
        torch.cuda.synchronize()

        total_ms: list[float] = []
        forward_ms: list[float] = []
        sample_ms: list[float] = []
        for i in range(repeat):
            state, prepared_tokens = self._prepare_case(task, tokens)
            start = torch.cuda.Event(enable_timing=True)
            mid = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            logits = self._run_once(task, prepared_tokens, state)
            mid.record()
            self.sample(logits)
            end.record()
            torch.cuda.synchronize()
            forward_ms.append(float(start.elapsed_time(mid)))
            sample_ms.append(float(mid.elapsed_time(end)))
            total_ms.append(float(start.elapsed_time(end)))

        p10 = percentile(total_ms, 0.10)
        p50 = percentile(total_ms, 0.50)
        p90 = percentile(total_ms, 0.90)
        return ok_row(
            **metadata,
            task=task,
            B=B,
            T=T,
            total_time_s=p50 / 1000.0,
            forward_time_s=percentile(forward_ms, 0.50) / 1000.0,
            sample_time_s=percentile(sample_ms, 0.50) / 1000.0,
            p10_ms=p10,
            p50_ms=p50,
            p90_ms=p90,
            entrypoint=self.entrypoint(task),
            measurement_boundary=MEASUREMENT_BOUNDARY,
            ended_at=iso_now(),
        )

    def _prepare_case(self, task: str, tokens: list[list[int]]):
        if task == "decode":
            state = self.model.generate_zero_state(0)
            prefix_logits = self.model.forward([tokens[0][0]], state)
            token = self._single_token(prefix_logits)
            return state, [[token]]
        if task == "prefill":
            return self.model.generate_zero_state(0), tokens
        if task == "batch_decode":
            state = self.model.generate_zero_state(len(tokens))
            prefix_logits = self.model.forward_batch([[int(row[0])] for row in tokens], state)
            sampled = self._batch_tokens(prefix_logits)
            return state, [[token] for token in sampled]
        if task == "batch_prefill":
            return self.model.generate_zero_state(len(tokens)), tokens
        raise ValueError(f"unknown task: {task}")

    def _run_once(self, task: str, tokens: list[list[int]], state):
        if task == "decode":
            return self.model.forward(int(tokens[0][0]), state)
        if task == "prefill":
            return self.model.forward([int(token) for token in tokens[0]], state)
        if task == "batch_decode":
            return self.model.forward_batch([[int(row[0])] for row in tokens], state)
        if task == "batch_prefill":
            return self.model.forward_batch([[int(token) for token in row] for row in tokens], state)
        raise ValueError(f"unknown task: {task}")

    def sample(self, logits):
        if logits.dim() == 1:
            return self.sampler_simple(logits, noise=0)
        if logits.shape[0] == 1:
            return self.sampler_simple(logits.view(-1), noise=0)
        return self.sampler_simple_batch(logits, noise=0)

    def _single_token(self, logits) -> int:
        token = self.sample(logits)
        if hasattr(token, "item"):
            return int(token.item())
        return int(token)

    def _batch_tokens(self, logits) -> list[int]:
        token = self.sample(logits)
        if hasattr(token, "detach"):
            token = token.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in token]

    def entrypoint(self, task: str) -> str:
        if task in {"decode", "prefill"}:
            return "RWKV_x070.forward+sampler_simple"
        return "RWKV_x070.forward_batch+sampler_simple_batch"

    @staticmethod
    def _model_name(model_path: Path) -> str:
        text = str(model_path)
        return text[:-4] if text.endswith(".pth") else text

    @staticmethod
    def _install_torch_load_mmap_fallback(torch_module) -> None:
        original_load = torch_module.load

        def load_with_fallback(*args, **kwargs):
            try:
                return original_load(*args, **kwargs)
            except RuntimeError as exc:
                if kwargs.get("mmap") is True and "mmap can only be used" in str(exc):
                    retry_kwargs = dict(kwargs)
                    retry_kwargs.pop("mmap", None)
                    return original_load(*args, **retry_kwargs)
                raise

        torch_module.load = load_with_fallback


class Faster4CppBench:
    BENCH_RE = re.compile(r"bench B(?P<B>\d+)T(?P<T>\d+)\s+.*?\bms=(?P<ms>-?\d+(?:\.\d+)?)\s+tok_s=(?P<tok_s>-?\d+(?:\.\d+)?)")

    def __init__(self, model_path: Path, version: str) -> None:
        self.model_path = model_path
        self.version_dir = Path(__file__).resolve().parent / version
        self.binary = self.version_dir / "bin" / "rwkv7_fast_v4"
        self._ensure_binary()

    def run_cases(
        self,
        metadata: dict[str, object],
        cases: list[tuple[str, int, int]],
        warmup: int,
        repeat: int,
    ) -> list[dict[str, object]]:
        case_text = ",".join(f"{B}x{T}" for _, B, T in cases)
        cmd = [
            str(self.binary),
            "--model",
            str(self.model_path),
            "--model-forward",
            "--cases",
            case_text,
            "--graph-bench",
            "--warmup",
            str(warmup),
            "--iters",
            str(repeat),
        ]
        run_metadata = {
            **metadata,
            "runner": "task5_core_forward_sample.py+rwkv7_fast_v4",
            "command": " ".join(cmd),
            "binary_path": str(self.binary),
        }
        completed = subprocess.run(
            cmd,
            cwd=self.version_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            error = (completed.stderr or completed.stdout).strip()[-2000:]
            return [
                failed_row(
                    **run_metadata,
                    task=task,
                    B=B,
                    T=T,
                    entrypoint=self.entrypoint(task),
                    error=f"faster4_2605_cpp failed: {error}",
                    measurement_boundary=MEASUREMENT_BOUNDARY,
                )
                for task, B, T in cases
            ]

        by_shape: dict[tuple[int, int], float] = {}
        for line in completed.stdout.splitlines():
            match = self.BENCH_RE.search(line)
            if not match:
                continue
            B = int(match.group("B"))
            T = int(match.group("T"))
            ms = float(match.group("ms"))
            if ms > 0:
                by_shape[(B, T)] = ms

        rows: list[dict[str, object]] = []
        for task, B, T in cases:
            ms = by_shape.get((B, T))
            if ms is None:
                rows.append(
                    failed_row(
                        **run_metadata,
                        task=task,
                        B=B,
                        T=T,
                        entrypoint=self.entrypoint(task),
                        error="faster4_2605_cpp did not emit a bench row for this B/T case",
                        measurement_boundary=MEASUREMENT_BOUNDARY,
                    )
                )
                continue
            rows.append(
                ok_row(
                    **run_metadata,
                    task=task,
                    B=B,
                    T=T,
                    total_time_s=ms / 1000.0,
                    forward_time_s=None,
                    sample_time_s=None,
                    p50_ms=ms,
                    entrypoint=self.entrypoint(task),
                    measurement_boundary=MEASUREMENT_BOUNDARY,
                    ended_at=iso_now(),
                )
            )
        return rows

    def _ensure_binary(self) -> None:
        if self.binary.exists():
            return
        configure = subprocess.run(
            ["cmake", "-S", str(self.version_dir), "-B", str(self.version_dir / "bin"), "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=120"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if configure.returncode != 0:
            raise RuntimeError(f"faster4_2605_cpp cmake configure failed: {(configure.stderr or configure.stdout).strip()[-2000:]}")
        build = subprocess.run(
            ["cmake", "--build", str(self.version_dir / "bin"), "-j"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(f"faster4_2605_cpp build failed: {(build.stderr or build.stdout).strip()[-2000:]}")

    @staticmethod
    def entrypoint(task: str) -> str:
        return "rwkv7_fast_v4 graph_forward+argmax_sampler"


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


if __name__ == "__main__":
    main()
