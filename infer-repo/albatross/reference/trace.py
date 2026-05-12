from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter_ns

import torch


def trace_enabled() -> bool:
    return os.environ.get("RWKV_TRACE_ONCE") == "1"


def case_root() -> Path:
    root = os.environ.get("RWKV_TRACE_ROOT")
    if not root:
        raise RuntimeError("RWKV_TRACE_ROOT must be set when RWKV_TRACE_ONCE=1")
    return Path(root) / "albatross" / "fp16" / "case_000000"


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def measure(fn, *inputs):
    if trace_enabled():
        for tensor in inputs:
            if isinstance(tensor, torch.Tensor):
                _sync_if_cuda(tensor)
        start = perf_counter_ns()
        output = fn()
        if isinstance(output, torch.Tensor):
            _sync_if_cuda(output)
        elif isinstance(output, tuple):
            for tensor in output:
                if isinstance(tensor, torch.Tensor):
                    _sync_if_cuda(tensor)
        elapsed_ns = perf_counter_ns() - start
        return output, elapsed_ns

    return fn(), 0


def trace(output_path: str | Path, filename: str, tensor: torch.Tensor, elapsed_ns: int = 0) -> None:
    from safetensors.torch import save_file

    path = Path(output_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    view = tensor.detach()
    if not view.is_contiguous():
        raise RuntimeError("trace requires a contiguous torch.Tensor")
    save_file({path.stem: view}, path)

    path.with_suffix(".time.json").write_text(
        json.dumps({"filename": filename, "elapsed_ns": elapsed_ns}),
        encoding="utf-8",
    )


def trace_tensor(filename: str, tensor: torch.Tensor, elapsed_ns: int = 0) -> None:
    trace(case_root(), filename, tensor.contiguous(), elapsed_ns)


def trace_token_ids(tokens: torch.Tensor) -> None:
    trace_tensor("embedding/token_ids.safetensors", tokens.to(dtype=torch.int64, device="cpu"))
