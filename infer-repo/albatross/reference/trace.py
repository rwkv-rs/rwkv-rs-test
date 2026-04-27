from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter_ns

import torch
from safetensors.torch import save_file


def trace_enabled() -> bool:
    return os.environ.get("RWKV_TRACE_ONCE") == "1"


def case_root() -> Path:
    root = os.environ.get("RWKV_TRACE_ROOT")
    if not root:
        raise RuntimeError("RWKV_TRACE_ROOT must be set when RWKV_TRACE_ONCE=1")
    return Path(root) / "albatross" / "fp16" / "case_000000"


def trace(output_path: str | Path, filename: str, tensor: torch.Tensor) -> None:
    path = Path(output_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter_ns()
    view = tensor.detach()
    if not view.is_contiguous():
        raise RuntimeError("trace requires a contiguous torch.Tensor")
    save_file({path.stem: view}, path)
    elapsed_ns = perf_counter_ns() - start

    path.with_suffix(".time.json").write_text(
        json.dumps({"filename": filename, "elapsed_ns": elapsed_ns}),
        encoding="utf-8",
    )


def trace_tensor(filename: str, tensor: torch.Tensor) -> None:
    trace(case_root(), filename, tensor.contiguous())


def trace_token_ids(tokens: torch.Tensor) -> None:
    trace_tensor("embedding/token_ids.safetensors", tokens.to(dtype=torch.int64, device="cpu"))
