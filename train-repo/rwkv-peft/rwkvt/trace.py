import json
import os
from pathlib import Path
from time import perf_counter_ns

import torch
from safetensors.torch import save_file


REPO_NAME = "rwkv_peft"
QUANTIZATION_NAME = "bf16"


def enabled() -> bool:
    if os.environ.get("RWKV_TRACE_ONCE") != "1":
        return False
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) == "0"


def case_root() -> Path:
    root = os.environ.get("RWKV_TRACE_ROOT")
    if not root:
        raise RuntimeError("RWKV_TRACE_ROOT must be set when RWKV_TRACE_ONCE=1")
    return Path(root) / REPO_NAME / QUANTIZATION_NAME / "case_000000"


def _sync_if_cuda(tensor: torch.Tensor) -> None:
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def measure(fn, *inputs):
    if enabled():
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


def timer_start(*inputs) -> int:
    if not enabled():
        return 0
    for tensor in inputs:
        if isinstance(tensor, torch.Tensor):
            _sync_if_cuda(tensor)
    return perf_counter_ns()


def timer_elapsed(start: int, *outputs) -> int:
    if start == 0 or not enabled():
        return 0
    for tensor in outputs:
        if isinstance(tensor, torch.Tensor):
            _sync_if_cuda(tensor)
    return perf_counter_ns() - start


def trace(filename: str, tensor: torch.Tensor, elapsed_ns: int = 0) -> None:
    if not enabled():
        return

    path = case_root() / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    view = tensor.detach()
    if not view.is_contiguous():
        view = view.contiguous()
    if not view.device.type == "cpu":
        view = view.cpu()
    save_file({path.stem: view}, path)

    path.with_suffix(".time.json").write_text(
        json.dumps({"filename": filename, "elapsed_ns": elapsed_ns}),
        encoding="utf-8",
    )


def trace_cell(layer_id: int, filename: str, tensor: torch.Tensor, elapsed_ns: int = 0) -> None:
    trace(f"cells/cell_{layer_id:04d}/{filename}", tensor, elapsed_ns)
