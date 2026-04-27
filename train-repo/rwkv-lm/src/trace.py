import json
import os
from pathlib import Path
from time import perf_counter_ns

import torch
from safetensors.torch import save_file


REPO_NAME = "rwkv_lm"
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


def trace(filename: str, tensor: torch.Tensor) -> None:
    if not enabled():
        return

    path = case_root() / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter_ns()
    view = tensor.detach()
    if not view.is_contiguous():
        view = view.contiguous()
    if not view.device.type == "cpu":
        view = view.cpu()
    save_file({path.stem: view}, path)
    elapsed_ns = perf_counter_ns() - start

    path.with_suffix(".time.json").write_text(
        json.dumps({"filename": filename, "elapsed_ns": elapsed_ns}),
        encoding="utf-8",
    )


def trace_cell(layer_id: int, filename: str, tensor: torch.Tensor) -> None:
    trace(f"cells/cell_{layer_id:04d}/{filename}", tensor)

