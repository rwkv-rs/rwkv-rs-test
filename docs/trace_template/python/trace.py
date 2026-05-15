from collections.abc import Callable, Mapping
import json
import os
from pathlib import Path
import re
from time import perf_counter_ns
from typing import Any

import torch
from safetensors.torch import save_file


TRACE = os.environ.get("RWKV_TRACE_ONCE") == "1"
TRACE_REPEAT = int(os.environ.get("RWKV_TRACE_REPEAT", "3"))
TRACE_WARMUP = int(os.environ.get("RWKV_TRACE_WARMUP", "1"))
OutputSelector = Callable[[Any], torch.Tensor] | int | tuple[int, ...]
TraceOutputs = str | Mapping[OutputSelector, str] | None
TensorKey = tuple[str, torch.dtype, tuple[int, ...], tuple[int, ...], int, int, int]

_SAVED_BY_FILENAME: dict[str, TensorKey] = {}
_LAST_SAMPLES_NS: list[int] = []
_CANONICAL_CELL_RE = re.compile(
    r"^cells/cell_\d{4}/("
    r"time_mixer/value_from_first_cell|"
    r"time_mixer/embedded_context|"
    r"embedded_context_after_time_mixer|"
    r"channel_mixer/embedded_context|"
    r"embedded_context_after_channel_mixer"
    r")\.safetensors$"
)
_CANONICAL_SINGLETONS = {
    "embedding/token_ids.safetensors",
    "embedding/embedded_context.safetensors",
    "layer_norm0/embedded_context.safetensors",
    "lm_head/embedded_context.safetensors",
    "lm_head/logits.safetensors",
    "loss/l2wrap_cross_entropy.safetensors",
    "loss/l2wrap_cross_entropy/lse.safetensors",
    "loss/l2wrap_cross_entropy/max_vals.safetensors",
    "loss/l2wrap_cross_entropy/argmax.safetensors",
    "loss/head_l2wrap_cross_entropy.safetensors",
    "loss/head_l2wrap_cross_entropy/grad_hidden.safetensors",
    "loss/head_l2wrap_cross_entropy/grad_weight.safetensors",
}


def is_canonical_path(filename: str) -> bool:
    return filename in _CANONICAL_SINGLETONS or _CANONICAL_CELL_RE.match(filename) is not None


def case_root() -> Path:
    root = os.environ.get("RWKV_TRACE_ROOT")
    if not root:
        raise RuntimeError("RWKV_TRACE_ROOT must be set when RWKV_TRACE_ONCE=1")
    return Path(root) / "repo_name" / "quantization" / "case_000000"


def _sync_tensors(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            torch.cuda.synchronize(value.device)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _sync_tensors(item)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _sync_tensors(item)


def _tensor_key(tensor: torch.Tensor) -> TensorKey:
    return (
        str(tensor.device),
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
        tensor._version,
    )


def _select(output: Any, selector: OutputSelector) -> torch.Tensor:
    if callable(selector):
        return selector(output)
    if isinstance(selector, int):
        return output[selector]

    value = output
    for index in selector:
        value = value[index]
    return value


def activation(filename: str, tensor: torch.Tensor) -> None:
    if not TRACE:
        return
    if not is_canonical_path(filename):
        raise RuntimeError(f"{filename} is not a canonical trace path")

    key = _tensor_key(tensor)
    saved_key = _SAVED_BY_FILENAME.get(filename)
    if saved_key is not None:
        if saved_key == key:
            return
        raise RuntimeError(f"{filename} already saved from a different tensor")

    path = case_root() / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    view = tensor.detach()
    if not view.is_contiguous():
        view = view.contiguous()
    if view.device.type != "cpu":
        view = view.cpu()
    save_file({path.stem: view}, path)
    _SAVED_BY_FILENAME[filename] = key


def timing(module: str, elapsed_ns: int) -> None:
    if not TRACE:
        return
    if elapsed_ns <= 0:
        raise RuntimeError(f"{module} timing must be a positive module forward duration")

    path = case_root() / "timing" / f"{module}.time.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "module": module,
                "elapsed_ns": elapsed_ns,
                "repeat": TRACE_REPEAT,
                "warmup": TRACE_WARMUP,
                "samples_ns": _LAST_SAMPLES_NS,
            }
        ),
        encoding="utf-8",
    )


def _write_outputs(outputs: TraceOutputs, result: Any) -> None:
    if outputs is None:
        return
    if isinstance(outputs, str):
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("single-output trace requires a torch.Tensor result")
        activation(outputs, result)
        return

    for selector, filename in outputs.items():
        activation(filename, _select(result, selector))


def trace(
    module: str,
    target: Callable[..., Any],
    *args: Any,
    outputs: TraceOutputs = None,
    **kwargs: Any,
) -> Any:
    if not TRACE:
        return target(*args, **kwargs)
    if TRACE_REPEAT <= 0:
        raise RuntimeError("RWKV_TRACE_REPEAT must be positive")
    if TRACE_WARMUP < 0:
        raise RuntimeError("RWKV_TRACE_WARMUP must be non-negative")

    for _ in range(TRACE_WARMUP):
        _sync_tensors(args)
        _sync_tensors(kwargs)
        warmup_result = target(*args, **kwargs)
        _sync_tensors(warmup_result)

    samples_ns: list[int] = []
    result = None
    for _ in range(TRACE_REPEAT):
        _sync_tensors(args)
        _sync_tensors(kwargs)
        start = perf_counter_ns()
        result = target(*args, **kwargs)
        _sync_tensors(result)
        samples_ns.append(perf_counter_ns() - start)

    elapsed_ns = round(sum(samples_ns) / len(samples_ns))
    global _LAST_SAMPLES_NS
    _LAST_SAMPLES_NS = samples_ns
    _write_outputs(outputs, result)
    timing(module, elapsed_ns)
    return result
