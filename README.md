# rwkv-rs-test

本仓库为rwkv的推理框架和训练框架提供完整的单元测试套件和基准测试套件. 

1. 导出激活值的目录结构契约如下, 生成数据用于单元测试.

```
test_gen
└── <repo_name>/
    └── <quantization_name>/
        └── case_000000/
            ├── embedding/
            │   ├── token_ids.safetensors
            │   └── embedded_context.safetensors
            ├── layer_norm0/
            │   └── embedded_context.safetensors
            ├── cells/
            │   ├── cell_0000/
            │   │   ├── pre_layer_norm_for_time_mix/
            │   │   │   └── embedded_context.safetensors
            │   │   ├── time_mixer/
            │   │   │   ├── value_from_first_cell.safetensors
            │   │   │   └── embedded_context.safetensors
            │   │   ├── embedded_context_after_time_mixer.safetensors
            │   │   ├── pre_layer_norm_for_channel_mix/
            │   │   │   └── embedded_context.safetensors
            │   │   ├── channel_mixer/
            │   │   │   └── embedded_context.safetensors
            │   │   └── embedded_context_after_channel_mixer.safetensors
            │   ├── cell_0001/
            │   │   └── ...
            │   └── cell_<n_layer_minus_1>/
            │       └── ...
            └── lm_head/
                ├── embedded_context.safetensors
                └── logits.safetensors

命名规则:
- rwkv_lm只使用bf16: test_gen/rwkv_lm/bf16/...
- albatross只使用fp16: test_gen/albatross/fp16/...
- 其它量化方案使用llama.cpp风格snake_case命名, 例如q8_0, q4_k_m, q5_k_m.
- 每个.safetensors文件只保存一个同名tensor, dtype必须保持导出时原样.

语义约定:
- embedding/embedded_context是Embedding输出, 也是layer_norm0输入, 不重复保存.
- layer_norm0/embedded_context是cell_0000的残差前输入.
- pre_layer_norm_for_time_mix/embedded_context是TMix的norm后输入.
- time_mixer/embedded_context是TMix残差分支输出.
- embedded_context_after_time_mixer是TMix残差后的CMix norm前输入.
- pre_layer_norm_for_channel_mix/embedded_context是CMix的norm后输入.
- channel_mixer/embedded_context是CMix残差分支输出.
- embedded_context_after_channel_mixer是当前cell输出, 也是下一层cell输入.
- lm_head/embedded_context是LMHead输入, lm_head/logits是LMHead输出.
```

2. 以下是提供的插桩函数设计, 直接复制到repo中调用完成数据和耗时的导出

```rust
use std::{
    fs::{create_dir_all, write},
    path::Path,
    time::Instant,
};

use burn::prelude::{Backend, Tensor};
use burn::tensor::{DType as BurnDType, TensorKind};
use safetensors::{Dtype as SafeDtype, TensorView, serialize_to_file};
use web_rwkv::{
    num::Scalar,
    tensor::{TensorGpu, TensorShape, kind::Kind},
};

fn tensor_name(filename: &str) -> String {
    Path::new(filename)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

fn write_trace_time(path: &Path, filename: &str, elapsed_ns: u128) {
    let mut time_path = path.to_owned();
    time_path.set_extension("time.json");
    write(
        time_path,
        format!(r#"{{"filename":"{}","elapsed_ns":{}}}"#, filename, elapsed_ns),
    )
    .unwrap();
}

fn write_safetensor(path: &Path, name: String, dtype: SafeDtype, shape: Vec<usize>, bytes: &[u8]) {
    create_dir_all(path.parent().unwrap()).unwrap();
    let view = TensorView::new(dtype, shape, bytes).unwrap();
    serialize_to_file([(name, view)], None, path).unwrap();
}

fn burn_dtype_to_safetensors_dtype(dtype: BurnDType) -> SafeDtype {
    match dtype {
        BurnDType::F64 => SafeDtype::F64,
        BurnDType::F32 | BurnDType::Flex32 => SafeDtype::F32,
        BurnDType::F16 => SafeDtype::F16,
        BurnDType::BF16 => SafeDtype::BF16,
        BurnDType::I64 => SafeDtype::I64,
        BurnDType::I32 => SafeDtype::I32,
        BurnDType::I16 => SafeDtype::I16,
        BurnDType::I8 => SafeDtype::I8,
        BurnDType::U64 => SafeDtype::U64,
        BurnDType::U32 => SafeDtype::U32,
        BurnDType::U16 => SafeDtype::U16,
        BurnDType::U8 => SafeDtype::U8,
        BurnDType::Bool => SafeDtype::BOOL,
        BurnDType::QFloat(_) => panic!("quantized Burn TensorData has backend-specific layout"),
    }
}

// for webgpu based by web-rwkv
fn trace_webgpu<T, K>(output_path: &Path, filename: &str, tensor: TensorGpu<T, K>)
where
    T: Scalar,
    K: Kind,
{
    let path = output_path.join(filename);
    let start = Instant::now();
    let cpu = tensor.back_in_place();
    let shape: [usize; 4] = cpu.shape().into();
    let bytes = bytemuck::cast_slice(cpu.data().as_ref());
    write_safetensor(&path, tensor_name(filename), T::DATA_TYPE, shape.to_vec(), bytes);
    write_trace_time(&path, filename, start.elapsed().as_nanos());
}

// for burn based by rwkv-rs
fn trace_burn<B, const D: usize, K>(output_path: &Path, filename: &str, tensor: Tensor<B, D, K>)
where
    B: Backend,
    K: TensorKind<B>,
{
    let path = output_path.join(filename);
    let start = Instant::now();
    let data = tensor.into_data();
    let dtype = burn_dtype_to_safetensors_dtype(data.dtype);
    let shape = data.shape.clone();
    let bytes = data.as_bytes();
    write_safetensor(
        &path,
        tensor_name(filename),
        dtype,
        shape,
        bytes,
    );
    write_trace_time(&path, filename, start.elapsed().as_nanos());
}

// usage:
// trace_webgpu(case_root, "cells/cell_0000/time_mixer/embedded_context.safetensors", x);
// trace_burn(case_root, "cells/cell_0000/time_mixer/embedded_context.safetensors", x);
```

```cpp
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "safetensors.hh" // syoyo/safetensors-cpp
#include <torch/torch.h>

static std::string tensor_name(const std::string & filename) {
    return std::filesystem::path(filename).stem().string();
}

static void write_trace_time(
    const std::filesystem::path & path,
    const std::string & filename,
    long long elapsed_ns
) {
    auto time_path = path;
    time_path.replace_extension("time.json");
    std::ofstream out(time_path);
    out << "{\"filename\":\"" << filename << "\",\"elapsed_ns\":" << elapsed_ns << "}";
}

static safetensors::dtype ggml_dtype(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return safetensors::dtype::kFLOAT32;
        case GGML_TYPE_F16:  return safetensors::dtype::kFLOAT16;
        case GGML_TYPE_BF16: return safetensors::dtype::kBFLOAT16;
        case GGML_TYPE_I32:  return safetensors::dtype::kINT32;
        case GGML_TYPE_I16:  return safetensors::dtype::kINT16;
        case GGML_TYPE_I8:   return safetensors::dtype::kINT8;
        default: throw std::runtime_error("unsupported ggml dtype for safetensors trace");
    }
}

static safetensors::dtype torch_dtype(torch::ScalarType type) {
    switch (type) {
        case torch::kFloat64:  return safetensors::dtype::kFLOAT64;
        case torch::kFloat32:  return safetensors::dtype::kFLOAT32;
        case torch::kFloat16:  return safetensors::dtype::kFLOAT16;
        case torch::kBFloat16: return safetensors::dtype::kBFLOAT16;
        case torch::kInt64:    return safetensors::dtype::kINT64;
        case torch::kInt32:    return safetensors::dtype::kINT32;
        case torch::kInt16:    return safetensors::dtype::kINT16;
        case torch::kInt8:     return safetensors::dtype::kINT8;
        case torch::kUInt8:    return safetensors::dtype::kUINT8;
        case torch::kBool:     return safetensors::dtype::kBOOL;
        default: throw std::runtime_error("unsupported torch dtype for safetensors trace");
    }
}

static void save_safetensor(
    const std::filesystem::path & path,
    const std::string & name,
    safetensors::dtype dtype,
    const std::vector<size_t> & shape,
    const std::vector<uint8_t> & bytes
) {
    std::filesystem::create_directories(path.parent_path());
    safetensors::safetensors_t st;
    safetensors::tensor_t t;
    t.dtype = dtype;
    t.shape = shape;
    t.data_offsets = {0, bytes.size()};
    st.tensors[name] = t;
    st.storage = bytes;

    std::string warn;
    std::string err;
    const bool ok = safetensors::save_to_file(st, path.string(), &warn, &err);
    if (!ok) {
        throw std::runtime_error(err);
    }
}

// for llama.cpp / ggml
static void trace_ggml(
    const std::filesystem::path & output_path,
    const std::string & filename,
    const ggml_tensor * tensor
) {
    const auto path = output_path / filename;
    const auto start = std::chrono::steady_clock::now();

    std::vector<size_t> shape;
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    save_safetensor(path, tensor_name(filename), ggml_dtype(tensor->type), shape, bytes);

    const auto end = std::chrono::steady_clock::now();
    write_trace_time(path, filename, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

// for libtorch
static void trace_libtorch(
    const std::filesystem::path & output_path,
    const std::string & filename,
    const torch::Tensor & tensor
) {
    const auto path = output_path / filename;
    const auto start = std::chrono::steady_clock::now();

    TORCH_CHECK(tensor.is_contiguous(), "trace_libtorch requires a contiguous tensor");
    const torch::Tensor host = tensor.device().is_cpu() ? tensor : tensor.cpu();

    std::vector<size_t> shape;
    for (const auto dim : host.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }

    const auto nbytes = static_cast<size_t>(host.nbytes());
    const auto * ptr = static_cast<const uint8_t *>(host.const_data_ptr());
    std::vector<uint8_t> bytes(ptr, ptr + nbytes);
    save_safetensor(path, tensor_name(filename), torch_dtype(host.scalar_type()), shape, bytes);

    const auto end = std::chrono::steady_clock::now();
    write_trace_time(path, filename, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

// usage:
// trace_ggml(case_root, "lm_head/logits.safetensors", logits);
// trace_libtorch(case_root, "lm_head/logits.safetensors", logits);
```

```python
from pathlib import Path
from time import perf_counter_ns
import json

import torch
from safetensors.torch import save_file


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


# usage:
# trace(case_root, "cells/cell_0000/time_mixer/embedded_context.safetensors", x)
```

3. 所有数据都需要是真实训练和推理过程中导出的, 不可以自己建立任何的服务于导出数据的程序入口, 只可以添加插桩, rwkv-lm和rwkv-peft均有Dataloader, 需要复用, 使用data目录下的binidx, 这同样也是rwkv-lm和rwkv-peft所默认使用的训练数据,无需更改任何脚本参数就能运行; 

4. 在G:\Projects\Packages\rwkv-rs-stable\crates\rwkv-test设计CLI工具, 导入两个目录路径, 一个是待测的导出数据的根, 另一个是baseline导出数据的根, 按照1中所设计的契约读取safetensor, 和baseline对比绝对误差, 相对误差, 以及向量相似度; 

5. 运行不同的推理引擎, [1]记录它们各自的吞吐量Prefill TokenPerSecond和Decode TokenPerSecond [2] 记录延迟指标: TTFT, E2EL, TokenGenerationTime(E2EL-TTFT),TimePerOutputToken(TokenGenerationTime/(token数-1)), ITL, 分位数延迟 . 并且绘制出图像来对比不同的推理后端(需要选择合适的绘图形式来清晰对比).
