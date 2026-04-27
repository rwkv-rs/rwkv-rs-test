# rwkv-rs-test

本仓库为 rwkv 的推理框架和训练框架提供完整的单元测试套件和基准测试套件。

## Trace 数据目录契约

导出激活值的目录结构契约如下，生成数据用于单元测试。

```text
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
```

## 命名规则

- `rwkv_lm` 只使用 `bf16`：`test_gen/rwkv_lm/bf16/...`
- `albatross` 只使用 `fp16`：`test_gen/albatross/fp16/...`
- 其它量化方案使用 `llama.cpp` 风格 `snake_case` 命名，例如 `q8_0`、`q4_k_m`、`q5_k_m`。
- 每个 `.safetensors` 文件只保存一个同名 tensor，`dtype` 必须保持导出时原样。

## 语义约定

- `embedding/embedded_context` 是 Embedding 输出，也是 `layer_norm0` 输入，不重复保存。
- `layer_norm0/embedded_context` 是 `cell_0000` 的残差前输入。
- `pre_layer_norm_for_time_mix/embedded_context` 是 TMix 的 norm 后输入。
- `time_mixer/embedded_context` 是 TMix 残差分支输出。
- `embedded_context_after_time_mixer` 是 TMix 残差后的 CMix norm 前输入。
- `pre_layer_norm_for_channel_mix/embedded_context` 是 CMix 的 norm 后输入。
- `channel_mixer/embedded_context` 是 CMix 残差分支输出。
- `embedded_context_after_channel_mixer` 是当前 cell 输出，也是下一层 cell 输入。
- `lm_head/embedded_context` 是 LMHead 输入，`lm_head/logits` 是 LMHead 输出。

## 插桩函数设计

以下是提供的插桩函数设计，直接复制到 repo 中调用完成数据和耗时的导出。

### Rust

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

### C++

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

### Python

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

## Trace 运行规则

给各 repo 加 trace 插桩，运行规则统一：

- 每个 repo 使用 `uv` 管理自己的环境。（训练需要最新版本的 `torch` + 最新版本的 `deepspeed`、`pytorch_lightning==1.9.5`、`ds_bucket_mb=200`。）不要切 `UV_CACHE_DIR` 等缓存目录，遇到沙箱只读问题找用户提权。
- 训练 repo：只跑 1 个真实 train step，完成该 step 的激活导出后退出（`L12-D768-CTX512-BSZ16`）。
- 推理 repo：只跑 1 个真实 prefill step，完成该 prefill 的激活导出后退出。（`weights/rwkv7-g1f-1.5b-20260419-ctx8192.pth`，以及 `rwkv7-g1f-1.5b-*.gguf`。）
- `rwkv-peft` 只跑 pretrain 路径，不是 LoRA / state tuning / SFT。
- 所有 repo 使用同一个环境变量，例如：

```bash
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen
RWKV_TRACE_ONCE=1
```

`RWKV_TRACE_ROOT` 负责指定导出根目录；`RWKV_TRACE_ONCE=1` 表示开启 trace，并在导出第一个完整训练 step 或第一个完整 prefill step 后退出。

### Trace 行为

- 输出目录仍为 `test_gen/<repo_name>/<quantization_name>/case_000000/...`。
- `RWKV_TRACE_ONCE` 在所有 repo 中语义一致：只导出第一个 case，然后立刻结束当前真实运行流程。
- 训练侧不要在 forward 中途退出；必须等 `training_step` 完成 loss 计算，必要时等 Lightning 完成当前 batch step，再停止 trainer。
- 推理侧不要进入 decode loop；prefill logits 导出完成后退出。

### 训练 Repo

- `rwkv-lm`：继续用原 `train.py` + DataLoader + `trainer.fit`，数据来自 `data/minipile.bin/.idx`。在第一个 `training_step` 的 forward 中导出所有 README 激活点，step 完成后通过 `RWKV_TRACE_ONCE` 停止训练。
- `rwkv-peft`：使用 `train.py` 的 pretrain 配置，明确传：

```bash
--peft none --data_type binidx --data_file ../../data/minipile --my_testing x070
```

不使用 `scripts/lora.sh`、`scripts/state tuning.sh`、`run_sft.sh` 的 PEFT 参数。插桩位置是 `rwkvt/rwkv7/model.py`、`block.py`、`att.py`、`ffn.py`，退出点放在 Lightning `training_step` / callback 层，保证只完成 1 个 train step。

### 推理 Repo

- albatross / rwkv-lightning：用现有 `benchmark.py` 或 demo 入口，调用真实 tokenizer 和 `model.forward(prompt_tokens, state)`。只导出 prompt prefill，不进入后续 token decode 循环。
- nano-vllm：在真实 scheduler / `ModelRunner.run_logits(..., is_prefill=True)` 路径导出第一个 prefill batch，完成后退出，不跑 decode。
- web-rwkv：用现有 example/runtime infer 入口，通过 hook 读取第一个 prefill chunk 的激活，导出后退出。
- llama.cpp / rwkv-mobile：通过真实 `llama_decode` / backend eval prefill 路径导出 graph tensor；第一个 prefill 完成后退出，不继续采样生成。
- prompt 统一设为：

```text
User: You are a very talented expert in aime24. Solve the problem and output the final answer in \\boxed{}. Problem: Let AB​CD be a tetrahedron such that AB = CD = \\sqrt{41}, AC = BD = \\sqrt{80}, and BC = AD = \\sqrt{89}. There exists a point I inside the tetrahedron such that the distances from I to each of the faces of the tetrahedron are all equal. This distance can be written in the form \\frac{m\\sqrt{n}}{p}, where m, n, and p are positive integers, m and p are relatively prime, and n is not divisible by the square of any prime. Find m + n + p. Assistant: <think
```

## 启动命令默认

### rwkv-lm

```bash
cd train-repo/rwkv-lm
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen bash trace-train.sh
```

### rwkv-peft pretrain

```bash
cd train-repo/rwkv-peft
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen bash trace-train.sh
```

### 推理 Repo

- 用各自现有 benchmark/demo/server 入口。
- 设置同样的：

```bash
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen
RWKV_TRACE_ONCE=1
```

- 入口检测到 `RWKV_TRACE_ONCE=1` 后，只执行第一个真实 prefill，并在导出后结束进程或返回主函数。

## rwkv-test CLI

`rwkv-test` 应该做成一个很薄的对比工具，放在 `/mnt/g/Projects/Packages/rwkv-rs-stable/crates/rwkv-test/src/main.rs`。

输入待测 trace 根目录和 baseline trace 根目录，按 README 的 safetensors 契约逐文件对比 abs error、relative error、cosine similarity。这个工具不应该很大。核心实现 150 行左右合理；加上 CLI、错误信息、阈值和表格输出，大概 220-300 行。

- `Cargo.toml` 添加最少依赖：`clap`、`anyhow`、`safetensors`、`half`。
- 输入目录不硬编码 repo 名、量化名或层数，递归扫描 `actual` 下所有 `.safetensors`。
- 用相对路径去 baseline 找同名文件。
- 忽略 `*.time.json`。
- 对 extra/missing 文件报错。
- 这样可以自然支持 `cell_0000`..`cell_N` 和不同 engine 的目录差异检查。

示例：

```bash
cargo run -p rwkv-test -- compare \
  --actual /path/to/test_gen/albatross/fp16/case_000000 \
  --baseline /path/to/test_gen/rwkv_lm/bf16/case_000000 \
  --atol 1e-3 --rtol 1e-2 --cos-min 0.999
```

### 文件契约

每个 `.safetensors` 文件必须满足 README 契约：

- 文件里只有一个 tensor。
- tensor 名等于文件 stem，例如 `logits.safetensors` 内 tensor 名为 `logits`。
- baseline 和 actual 的 `dtype`、shape、元素个数必须一致，否则直接失败。

### 统计规则

tensor 统一转成 `f64` 做统计：

- 支持 `F64` / `F32` / `F16` / `BF16`。
- `I64` / `I32` / `I16` / `I8` / `U64` / `U32` / `U16` / `U8` / `BOOL` 也可转成 `f64`，主要用于 `token_ids`。
- 不支持 safetensors 新增的 packed / FP8 / MX 类型，遇到时给清晰错误。

每个 tensor 输出：

- `max_abs`
- `mean_abs`
- `max_rel = max(abs(a-b) / max(abs(b), eps))`
- `mean_rel`
- `cosine = dot(a,b) / (norm(a)*norm(b))`
- `count`

### 失败规则

- shape / dtype / missing / extra 文件直接失败。
- 数值文件若 `max_abs > atol && max_rel > rtol` 则失败。
- `cosine < cos_min` 则失败。
- `token_ids` 这类整数 tensor 建议要求完全一致，也就是任何 abs diff 都失败。

### 输出格式

默认输出人类可读表格，按 `max_abs` 或失败项排序：

```text
status path dtype shape max_abs max_rel cosine
PASS lm_head/logits.safetensors F16 [1,512,...] 3.1e-4 8.2e-3 0.99998
FAIL cells/cell_0003/time_mixer/embedded... F16 [1,512,768] 1.7e-2 2.4e-1 0.99120
```

最后输出 summary：

```text
compared=84 passed=83 failed=1 missing=0 extra=0
worst_abs=...
worst_rel=...
worst_cosine=...
```

退出码：

- `0`：全部通过。
- `1`：存在数值差异或契约不匹配。
- `2`：CLI 参数、文件读取、safetensors 解析错误。

## 推理引擎 Benchmark

运行不同的推理引擎：

- 记录它们各自的吞吐量 Prefill TokenPerSecond 和 Decode TokenPerSecond。
- 记录延迟指标：TTFT、E2EL、TokenGenerationTime（E2EL-TTFT）、TimePerOutputToken（TokenGenerationTime / (token 数 - 1)）、ITL、分位数延迟。
- 绘制图像来对比不同的推理后端（需要选择合适的绘图形式来清晰对比）。

优先直接使用成熟推理引擎已有 benchmark/server/API 路径测量，不新增推理入口、不改核心调度。只为 albatross、rwkv-lightning 这类 demo 型代码，或 web-rwkv / rwkv-mobile 这种只有库/example bench、缺少完整服务压测指标的后端，添加薄测量脚本。

### 固定测试矩阵

- `bsz = [1, 16, 64, 128, 256, 512, 1024]`
- `prompt_len = [16, 256, 512, 1024, 4096]`
- `decode_len = 16`
- 数据使用：`data/gsm8k.jsonl`
- 允许 warmup。
- 每个仓库输出统一 CSV，并添加绘图脚本。
- 不要跑任何 smoke，直接跑出最终结果。

### Measurement Approach

全部使用本机 GPU 5090 运行。没权限找用户提权。

- nano-vllm：
  - 优先使用现有 OpenAI-compatible server + `benchmark_openai_api_perf.py` / `benchmark_openai_api.py`。
  - 用 `users_sweep` 映射 `bsz`，`max_tokens=16`，开启 streaming 以获得 TTFT / E2EL / ITL。
  - 如需严格 token 长度，用现有 `benchmark_rwkv.py` 的 direct engine 路径补充 exact `prompt_len` 的 prefill/decode TPS，不新增服务入口。
- llama.cpp：
  - 吞吐量优先使用原生 `llama-batched-bench`：`pl=bsz`、`pp=prompt_len`、`tg=16`、JSONL 输出。
  - 延迟指标优先使用现有 `llama-server` + `scripts/server-bench.py` 或 OpenAI-compatible streaming 请求测 TTFT / E2EL / ITL。
  - 只添加结果转换脚本，把原生 JSONL / server-bench 输出规整成统一 CSV。
- albatross / rwkv-lightning：
  - 作为 demo 型后端，直接打印时间来完成测量。
  - 直接调用真实 tokenizer/model forward 路径；prefill 用 batch prompt，decode 循环 16 token。
  - 只测原始代码可提供的路径；不实现完整队列调度。
- web-rwkv：
  - 基于现有 `examples/bench.rs` 扩展为 CSV benchmark example。
  - 不实现 server/scheduler，只测 library runtime 的 prefill/decode 原始路径。
- rwkv-mobile：
  - 基于已有 `simple_benchmark.cpp` / `batch_benchmark.cpp` 增加 CSV 输出版本。
  - 优先用已有 runtime API 和 batch decode；不补完整请求队列。
  - 不支持的 `bsz` / `prompt_len` 组合写 `status=failed` 或 `status=unsupported`。

## CSV Schema

统一字段：

- `repo,backend,model_path,model_format,device,dtype,quantization`
- `bsz,prompt_len,decode_len,warmup,repeat,seed,status,error`
- `prefill_tokens,output_tokens`
- `prefill_time_s,ttft_s,e2el_s,token_generation_time_s`
- `prefill_tps,decode_tps,e2e_tps,time_per_output_token_ms`
- `itl_mean_ms,itl_p50_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms`

指标定义：

- TTFT：请求开始到首个输出 token 到达。
- E2EL：请求开始到 16 个输出 token 完成。
- TokenGenerationTime = E2EL - TTFT。
- TimePerOutputToken = TokenGenerationTime / (decode_len - 1)。
- ITL：首 token 后相邻 token 到达间隔。
- Prefill TokenPerSecond = bsz \* prompt_len / prefill_time_s。
- Decode TokenPerSecond = bsz \* (decode_len - 1) / TokenGenerationTime。

## Plotting

每个仓库添加 `plot_task5.py`，读取本仓库 CSV 并输出：

- `results/task5_prefill_tps.png`
- `results/task5_decode_tps.png`
- `results/task5_ttft.png`
- `results/task5_e2el.png`
- `results/task5_itl_p95.png`

图形规则：

- x 轴为 `bsz`。
- 按 `prompt_len` 分面或分图。
- 不同后端/模式用不同颜色。
- 吞吐图使用 log y 轴。
- 延迟图使用 ms 单位。
- 失败组合不绘图，但保留在 CSV。
