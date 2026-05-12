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
- 训练 trace 和推理 trace 的导出集合不同。训练 trace 必须仍然从原训练入口跑一个真实
  `training_step`，但测试目标是训练路径 custom CUDA kernel 的输出，而不是推理式 logits
  对齐。
- `lm_head/logits` 只属于推理 prefill 或显式 logits 对齐 case；`rwkv_lm` 训练 trace 不要求导出 `lm_head/logits`。
- `rwkv_lm` 训练 trace 至少导出 custom kernel 主输出：
  `cells/cell_*/time_mixer/embedded_context.safetensors`、
  `cells/cell_*/channel_mixer/embedded_context.safetensors`、
  `loss/l2wrap_cross_entropy.safetensors`。启用 `--head_chunk > 0` 时，训练 trace 改为导出
  `loss/head_l2wrap_cross_entropy.safetensors` 作为 fused head+CE kernel 主输出。
- 一个 CUDA kernel 返回多个 tensor 时，所有导出的 tensor 都必须写同一个真实 kernel compute
  边界耗时；测试/汇总层可以只选择主输出做性能聚合，但不得通过给辅助输出写 `0` 来避免重复汇总。

## 插桩函数设计

以下是提供的插桩函数设计，直接复制到 repo 中调用完成数据导出。

`*.time.json` 的 `elapsed_ns` 只允许记录产生该 tensor 的被测 compute 区间耗时。不要在
trace/export helper 内部计时；不要把 GPU readback、CPU copy、safetensors 序列化或文件
写入算进去；也不要把整段 prefill / train step 的耗时挂到某个中间 tensor 上。

只要为某个被测输出写了插桩代码，就必须保证该输出的 `elapsed_ns` 来自正确的 compute
边界和必要的设备同步。不会正确计时就不要写成 `0` 冒充有效 trace；该后端必须修正计时
实现，或把对应测项标记为 unsupported 并拒绝产出 timing 结果。`elapsed_ns=0` 不是
“测不了时的占位方案”，只能用于确实没有 compute 区间的输入元数据。

`elapsed_ns` 必须写平均耗时，不允许写单次运行的总耗时。平均口径是同一原始程序入口、
同一输入、同一模型、同一设备同步边界下，跳过 `warmup` 次完整 trace run 后，对 `repeat`
次完整 trace run 中同名 `.time.json` 的有效 `elapsed_ns` 做算术平均：

```text
elapsed_ns = round(sum(samples_ns) / repeat)
```

平均只跨完整 trace run，不在模型 forward 内重复执行某个子模块，也不按 token、batch
element、hidden size 或输出元素数再除一次。每个 `.time.json` 必须同时写入：

- `elapsed_ns`：平均后的 ns。
- `repeat`：参与平均的有效 trace run 数。
- `warmup`：未参与平均的预热 trace run 数。
- `samples_ns`：参与平均的每次完整 trace run 原始样本。

`elapsed_ns=0` 只允许用于确实没有被测 compute 区间的输入元数据，例如
`embedding/token_ids.safetensors`。别名/透传 tensor、辅助输出和 unsupported 测项不得
靠写 `0` 通过 timing 契约；需要保留数值 trace 时，应在测试/汇总层显式排除其 timing，
或修正为可验证的真实计时。

### Rust

```rust
use std::{
    fs::{create_dir_all, write},
    path::Path,
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
        format!(
            r#"{{"filename":"{}","elapsed_ns":{},"repeat":1,"warmup":0,"samples_ns":[{}]}}"#,
            filename, elapsed_ns, elapsed_ns
        ),
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
fn trace_webgpu<T, K>(output_path: &Path, filename: &str, tensor: TensorGpu<T, K>, elapsed_ns: u128)
where
    T: Scalar,
    K: Kind,
{
    let path = output_path.join(filename);
    let cpu = tensor.back_in_place();
    let shape: [usize; 4] = cpu.shape().into();
    let bytes = bytemuck::cast_slice(cpu.data().as_ref());
    write_safetensor(&path, tensor_name(filename), T::DATA_TYPE, shape.to_vec(), bytes);
    write_trace_time(&path, filename, elapsed_ns);
}

// for burn based by rwkv-rs
fn trace_burn<B, const D: usize, K>(
    output_path: &Path,
    filename: &str,
    tensor: Tensor<B, D, K>,
    elapsed_ns: u128,
)
where
    B: Backend,
    K: TensorKind<B>,
{
    let path = output_path.join(filename);
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
    write_trace_time(&path, filename, elapsed_ns);
}

// usage:
// let start = Instant::now();
// let y = run_time_mixer(x);
// let elapsed_ns = start.elapsed().as_nanos();
// trace_webgpu(case_root, "cells/cell_0000/time_mixer/embedded_context.safetensors", y, elapsed_ns);
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
    out << "{\"filename\":\"" << filename << "\",\"elapsed_ns\":" << elapsed_ns
        << ",\"repeat\":1,\"warmup\":0,\"samples_ns\":[" << elapsed_ns << "]}";
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
    const ggml_tensor * tensor,
    long long elapsed_ns
) {
    const auto path = output_path / filename;

    std::vector<size_t> shape;
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    save_safetensor(path, tensor_name(filename), ggml_dtype(tensor->type), shape, bytes);
    write_trace_time(path, filename, elapsed_ns);
}

// for libtorch
static void trace_libtorch(
    const std::filesystem::path & output_path,
    const std::string & filename,
    const torch::Tensor & tensor,
    long long elapsed_ns
) {
    const auto path = output_path / filename;

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
    write_trace_time(path, filename, elapsed_ns);
}

// usage:
// trace_ggml(case_root, "lm_head/logits.safetensors", logits, lm_head_elapsed_ns);
// trace_libtorch(case_root, "lm_head/logits.safetensors", logits, lm_head_elapsed_ns);
```

llama.cpp 的 `cb_eval` 只能在 scheduler 观察到某个 graph tensor 时回调。用于逐模块耗时
对比时，计时必须夹在命中 observed node 的 `ask=true` 和对应 `ask=false` 之间，也就是
记录 scheduler 计算到该 tensor 并同步完成的 graph segment 耗时；不要在
`trace_ggml` / `trace_bytes` / safetensors 写入 helper 内部计时。旧版
`test_gen/llama_cpp/fp16/case_000000` 若由写文件 helper 计时生成，其 `.time.json` 只代表
导出耗时，必须重新导出。

### Python

```python
from pathlib import Path
import json

import torch
from safetensors.torch import save_file


def trace(output_path: str | Path, filename: str, tensor: torch.Tensor, elapsed_ns: int) -> None:
    path = Path(output_path) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    view = tensor.detach()
    if not view.is_contiguous():
        raise RuntimeError("trace requires a contiguous torch.Tensor")
    save_file({path.stem: view}, path)

    path.with_suffix(".time.json").write_text(
        json.dumps(
            {
                "filename": filename,
                "elapsed_ns": elapsed_ns,
                "repeat": 1,
                "warmup": 0,
                "samples_ns": [elapsed_ns],
            }
        ),
        encoding="utf-8",
    )


# usage:
# output, elapsed_ns = measure(lambda: time_mixer(x))
# trace(case_root, "cells/cell_0000/time_mixer/embedded_context.safetensors", output, elapsed_ns)
```

## Trace 运行规则

给各 repo 加 trace 插桩，运行规则统一：

- 每个 repo 使用 `uv` 管理自己的环境。（训练需要最新版本的 `torch` + 最新版本的 `deepspeed`、`pytorch_lightning==1.9.5`、`ds_bucket_mb=200`。）不要切 `UV_CACHE_DIR` 等缓存目录，遇到沙箱只读问题找用户提权。
- 训练 repo：只跑 1 个真实 train step，完成该 step 的训练 kernel 输出和必要激活导出后退出（`L12-D768-CTX512-BSZ16`）。
- 推理 repo：只跑 1 个真实 prefill step，完成该 prefill 的激活导出后退出。（`weights/rwkv7-g1f-1.5b-20260419-ctx8192.pth`，以及 `rwkv7-g1f-1.5b-*.gguf`。）
- `rwkv-peft` 只跑 pretrain 路径，不是 LoRA / state tuning / SFT。
- 所有 repo 使用同一个环境变量，例如：

```bash
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen
RWKV_TRACE_ONCE=1
```

`RWKV_TRACE_ROOT` 负责指定导出根目录；`RWKV_TRACE_ONCE=1` 表示开启 trace，并在导出第一个完整训练 step 或第一个完整 prefill step 后退出。
最终 trace 结果必须通过仓库根目录的平均导出脚本生成：

```bash
TRACE_REPEAT=3 TRACE_WARMUP=1 bash scripts/export_trace_average.sh
```

该脚本逐后端运行原有真实入口多次，用仓库内 staging 目录收集中间结果，校验每次导出的
`.time.json` 集合一致，然后把最后一次 tensor 数据和平均后的 `.time.json` 写回
`test_gen/<repo_name>/<quantization_name>/case_000000`。

### Trace 行为

- 输出目录仍为 `test_gen/<repo_name>/<quantization_name>/case_000000/...`。
- `RWKV_TRACE_ONCE` 在所有 repo 中语义一致：只导出第一个 case，然后立刻结束当前真实运行流程。
- 训练侧不要在 forward 中途退出；必须等 `training_step` 完成 loss kernel 计算，必要时等 Lightning 完成当前 batch step，再停止 trainer。
- 推理侧不要进入 decode loop；prefill logits 导出完成后退出。

### 训练 Repo

- `rwkv-lm`：继续用原 `train.py` + DataLoader + `trainer.fit`，数据来自 `data/minipile.bin/.idx`。在第一个 `training_step` 中导出训练 kernel 输出；不为训练契约强制导出 `lm_head/logits`。step 完成后通过 `RWKV_TRACE_ONCE` 停止训练。
- `rwkv-peft`：使用 `train.py` 的 pretrain 配置，明确传：

```bash
--peft none --data_type binidx --data_file ../../data/minipile --my_testing x070
```

不使用 `scripts/lora.sh`、`scripts/state tuning.sh`、`run_sft.sh` 的 PEFT 参数。插桩位置是 `rwkvt/rwkv7/model.py`、`block.py`、`att.py`、`ffn.py`，退出点放在 Lightning `training_step` / callback 层，保证只完成 1 个 train step。

### 推理 Repo

- albatross / rwkv-lightning：用现有 `benchmark.py` 或 demo 入口，调用真实 tokenizer 和 `model.forward(prompt_tokens, state)`。只导出 prompt prefill，不进入后续 token decode 循环。
- nano-vllm：在真实 scheduler / `ModelRunner.run_logits(..., is_prefill=True)` 路径导出第一个 prefill batch，完成后退出，不跑 decode。
- web-rwkv：用现有 example/runtime infer 入口，通过 hook 读取第一个 prefill chunk 的激活，导出后退出。
  `web_rwkv` trace 必须在 Windows WebGPU 环境运行；WSL 里即使 CUDA / `nvidia-smi` 可用，
  `wgpu` 也可能只能枚举到 llvmpipe CPU adapter，不能作为正式 `web_rwkv` 导出环境。
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

上面的命令只用于单次调试。正式结果使用：

```bash
TRACE_REPEAT=3 TRACE_WARMUP=1 bash scripts/export_trace_average.sh rwkv_lm
```

### rwkv-peft pretrain

```bash
cd train-repo/rwkv-peft
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen bash trace-train.sh
```

上面的命令只用于单次调试。正式结果使用：

```bash
TRACE_REPEAT=3 TRACE_WARMUP=1 bash scripts/export_trace_average.sh rwkv_peft
```

### 推理 Repo

- 用各自现有 benchmark/demo/server 入口。
- 设置同样的：

```bash
RWKV_TRACE_ROOT=/mnt/g/Projects/Packages/rwkv-rs-test/test_gen
RWKV_TRACE_ONCE=1
```

- 入口检测到 `RWKV_TRACE_ONCE=1` 后，只执行第一个真实 prefill，并在导出后结束进程或返回主函数。
- Linux / WSL 中正式结果通过 `scripts/export_trace_average.sh albatross llama_cpp` 生成平均耗时。
- `web_rwkv` 正式结果在 Windows 环境运行同一平均导出契约；不要用 WSL 下的 llvmpipe
  WebGPU fallback 结果覆盖 `test_gen/web_rwkv/fp16/case_000000`。
  Windows 下如果 Dx12 adapter 不暴露 `SUBGROUP` feature，`web_rwkv` trace 使用
  `cargo run --no-default-features --features tokio --example trace_infer -- --model ..\..\weights\rwkv7-g1f-1.5b-20260419-ctx8192.st`，
  避免默认 `native` feature 请求不可用的 subgroup capability。

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

Benchmark 必须先冻结测量合同，再开始长跑。不要把 synthetic throughput、server
latency、GSM8K workload、trace/export timing 混进同一个速度结论。每次跑分都必须能回答：
测的是什么、使用哪个 runner、输入样本如何生成、是否固定 RTX 5090、失败组合为什么失败。

### 硬件与 Preflight

所有性能数据只允许使用本机 RTX 5090。运行前必须写入 preflight 记录：

- `gpu_name`、`gpu_uuid`、driver version、CUDA / backend runtime version。
- benchmark 二进制路径、commit / build id、完整命令行。
- 模型文件路径、量化类型、文件大小，建议记录 sha256。
- 结果 CSV 的 `device` 字段必须是可追踪的硬件标识，至少为 `cuda0`，聚合报告必须同时包含
  RTX 5090 的 `gpu_uuid`。只写 `cuda`、`unknown` 或 CPU 结果不能参与最终对比。
- WSL 下 `web-rwkv` 如果只能看到 `llvmpipe` 或 `unknown`，不得参与 5090 对比；必须改用能看到
  RTX 5090 的 Windows native / Dx12 路径，或明确标记 `unsupported`。
- 没权限或 GPU 不可见时停止测试，不允许改用 CPU 产出性能数据。

### 模型与量化矩阵

Task 5 的 llama.cpp/RWKV 模型规模矩阵为：

- `0.1B`: `rwkv7-g1d-0.1b-20260129-ctx8192`
- `0.4B`: `rwkv7-g1d-0.4b-20260210-ctx8192`
- `2.9B`: `rwkv7-g1f-2.9b-20260420-ctx8192`
- `7.2B`: `rwkv7-g1f-7.2b-20260414-ctx8192`
- `13.3B`: `rwkv7-g1f-13.3b-20260415-ctx8192`

每个规模至少导出并测试：

- `FP16`
- `Q4_K_M`
- `Q5_K_M`
- `Q6_K`
- `Q8_0`

`1.5B` 可作为额外参考点，但不能替代上述五个规模，也不能让图表只围绕 `1.5B` 展开。

### Benchmark 类型

#### 1. Synthetic Throughput

目标：测后端核心 prefill/decode 计算吞吐，尽量暴露 RTX 5090 能力。它不是 GSM8K 测试。

矩阵：

- `bsz = [1, 16, 64, 128, 256, 320, 512, 960, 1024]`
- `prompt_len = [16, 256, 512, 1024, 4096]`
- `decode_len = 16`

实现要求：

- llama.cpp 使用 `llama-batched-bench`，固定 `CUDA_VISIBLE_DEVICES=0` 和 `-ngl 999`。
- nano-vllM / albatross / web-rwkv / rwkv-mobile 必须使用各自最接近“核心模型计算”的 batch
  prefill + decode 路径，不要通过 HTTP server 路径冒充核心吞吐。
- 不支持的组合写 `status=unsupported`，真实运行失败写 `status=failed` 并保存错误。

导出：

- `results/task5_throughput.csv`
- 原始 runner 输出保存到 `results/raw/throughput/`

#### 2. Synthetic Server Latency

目标：测服务/调度路径的 TTFT、E2EL、ITL。它可以 GPU 利用率较低，不能拿来表示峰值吞吐。

矩阵同 synthetic throughput，但必须真实执行 `repeat`，不能只把 `repeat` 写进 CSV。

实现要求：

- 每个 `(model, quant, bsz, prompt_len, decode_len)` 至少运行 `repeat >= 3`。
- prompt token 可以由固定 fixture 生成，也可以由 GSM8K 文本 token 化后重复/截断，但必须在 CSV
  或 manifest 里标明 `prompt_source`。
- 每个请求要保存请求级原始数据，聚合指标由请求级数据计算。

导出：

- `results/task5_latency_synthetic.csv`
- `results/raw/latency_synthetic/*.jsonl`

#### 3. GSM8K Workload Latency

目标：测真实 GSM8K prompt 分布下的服务延迟。它不是固定 `prompt_len` 矩阵。

要求：

- 默认使用 `data/gsm8k.jsonl` 全量 1319 条问题。
- 不强行把 prompt 重复/截断到固定长度；记录自然 token 长度。
- 并发档位建议从 `[1, 16, 64, 128]` 开始；更高并发只有在后端明确支持时再加。
- `decode_len` 应按真实服务目标单独固定，例如 `16` 用于短 decode，`128` 用于更稳定的生成延迟。

导出：

- `results/task5_latency_gsm8k_requests.csv`：一请求一行。
- `results/task5_latency_gsm8k_summary.csv`：按模型/量化/并发聚合。
- `results/raw/latency_gsm8k/*.jsonl`

### CSV Schema

所有 CSV 至少包含：

- `run_id,repo,backend,runner,benchmark_kind`
- `model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization`
- `bsz,prompt_len,decode_len,warmup,repeat,seed,status,error`
- `prompt_source,prompt_count,prompt_tokens,output_tokens`
- `prefill_time_s,ttft_s,e2el_s,token_generation_time_s`
- `prefill_tps,decode_tps,e2e_tps,time_per_output_token_ms`
- `itl_mean_ms,itl_p50_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms`
- `command,binary_path,binary_build_id,started_at,ended_at`

指标定义：

- TTFT：请求开始到首个输出 token 到达。
- E2EL：请求开始到请求完成。
- TokenGenerationTime = E2EL - TTFT。
- TimePerOutputToken = TokenGenerationTime / max(`decode_len - 1`, 1)。
- ITL：首 token 后相邻 token 到达间隔。
- Prefill TokenPerSecond = `prefill_tokens / prefill_time_s`。
- Decode TokenPerSecond = `output_tokens / TokenGenerationTime`，或 runner 原生 decode TPS；必须在
  `runner` / `benchmark_kind` 中区分。

### GPU Telemetry

每次性能测试必须旁路采集 GPU telemetry：

- `results/gpu_telemetry.csv`
- 字段：`timestamp,run_id,gpu_uuid,gpu_util,mem_used,mem_total,power_w,sm_clock,mem_clock,pstate,process_name`

没有 telemetry 的结果只能作为开发调试数据，不能作为最终性能报告。

### Plotting

不同 benchmark kind 分开画，不能混图。

Synthetic throughput：

- `plots/throughput_prefill_heatmap.png`：`bsz x prompt_len` 的 `prefill_tps`。
- `plots/throughput_decode_heatmap.png`：`bsz x prompt_len` 的 `decode_tps`。
- `plots/throughput_frontier.png`：每个模型/量化能成功运行的最大 `bsz * (prompt_len + decode_len)`。
- `plots/throughput_pareto.png`：模型大小 / 量化大小 vs `decode_tps`。

Synthetic server latency：

- `plots/latency_synthetic_ttft_p95.png`
- `plots/latency_synthetic_e2el_p95.png`
- `plots/latency_synthetic_itl_p95.png`

GSM8K workload latency：

- `plots/gsm8k_ttft_cdf.png`
- `plots/gsm8k_e2el_cdf.png`
- `plots/gsm8k_concurrency_reqps.png`
- `plots/gsm8k_concurrency_p95_latency.png`

GPU telemetry：

- `plots/gpu_util_time.png`
- `plots/gpu_vram_time.png`
- `plots/gpu_power_time.png`

### 无效结果处理

以下结果必须删除或移入明确的 rejected 目录，不得参与最终报告：

- CPU 性能数据。
- 只写 `cuda` / `unknown` 且没有 RTX 5090 `gpu_uuid` 的性能数据。
- smoke test 数据。
- partial run 数据，除非文件名和 manifest 明确标记 `partial`。
- trace/export 模式数据。
- 把 `llama-batched-bench` throughput 和 `llama-server` latency 混成一个速度结论的数据。
