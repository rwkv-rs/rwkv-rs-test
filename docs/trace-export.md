## 插桩函数设计

以下是提供的插桩函数设计，直接复制到 repo 中调用完成数据导出。

激活值和耗时是两个独立契约：

- 激活值描述模块边界上的输入/输出 tensor，写入 canonical `.safetensors` 路径。
- 耗时描述某个模块 forward 的 compute 区间，写入 `timing/<module>.time.json`。

禁止给某个激活值路径写同名 `.time.json`，例如
`embedding/token_ids.time.json`、`cells/cell_0000/time_mixer/embedded_context.time.json`
都不是合法 timing 文件。输入激活没有模块 forward 耗时，不允许用 `elapsed_ns=0`
伪造 timing；它只应该保存 `.safetensors`。

`*.time.json` 的 `elapsed_ns` 只允许记录对应模块 forward 的被测 compute 区间耗时。计时
边界必须包住被测模块调用本身，而不是插到 kernel 内部，也不要把 GPU readback、CPU copy、
safetensors 序列化或文件写入算进去；更不要把整段 prefill / train step 的耗时挂到某个
模块上。

只要为某个模块写了 timing，就必须保证该模块的 `elapsed_ns` 来自正确的 compute
边界和必要的设备同步。不会正确计时就不要写成 `0` 冒充有效 trace；该后端必须修正计时
实现，或把对应模块标记为 unsupported 并拒绝产出 timing 结果。

`elapsed_ns` 必须写平均耗时，不允许写单次运行的总耗时。平均口径是同一原始程序入口、
同一输入、同一模型、同一设备同步边界下，跳过 `warmup` 次完整 trace run 后，对 `repeat`
次完整 trace run 中同名模块 `.time.json` 的有效 `elapsed_ns` 做算术平均：

```text
elapsed_ns = round(sum(samples_ns) / repeat)
```

平均只跨完整 trace run，不在模型 forward 内重复执行某个子模块，也不按 token、batch
element、hidden size 或输出元素数再除一次。每个模块 `.time.json` 必须同时写入：

- `module`：模块名，例如 `embedding`、`cells/cell_0000/time_mixer`。
- `elapsed_ns`：平均后的 ns。
- `repeat`：参与平均的有效 trace run 数。
- `warmup`：未参与平均的预热 trace run 数。
- `samples_ns`：参与平均的每次完整 trace run 原始样本。

多输出模块只能写一个模块 timing。例如 `loss/l2wrap_cross_entropy` 同时导出 loss、lse、
max_vals、argmax 等激活值时，只写 `timing/loss/l2wrap_cross_entropy.time.json`，
不得把同一个耗时复制到多个激活值的同名 `.time.json`。

### 模板文件

代码模板已经拆到 `docs/trace_template`，主文档只保留行为约束和调用入口：

- Rust Vulkan helper: `docs/trace_template/rust/vulkan/trace.rs`
- Rust Vulkan usage: `docs/trace_template/rust/vulkan/usage.rs`
- Rust Burn helper: `docs/trace_template/rust/burn/trace.rs`
- Rust Burn usage: `docs/trace_template/rust/burn/usage.rs`
- C++ helper: `docs/trace_template/cpp/trace.cpp`
- C++ usage: `docs/trace_template/cpp/usage.cpp`
- Python helper: `docs/trace_template/python/trace.py`
- Python usage: `docs/trace_template/python/usage.py`

Rust Vulkan helper 只描述 web-rwkv / WebGPU 插桩。web-rwkv 的 `TensorGpu` 有 runtime
id，可以自动跳过重复输入/别名。Rust Burn helper 单独放在 `rust/burn` 下；Burn 的泛型
`Tensor<B, D, K>` 不保证暴露稳定 storage id，因此只有传入 backend-specific key 时才启用
自动去重，否则只做 canonical path 校验。调用点传入被测 closure 和 declarative output
spec，例如 `outputs! { value => "..." }` 或 `outputs! { 0 => "...", 1 => "..." }`；
`trace_*` 在内部执行 closure、测量模块 forward、保存输出激活并写
`timing/<module>.time.json`。所有 readback / `into_data()` 都必须发生在 canonical /
duplicate 判断之后。

C++ helper 也使用 `TraceWriter` 维护已保存 tensor key。ggml / libtorch 都能构造 runtime
identity，不需要读回 tensor 内容；但 libtorch 没有统一暴露 Python `_version` 等价物，
已登记 tensor 不应再被 in-place 改写，或由调用点扩展 key 加显式 generation。libtorch
调用点传入被测 lambda、`outputs(out<0>("..."))` 和需要同步的输入 tensor，
`trace_libtorch` 在内部完成同步、计时、执行、输出保存和模块 timing 写入。

llama.cpp 的 `cb_eval` 只能在 scheduler 观察到某个 graph tensor 时回调。用于逐模块耗时
对比时，计时必须夹在命中 observed node 的 `ask=true` 和对应 `ask=false` 之间，也就是
记录 scheduler 计算到该 tensor 并同步完成的 graph segment 耗时；`trace_ggml`
只能接收这个边界已经测出的 module elapsed_ns，再负责输出激活保存和模块 timing
写入；不要在 `activation_ggml` / `trace_bytes` / safetensors 写入 helper 内部计时。旧版
`test_gen/llama_cpp/fp16/case_000000` 若由写文件 helper 计时生成，其 `.time.json` 只代表
导出耗时，必须重新导出。

Python helper 必须以 trace contract 的 canonical path 为准，不能用“谁先保存谁赢”决定输出
文件名。已保存 tensor 再以输入/别名路径传入时可以自动跳过；但一个未保存 tensor 如果传入
非 canonical path，必须报错。这样导出的 `.safetensors` 相对路径集合仍然天然适配
`rwkv-test compare` 的同名文件对齐模型。

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
`timing/**/*.time.json` 集合一致，然后把最后一次 tensor 数据和平均后的模块 timing 写回
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
