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
