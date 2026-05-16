# rwkv-rs-test

本仓库为 rwkv 的推理框架和训练框架提供完整的单元测试套件和基准测试套件。

## Trace Contract

训练 trace 和推理 trace 的导出集合不同：训练 baseline 以 `rwkv_lm` 的真实训练路径为准，
推理 baseline 以推理入口的真实 prefill 路径为准。
`lm_head/logits` 只属于推理 prefill 或显式 logits 对齐 case；`rwkv_lm` 训练 trace 不要求导出 `lm_head/logits`。

训练 trace 的 loss 边界包含 `loss/l2wrap_cross_entropy.safetensors`，以及启用 fused
head+CE 路径时的 `loss/head_l2wrap_cross_entropy.safetensors`。如果一个真实 kernel
返回辅助 tensor，这些辅助 tensor 也属于同一模块的激活契约，但耗时只写一个模块 timing。

耗时统一写到 `timing/<module>.time.json`。激活值旁边不能再生成同名 `.time.json`；
输入 metadata 例如 `embedding/token_ids.safetensors` 只保存激活，不写 `elapsed_ns=0`。
`elapsed_ns` 必须写平均耗时：

```text
elapsed_ns = round(sum(samples_ns) / repeat)
```

`repeat`：参与平均的有效真实 run 样本数。平均导出使用真实入口：

```bash
RWKV_TRACE_WARMUP=1 RWKV_TRACE_REPEAT=3 bash trace-train.sh
```

trace 必须从真实训练或推理入口导出；禁止创建重建 Trainer、DataLoader、ModelRunner
或 scheduler 的专用 trace 程序入口。
