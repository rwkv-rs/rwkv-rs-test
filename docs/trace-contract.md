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