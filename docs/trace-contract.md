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
            ├── lm_head/
            │   ├── embedded_context.safetensors
            │   └── logits.safetensors
            └── timing/
                ├── embedding.time.json
                ├── layer_norm0.time.json
                ├── cells/
                │   ├── cell_0000/
                │   │   ├── pre_layer_norm_for_time_mix.time.json
                │   │   ├── time_mixer.time.json
                │   │   ├── embedded_context_after_time_mixer.time.json
                │   │   ├── pre_layer_norm_for_channel_mix.time.json
                │   │   ├── channel_mixer.time.json
                │   │   └── embedded_context_after_channel_mixer.time.json
                │   └── ...
                └── lm_head.time.json
```

## 命名规则

- `rwkv_lm` 只使用 `bf16`：`test_gen/rwkv_lm/bf16/...`
- `albatross` 只使用 `fp16`：`test_gen/albatross/fp16/...`
- 其它量化方案使用 `llama.cpp` 风格 `snake_case` 命名，例如 `q8_0`、`q4_k_m`、`q5_k_m`。
- 每个 `.safetensors` 文件只保存一个同名 tensor，`dtype` 必须保持导出时原样。
- `.time.json` 只允许出现在 `timing/` 目录下，文件名来自模块名，不来自激活值名。
- 禁止生成激活值同名 timing，例如 `embedding/token_ids.time.json` 或
  `cells/cell_0000/time_mixer/embedded_context.time.json`。
- timing JSON 使用 `module` 字段标识模块，例如 `cells/cell_0000/time_mixer`，不使用
  `.safetensors` 的 `filename` 字段。

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

## 耗时约定

- 耗时描述模块 forward，不描述某个激活值。
- 输入激活没有独立 forward 耗时，只导出 `.safetensors`，不写 `elapsed_ns=0`。
- 一个模块返回多个激活值时，只写一个模块 timing；不要把同一个耗时复制到多个输出 tensor。
- 别名/透传 tensor 和辅助输出不靠 `0` 耗时占位；需要性能比较时，必须定义清楚对应模块边界。


## 边界原则
- 训练 baseline 是 rwkv-lm，推理 baseline 是 albatross
- 导出集合由 baseline 的真实执行路径决定，不强行制造某个 backend 没有自然暴露的中间结果
- 不导出某个 kernel 的内部中间运行结果；只导出真实被调用 kernel 的输入/输出边界
- 一个被使用的 kernel 返回多个 tensor，这些 tensor 都属于契约，必须导出并参与同名对比
