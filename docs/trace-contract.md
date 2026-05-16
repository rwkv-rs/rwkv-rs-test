## Trace 数据目录契约

训练 trace 和推理 trace 的导出集合不同。compare 只对已经对齐的同名文件集合做数值和耗时
比较；融合差异必须在插桩边界选择阶段解决，compare 输出里不出现 fused / skip 状态。

## 训练导出树

训练 trace 以 `rwkv_lm` / `rwkv_peft` 的真实训练路径为准。训练真实导出的模块边界和 kernel
边界必须保留；推理 compare 不导出的训练节点在树内标注“推理已融合，跳过”。

```text
test_gen/
└── rwkv_lm/
    └── bf16/
        └── case_000000/
            ├── embedding/
            │   ├── token_ids.safetensors
            │   └── embedded_context.safetensors
            ├── layer_norm0/
            │   └── embedded_context.safetensors  # 推理已融合，跳过
            ├── cells/
            │   ├── cell_0000/
            │   │   ├── time_mixer/
            │   │   │   ├── value_from_first_cell.safetensors  # 推理已融合，跳过
            │   │   │   └── embedded_context.safetensors
            │   │   ├── embedded_context_after_time_mixer.safetensors  # 推理已融合，跳过
            │   │   ├── channel_mixer/
            │   │   │   └── embedded_context.safetensors
            │   │   └── embedded_context_after_channel_mixer.safetensors  # 推理已融合，跳过
            │   ├── cell_0001/
            │   │   └── ...
            │   └── cell_<n_layer_minus_1>/
            │       └── ...
            ├── loss/
            │   ├── l2wrap_cross_entropy.safetensors  # 推理已融合，跳过
            │   └── head_l2wrap_cross_entropy.safetensors  # 仅融合 head+CE 路径；推理已融合，跳过
            └── timing/
                ├── embedding.time.json
                ├── layer_norm0.time.json  # 推理已融合，跳过
                ├── cells/
                │   ├── cell_0000/
                │   │   ├── pre_layer_norm_for_time_mix.time.json  # 推理已融合，跳过
                │   │   ├── time_mixer.time.json
                │   │   ├── embedded_context_after_time_mixer.time.json  # 推理已融合，跳过
                │   │   ├── pre_layer_norm_for_channel_mix.time.json  # 推理已融合，跳过
                │   │   ├── channel_mixer.time.json
                │   │   └── embedded_context_after_channel_mixer.time.json  # 推理已融合，跳过
                │   └── ...
                └── loss/
                    ├── l2wrap_cross_entropy.time.json  # 推理已融合，跳过
                    └── head_l2wrap_cross_entropy.time.json  # 仅融合 head+CE 路径；推理已融合，跳过
```

`lm_head/logits.safetensors` 不要求出现在 `rwkv_lm` 训练 trace；它属于推理 prefill 或显式
logits 对齐 case。

## 推理 Compare 导出树

推理 trace 以 albatross、llama.cpp、web-rwkv、rwkv-mobile 等真实 prefill 路径为准。若一个
后端融合相邻模块 A+B，另一个后端未融合，所有推理后端统一选择 A 前和 B 后作为导出/计时
边界，A/B 中间点不进入这棵树。

```text
test_gen/
└── <infer_repo_name>/
    └── <quantization_name>/
        └── case_000000/
            ├── embedding/
            │   ├── token_ids.safetensors
            │   └── embedded_context.safetensors
            ├── cells/
            │   ├── cell_0000/
            │   │   ├── time_mixer/
            │   │   │   └── embedded_context.safetensors
            │   │   └── channel_mixer/
            │   │       └── embedded_context.safetensors
            │   ├── cell_0001/
            │   │   └── ...
            │   └── cell_<n_layer_minus_1>/
            │       └── ...
            ├── lm_head/
            │   ├── embedded_context.safetensors
            │   └── logits.safetensors
            └── timing/
                ├── embedding.time.json
                ├── cells/
                │   ├── cell_0000/
                │   │   ├── time_mixer.time.json
                │   │   └── channel_mixer.time.json
                │   └── ...
                └── lm_head.time.json
```

## 命名规则

- `rwkv_lm` 只使用 `bf16`：`test_gen/rwkv_lm/bf16/...`
- `albatross` 只使用 `fp16`：`test_gen/albatross/fp16/...`
- 其它量化方案使用 `llama.cpp` 风格 `snake_case` 命名，例如 `q8_0`、`q4_k_m`、`q5_k_m`。
- 每个 `.safetensors` 文件只保存一个同名 tensor，`dtype` 必须保持导出时原样。
- `.time.json` 只允许出现在 `timing/` 目录下，文件名来自模块名，不来自激活值名。
- timing JSON 的 `module` 字段必须等于去掉 `timing/` 前缀和 `.time.json` 后缀后的模块路径。

## 边界规则

- 激活值描述模块边界 tensor；耗时描述模块 forward compute 区间。
- `embedded_context_after_channel_mixer` 是当前 cell 输出，也是下一层 cell 输入。
- pre-LN 的输入已由上一个 canonical 激活表示，不再另存一份输入激活。
- 输入激活没有独立 forward 耗时，不写 `elapsed_ns=0`。
- 一个模块返回多个激活值时，只写一个模块 timing。
- 推理 timing 只允许来自真实同步点、kernel 边界、observed graph node 或原生 timed submission。
- 不为补齐推理 compare 文件集合新增 GPU 同步点、拷贝其它耗时或写 `0` 占位。
- 不导出 kernel 内部临时结果；只导出真实被调用 kernel 的输入/输出边界。
- stateful / mutating kernel 不能在 trace helper 内 repeat 同一个 callable；repeat 必须重建等价
  输入和 state，或使用后端已有 benchmark / graph timing 机制。
