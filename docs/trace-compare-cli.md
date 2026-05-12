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