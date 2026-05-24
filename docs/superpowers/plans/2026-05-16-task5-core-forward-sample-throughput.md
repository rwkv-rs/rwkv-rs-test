# Task5 Core Forward+Sample Throughput Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old Task5 throughput benchmark with a core inference benchmark that strips framework scheduling and measures each backend's real task-specific `forward + sample` path.

**Architecture:** The benchmark contract is task-based, not unified-forward-based. Each backend owns its true entrypoints for `decode`, `prefill`, `batch_decode`, and `batch_prefill`; shared code only covers CSV schema, status rows, common metadata, and frontend ingestion.

**Tech Stack:** Python benchmark runners, Rust `web-rwkv` example runner, CSV/JSON result ingestion, Next.js benchmark UI, DevPod RTX 5090 execution.

---

## Correction From Previous Misunderstanding

The earlier plan was wrong because it still tried to normalize all workloads into one abstract `forward(B,T)` benchmark. That is not the target.

The correct contract is:

- `decode` is a single-stream stateful one-token step: `B=1,T=1`. It must use the backend's true decode entrypoint and then sample one token.
- `prefill` is a single-stream sequence scan: `B=1,T=n`. It must use the backend's true prefill/sequence entrypoint and then sample from the final logits.
- `batch_decode` is many independent states advanced by one token: `B=n,T=1`. It must use a true batch decode entrypoint; repeating single decode `n` times is not acceptable.
- `batch_prefill` is many independent sequences prefilled together: `B=n,T=n`. It must use a true batch prefill entrypoint; repeating single prefill `n` times is not acceptable.

Sampling is part of the measured workload. Tokenizer decode, text printing, HTTP APIs, OpenAI server paths, request queues, scheduler loops, continuous batching frameworks, and state-cache scheduling policies are not part of this benchmark.

## Correction From Failed Completion Attempt

The benchmark is not complete when most rows are merely schema-valid `unsupported` rows. `unsupported` is only valid for task families that a backend truly cannot express without changing semantics. It is not a placeholder for "runner not wired yet", "model path was wrong", "only smoke-tested one backend", or "the direct path has not been investigated".

Hard requirements:

- `B1T1` / `decode` must produce `status=ok` for every backend.
- `B1Tn` / `prefill` must produce `status=ok` for every backend across the canonical `T` grid, unless a specific row genuinely runs and fails; a missing implementation is not `unsupported`.
- `BnT1` / `batch_decode` is partial. A backend may emit `unsupported` only when it lacks a true batch decode entrypoint and the only alternative is looping smaller decode calls or going through a scheduler/request queue.
- `BnTn` / `batch_prefill` is the fast-path target. It is expected to be supported at least by Albatross `faster3_2605` / `faster3a_2605`; other backends may emit `unsupported` only after their direct batch-prefill capability has been inspected.
- Albatross 7B is the sanity gate for the implementation. The `7.2B` `faster3`/`faster3a` run must include a real forward+sample row above `17,000` TPS on an expected fast path. If the 7B run is far below that, the timing boundary, batching path, CUDA extension configuration, or model loading is wrong and the benchmark must not be accepted.
- All supported model sizes must be measured, not just one convenient checkpoint.

## Benchmark Case Matrix

Use two grids: a required canonical grid for all result comparisons, and optional stress cases for backend-specific frontier exploration. Unsupported combinations must be written as `status=unsupported` with a concrete reason, not silently skipped. `decode` and `prefill` are not optional backend features for this benchmark.

### Required Canonical Grid

| Contract label | task | B values | T values | Support expectation | Notes |
| --- | --- | --- | --- | --- | --- |
| `B1T1` | `decode` | `1` | `1` | all backends must be `ok` | True one-token decode plus sampler. Do not use `unsupported` as a "not wired" placeholder. |
| `B1Tn` | `prefill` | `1` | `16,64,256,1024,4096` | all backends must be `ok` or real `failed` rows | True single-sequence prefill plus sampler from final logits. If a row OOMs or errors, record `failed`; do not call prefill unsupported. |
| `BnT1` | `batch_decode` | `2,4,8,16,32,64,128,256,512,960,1024` | `1` | partial | True batch decode only; `unsupported` is valid only if the backend lacks a direct batch decode path. |
| `BnTn` | `batch_prefill` | `2,4,8,16,32` | diagonal pairs `2x2,4x4,8x8,16x16,32x32` | Albatross faster3/faster3a expected; others inspected individually | This is the expected fastest task family. True batch prefill only; unsupported if backend only loops prefill. |

### Required Model Coverage

Every backend must be attempted for every model size it can load in its native format. Missing model files are blockers to resolve, not a reason to silently shrink the matrix.

Required parameter sizes:

| size | canonical RWKV checkpoint | Python `.pth` consumers | `web-rwkv` `.st` consumers | `rwkv-mobile` GGUF consumers |
| --- | --- | --- | --- | --- |
| `0.1B` | `rwkv7-g1d-0.1b-20260129-ctx8192` | required | required if `.st` exists or can be produced | required for available quantizations |
| `0.4B` | `rwkv7-g1d-0.4b-20260210-ctx8192` | required | required if `.st` exists or can be produced | required for available quantizations |
| `1.5B` | `rwkv7-g1f-1.5b-20260419-ctx8192` | required; recover a real nonzero `.pth` if the Pod only has a zero-byte placeholder | required | required for available quantizations |
| `2.9B` | `rwkv7-g1f-2.9b-20260420-ctx8192` | required | required if `.st` exists or can be produced | required for available quantizations |
| `7.2B` | `rwkv7-g1f-7.2b-20260414-ctx8192` | required; Albatross must pass the `>17,000 TPS` sanity gate | required if `.st` exists or can be produced | required for available quantizations |
| `13.3B` | `rwkv7-g1f-13.3b-20260415-ctx8192` | required where memory allows; otherwise real `failed` rows with error | required if `.st` exists or can be produced | required for available quantizations |

Result layout must include the model size in the file or directory name, for example:

```text
results/core-forward-sample/<pod>/<backend>/<model-size>/task5_core_forward_sample.csv
```

Do not overwrite one model size's result with another model size's result.

### Optional Stress Grid

Use this only after the canonical grid succeeds:

- `prefill`: `B=1`, `T=512,2048,8192` if model context and memory allow.
- `batch_decode`: `B=320,768,1536,2048`, `T=1` if backend supports it.
- `batch_prefill`: `B,T` in `16x32,32x16,32x64,64x16,64x32`, plus `32x32` as the expected fast-path headline case.

### Albatross Existing Case Mapping

Albatross already uses a shape list close to:

```text
1x1,
1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,
2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,
2x2,4x4,8x8,16x16
```

Do not treat this as a universal `forward(B,T)` abstraction. Map these cases into the four task categories above, and add the missing canonical cases such as `1x1024`, `1x4096`, `512x1`, `960x1`, `1024x1`, and `32x32` where the backend supports them.

For Albatross, run at least `faster3_2605` or `faster3a_2605` across all required model sizes. The 7.2B model is the correctness/performance sentinel: an all-success but slow 7.2B result below the `17,000` TPS sanity gate is not acceptable until the implementation has been debugged.

## Result Schema

Add a new benchmark kind:

```text
core_forward_sample_throughput
```

Each CSV row represents exactly one `(backend, model, task, B, T)` result.

Required fields:

```text
run_id,repo,backend,runner,benchmark_kind,task,
model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,
B,T,warmup,repeat,seed,status,error,
input_tokens,measured_tokens,total_time_s,forward_time_s,sample_time_s,
p10_ms,p50_ms,p90_ms,forward_sample_tps,
entrypoint,measurement_boundary,
command,binary_path,binary_build_id,model_bytes,model_sha256,started_at,ended_at
```

Metric rules:

- `total_time_s` includes forward and sample.
- `forward_sample_tps = measured_tokens / total_time_s`.
- `measured_tokens = B*T` for prefill-like tasks and `B` for decode-like tasks.
- `forward_time_s` and `sample_time_s` may be empty when a backend cannot split them cleanly, but `total_time_s`, `p50_ms`, and `forward_sample_tps` must be present for `status=ok`.
- `entrypoint` must name the real backend method or binary path used, for example `forward_one`, `forward_seq`, `forward_seq_batch_1`, `forward_batch`, `forward_from_x+sampler_simple_batch`, or `web-rwkv Rnn::run`.
- `measurement_boundary` must explicitly say what is included, for example `forward+sampler; no tokenizer decode; no scheduler; no server`.

## File Scope

### Documentation

- Modify `docs/benchmark.md`.
- Add or update any Task5 result notes if the repo has a dedicated results README.

### Shared Benchmark Utilities

- Create `scripts/task5_core_schema.py`.
- Create `tests/test_task5_core_schema.py`.

### Backend Runners

- Create `infer-repo/albatross/task5_core_forward_sample.py`.
- Create `infer-repo/nano-vllm/task5_core_forward_sample.py`.
- Create `infer-repo/rwkv-lightning/task5_core_forward_sample.py`.
- Create `infer-repo/web-rwkv/examples/task5_core_forward_sample.rs`.
- Create or replace `infer-repo/rwkv-mobile/task5_core_forward_sample.py`.

### Frontend

- Modify `benchmark-ui/src/lib/types.ts`.
- Modify `benchmark-ui/src/lib/ingest.ts`.
- Modify `benchmark-ui/src/lib/analytics.ts`.
- Modify `benchmark-ui/src/components/benchmark-explorer.tsx`.
- Modify `benchmark-ui/src/components/throughput-chart.tsx`.
- Modify `benchmark-ui/src/lib/ingest.test.ts`.
- Modify `benchmark-ui/src/lib/analytics.test.ts`.
- Modify `benchmark-ui/src/components/throughput-chart.test.ts`.

## Implementation Tasks

### Task 1: Freeze The New Benchmark Contract

**Files:**
- Modify: `docs/benchmark.md`
- Create: `scripts/task5_core_schema.py`
- Create: `tests/test_task5_core_schema.py`

- [ ] Add the correction section to `docs/benchmark.md`: explicitly state that `decode`, `prefill`, `batch_decode`, and `batch_prefill` cannot be collapsed into one `forward(B,T)` abstraction.
- [ ] Document the canonical and optional stress B/T grids from this plan.
- [ ] Document the new CSV fields and metric rules.
- [ ] Implement `scripts/task5_core_schema.py` with:
  - `TASKS = ("decode", "prefill", "batch_decode", "batch_prefill")`
  - `CSV_FIELDS`
  - `task_shape(task, B, T)` validation
  - `unsupported_row(...)`
  - `ok_row(...)`
  - `write_csv(path, rows)`
- [ ] Add tests that prove:
  - each task enforces the right B/T shape class,
  - unsupported rows still contain `task`, `B`, `T`, and `entrypoint`,
  - no old `prefill_tps`, `decode_tps`, or `e2e_tps` field is required.
- [ ] Run:

```bash
rtk python -m pytest tests/test_task5_core_schema.py -q
```

### Task 2: Remove Old Task5 Semantics From New Paths

**Files:**
- Modify only if needed: `infer-repo/nano-vllm/task5_collect_nano_vllm.py`
- Modify only if needed: `infer-repo/web-rwkv/examples/task5_benchmark.rs`
- Create new runners instead of overloading old ones where possible.

- [ ] Keep old files runnable for historical CSVs unless they block tests.
- [ ] Do not extend `synthetic_throughput` for this benchmark.
- [ ] Ensure all new code writes `benchmark_kind=core_forward_sample_throughput`.
- [ ] Ensure no new runner imports or launches:
  - OpenAI API benchmark clients,
  - HTTP server,
  - scheduler request queue,
  - framework-level continuous batching driver.

### Task 3: Add Albatross Runner

**Files:**
- Create: `infer-repo/albatross/task5_core_forward_sample.py`

- [ ] Implement CLI:

```text
--version faster3a_2605
--model /path/to/model.pth
--tasks decode,prefill,batch_decode,batch_prefill
--prefill-t 16,64,256,1024,4096
--batch-decode-b 2,4,8,16,32,64,128,256,512,960,1024
--batch-prefill-pairs 2x2,4x4,8x8,16x16,32x32
--warmup 3
--repeat 10
--out infer-repo/albatross/results/task5_core_forward_sample.csv
```

- [ ] For `_ref_slower_`, `faster_251101`, and `faster2_251201`, use their real methods such as `forward_one`, `forward_seq`, `forward_batch`, `forward_seq_batch_1`, or `forward_seq_batch`.
- [ ] For `faster3_2605` and `faster3a_2605`, use their real shape-selected path, but keep the result task-labeled. Include sampler timing after logits are produced.
- [ ] Produce `ok` rows for `decode` and `prefill` for every required model size. If a model file is missing or zero bytes, recover the real checkpoint before claiming completion.
- [ ] For `batch_prefill`, return `unsupported` only for Albatross versions that cannot run true batch prefill without looping single prefill.
- [ ] Reuse albatross sampler functions such as `sampler_simple` and `sampler_simple_batch`; do not replace sampling with argmax unless that is the backend's real configured sampler.
- [ ] Validate the `7.2B` `faster3`/`faster3a` result against the `>17,000` TPS sanity gate. If it fails, inspect CUDA extension arch (`TORCH_CUDA_ARCH_LIST=12.0`), CUDAGraph reuse, prefill chunking, model dtype, and whether the timed path accidentally includes setup or scheduler work.

### Task 4: Add nano-vllm Runner

**Files:**
- Create: `infer-repo/nano-vllm/task5_core_forward_sample.py`

- [ ] Load the model directly.
- [ ] Do not use `benchmark_openai_api.py`.
- [ ] Do not start the server.
- [ ] Do not use scheduler/request queue code to synthesize batch modes.
- [ ] Implement direct `B1T1` decode through `model_runner` without the OpenAI API or HTTP server.
- [ ] Implement direct `B1Tn` prefill through `model_runner` for `T=16,64,256,1024,4096`.
- [ ] Mark `batch_decode` or `batch_prefill` as `unsupported` only when the only available path is a scheduler loop or repeated single-item calls.
- [ ] It is invalid for nano-vllm to emit all rows as `unsupported`; at minimum `decode` and `prefill` must be real `ok` or real `failed` rows after attempting the direct path.

### Task 5: Add rwkv-lightning Runner

**Files:**
- Create: `infer-repo/rwkv-lightning/task5_core_forward_sample.py`

- [ ] Load model directly from existing inference modules.
- [ ] Use true single-stream `forward` for `B1T1` decode and `B1Tn` prefill.
- [ ] Use true `forward_batch` paths for `BnT1` and any supported `BnTn` case.
- [ ] Preserve the existing sampler used by its direct generation code.
- [ ] Do not use `webui_rwkv.py`, API routes, queues, task pools, or server streaming code.
- [ ] It is invalid for rwkv-lightning to emit all rows as `unsupported`; at minimum `decode` and `prefill` must be real `ok` or real `failed` rows after attempting the direct path.

### Task 6: Add web-rwkv Runner

**Files:**
- Create: `infer-repo/web-rwkv/examples/task5_core_forward_sample.rs`

- [ ] Reuse model loading and runtime setup from `examples/task5_benchmark.rs`.
- [ ] Replace old prefill/decode/e2e metrics with task rows.
- [ ] Include sampler logic used by the example generation path.
- [ ] Produce real `ok` rows for `B1T1` decode and `B1Tn` prefill using the direct runtime path.
- [ ] Mark unsupported tasks honestly only where `web-rwkv` lacks a true batch mode.
- [ ] It is invalid for web-rwkv to emit all rows as `unsupported`; if Linux/WSL lacks a compatible WebGPU adapter, run the required rows in the correct Windows/Dx12 or DevPod GPU environment instead of accepting unsupported smoke output.
- [ ] Validate with:

```bash
rtk cargo check --manifest-path infer-repo/web-rwkv/Cargo.toml --example task5_core_forward_sample
```

### Task 7: Add rwkv-mobile Runner

**Files:**
- Create: `infer-repo/rwkv-mobile/task5_core_forward_sample.py`

- [ ] Produce real `ok` rows for `B1T1` decode and `B1Tn` prefill using the direct rwkv-mobile core path.
- [ ] Mark `BnT1` and `BnTn` unsupported only if the backend lacks a true batch path; do not mark `decode` or `prefill` unsupported because the runner has not been wired.
- [ ] Use the new schema and preserve GPU preflight.
- [ ] Do not emit old `synthetic_throughput` rows.

### Task 8: Update Frontend Ingestion And UI

**Files:**
- Modify: `benchmark-ui/src/lib/types.ts`
- Modify: `benchmark-ui/src/lib/ingest.ts`
- Modify: `benchmark-ui/src/lib/analytics.ts`
- Modify: `benchmark-ui/src/components/benchmark-explorer.tsx`
- Modify: `benchmark-ui/src/components/throughput-chart.tsx`

- [ ] Replace `decodeTps/prefillTps/e2eTps` primary model with `forwardSampleTps` and `p50Ms`.
- [ ] Add `task`, `B`, `T`, `entrypoint`, and `measurementBoundary` to the row type.
- [ ] Ingest only `benchmark_kind=core_forward_sample_throughput` for the new view.
- [ ] Add task segmented control:
  - Decode
  - Prefill
  - Batch Decode
  - Batch Prefill
- [ ] Use task-aware x axis:
  - `decode`: single point or `B/T` label.
  - `prefill`: x = `T`.
  - `batch_decode`: x = `B`.
  - `batch_prefill`: x = `B*T`, while tooltip and table show both `B` and `T`.
- [ ] Table columns:
  - Model
  - Backend
  - Task
  - B
  - T
  - Quant
  - Status
  - Forward+Sample TPS
  - p50 ms
  - Entrypoint
  - Error

### Task 9: Update Frontend Tests

**Files:**
- Modify: `benchmark-ui/src/lib/ingest.test.ts`
- Modify: `benchmark-ui/src/lib/analytics.test.ts`
- Modify: `benchmark-ui/src/components/throughput-chart.test.ts`

- [ ] Add sample CSV rows for all four tasks.
- [ ] Test unsupported task rows.
- [ ] Test task filtering.
- [ ] Test task-aware x-axis mapping.
- [ ] Run:

```bash
cd benchmark-ui
rtk bun test
rtk bun run build
```

### Task 10: Run On DevPod And Bring Results Back

**Files/Outputs:**
- Create results on DevPod under model-qualified backend result directories.
- Copy results back to local `results/core-forward-sample/<pod>/<backend>/<model-size>/`.
- Regenerate `benchmark-ui/public/data/task5.json`.

- [ ] Sync code to DevPod using scoped repo sync only:

```bash
rsync -av --delete \
  --exclude target \
  --exclude results \
  --exclude weights \
  /mnt/g/Projects/Packages/rwkv-rs-test/ \
  <devpod2>:/workspace/Projects/Packages/rwkv-rs-test/
```

- [ ] On DevPod, run:

```bash
nvidia-smi
```

Expected: RTX 5090 is visible. Record GPU UUID in manifests.

- [ ] Inventory required cloud model files before running. Any missing or zero-byte required checkpoint is a blocker to fix before completion:

```bash
cd /workspace/Projects/Packages/rwkv-rs-test
find /workspace/Weights/RWKV -maxdepth 1 -type f \
  \( -iname "*.pth" -o -iname "*.st" -o -iname "*.gguf" \) \
  -printf "%s %p\n" | sort -nr
```

- [ ] Run Python `.pth` backends for every required size. Do not substitute a different size when one checkpoint is missing:

```bash
cd /workspace/Projects/Packages/rwkv-rs-test
export TORCH_CUDA_ARCH_LIST=12.0

declare -A PTH_MODELS=(
  [0.1B]="/workspace/Weights/RWKV/rwkv7-g1d-0.1b-20260129-ctx8192.pth"
  [0.4B]="/workspace/Weights/RWKV/rwkv7-g1d-0.4b-20260210-ctx8192.pth"
  [1.5B]="/workspace/Weights/RWKV/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
  [2.9B]="/workspace/Weights/RWKV/rwkv7-g1f-2.9b-20260420-ctx8192.pth"
  [7.2B]="/workspace/Weights/RWKV/rwkv7-g1f-7.2b-20260414-ctx8192.pth"
  [13.3B]="/workspace/Weights/RWKV/rwkv7-g1f-13.3b-20260415-ctx8192.pth"
)

for size in 0.1B 0.4B 1.5B 2.9B 7.2B 13.3B; do
  model="${PTH_MODELS[$size]}"
  test -s "$model"

  python infer-repo/albatross/task5_core_forward_sample.py \
    --version faster3a_2605 \
    --model "$model" \
    --tasks decode,prefill,batch_decode,batch_prefill \
    --out "infer-repo/albatross/results/$size/task5_core_forward_sample.csv"

  python infer-repo/nano-vllm/task5_core_forward_sample.py \
    --model-pth "$model" \
    --tasks decode,prefill,batch_decode,batch_prefill \
    --out "infer-repo/nano-vllm/results/$size/task5_core_forward_sample.csv"

  python infer-repo/rwkv-lightning/task5_core_forward_sample.py \
    --model "$model" \
    --tasks decode,prefill,batch_decode,batch_prefill \
    --out "infer-repo/rwkv-lightning/results/$size/task5_core_forward_sample.csv"
done
```

- [ ] Validate the Albatross 7.2B sanity gate before moving on:

```bash
python - <<'PY'
import csv
path = "infer-repo/albatross/results/7.2B/task5_core_forward_sample.csv"
rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
ok = [float(r["forward_sample_tps"]) for r in rows if r["status"] == "ok" and r["forward_sample_tps"]]
best = max(ok) if ok else 0.0
print(f"albatross_7.2B_best_tps={best:.2f}")
raise SystemExit(0 if best >= 17000 else 1)
PY
```

- [ ] Run `web-rwkv` for every required `.st` size. If an `.st` file is missing but the matching `.pth` exists, produce it through the existing conversion workflow before accepting the run:

```bash
declare -A ST_MODELS=(
  [0.1B]="/workspace/Weights/RWKV/rwkv7-g1d-0.1b-20260129-ctx8192.st"
  [0.4B]="/workspace/Weights/RWKV/rwkv7-g1d-0.4b-20260210-ctx8192.st"
  [1.5B]="/workspace/Weights/RWKV/rwkv7-g1f-1.5b-20260419-ctx8192.st"
  [2.9B]="/workspace/Weights/RWKV/rwkv7-g1f-2.9b-20260420-ctx8192.st"
  [7.2B]="/workspace/Weights/RWKV/rwkv7-g1f-7.2b-20260414-ctx8192.st"
  [13.3B]="/workspace/Weights/RWKV/rwkv7-g1f-13.3b-20260415-ctx8192.st"
)

for size in 0.1B 0.4B 1.5B 2.9B 7.2B 13.3B; do
  model="${ST_MODELS[$size]}"
  test -s "$model"
  cargo run --release \
    --manifest-path infer-repo/web-rwkv/Cargo.toml \
    --example task5_core_forward_sample \
    -- \
    --model "$model" \
    --output "infer-repo/web-rwkv/results/$size/task5_core_forward_sample.csv"
done
```

- [ ] Run `rwkv-mobile` for every required GGUF size and quantization available. At minimum run the canonical `Q4_K_M` matrix for each size:

```bash
for model in /workspace/Weights/RWKV/rwkv7-*-Q4_K_M.gguf; do
  test -s "$model"
  size="$(basename "$model" | sed -E 's/.*-([0-9]+(\.[0-9]+)?b).*/\1/I' | tr '[:lower:]' '[:upper:]')"
  python infer-repo/rwkv-mobile/task5_core_forward_sample.py \
    --model "$model" \
    --tasks decode,prefill,batch_decode,batch_prefill \
    --output "infer-repo/rwkv-mobile/results/$size-Q4_K_M/task5_core_forward_sample.csv"
done
```

- [ ] Copy results back:

```bash
POD="<devpod-pod-name>"
mkdir -p "/mnt/g/Projects/Packages/rwkv-rs-test/results/core-forward-sample/$POD"
rsync -av --relative \
  <devpod2>:/workspace/Projects/Packages/rwkv-rs-test/./infer-repo/*/results/ \
  "/mnt/g/Projects/Packages/rwkv-rs-test/results/core-forward-sample/$POD/"
```

- [ ] Regenerate frontend data:

```bash
cd /mnt/g/Projects/Packages/rwkv-rs-test/benchmark-ui
rtk bun run ingest -- --roots=../results/core-forward-sample
```

- [ ] Start frontend after implementation:

```bash
cd /mnt/g/Projects/Packages/rwkv-rs-test/benchmark-ui
rtk bun run dev
```

## Acceptance Criteria

- The docs explicitly state why the old unified-forward interpretation was wrong.
- The canonical B/T grid is documented and used by every backend.
- Every required model size is attempted: `0.1B`, `0.4B`, `1.5B`, `2.9B`, `7.2B`, and `13.3B`.
- Missing or zero-byte model files are resolved before completion; they are not replaced by a different parameter size.
- Every backend emits `ok` rows for `B1T1` decode.
- Every backend emits `ok` rows for `B1Tn` prefill across the canonical `T` grid, or real `failed` rows if execution was attempted and failed. `unsupported` is not valid for decode or prefill.
- `BnT1` `batch_decode` may be `unsupported` only after verifying that the backend lacks a true batch decode path.
- `BnTn` `batch_prefill` may be `unsupported` outside Albatross only after verifying that the backend lacks a true batch prefill path.
- Albatross `7.2B` produces at least one real forward+sample result above `17,000` TPS on the expected fast path.
- Every backend emits one row per required `(model_size, task, B, T)` with `ok`, `failed`, or justified `unsupported`.
- No new core benchmark path uses HTTP API, server routes, request schedulers, or framework batching.
- Sampling is included in the measured primary metric.
- `batch_decode` and `batch_prefill` are never emulated by loops over smaller tasks.
- Frontend can filter and display all four tasks.
- DevPod results are copied back with backend and model-size directory boundaries intact and ingested into the UI dataset.
