#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPEAT="${TRACE_REPEAT:-3}"
WARMUP="${TRACE_WARMUP:-1}"

if [[ "$#" -eq 0 ]]; then
  BACKENDS=(rwkv_lm rwkv_peft albatross llama_cpp web_rwkv)
else
  BACKENDS=("$@")
fi

run_case() {
  local backend="$1"
  local case_root="$2"
  local workdir="$3"
  shift 3

  echo "==> ${backend}: repeat=${REPEAT} warmup=${WARMUP}"
  (
    cd "$workdir"
    python3 "$ROOT/scripts/trace_average.py" \
      --case-root "$case_root" \
      --repeat "$REPEAT" \
      --warmup "$WARMUP" \
      -- "$@"
  )
}

for backend in "${BACKENDS[@]}"; do
  case "$backend" in
    rwkv_lm)
      run_case \
        "$backend" \
        "$ROOT/test_gen/rwkv_lm/bf16/case_000000" \
        "$ROOT/train-repo/rwkv-lm" \
        bash trace-train.sh
      ;;
    rwkv_peft)
      run_case \
        "$backend" \
        "$ROOT/test_gen/rwkv_peft/bf16/case_000000" \
        "$ROOT/train-repo/rwkv-peft" \
        bash trace-run.sh
      ;;
    albatross)
      run_case \
        "$backend" \
        "$ROOT/test_gen/albatross/fp16/case_000000" \
        "$ROOT/infer-repo/albatross" \
        env TORCH_CUDA_ARCH_LIST="${ALBATROSS_TORCH_CUDA_ARCH_LIST:-12.0}" \
        uv run python trace_infer.py --model ../../weights/rwkv7-g1f-1.5b-20260419-ctx8192.pth
      ;;
    llama_cpp)
      run_case \
        "$backend" \
        "$ROOT/test_gen/llama_cpp/fp16/case_000000" \
        "$ROOT/infer-repo/llama.cpp" \
        ./build-gpu/bin/llama-trace-rwkv --model ../../weights/rwkv7-g1f-1.5b-20260419-ctx8192-FP16.gguf -ngl 999
      ;;
    web_rwkv)
      run_case \
        "$backend" \
        "$ROOT/test_gen/web_rwkv/fp16/case_000000" \
        "$ROOT/infer-repo/web-rwkv" \
        cargo run --no-default-features --features tokio --example trace_infer -- --model ../../weights/rwkv7-g1f-1.5b-20260419-ctx8192.st
      ;;
    *)
      echo "unknown backend: $backend" >&2
      exit 2
      ;;
  esac
done
