#!/bin/bash
set -euo pipefail

TRACE_ROOT="${RWKV_TRACE_ROOT:-/mnt/g/Projects/Packages/rwkv-rs-test/test_gen}"
LOAD_MODEL="${RWKV_TRACE_LOAD_MODEL:-out_trace/L12-D768-x070/rwkv-init.pth}"

RWKV_TRACE_ROOT="$TRACE_ROOT" RWKV_TRACE_ONCE=1 \
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions/rwkv-peft}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-rwkv-peft}" \
uv run python scripts/trace_run_pretrain.py \
  --load_model "$LOAD_MODEL" --proj_dir out_trace/L12-D768-x070 \
  --data_file ../../data/minipile --vocab_size 65536 \
  --n_layer 12 --n_embd 768 --ctx_len 512 --micro_bsz 16 \
  --epoch_steps 2520 --magic_prime 2926181 --ds_bucket_mb 200
