#!/bin/bash
set -euo pipefail

TRACE_ROOT="${RWKV_TRACE_ROOT:-/mnt/g/Projects/Packages/rwkv-rs-test/test_gen}"
LOAD_MODEL="${RWKV_TRACE_LOAD_MODEL:-out_trace/L12-D768-x070/rwkv-init.pth}"
CACHE_ROOT="${RWKV_TRACE_CACHE_ROOT:-$PWD/.trace_cache}"

RWKV_TRACE_ROOT="$TRACE_ROOT" RWKV_TRACE_ONCE=1 \
RWKV_TRACE_REPEAT="${RWKV_TRACE_REPEAT:-3}" \
RWKV_TRACE_WARMUP="${RWKV_TRACE_WARMUP:-1}" \
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}" \
MPLCONFIGDIR="${MPLCONFIGDIR:-$CACHE_ROOT/matplotlib}" \
uv run python train.py \
  --load_model "$LOAD_MODEL" --proj_dir out_trace/L12-D768-x070 \
  --data_file ../../data/minipile --vocab_size 65536 \
  --n_layer 12 --n_embd 768 --ctx_len 512 --micro_bsz 16 \
  --epoch_steps 2520 --epoch_count 1 --magic_prime 2926181 --ds_bucket_mb 200 \
  --peft none --data_type binidx --my_testing x070 --head_size_a 64 \
  --precision bf16 --accelerator gpu --devices 1 --strategy deepspeed_stage_2 \
  --op cuda --num_workers 0
