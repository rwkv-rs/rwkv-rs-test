#!/bin/bash
set -euo pipefail

TRACE_ROOT="${RWKV_TRACE_ROOT:-/mnt/g/Projects/Packages/rwkv-rs-test/test_gen}"
STABLE_ROOT="${RWKV_RS_STABLE_ROOT:-/mnt/g/Projects/Packages/rwkv-rs-stable}"
SOURCE_MODEL="${RWKV_TRACE_LOAD_MODEL:-$STABLE_ROOT/weights/rwkv-init-0.1b-ctx512-test.pth}"
PROJ_DIR="${RWKV_TRACE_PROJ_DIR:-/tmp/rwkv-rs-stable-trace-L12-D768-x070}"

rm -rf "$PROJ_DIR"
mkdir -p "$PROJ_DIR"
cp "$SOURCE_MODEL" "$PROJ_DIR/rwkv-init.pth"

RWKV_TRACE_ROOT="$TRACE_ROOT" RWKV_TRACE_ONCE=1 \
uv run python train.py \
  --load_model "$PROJ_DIR/rwkv-init.pth" --wandb "" --proj_dir "$PROJ_DIR" \
  --my_testing x070 --ctx_len 512 --train_stage 3 --epoch_begin 0 \
  --data_file ../../data/minipile --data_type binidx --vocab_size 65536 \
  --my_exit_tokens 1498226207 --magic_prime 2926181 \
  --num_nodes 1 --micro_bsz 16 --n_layer 12 --n_embd 768 \
  --lr_init 6e-4 --lr_final 6e-5 --warmup_steps 10 \
  --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
  --weight_decay 0.001 --epoch_save 10 --head_size 64 \
  --accelerator gpu --devices 1 --precision bf16 \
  --strategy deepspeed_stage_2 --ds_bucket_mb 200 --grad_cp 0 \
  --enable_progress_bar True
