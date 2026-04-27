import argparse
import os
import sys
from types import SimpleNamespace

import torch


def main() -> None:
    sys.path.insert(0, os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="out_trace/L12-D768-x070/rwkv-init.pth")
    parser.add_argument("--vocab_size", default=65536, type=int)
    parser.add_argument("--n_layer", default=12, type=int)
    parser.add_argument("--n_embd", default=768, type=int)
    parser.add_argument("--ctx_len", default=512, type=int)
    parser.add_argument("--head_size_a", default=64, type=int)
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()

    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_TRAIN_TYPE"] = "none"
    os.environ["WKV"] = "cuda"
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    os.environ["RWKV_JIT_ON"] = "0"

    model_args = SimpleNamespace(
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        ctx_len=args.ctx_len,
        dim_att=args.n_embd,
        dim_ffn=int((args.n_embd * 3.5) // 32 * 32),
        head_size_a=args.head_size_a,
        head_size_divisor=8,
        grad_cp=0,
        train_type="none",
        peft="none",
        my_testing="x070",
    )

    from rwkvt.rwkv7.model import RWKV7

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    model = RWKV7(model_args)
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    if args.precision == "bf16":
        state = {name: tensor.bfloat16() if tensor.is_floating_point() else tensor for name, tensor in state.items()}
    elif args.precision == "fp16":
        state = {name: tensor.half() if tensor.is_floating_point() else tensor for name, tensor in state.items()}
    torch.save(state, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
