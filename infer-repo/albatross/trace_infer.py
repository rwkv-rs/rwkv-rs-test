import argparse
import random
import types

import numpy as np
import torch

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER


PROMPT = r"User: You are a very talented expert in aime24. Solve the problem and output the final answer in \boxed{}. Problem: Let AB​CD be a tetrahedron such that AB = CD = \sqrt{41}, AC = BD = \sqrt{80}, and BC = AD = \sqrt{89}. There exists a point I inside the tetrahedron such that the distances from I to each of the faces of the tetrahedron are all equal. This distance can be written in the form \frac{m\sqrt{n}}{p}, where m, n, and p are positive integers, m and p are relatively prime, and n is not divisible by the square of any prime. Find m + n + p. Assistant: <think"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model path without the .pth suffix")
    args_cli = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = args_cli.model.removesuffix(".pth")

    tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")
    tokens = tokenizer.encode(PROMPT)
    print(f"albatross trace prefill tokens={len(tokens)}")

    model = RWKV_x070(args)

    warmup_state = model.generate_zero_state(0)
    _ = model.forward_seq(torch.tensor(tokens), warmup_state)
    torch.cuda.synchronize()

    state = model.generate_zero_state(0)
    logits = model.forward(tokens, state)
    torch.cuda.synchronize()
    print(f"albatross trace prefill complete logits_shape={tuple(logits.shape)}")


if __name__ == "__main__":
    main()
