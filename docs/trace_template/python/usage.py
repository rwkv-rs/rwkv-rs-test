"""
Add trace around the real module call. The traced call still returns the original
result, while trace() records module timing and selected output activations.

```diff
  activation("embedding/token_ids.safetensors", token_ids.to(dtype=torch.int64, device="cpu"))

- x = time_mixer(x)
+ x = trace(
+     "cells/cell_0000/time_mixer",
+     time_mixer,
+     x,
+     outputs="cells/cell_0000/time_mixer/embedded_context.safetensors",
+ )

  # This is skipped automatically if x was already saved by the previous module.
  activation("cells/cell_0001/input.safetensors", x)

- loss, lse, max_vals, argmax = L2WRAP_CE_CUDA_V2.forward(logits, targets)
+ loss, lse, max_vals, argmax = trace(
+     "loss/l2wrap_cross_entropy",
+     L2WRAP_CE_CUDA_V2.forward,
+     logits,
+     targets,
+     outputs={
+         0: "loss/l2wrap_cross_entropy.safetensors",
+         1: "loss/l2wrap_cross_entropy/lse.safetensors",
+         2: "loss/l2wrap_cross_entropy/max_vals.safetensors",
+         3: "loss/l2wrap_cross_entropy/argmax.safetensors",
+     },
+ )
```
"""
