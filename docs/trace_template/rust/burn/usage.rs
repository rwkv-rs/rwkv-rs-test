/*
Add trace around the real Burn module call. The traced call still returns the
original result, while trace_burn() records module timing and selected output
activations.

```diff
  let mut writer = TraceWriter::new();

- let (loss, lse, max_vals, argmax) = l2wrap_cross_entropy(logits, targets);
+ let (loss, lse, max_vals, argmax) = trace_burn(
+     &mut writer,
+     case_root,
+     "loss/l2wrap_cross_entropy",
+     || l2wrap_cross_entropy(logits, targets),
+     outputs! {
+         0 => "loss/l2wrap_cross_entropy.safetensors",
+         1 => "loss/l2wrap_cross_entropy/lse.safetensors",
+         2 => "loss/l2wrap_cross_entropy/max_vals.safetensors",
+         3 => "loss/l2wrap_cross_entropy/argmax.safetensors",
+     },
+ );
```
*/
