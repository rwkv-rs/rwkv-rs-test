/*
Add trace around the real module call. The traced call still returns the original
result, while trace_libtorch()/trace_ggml() records module timing and selected
output activations.

```diff
  TraceWriter writer;

- auto logits = head(hidden);
+ auto logits = trace_libtorch(
+     writer,
+     case_root,
+     "lm_head",
+     [&] {
+         return head(hidden);
+     },
+     outputs(out<0>("lm_head/logits.safetensors")),
+     hidden
+ );

  // For ggml / llama.cpp scheduler callbacks, keep the scheduler ask=true/ask=false
  // boundary as the timing source, then save activation and module timing together.
- activation_ggml(writer, case_root, "lm_head/logits.safetensors", logits_node);
+ trace_ggml(
+     writer,
+     case_root,
+     "lm_head",
+     lm_head_elapsed_ns,
+     outputs(node("lm_head/logits.safetensors", logits_node))
+ );
```
*/
