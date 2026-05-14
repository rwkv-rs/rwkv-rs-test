/*
Add trace around the real module call. The traced call still returns the original
result, while trace_webgpu() records module timing and selected output
activations for web-rwkv / WebGPU.

```diff
  let mut writer = TraceWriter::new();

- let y = run_time_mixer(x);
+ let y = trace_webgpu(
+     &mut writer,
+     case_root,
+     "cells/cell_0000/time_mixer",
+     || run_time_mixer(x),
+     outputs! {
+         value => "cells/cell_0000/time_mixer/embedded_context.safetensors",
+     },
+ );
```
*/
