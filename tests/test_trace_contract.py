from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class TraceContractTests(unittest.TestCase):
    def test_readme_requires_real_entrypoints_and_warm_timing(self):
        readme = read("README.md")
        docs = read("docs/trace-export.md")

        self.assertIn("`elapsed_ns` 必须写平均耗时", readme)
        self.assertIn("elapsed_ns = round(sum(samples_ns) / repeat)", readme)
        self.assertIn("禁止创建重建 Trainer、DataLoader、ModelRunner", readme)
        self.assertIn("RWKV_TRACE_WARMUP=1 RWKV_TRACE_REPEAT=3", readme)
        self.assertIn("禁止创建专用 trace 程序入口", docs)
        self.assertIn("RWKV_TRACE_REPEAT=3", docs)
        self.assertIn("RWKV_TRACE_WARMUP=1", docs)
        self.assertNotIn("scripts/export_trace_average.sh", readme + docs)

    def test_average_export_scripts_are_removed(self):
        self.assertFalse((ROOT / "scripts" / "trace_average.py").exists())
        self.assertFalse((ROOT / "scripts" / "export_trace_average.sh").exists())

    def test_python_template_owns_warmup_repeat_schema(self):
        trace = read("docs/trace_template/python/trace.py")

        self.assertIn('TRACE_REPEAT = int(os.environ.get("RWKV_TRACE_REPEAT", "3"))', trace)
        self.assertIn('TRACE_WARMUP = int(os.environ.get("RWKV_TRACE_WARMUP", "1"))', trace)
        self.assertIn("for _ in range(TRACE_WARMUP):", trace)
        self.assertIn("for _ in range(TRACE_REPEAT):", trace)
        self.assertIn("samples_ns.append(perf_counter_ns() - start)", trace)
        self.assertIn("elapsed_ns = round(sum(samples_ns) / len(samples_ns))", trace)
        self.assertIn('"samples_ns": _LAST_SAMPLES_NS', trace)
        self.assertNotIn('"samples_ns": [elapsed_ns]', trace)

    def test_pre_layer_norm_inputs_are_not_exported_as_duplicate_activations(self):
        checked_paths = [
            "train-repo/rwkv-lm/src/model.py",
            "train-repo/rwkv-peft/rwkvt/rwkv7/block.py",
            "infer-repo/albatross/reference/rwkv7.py",
            "infer-repo/web-rwkv/examples/trace_infer.rs",
            "infer-repo/llama.cpp/examples/trace-rwkv/trace-rwkv.cpp",
            "docs/trace-contract.md",
            "docs/trace_template/python/trace.py",
            "docs/trace_template/cpp/trace.cpp",
            "docs/trace_template/rust/burn/trace.rs",
            "docs/trace_template/rust/vulkan/trace.rs",
        ]
        combined = "\n".join(read(path) for path in checked_paths)

        self.assertNotIn("pre_layer_norm_for_time_mix/embedded_context.safetensors", combined)
        self.assertNotIn("pre_layer_norm_for_channel_mix/embedded_context.safetensors", combined)
        self.assertIn("pre_layer_norm_for_time_mix.time.json", read("docs/trace-contract.md"))
        self.assertIn("pre_layer_norm_for_channel_mix.time.json", read("docs/trace-contract.md"))

    def test_cell_output_is_the_next_cell_input_without_duplicate_file(self):
        contract = read("docs/trace-contract.md")

        self.assertIn(
            "`embedded_context_after_channel_mixer` 是当前 cell 输出，也是下一层 cell 输入",
            contract,
        )
        self.assertIn(
            "canonical 激活表示，不再另存一份输入激活",
            contract,
        )


if __name__ == "__main__":
    unittest.main()
