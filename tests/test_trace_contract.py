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
        self.assertIn("elapsed_ns = (2 * sum(samples_ns) + len(samples_ns)) // (2 * len(samples_ns))", trace)
        self.assertIn('"samples_ns": _LAST_SAMPLES_NS', trace)
        self.assertNotIn('"samples_ns": [elapsed_ns]', trace)

    def test_pre_layer_norm_inputs_are_not_exported_as_duplicate_activations(self):
        checked_paths = [
            "train-repo/rwkv-lm/src/model.py",
            "train-repo/rwkv-peft/rwkvt/rwkv7/block.py",
            "infer-repo/albatross/_ref_slower_/reference/rwkv7.py",
            "infer-repo/albatross/faster_251101/reference/rwkv7.py",
            "infer-repo/albatross/faster2_251201/reference/rwkv7.py",
            "infer-repo/albatross/faster3_2605/rwkv7_fast_v3.py",
            "infer-repo/albatross/faster3a_2605/rwkv7_fast_v3a.py",
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

    def test_albatross_versions_have_isolated_inference_trace_contract(self):
        python_versions = {
            "_ref_slower_/reference": "albatross__ref_slower_",
            "faster_251101/reference": "albatross_faster_251101",
            "faster2_251201/reference": "albatross_faster2_251201",
            "faster3_2605": "albatross_faster3_2605",
            "faster3a_2605": "albatross_faster3a_2605",
        }
        for relative, repo_name in python_versions.items():
            trace = read(f"infer-repo/albatross/{relative}/trace.py")
            self.assertIn(f'REPO_NAME = "{repo_name}"', trace)
            self.assertIn('QUANTIZATION = "fp16"', trace)
            self.assertIn('Path(root) / REPO_NAME / QUANTIZATION / "case_000000"', trace)
            self.assertIn('"elapsed_ns": averaged_ns', trace)
            self.assertIn('"repeat": TRACE_REPEAT', trace)
            self.assertIn('"warmup": TRACE_WARMUP', trace)
            self.assertIn('"samples_ns": _LAST_SAMPLES_NS', trace)
            self.assertIn('path = case_root() / "timing" / f"{module}.time.json"', trace)
            self.assertIn("is_canonical_module", trace)
            self.assertNotIn("tensor.squeeze(0)", trace)
            self.assertIn("configure_trace_capture", trace)
            self.assertNotIn("for _ in range(TRACE_REPEAT):", trace)
            self.assertIn("_TIMING_SAMPLES_BY_MODULE", trace)
            self.assertIn("averaged_ns = (2 * sum(samples) + len(samples)) // (2 * len(samples))", trace)
            self.assertNotIn("averaged_ns = round(sum(samples) / len(samples))", trace)

    def test_albatross_python_forwards_are_instrumented_at_prefill_boundaries(self):
        model_paths = [
            "infer-repo/albatross/_ref_slower_/reference/rwkv7.py",
            "infer-repo/albatross/faster_251101/reference/rwkv7.py",
            "infer-repo/albatross/faster2_251201/reference/rwkv7.py",
            "infer-repo/albatross/faster3_2605/rwkv7_fast_v3.py",
            "infer-repo/albatross/faster3a_2605/rwkv7_fast_v3a.py",
        ]
        required = [
            "TRACE",
            "trace_token_ids",
            'trace("embedding"',
            'outputs="embedding/embedded_context.safetensors"',
            'f"cells/cell_{layer:04d}/time_mixer"',
            'f"cells/cell_{layer:04d}/time_mixer/embedded_context.safetensors"',
            'f"cells/cell_{layer:04d}/channel_mixer"',
            'f"cells/cell_{layer:04d}/channel_mixer/embedded_context.safetensors"',
            'trace("lm_head"',
            'outputs="lm_head/logits.safetensors"',
        ]
        for path in model_paths:
            source = read(path)
            for needle in required:
                self.assertIn(needle, source, f"{needle} missing from {path}")
            self.assertNotIn("value_from_first_cell.safetensors", source)
            self.assertNotIn("embedded_context_after_time_mixer.safetensors", source)
            self.assertNotIn("embedded_context_after_channel_mixer.safetensors", source)
            self.assertNotIn("lm_head_layer_norm", source)

    def test_albatross_cpp_model_forward_has_trace_contract(self):
        source = read("infer-repo/albatross/faster4_2605_cpp/src/rwkv7_fast_v4.cu")
        cmake = read("infer-repo/albatross/faster4_2605_cpp/CMakeLists.txt")

        self.assertIn('repo_name = "albatross_faster4_2605_cpp"', source)
        self.assertIn('quantization = "fp16"', source)
        self.assertIn("RWKV_TRACE_ONCE", source)
        self.assertIn("RWKV_TRACE_ROOT", source)
        self.assertIn("write_time_json", source)
        self.assertIn("write_f16_safetensor", source)
        self.assertIn("embedding/token_ids.safetensors", source)
        self.assertIn("embedding/embedded_context.safetensors", source)
        self.assertIn("time_mixer/embedded_context.safetensors", source)
        self.assertIn("channel_mixer/embedded_context.safetensors", source)
        self.assertIn("lm_head/logits.safetensors", source)
        self.assertIn("timing/", source)
        self.assertIn("std::filesystem", cmake + source)


if __name__ == "__main__":
    unittest.main()
