from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def read_workspace(relative: str) -> str:
    return (WORKSPACE_ROOT / relative).read_text(encoding="utf-8")


class RwkvLmStaticContractTests(unittest.TestCase):
    def test_train_entrypoint_exposes_new_kernel_and_head_chunk_flags(self):
        train = read("train.py")

        self.assertIn('parser.add_argument("--head_chunk"', train)
        self.assertIn('parser.add_argument("--kernel"', train)
        self.assertIn('os.environ["RWKV_KERNEL"] = args.kernel', train)
        self.assertIn('os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = str(args.head_chunk)', train)
        self.assertIn('trace_once = os.environ.get("RWKV_TRACE_ONCE") == "1"', train)
        self.assertIn("args.grad_cp = 0", train)
        self.assertIn('os.environ["RWKV_JIT_ON"] = "0"', train)

    def test_new_cuda_sources_are_available(self):
        expected = [
            "cuda/rwkv7_clampw128_v2.cpp",
            "cuda/rwkv7_clampw128_v2.cu",
            "cuda/rwkv7_head_l2wrap_ce_bf16_v4.cpp",
            "cuda/rwkv7_head_l2wrap_ce_bf16_v4.cu",
            "cuda/rwkv7_l2wrap_ce_bf16_v2.cpp",
            "cuda/rwkv7_l2wrap_ce_bf16_v2.cu",
        ]

        missing = [path for path in expected if not (ROOT / path).exists()]
        self.assertEqual(missing, [])

        obsolete = [
            "cuda/rwkv7_l2wrap_ce_bf16_v1.cpp",
            "cuda/rwkv7_l2wrap_ce_bf16_v1.cu",
        ]
        present = [path for path in obsolete if (ROOT / path).exists()]
        self.assertEqual(present, [])

    def test_trace_helper_owns_module_timing_and_activations(self):
        trace = read("src/trace.py")

        self.assertIn("def activation(filename: str, tensor: torch.Tensor) -> None:", trace)
        self.assertIn("def timing(module: str, elapsed_ns: int) -> None:", trace)
        self.assertIn("def trace(", trace)
        self.assertIn('case_root() / "timing" / f"{module}.time.json"', trace)
        self.assertIn('"module": module', trace)
        self.assertIn('"samples_ns": _LAST_SAMPLES_NS', trace)
        self.assertIn('TRACE_REPEAT = int(os.environ.get("RWKV_TRACE_REPEAT", "3"))', trace)
        self.assertIn('TRACE_WARMUP = int(os.environ.get("RWKV_TRACE_WARMUP", "1"))', trace)
        self.assertIn("for _ in range(TRACE_WARMUP):", trace)
        self.assertIn("for _ in range(TRACE_REPEAT):", trace)
        self.assertNotIn("def trace_cell(", trace)
        self.assertNotIn("def measure(", trace)
        self.assertNotIn("def timer_start(", trace)
        self.assertNotIn("def timer_elapsed(", trace)

    def test_model_uses_callable_trace_without_old_elapsed_api(self):
        model = read("src/model.py")

        self.assertIn("def _forward_features(self, idx):", model)
        self.assertIn("def head_l2wrap_cross_entropy(hidden, weight, targets):", model)
        self.assertIn('if int(os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"]) > 0:', model)
        self.assertIn('activation("embedding/token_ids.safetensors", idx)', model)
        self.assertIn('trace("embedding"', model)
        self.assertIn('f"cells/cell_{self.layer_id:04d}/time_mixer"', model)
        self.assertIn('f"cells/cell_{self.layer_id:04d}/channel_mixer"', model)
        self.assertIn('"loss/l2wrap_cross_entropy"', model)
        self.assertIn('"loss/head_l2wrap_cross_entropy"', model)
        self.assertNotIn("trace(\"lm_head/logits.safetensors\"", model)
        self.assertNotIn("measure(", model)
        self.assertNotIn("timer_start(", model)
        self.assertNotIn("timer_elapsed(", model)
        self.assertNotIn("trace_cell(", model)
        self.assertNotIn("elapsed_ns", model)

        single_use_forward_helpers = re.findall(r"def _[a-z0-9_]+_forward_op\(", model)
        self.assertEqual(single_use_forward_helpers, [])

    def test_no_activation_sidecar_timing_contract_remains(self):
        trace = read("src/trace.py")

        self.assertNotIn('with_suffix(".time.json")', trace)
        self.assertNotIn('"filename": filename', trace)

    def test_readme_defines_training_kernel_trace_contract(self):
        readme = read_workspace("README.md")

        self.assertIn("训练 trace 和推理 trace 的导出集合不同", readme)
        self.assertIn("`lm_head/logits` 只属于推理 prefill 或显式 logits 对齐 case", readme)
        self.assertIn("`rwkv_lm` 训练 trace 不要求导出 `lm_head/logits`", readme)
        self.assertIn("`loss/l2wrap_cross_entropy.safetensors`", readme)
        self.assertIn("`loss/head_l2wrap_cross_entropy.safetensors`", readme)


if __name__ == "__main__":
    unittest.main()
