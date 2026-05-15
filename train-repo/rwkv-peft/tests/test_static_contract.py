from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


class RwkvPeftStaticContractTests(unittest.TestCase):
    def test_trace_helper_matches_module_timing_contract(self):
        trace = read("rwkvt/trace.py")

        self.assertIn("def activation(filename: str, tensor: torch.Tensor) -> None:", trace)
        self.assertIn("def timing(module: str, elapsed_ns: int) -> None:", trace)
        self.assertIn("def trace(", trace)
        self.assertIn('case_root() / "timing" / f"{module}.time.json"', trace)
        self.assertIn('"module": module', trace)
        self.assertIn('"samples_ns": _LAST_SAMPLES_NS', trace)
        self.assertIn('TRACE_REPEAT = int(os.environ.get("RWKV_TRACE_REPEAT", "3"))', trace)
        self.assertIn('TRACE_WARMUP = int(os.environ.get("RWKV_TRACE_WARMUP", "1"))', trace)
        self.assertNotIn("def measure(", trace)
        self.assertNotIn("def timer_start(", trace)
        self.assertNotIn("def timer_elapsed(", trace)
        self.assertNotIn("def trace_cell(", trace)
        self.assertNotIn('with_suffix(".time.json")', trace)

    def test_block_owns_canonical_mixer_timing(self):
        block = read("rwkvt/rwkv7/block.py")
        att = read("rwkvt/rwkv7/att.py")

        self.assertNotIn("measure", block)
        self.assertNotIn("trace_cell", block)
        self.assertIn('f"cells/cell_{self.layer_id:04d}/time_mixer"', block)
        self.assertIn('f"cells/cell_{self.layer_id:04d}/channel_mixer"', block)
        self.assertNotIn("timer_start", att)
        self.assertNotIn("timer_elapsed", att)
        self.assertNotIn("trace_cell", att)
        self.assertNotIn(".time.json", att)

    def test_model_uses_rwkv_lm_training_trace_layout(self):
        model = read("rwkvt/rwkv7/model.py")

        self.assertIn('activation("embedding/token_ids.safetensors", input_ids)', model)
        self.assertIn('trace("embedding"', model)
        self.assertIn('trace("lm_head"', model)
        self.assertNotIn('trace("lm_head/logits.safetensors"', model)
        self.assertNotIn("measure", model)

    def test_trace_wrapper_uses_real_training_entrypoint(self):
        wrapper = read("trace-run.sh")

        self.assertIn("uv run python train.py", wrapper)
        self.assertIn("--peft none", wrapper)
        self.assertIn("--data_type binidx", wrapper)
        self.assertIn("--my_testing x070", wrapper)
        self.assertNotIn("trace_run_pretrain.py", wrapper)
        self.assertFalse((ROOT / "scripts" / "trace_run_pretrain.py").exists())


if __name__ == "__main__":
    unittest.main()
