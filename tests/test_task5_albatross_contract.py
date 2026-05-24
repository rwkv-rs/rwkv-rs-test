import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "infer-repo" / "albatross" / "task5_core_forward_sample.py"
SPEC = importlib.util.spec_from_file_location("task5_core_forward_sample", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
task5_core_forward_sample = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task5_core_forward_sample)


class AlbatrossTask5ContractTest(unittest.TestCase):
    def test_legacy_versions_do_not_claim_task_native_batch_prefill(self) -> None:
        for version in ("_ref_slower_", "faster_251101", "faster2_251201"):
            with self.subTest(version=version):
                self.assertTrue(task5_core_forward_sample.supports_task(version, "decode"))
                self.assertTrue(task5_core_forward_sample.supports_task(version, "prefill"))
                self.assertTrue(task5_core_forward_sample.supports_task(version, "batch_decode"))
                self.assertFalse(task5_core_forward_sample.supports_task(version, "batch_prefill"))

    def test_faster3_versions_claim_task_native_batch_prefill(self) -> None:
        self.assertTrue(task5_core_forward_sample.supports_task("faster3_2605", "batch_prefill"))
        self.assertTrue(task5_core_forward_sample.supports_task("faster3a_2605", "batch_prefill"))

    def test_partitions_unsupported_cases_before_model_setup(self) -> None:
        supported, unsupported = task5_core_forward_sample.partition_supported_cases(
            "faster_251101",
            [("decode", 1, 1), ("batch_prefill", 32, 32)],
        )

        self.assertEqual(supported, [("decode", 1, 1)])
        self.assertEqual(unsupported, [("batch_prefill", 32, 32)])

    def test_faster_251101_uses_original_flag_gems_sparsity_path(self) -> None:
        source = (MODULE_PATH.parent / "faster_251101" / "reference" / "rwkv7.py").read_text()

        self.assertIn("torch.ops.flag_gems.rwkv_mm_sparsity(k, V_)", source)
        self.assertNotIn("ALBATROSS_DISABLE_FLAG_GEMS_SPARSITY", source)
        self.assertNotIn("def rwkv_mm_sparsity", source)


if __name__ == "__main__":
    unittest.main()
