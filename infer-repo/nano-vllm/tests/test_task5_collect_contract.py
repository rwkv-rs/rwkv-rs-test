import unittest
from pathlib import Path
from types import SimpleNamespace

import task5_collect_nano_vllm as task5


class Task5CollectContractTest(unittest.TestCase):
    def test_csv_fields_cover_common_task5_contract(self):
        required = {
            "run_id",
            "repo",
            "backend",
            "runner",
            "benchmark_kind",
            "model_size",
            "device",
            "gpu_name",
            "gpu_uuid",
            "prompt_source",
            "prompt_count",
            "prompt_tokens",
            "command",
            "binary_path",
            "binary_build_id",
            "model_bytes",
            "model_sha256",
            "started_at",
            "ended_at",
        }
        self.assertTrue(required.issubset(set(task5.CSV_FIELDS)))

    def test_build_row_records_gpu_uuid_and_cuda0_device(self):
        args = SimpleNamespace(warmup=0, repeat=1, seed=0)
        row = task5.build_row(
            model_pth=Path("../../weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth"),
            device="cuda0",
            bsz=1,
            prompt_len=16,
            decode_len=4,
            args=args,
            direct=task5.DirectResult(prefill_tokens=16, output_tokens=4, prefill_time_s=1.0, prefill_tps=16.0, decode_tps=3.0),
            api_summary=None,
            metrics=[],
            api_error=None,
            run_id="task5-nano-test",
            gpu_name="NVIDIA GeForce RTX 5090",
            gpu_uuid="GPU-test",
            command=["python", "task5_collect_nano_vllm.py"],
            binary_path="python",
            binary_build_id="driver=596.36",
            started_at="2026-05-09T00:00:00Z",
            ended_at="2026-05-09T00:00:01Z",
            prompt_source="../../data/gsm8k.jsonl",
            prompt_count=1,
        )
        self.assertEqual(row["device"], "cuda0")
        self.assertEqual(row["gpu_uuid"], "GPU-test")
        self.assertEqual(row["benchmark_kind"], "synthetic_throughput")
        self.assertEqual(row["status"], "ok")

    def test_status_row_records_unsupported_prefill_limit(self):
        args = SimpleNamespace(warmup=0, repeat=1, seed=0)
        row = task5.build_status_row(
            model_pth=Path("../../weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth"),
            device="cuda0",
            bsz=1024,
            prompt_len=4096,
            decode_len=16,
            args=args,
            status="unsupported",
            error="prefill token count 4194304 exceeds max-prefill-tokens 65536",
            run_id="task5-nano-unsupported-test",
            gpu_name="NVIDIA GeForce RTX 5090",
            gpu_uuid="GPU-test",
            command=["python", "task5_collect_nano_vllm.py"],
            binary_path="python",
            binary_build_id="driver=596.36",
            started_at="2026-05-09T00:00:00Z",
            ended_at="2026-05-09T00:00:01Z",
            prompt_source="../../data/gsm8k.jsonl",
            prompt_count=1024,
        )
        self.assertEqual(row["status"], "unsupported")
        self.assertEqual(row["device"], "cuda0")
        self.assertEqual(row["gpu_uuid"], "GPU-test")
        self.assertIn("max-prefill-tokens", row["error"])


if __name__ == "__main__":
    unittest.main()
