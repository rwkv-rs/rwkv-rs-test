#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import task5_common as common
import run_llama_task5 as runner
import stream_latency_bench as latency


class Task5ContractTest(unittest.TestCase):
    def test_csv_fields_cover_readme_contract(self) -> None:
        required = {
            "run_id",
            "repo",
            "backend",
            "runner",
            "benchmark_kind",
            "model_size",
            "model_path",
            "model_format",
            "device",
            "gpu_name",
            "gpu_uuid",
            "dtype",
            "quantization",
            "bsz",
            "prompt_len",
            "decode_len",
            "warmup",
            "repeat",
            "seed",
            "status",
            "error",
            "prompt_source",
            "prompt_count",
            "prompt_tokens",
            "output_tokens",
            "prefill_time_s",
            "ttft_s",
            "ttft_p95_s",
            "e2el_s",
            "e2el_p95_s",
            "token_generation_time_s",
            "prefill_tps",
            "decode_tps",
            "e2e_tps",
            "time_per_output_token_ms",
            "requests_per_s",
            "itl_mean_ms",
            "itl_p50_ms",
            "itl_p90_ms",
            "itl_p95_ms",
            "itl_p99_ms",
            "command",
            "binary_path",
            "binary_build_id",
            "started_at",
            "ended_at",
        }
        self.assertTrue(required.issubset(set(common.CSV_FIELDS)))

    def test_base_row_includes_hardware_runner_and_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "rwkv7-g1d-0.1b-20260129-ctx8192-Q4_K_M.gguf"
            model.write_bytes(b"GGUFpayload")
            row = common.base_row(
                model_path=model,
                backend="llama.cpp",
                runner="llama-batched-bench",
                benchmark_kind="synthetic_throughput",
                device="cuda0",
                gpu_name="NVIDIA GeForce RTX 5090",
                gpu_uuid="GPU-test",
                bsz=16,
                prompt_len=256,
                decode_len=16,
                warmup=1,
                repeat=3,
                seed=0,
                status="ok",
                command=["bench", "-m", str(model)],
                binary_path=Path("/bin/bench"),
                binary_build_id="build-id",
                started_at="2026-05-09T00:00:00Z",
                ended_at="2026-05-09T00:00:01Z",
                prompt_source="synthetic_fixed_tokens",
                prompt_count=16,
            )

            self.assertEqual(row["run_id"][:5], "task5")
            self.assertEqual(row["model_size"], "0.1B")
            self.assertEqual(row["quantization"], "q4_k_m")
            self.assertEqual(row["gpu_uuid"], "GPU-test")
            self.assertIn("bench -m", row["command"])
            self.assertEqual(row["binary_build_id"], "build-id")
            self.assertRegex(row["model_sha256"], r"^[0-9a-f]{64}$")

    def test_manifest_and_telemetry_writers_use_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            telemetry = root / "gpu_telemetry.csv"

            common.append_manifest(
                manifest,
                {
                    "run_id": "task5-test",
                    "benchmark_kind": "synthetic_throughput",
                    "gpu_uuid": "GPU-test",
                    "command": "bench",
                },
            )
            common.append_gpu_telemetry(
                telemetry,
                [
                    {
                        "timestamp": "2026-05-09T00:00:00Z",
                        "run_id": "task5-test",
                        "gpu_uuid": "GPU-test",
                        "gpu_util": "90",
                        "mem_used": "1024",
                        "mem_total": "32607",
                        "power_w": "450",
                        "sm_clock": "3000",
                        "mem_clock": "14001",
                        "pstate": "P0",
                        "process_name": "llama-batched-bench",
                    }
                ],
            )

            self.assertEqual(json.loads(manifest.read_text())["gpu_uuid"], "GPU-test")
            with telemetry.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["process_name"], "llama-batched-bench")


class LatencyContractTest(unittest.TestCase):
    def test_load_gsm8k_questions_defaults_to_full_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsm8k.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"question": f"question {i}", "answer": str(i)}) for i in range(3)
                ),
                encoding="utf-8",
            )

            questions = latency.load_gsm8k_questions(path)
            self.assertEqual(questions, ["question 0", "question 1", "question 2"])

    def test_synthetic_prompts_repeat_real_repeat_count(self) -> None:
        prompts = latency.make_synthetic_prompt_tokens(prompt_len=4, bsz=2, repeat=3, seed=7)
        self.assertEqual(len(prompts), 6)
        self.assertTrue(all(len(prompt) == 4 for prompt in prompts))

    def test_runnable_or_error_rejects_context_overflow(self) -> None:
        self.assertEqual(
            runner.runnable_or_error(bsz=2, prompt_len=16, decode_len=4, max_parallel=4, max_ctx_size=32),
            "required ctx 40 exceeds max ctx 32",
        )
        self.assertIn(
            "n_seq_max",
            runner.runnable_or_error(bsz=8, prompt_len=1, decode_len=1, max_parallel=4, max_ctx_size=32),
        )


if __name__ == "__main__":
    unittest.main()
