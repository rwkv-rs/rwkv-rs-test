import unittest

import task5_benchmark as task5


class Task5BenchmarkHelpersTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(task5.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(task5.percentile([10.0, 20.0], 0.95), 19.5)

    def test_unsupported_row_uses_readme_schema(self):
        row = task5.make_row(
            model_path="../../weights/model.pth",
            bsz=1024,
            prompt_len=4096,
            decode_len=16,
            warmup=0,
            repeat=1,
            seed=42,
            status="unsupported",
            error="prefill token count exceeds limit",
            gpu_name="NVIDIA GeForce RTX 5090",
            gpu_uuid="GPU-test",
            command=["python", "task5_benchmark.py"],
            started_at="2026-05-09T00:00:00Z",
            ended_at="2026-05-09T00:00:01Z",
        )
        self.assertEqual(set(row), set(task5.CSV_FIELDS))
        self.assertEqual(row["repo"], "albatross")
        self.assertEqual(row["backend"], "albatross-direct")
        self.assertEqual(row["runner"], "albatross-direct")
        self.assertEqual(row["benchmark_kind"], "synthetic_throughput")
        self.assertEqual(row["gpu_uuid"], "GPU-test")
        self.assertEqual(row["model_format"], "pth")
        self.assertEqual(row["dtype"], "fp16")
        self.assertEqual(row["prefill_tokens"], 1024 * 4096)
        self.assertEqual(row["output_tokens"], 1024 * 16)

    def test_benchmark_cases_do_not_filter_large_prefill_pairs(self):
        cases = task5.benchmark_cases([1, 1024], [16, 4096])

        self.assertEqual(cases, [(1, 16), (1, 4096), (1024, 16), (1024, 4096)])

    def test_chunk_prompt_tokens_preserves_batch_order(self):
        tokens = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]

        chunks = list(task5.chunk_prompt_tokens(tokens, 2))

        self.assertEqual(chunks, [[[1, 2], [6, 7]], [[3, 4], [8, 9]], [[5], [10]]])

    def test_run_once_uses_tensor_forward_when_available(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()
            long = "long"

            @staticmethod
            def tensor(data, *, dtype=None, device=None):
                return FakeTensor(data)

        class FakeTensor:
            def __init__(self, data):
                self.data = data

            @property
            def shape(self):
                if self.data and isinstance(self.data[0], list):
                    return (len(self.data), len(self.data[0]))
                return (len(self.data),)

            def __getitem__(self, item):
                rows, cols = item
                row_data = self.data[rows]
                return FakeTensor([row[cols] for row in row_data])

            def reshape(self, *shape):
                if shape == (-1,):
                    return self
                if len(shape) == 2 and shape[1] == 1:
                    return FakeTensor([[value] for value in self.data])
                raise AssertionError(f"unexpected reshape {shape}")

            def float(self):
                return self

        class FakeModel:
            def __init__(self):
                self.shapes = []

            def generate_zero_state(self, bsz):
                return {"bsz": bsz}

            def forward_seq_batch_1(self, tokens, state):
                self.shapes.append(tokens.shape)
                return FakeTensor([0] * tokens.shape[0])

            def forward_batch(self, tokens, state):
                raise AssertionError("run_once should use tensor fast path")

        def fake_sampler(logits, noise=0):
            return FakeTensor([1] * logits.shape[0])

        model = FakeModel()

        task5.run_once(
            model,
            fake_sampler,
            FakeTorch,
            tokens=[[1, 2, 3], [4, 5, 6]],
            decode_len=2,
            prefill_chunk_size=2,
        )

        self.assertEqual(model.shapes, [(2, 2), (2, 1), (2, 1)])

    def test_run_once_rejects_microbatch_split(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        class FakeTorch:
            cuda = FakeCuda()
            long = "long"

            @staticmethod
            def tensor(data, *, dtype=None, device=None):
                return FakeTensor(data)

        class FakeTensor:
            def __init__(self, data):
                self.data = data

            @property
            def shape(self):
                if self.data and isinstance(self.data[0], list):
                    return (len(self.data), len(self.data[0]))
                return (len(self.data),)

            def __getitem__(self, item):
                rows, cols = item
                row_data = self.data[rows]
                return FakeTensor([row[cols] for row in row_data])

            def reshape(self, *shape):
                if shape == (-1,):
                    return self
                if len(shape) == 2 and shape[1] == 1:
                    return FakeTensor([[value] for value in self.data])
                raise AssertionError(f"unexpected reshape {shape}")

            def float(self):
                return self

        class FakeModel:
            def __init__(self):
                self.shapes = []

            def generate_zero_state(self, bsz):
                return {"bsz": bsz}

            def forward_seq_batch_1(self, tokens, state):
                self.shapes.append(tokens.shape)
                return FakeTensor([0] * tokens.shape[0])

        def fake_sampler(logits, noise=0):
            return FakeTensor([1] * logits.shape[0])

        model = FakeModel()

        with self.assertRaisesRegex(ValueError, "would split real bsz"):
            task5.run_once(
                model,
                fake_sampler,
                FakeTorch,
                tokens=[[1, 2, 3], [4, 5, 6]],
                decode_len=2,
                prefill_chunk_size=2,
                micro_batch_size=1,
            )

        self.assertEqual(model.shapes, [])


if __name__ == "__main__":
    unittest.main()
