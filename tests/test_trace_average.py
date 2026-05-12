import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "trace_average.py"


class TraceAverageTests(unittest.TestCase):
    def test_readme_requires_average_elapsed_ns(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("`elapsed_ns` 必须写平均耗时", readme)
        self.assertIn("elapsed_ns = round(sum(samples_ns) / repeat)", readme)
        self.assertIn("`repeat`：参与平均的有效 trace run 数", readme)
        self.assertIn("scripts/export_trace_average.sh", readme)

    def test_average_rewrites_elapsed_ns_and_metadata(self):
        (ROOT / ".trace_tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".trace_tmp") as tmp:
            tmp = Path(tmp)
            case_root = tmp / "test_gen" / "demo" / "fp16" / "case_000000"
            runner = tmp / "runner.py"
            runner.write_text(
                """import json, os
from pathlib import Path
root = Path(os.environ["RWKV_TRACE_ROOT"]) / "demo" / "fp16" / "case_000000"
(root / "x").mkdir(parents=True, exist_ok=True)
run_file = Path(os.environ["RUN_FILE"])
count = int(run_file.read_text()) if run_file.exists() else 0
run_file.write_text(str(count + 1))
(root / "x" / "tensor.safetensors").write_bytes(b"last-run" + bytes([count]))
(root / "x" / "tensor.time.json").write_text(json.dumps({"filename": "x/tensor.safetensors", "elapsed_ns": (count + 1) * 10}))
""",
                encoding="utf-8",
            )
            env = dict(os.environ, RUN_FILE=str(tmp / "run_count.txt"))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-root",
                    str(case_root),
                    "--repeat",
                    "3",
                    "--warmup",
                    "1",
                    "--",
                    sys.executable,
                    str(runner),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            timing = json.loads((case_root / "x" / "tensor.time.json").read_text())
            self.assertEqual(timing["filename"], "x/tensor.safetensors")
            self.assertEqual(timing["elapsed_ns"], 30)
            self.assertEqual(timing["repeat"], 3)
            self.assertEqual(timing["warmup"], 1)
            self.assertEqual(timing["samples_ns"], [20, 30, 40])
            self.assertEqual((case_root / "x" / "tensor.safetensors").read_bytes(), b"last-run" + bytes([3]))

    def test_fails_when_measured_runs_do_not_match(self):
        (ROOT / ".trace_tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".trace_tmp") as tmp:
            tmp = Path(tmp)
            case_root = tmp / "test_gen" / "demo" / "fp16" / "case_000000"
            runner = tmp / "runner.py"
            runner.write_text(
                """import json, os
from pathlib import Path
root = Path(os.environ["RWKV_TRACE_ROOT"]) / "demo" / "fp16" / "case_000000"
root.mkdir(parents=True, exist_ok=True)
run_file = Path(os.environ["RUN_FILE"])
count = int(run_file.read_text()) if run_file.exists() else 0
run_file.write_text(str(count + 1))
name = "a" if count == 0 else "b"
(root / f"{name}.safetensors").write_bytes(b"x")
(root / f"{name}.time.json").write_text(json.dumps({"filename": f"{name}.safetensors", "elapsed_ns": 1}))
""",
                encoding="utf-8",
            )
            env = dict(os.environ, RUN_FILE=str(tmp / "run_count.txt"))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-root",
                    str(case_root),
                    "--repeat",
                    "2",
                    "--warmup",
                    "0",
                    "--",
                    sys.executable,
                    str(runner),
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("time file set mismatch", result.stderr + result.stdout)

    def test_fails_when_non_input_timing_is_zero(self):
        (ROOT / ".trace_tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".trace_tmp") as tmp:
            tmp = Path(tmp)
            case_root = tmp / "test_gen" / "demo" / "fp16" / "case_000000"
            runner = tmp / "runner.py"
            runner.write_text(
                """import json, os
from pathlib import Path
root = Path(os.environ["RWKV_TRACE_ROOT"]) / "demo" / "fp16" / "case_000000"
root.mkdir(parents=True, exist_ok=True)
(root / "x.safetensors").write_bytes(b"x")
(root / "x.time.json").write_text(json.dumps({"filename": "x.safetensors", "elapsed_ns": 0}))
""",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--case-root",
                    str(case_root),
                    "--repeat",
                    "1",
                    "--warmup",
                    "0",
                    "--",
                    sys.executable,
                    str(runner),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("zero elapsed_ns is only allowed", result.stderr + result.stdout)

    def test_current_trace_outputs_do_not_use_zero_timing_except_token_ids(self):
        allowed_zero = {"embedding/token_ids.time.json"}
        bad = []
        for case in sorted((ROOT / "test_gen").glob("*/*/case_000000")):
            for path in sorted(case.rglob("*.time.json")):
                rel = path.relative_to(case).as_posix()
                data = json.loads(path.read_text(encoding="utf-8"))
                if int(data.get("elapsed_ns", -1)) == 0 and rel not in allowed_zero:
                    bad.append(f"{case.relative_to(ROOT)}:{rel}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
