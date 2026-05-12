import os
import sys
import unittest
from unittest import mock

from nanovllm.ops import rwkv7_cuda


class RWKV7CudaPathTest(unittest.TestCase):
    def test_ensure_python_bin_on_path_adds_venv_scripts_dir(self):
        python_bin = os.path.dirname(sys.executable)
        with mock.patch.dict(os.environ, {"PATH": os.pathsep.join(part for part in os.environ.get("PATH", "").split(os.pathsep) if part != python_bin)}):
            rwkv7_cuda.ensure_python_bin_on_path()
            self.assertIn(python_bin, os.environ["PATH"].split(os.pathsep))


if __name__ == "__main__":
    unittest.main()
