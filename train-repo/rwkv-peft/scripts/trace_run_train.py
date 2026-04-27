import runpy
import sys
import os

import peft


if not hasattr(peft, "BoneConfig") and hasattr(peft, "MissConfig"):
    peft.BoneConfig = peft.MissConfig


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    sys.argv = ["train.py", *sys.argv[1:]]
    runpy.run_path("train.py", run_name="__main__")
