#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "core-forward-sample" / "merged-current"


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Task5 core result CSVs into the frontend default result root.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output == ROOT or ROOT not in output.parents:
        raise SystemExit(f"refusing to replace unsafe output path: {output}")

    sources = [
        (
            ROOT / "results/core-forward-sample/devpod-default-py-17326/albatross",
            Path("devpod-default-py-17326/albatross"),
        ),
        (
            ROOT / "results/core-forward-sample/devpod-default-py-17326/infer-repo/albatross",
            Path("devpod-default-py-17326/infer-repo/albatross"),
        ),
        (ROOT / "results/core-forward-sample/devpod-default-py-73e40", Path("devpod-default-py-73e40")),
    ]
    web_results = ROOT / "infer-repo/web-rwkv/results"
    for source in sorted(web_results.glob("windows-*")):
        if source.is_dir():
            sources.append((source, Path("windows-web-rwkv") / source.name))

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied = 0
    for source_root, destination_root in sources:
        if not source_root.exists():
            continue
        for csv_path in sorted(source_root.rglob("task5_core_forward_sample.csv")):
            if not args.include_smoke and "smoke" in csv_path.as_posix().lower():
                continue
            destination = output / destination_root / csv_path.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, destination)
            copied += 1

    print(f"wrote {output}")
    print(f"copied {copied} task5_core_forward_sample.csv files")


if __name__ == "__main__":
    main()
