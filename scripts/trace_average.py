#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ALLOWED_ZERO_TIME_FILES = {"embedding/token_ids.time.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a real trace entrypoint repeatedly and rewrite .time.json files with average elapsed_ns."
    )
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not args.command or args.command[0] != "--" or len(args.command) == 1:
        parser.error("pass the trace command after --")
    args.command = args.command[1:]
    return args


def relative_case_parts(case_root: Path) -> tuple[Path, Path]:
    parts = case_root.parts
    if len(parts) < 4 or parts[-1] != "case_000000":
        raise SystemExit(f"--case-root must end with <trace-root>/<repo>/<quantization>/case_000000: {case_root}")
    relative = Path(*parts[-3:])
    trace_root = Path(*parts[:-3]) if case_root.is_absolute() else Path(*parts[:-3] or ["."])
    return trace_root, relative


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_times(case_root: Path) -> dict[str, dict]:
    times: dict[str, dict] = {}
    for path in sorted(case_root.rglob("*.time.json")):
        rel = path.relative_to(case_root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc
        if "filename" not in data or "elapsed_ns" not in data:
            raise RuntimeError(f"{path} must contain filename and elapsed_ns")
        elapsed_ns = int(data["elapsed_ns"])
        if elapsed_ns == 0 and rel not in ALLOWED_ZERO_TIME_FILES:
            raise RuntimeError(f"zero elapsed_ns is only allowed for input metadata, got {rel}")
        times[rel] = data
    if not times:
        raise RuntimeError(f"no .time.json files exported under {case_root}")
    return times


def run_once(command: list[str], trace_root: Path, relative_case: Path) -> dict[str, dict]:
    clean_dir(trace_root)
    env = os.environ.copy()
    env["RWKV_TRACE_ROOT"] = str(trace_root)
    env["RWKV_TRACE_ONCE"] = "1"
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"trace command failed with exit code {result.returncode}: {' '.join(command)}")
    case_root = trace_root / relative_case
    return read_times(case_root)


def rewrite_average_times(
    final_case_root: Path,
    samples: list[dict[str, dict]],
    repeat: int,
    warmup: int,
) -> None:
    baseline_keys = set(samples[0])
    for sample in samples[1:]:
        keys = set(sample)
        if keys != baseline_keys:
            missing = sorted(baseline_keys - keys)
            extra = sorted(keys - baseline_keys)
            raise RuntimeError(f"time file set mismatch: missing={missing[:10]} extra={extra[:10]}")

    for rel in sorted(baseline_keys):
        values = []
        filename = samples[0][rel]["filename"]
        for sample in samples:
            data = sample[rel]
            if data["filename"] != filename:
                raise RuntimeError(f"filename mismatch for {rel}: {filename} vs {data['filename']}")
            values.append(int(data["elapsed_ns"]))

        average = round(sum(values) / len(values))
        path = final_case_root / rel
        path.write_text(
            json.dumps(
                {
                    "filename": filename,
                    "elapsed_ns": average,
                    "repeat": repeat,
                    "warmup": warmup,
                    "samples_ns": values,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    trace_root, relative_case = relative_case_parts(args.case_root)
    staging_root = args.staging_root or (trace_root / ".trace_average" / "_".join(relative_case.parts[:-1]))
    clean_dir(staging_root)

    measured_samples: list[dict[str, dict]] = []
    total_runs = args.warmup + args.repeat
    final_run_case_root = staging_root / f"run_{total_runs - 1:04d}" / relative_case

    try:
        for index in range(total_runs):
            run_root = staging_root / f"run_{index:04d}"
            sample = run_once(args.command, run_root, relative_case)
            if index >= args.warmup:
                measured_samples.append(sample)
            print(
                f"trace-average run {index + 1}/{total_runs} "
                f"({'warmup' if index < args.warmup else 'measured'}) files={len(sample)}",
                flush=True,
            )

        clean_dir(args.case_root)
        shutil.copytree(final_run_case_root, args.case_root, dirs_exist_ok=True)
        rewrite_average_times(args.case_root, measured_samples, args.repeat, args.warmup)
    finally:
        if not args.keep_staging and staging_root.exists():
            shutil.rmtree(staging_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"trace-average: {exc}", file=sys.stderr)
        raise SystemExit(1)
