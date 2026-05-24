#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task5_core_schema import (
    BATCH_DECODE_B_DEFAULT,
    BATCH_PREFILL_PAIRS_DEFAULT,
    BENCHMARK_KIND,
    PREFILL_T_DEFAULT,
)


REQUIRED_BACKENDS = (
    "albatross",
    "nano-vllm",
    "rwkv-lightning",
    "rwkv-mobile",
    "web-rwkv",
)
REQUIRED_MODEL_SIZES = ("0.1B", "0.4B", "1.5B", "2.9B", "7.2B", "13.3B")
ALBATROSS_7B_TPS_GATE = 17_000.0


@dataclass(frozen=True)
class SourceRow:
    source: Path
    row: dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task5 core forward+sample result completeness.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--backends", default=",".join(REQUIRED_BACKENDS))
    parser.add_argument("--model-sizes", default=",".join(REQUIRED_MODEL_SIZES))
    parser.add_argument("--allow-required-failed", action="store_true")
    parser.add_argument("--skip-layout-check", action="store_true")
    parser.add_argument("--albatross-7b-tps-gate", type=float, default=ALBATROSS_7B_TPS_GATE)
    args = parser.parse_args()

    rows = load_rows(args.roots)
    errors = validate_rows(
        rows,
        required_backends=split_csv(args.backends),
        required_model_sizes=split_csv(args.model_sizes),
        allow_required_failed=args.allow_required_failed,
        require_layout=not args.skip_layout_check,
        albatross_7b_tps_gate=args.albatross_7b_tps_gate,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"validated {len(rows)} Task5 core rows")


def load_rows(roots: list[Path]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for root in roots:
        files = [root] if root.is_file() else sorted(root.rglob("task5*.csv"))
        for path in files:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if row.get("benchmark_kind") == BENCHMARK_KIND:
                        rows.append(SourceRow(path, row))
    return rows


def validate_rows(
    rows: list[SourceRow],
    *,
    required_backends: list[str],
    required_model_sizes: list[str],
    allow_required_failed: bool,
    require_layout: bool,
    albatross_7b_tps_gate: float,
) -> list[str]:
    errors: list[str] = []
    if not rows:
        return ["no Task5 core rows found"]

    cases = required_cases()
    by_key: dict[tuple[str, str, str, int, int], list[SourceRow]] = {}
    for item in rows:
        row = item.row
        backend = normalize_backend(row.get("repo") or row.get("backend") or "")
        model_size = normalize_model_size(row.get("model_size") or row.get("model_path") or "")
        task = row.get("task", "")
        B = int_or_none(row.get("B"))
        T = int_or_none(row.get("T"))
        status = row.get("status", "")
        if B is None or T is None:
            errors.append(f"{item.source}: row has invalid B/T: B={row.get('B')!r} T={row.get('T')!r}")
            continue
        if task in {"decode", "prefill"} and status == "unsupported":
            errors.append(f"{item.source}: required task {task} B={B} T={T} is unsupported")
        if status == "ok":
            if positive_float(row.get("total_time_s")) is None:
                errors.append(f"{item.source}: ok row missing positive total_time_s for {backend} {model_size} {task} {B}x{T}")
            if positive_float(row.get("p50_ms")) is None:
                errors.append(f"{item.source}: ok row missing positive p50_ms for {backend} {model_size} {task} {B}x{T}")
            if positive_float(row.get("forward_sample_tps")) is None:
                errors.append(f"{item.source}: ok row missing positive forward_sample_tps for {backend} {model_size} {task} {B}x{T}")
        if require_layout:
            expected = model_size.lower()
            source_text = item.source.as_posix().lower()
            if expected != "unknown" and expected not in source_text:
                errors.append(f"{item.source}: result path does not include model size {model_size}")
        by_key.setdefault((backend, model_size, task, B, T), []).append(item)

    for backend in required_backends:
        for model_size in required_model_sizes:
            for task, B, T in cases:
                matches = by_key.get((backend, model_size, task, B, T), [])
                if not matches:
                    errors.append(f"missing row for {backend} {model_size} {task} {B}x{T}")
                    continue
                statuses = {item.row.get("status", "") for item in matches}
                if task in {"decode", "prefill"}:
                    ok = "ok" in statuses
                    allowed_failed = allow_required_failed and statuses <= {"failed"}
                    if not ok and not allowed_failed:
                        errors.append(f"{backend} {model_size} {task} {B}x{T} must be ok, got {sorted(statuses)}")
                if task == "batch_prefill" and backend == "albatross" and "ok" not in statuses:
                    errors.append(f"albatross {model_size} batch_prefill {B}x{T} must be ok, got {sorted(statuses)}")

    gate_rows = [
        item.row
        for item in rows
        if normalize_backend(item.row.get("repo") or item.row.get("backend") or "") == "albatross"
        and normalize_model_size(item.row.get("model_size") or item.row.get("model_path") or "") == "7.2B"
        and item.row.get("status") == "ok"
        and positive_float(item.row.get("forward_sample_tps")) is not None
    ]
    best_7b = max((float(row["forward_sample_tps"]) for row in gate_rows), default=0.0)
    if best_7b < albatross_7b_tps_gate:
        errors.append(f"albatross 7.2B TPS gate not met: best={best_7b:.2f}, required>={albatross_7b_tps_gate:.2f}")

    return errors


def required_cases() -> list[tuple[str, int, int]]:
    return (
        [("decode", 1, 1)]
        + [("prefill", 1, T) for T in PREFILL_T_DEFAULT]
        + [("batch_decode", B, 1) for B in BATCH_DECODE_B_DEFAULT]
        + [("batch_prefill", B, T) for B, T in BATCH_PREFILL_PAIRS_DEFAULT]
    )


def normalize_backend(value: str) -> str:
    value = value.lower()
    for backend in REQUIRED_BACKENDS:
        if backend in value:
            return backend
    return value or "unknown"


def normalize_model_size(value: str) -> str:
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*b", value, re.IGNORECASE)
    if not match:
        return "unknown"
    return f"{match.group(1)}B"


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def int_or_none(value: str | None) -> int | None:
    try:
        return int(value or "")
    except ValueError:
        return None


def positive_float(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except ValueError:
        return None
    return parsed if parsed > 0 else None


if __name__ == "__main__":
    main()
