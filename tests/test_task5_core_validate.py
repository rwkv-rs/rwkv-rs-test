from __future__ import annotations

import csv
from pathlib import Path

from scripts.task5_core_schema import ok_row, unsupported_row, write_csv
from scripts.task5_core_validate import required_cases, validate_rows, SourceRow


def test_validator_rejects_required_task_unsupported(tmp_path: Path) -> None:
    row = unsupported_row(
        task="decode",
        B=1,
        T=1,
        entrypoint="placeholder",
        error="not wired",
        allow_required_task_unsupported=True,
        repo="nano-vllm",
        model_size="2.9B",
    )
    source = tmp_path / "2.9B" / "task5_core_forward_sample.csv"
    source.parent.mkdir()
    write_csv(source, [row])

    errors = validate_rows(
        [SourceRow(source, row)],
        required_backends=["nano-vllm"],
        required_model_sizes=["2.9B"],
        allow_required_failed=False,
        require_layout=True,
        albatross_7b_tps_gate=0,
    )

    assert any("required task decode" in error for error in errors)
    assert any("nano-vllm 2.9B decode 1x1 must be ok" in error for error in errors)


def test_validator_accepts_complete_minimal_matrix(tmp_path: Path) -> None:
    source = tmp_path / "albatross" / "7.2B" / "task5_core_forward_sample.csv"
    source.parent.mkdir(parents=True)
    rows = [
        ok_row(
            repo="albatross",
            backend="albatross-faster3a_2605",
            model_size="7.2B",
            task=task,
            B=B,
            T=T,
            total_time_s=max(0.001, (B * T) / 20000.0),
            p50_ms=1.0,
            entrypoint="forward+sample",
            measurement_boundary="forward+sampler; no tokenizer decode; no scheduler; no server",
        )
        for task, B, T in required_cases()
    ]
    write_csv(source, rows)

    loaded = []
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            loaded.append(SourceRow(source, row))

    errors = validate_rows(
        loaded,
        required_backends=["albatross"],
        required_model_sizes=["7.2B"],
        allow_required_failed=False,
        require_layout=True,
        albatross_7b_tps_gate=17_000,
    )

    assert errors == []
