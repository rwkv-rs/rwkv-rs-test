from __future__ import annotations

import csv

import pytest

from scripts.task5_core_schema import (
    BENCHMARK_KIND,
    CSV_FIELDS,
    failed_row,
    ok_row,
    task_shape,
    unsupported_row,
    write_csv,
)


@pytest.mark.parametrize(
    ("task", "B", "T"),
    [
        ("decode", 1, 1),
        ("prefill", 1, 4096),
        ("batch_decode", 1024, 1),
        ("batch_prefill", 32, 32),
    ],
)
def test_task_shape_accepts_canonical_classes(task: str, B: int, T: int) -> None:
    assert task_shape(task, B, T) == (task, B, T)


@pytest.mark.parametrize(
    ("task", "B", "T"),
    [
        ("decode", 2, 1),
        ("decode", 1, 16),
        ("prefill", 2, 16),
        ("batch_decode", 1, 1),
        ("batch_decode", 8, 16),
        ("batch_prefill", 1, 16),
        ("batch_prefill", 16, 1),
    ],
)
def test_task_shape_rejects_wrong_shape_classes(task: str, B: int, T: int) -> None:
    with pytest.raises(ValueError):
        task_shape(task, B, T)


def test_unsupported_rows_preserve_task_shape_and_entrypoint() -> None:
    row = unsupported_row(
        run_id="r1",
        repo="web-rwkv",
        backend="web-rwkv",
        runner="task5_core_forward_sample",
        task="batch_prefill",
        B=32,
        T=32,
        entrypoint="web-rwkv Rnn::run",
        error="true batch prefill is not exposed",
    )

    assert row["benchmark_kind"] == BENCHMARK_KIND
    assert row["task"] == "batch_prefill"
    assert row["B"] == 32
    assert row["T"] == 32
    assert row["status"] == "unsupported"
    assert row["entrypoint"] == "web-rwkv Rnn::run"


@pytest.mark.parametrize("task", ["decode", "prefill"])
def test_required_decode_and_prefill_cannot_be_unsupported(task: str) -> None:
    with pytest.raises(ValueError, match="required Task5 core workload"):
        unsupported_row(
            task=task,
            B=1,
            T=1 if task == "decode" else 16,
            entrypoint="not wired",
            error="placeholder unsupported is invalid",
        )


def test_ok_rows_compute_task_specific_measured_tokens() -> None:
    row = ok_row(
        task="batch_decode",
        B=512,
        T=1,
        total_time_s=0.25,
        p50_ms=25,
        entrypoint="forward_batch+sampler_simple_batch",
        measurement_boundary="forward+sampler; no tokenizer decode; no scheduler; no server",
    )

    assert row["input_tokens"] == 512
    assert row["measured_tokens"] == 512
    assert row["forward_sample_tps"] == 2048

    prefill = ok_row(
        task="prefill",
        B=1,
        T=1024,
        total_time_s=0.5,
        p50_ms=50,
        entrypoint="forward_seq+sampler_simple",
        measurement_boundary="forward+sampler; no tokenizer decode; no scheduler; no server",
    )
    assert prefill["measured_tokens"] == 1024
    assert prefill["forward_sample_tps"] == 2048


def test_failed_rows_are_status_rows_not_metric_rows() -> None:
    row = failed_row(task="prefill", B=1, T=4096, entrypoint="forward_seq", error="OOM")

    assert row["status"] == "failed"
    assert row["forward_sample_tps"] == ""
    assert row["total_time_s"] == ""


def test_schema_does_not_require_old_task5_tps_fields(tmp_path) -> None:
    assert "prefill_tps" not in CSV_FIELDS
    assert "decode_tps" not in CSV_FIELDS
    assert "e2e_tps" not in CSV_FIELDS

    out = tmp_path / "task5_core.csv"
    write_csv(
        out,
        [
            unsupported_row(
                task="batch_decode",
                B=64,
                T=1,
                entrypoint="direct model",
                error="no true batch decode",
            )
        ],
    )

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["benchmark_kind"] == BENCHMARK_KIND
    assert rows[0]["task"] == "batch_decode"
    assert "prefill_tps" not in rows[0]
