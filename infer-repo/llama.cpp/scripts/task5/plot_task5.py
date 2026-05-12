#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ok_rows(rows: list[dict[str, str]], kind: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "ok" and row.get("benchmark_kind") == kind]


def save_empty(path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.set_title(title)
    axis.text(0.5, 0.5, "No successful rows", ha="center", va="center")
    axis.set_axis_off()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows: list[dict[str, str]], metric: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    usable = [row for row in rows if number(row.get(metric)) is not None]
    if not usable:
        save_empty(path, title)
        return
    bsz_values = sorted({int(row["bsz"]) for row in usable})
    prompt_lens = sorted({int(row["prompt_len"]) for row in usable})
    matrix = []
    for prompt_len in prompt_lens:
        row_values = []
        for bsz in bsz_values:
            values = [
                number(row.get(metric))
                for row in usable
                if int(row["bsz"]) == bsz and int(row["prompt_len"]) == prompt_len
            ]
            values = [value for value in values if value is not None]
            row_values.append(max(values) if values else 0.0)
        matrix.append(row_values)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 5))
    image = axis.imshow(matrix, aspect="auto")
    axis.set_title(title)
    axis.set_xlabel("bsz")
    axis.set_ylabel("prompt_len")
    axis.set_xticks(range(len(bsz_values)), [str(v) for v in bsz_values])
    axis.set_yticks(range(len(prompt_lens)), [str(v) for v in prompt_lens])
    fig.colorbar(image, ax=axis)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_frontier(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    usable = ok_rows(rows, "synthetic_throughput")
    frontier: dict[str, float] = defaultdict(float)
    for row in usable:
        key = f"{row.get('model_size')} {row.get('quantization')}"
        value = int(row["bsz"]) * (int(row["prompt_len"]) + int(row["decode_len"]))
        frontier[key] = max(frontier[key], value)
    if not frontier:
        save_empty(path, "Throughput frontier")
        return
    labels, values = zip(*sorted(frontier.items(), key=lambda item: item[1], reverse=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(max(9, len(labels) * 0.45), 5))
    axis.bar(range(len(values)), values)
    axis.set_title("Throughput frontier")
    axis.set_ylabel("max bsz * (prompt_len + decode_len)")
    axis.set_xticks(range(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    usable = [row for row in ok_rows(rows, "synthetic_throughput") if number(row.get("decode_tps")) is not None]
    if not usable:
        save_empty(path, "Throughput Pareto")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    for row in usable:
        size = number(row.get("model_bytes")) or 0.0
        axis.scatter(size / (1024**3), number(row.get("decode_tps")), s=24)
    axis.set_title("Throughput Pareto")
    axis.set_xlabel("model file GiB")
    axis.set_ylabel("decode tokens/s")
    axis.set_yscale("log")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_latency_line(rows: list[dict[str, str]], metric: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    usable = [row for row in ok_rows(rows, "synthetic_latency") if number(row.get(metric)) is not None]
    if not usable:
        save_empty(path, title)
        return
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in usable:
        label = f"{row.get('model_size')} {row.get('quantization')}"
        grouped[label].append((int(row.get("bsz") or row.get("concurrency") or 0), number(row.get(metric)) or 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, points in sorted(grouped.items()):
        points = sorted(points)
        axis.plot([p[0] for p in points], [p[1] * 1000.0 for p in points], marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel("concurrency")
    axis.set_ylabel("ms")
    axis.set_xscale("log", base=2)
    if len(grouped) <= 12:
        axis.legend(fontsize=7)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cdf(rows: list[dict[str, str]], metric: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    values = sorted((number(row.get(metric)) or 0.0) * 1000.0 for row in ok_rows(rows, "gsm8k_latency") if number(row.get(metric)) is not None)
    if not values:
        save_empty(path, title)
        return
    y = [(i + 1) / len(values) for i in range(len(values))]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(values, y)
    axis.set_title(title)
    axis.set_xlabel("ms")
    axis.set_ylabel("CDF")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gsm8k_concurrency(rows: list[dict[str, str]], metric: str, path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    usable = [row for row in ok_rows(rows, "gsm8k_latency") if number(row.get(metric)) is not None]
    if not usable:
        save_empty(path, title)
        return
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in usable:
        label = f"{row.get('model_size')} {row.get('quantization')}"
        grouped[label].append((int(row.get("concurrency") or row.get("bsz") or 0), number(row.get(metric)) or 0.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    for label, points in sorted(grouped.items()):
        points = sorted(points)
        axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel("concurrency")
    axis.set_ylabel(ylabel)
    axis.set_xscale("log", base=2)
    if len(grouped) <= 12:
        axis.legend(fontsize=7)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_telemetry(rows: list[dict[str, str]], metric: str, path: Path, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    usable = [row for row in rows if number(row.get(metric)) is not None]
    if not usable:
        save_empty(path, title)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(range(len(usable)), [number(row.get(metric)) for row in usable])
    axis.set_title(title)
    axis.set_xlabel("sample")
    axis.set_ylabel(ylabel)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    default_root = Path(__file__).resolve().parents[2] / "results"
    parser = argparse.ArgumentParser(description="Plot README Task 5 benchmark outputs by benchmark kind.")
    parser.add_argument("--results-dir", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir / "plots"
    throughput = read_rows(results_dir / "task5_throughput.csv")
    latency_synthetic = read_rows(results_dir / "task5_latency_synthetic.csv")
    gsm8k_requests = read_rows(results_dir / "task5_latency_gsm8k_requests.csv")
    gsm8k_summary = read_rows(results_dir / "task5_latency_gsm8k_summary.csv")
    telemetry = read_rows(results_dir / "gpu_telemetry.csv")

    plot_heatmap(ok_rows(throughput, "synthetic_throughput"), "prefill_tps", output_dir / "throughput_prefill_heatmap.png", "Synthetic throughput prefill tokens/s")
    plot_heatmap(ok_rows(throughput, "synthetic_throughput"), "decode_tps", output_dir / "throughput_decode_heatmap.png", "Synthetic throughput decode tokens/s")
    plot_frontier(throughput, output_dir / "throughput_frontier.png")
    plot_pareto(throughput, output_dir / "throughput_pareto.png")

    plot_latency_line(latency_synthetic, "ttft_p95_s", output_dir / "latency_synthetic_ttft_p95.png", "Synthetic latency TTFT p95")
    plot_latency_line(latency_synthetic, "e2el_p95_s", output_dir / "latency_synthetic_e2el_p95.png", "Synthetic latency E2EL p95")
    plot_latency_line(latency_synthetic, "itl_p95_ms", output_dir / "latency_synthetic_itl_p95.png", "Synthetic latency ITL p95")

    plot_cdf(gsm8k_requests, "ttft_s", output_dir / "gsm8k_ttft_cdf.png", "GSM8K TTFT CDF")
    plot_cdf(gsm8k_requests, "e2el_s", output_dir / "gsm8k_e2el_cdf.png", "GSM8K E2EL CDF")
    plot_gsm8k_concurrency(gsm8k_summary, "requests_per_s", output_dir / "gsm8k_concurrency_reqps.png", "GSM8K request throughput", "requests/s")
    plot_gsm8k_concurrency(gsm8k_summary, "e2el_p95_s", output_dir / "gsm8k_concurrency_p95_latency.png", "GSM8K E2EL p95 latency", "s")

    plot_telemetry(telemetry, "gpu_util", output_dir / "gpu_util_time.png", "GPU utilization", "%")
    plot_telemetry(telemetry, "mem_used", output_dir / "gpu_vram_time.png", "GPU VRAM", "MiB")
    plot_telemetry(telemetry, "power_w", output_dir / "gpu_power_time.png", "GPU power", "W")


if __name__ == "__main__":
    main()
