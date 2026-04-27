#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path


PLOTS = [
    ("prefill_tps", "Prefill tokens/s", "task5_prefill_tps.png", True),
    ("decode_tps", "Decode tokens/s", "task5_decode_tps.png", True),
    ("ttft_s", "TTFT (ms)", "task5_ttft.png", False),
    ("e2el_s", "E2EL (ms)", "task5_e2el.png", False),
    ("itl_p95_ms", "ITL p95 (ms)", "task5_itl_p95.png", False),
]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_metric(rows: list[dict[str, str]], metric: str, ylabel: str, output: Path, log_y: bool) -> None:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib.pyplot as plt

    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        bsz = parse_float(row.get("bsz"))
        prompt_len = parse_float(row.get("prompt_len"))
        value = parse_float(row.get(metric))
        if bsz is None or prompt_len is None or value is None:
            continue
        if metric in {"ttft_s", "e2el_s"}:
            value *= 1000.0
        grouped[int(prompt_len)].append((int(bsz), value))

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for prompt_len in sorted(grouped):
        points = sorted(grouped[prompt_len])
        if not points:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, marker="o", label=f"prompt_len={prompt_len}")
        plotted = True

    ax.set_xlabel("bsz")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    if plotted:
        ax.legend(title="nano-vllm fp16")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot README task 5 nano-vllm benchmark CSV.")
    parser.add_argument("--csv", default="results/task5_nano_vllm_fp16.csv")
    parser.add_argument("--out-dir", default="results")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_rows(Path(args.csv))
    out_dir = Path(args.out_dir)
    for metric, ylabel, filename, log_y in PLOTS:
        plot_metric(rows, metric, ylabel, out_dir / filename, log_y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
