#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


PLOTS = [
    ("prefill_tps", "task5_prefill_tps.png", "Prefill tokens/s", True, 1.0),
    ("decode_tps", "task5_decode_tps.png", "Decode tokens/s", True, 1.0),
    ("ttft_s", "task5_ttft.png", "TTFT ms", False, 1000.0),
    ("e2el_s", "task5_e2el.png", "E2EL ms", False, 1000.0),
    ("itl_p95_ms", "task5_itl_p95.png", "ITL p95 ms", False, 1.0),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def plot_metric(rows: list[dict[str, Any]], metric: str, output: Path, ylabel: str, log_y: bool, scale: float) -> None:
    import matplotlib.pyplot as plt

    usable = []
    for row in rows:
        value = numeric(row.get(metric))
        accepted_statuses = {"ok"}
        if metric in {"decode_tps", "time_per_output_token_ms", "itl_mean_ms", "itl_p50_ms", "itl_p90_ms", "itl_p95_ms", "itl_p99_ms"}:
            accepted_statuses.add("decode_only")
        if row.get("status") not in accepted_statuses or value is None:
            continue
        usable.append(
            {
                "bsz": int(row["bsz"]),
                "prompt_len": int(row["prompt_len"]),
                "label": f"{row['backend']} {row['quantization']}",
                "value": value * scale,
            }
        )

    prompt_lens = sorted({row["prompt_len"] for row in usable})
    output.parent.mkdir(parents=True, exist_ok=True)
    if not prompt_lens:
        plt.figure(figsize=(8, 4))
        plt.title(f"No successful rows for {metric}")
        plt.savefig(output, dpi=180, bbox_inches="tight")
        plt.close()
        return

    fig, axes = plt.subplots(len(prompt_lens), 1, figsize=(10, max(3.2, 2.6 * len(prompt_lens))), sharex=True)
    if len(prompt_lens) == 1:
        axes = [axes]

    for axis, prompt_len in zip(axes, prompt_lens):
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in usable:
            if row["prompt_len"] == prompt_len:
                grouped[row["label"]].append((row["bsz"], row["value"]))
        for label, points in sorted(grouped.items()):
            points = sorted(points)
            axis.plot([point[0] for point in points], [point[1] for point in points], marker="o", linewidth=1.5, label=label)
        axis.set_title(f"prompt_len={prompt_len}")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        if log_y:
            axis.set_yscale("log")
        axis.set_xscale("log", base=2)

    axes[-1].set_xlabel("bsz")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot README Task 5 albatross benchmark CSV.")
    parser.add_argument("--csv", type=Path, default=Path("results/task5_albatross_fp16.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = load_rows(args.csv)
    for metric, filename, ylabel, log_y, scale in PLOTS:
        plot_metric(rows, metric, args.output_dir / filename, ylabel, log_y, scale)


if __name__ == "__main__":
    main()
