import { describe, expect, it } from "vitest";
import { buildThroughputChartOption, FAILED_SYMBOL, logAxisMin } from "./throughput-chart";
import type { ChartSeries } from "@/lib/types";

describe("throughput chart option", () => {
  it("keeps failed marker series out of the legend", () => {
    const series: ChartSeries[] = [
      {
        name: "albatross",
        points: [{ x: 1, y: 100, row: null as never }],
        statusPoints: [{ x: 16, status: "failed", row: null as never }]
      },
      {
        name: "llama.cpp · fp16",
        points: [{ x: 1, y: 80, row: null as never }],
        statusPoints: [{ x: 16, status: "failed", row: null as never }]
      }
    ];

    const option = buildThroughputChartOption({
      metric: "decodeTps",
      series,
      selectedName: null
    });

    expect(option.legend.show).toBe(false);
    expect(option.legend.data).toEqual(["albatross", "llama.cpp · fp16"]);
    expect(option.series.map((item) => item.name)).toEqual(["albatross", "albatross", "llama.cpp · fp16", "llama.cpp · fp16"]);
  });

  it("uses round log-axis minima instead of long decimal labels", () => {
    expect(logAxisMin(5.61555216)).toBe(1);
    expect(logAxisMin(56)).toBe(10);
    expect(logAxisMin(null)).toBeUndefined();
  });

  it("renders failed points as same-color x markers instead of a separate red backend", () => {
    const series: ChartSeries[] = [
      {
        name: "web-rwkv",
        points: [{ x: 16, y: 100, row: null as never }],
        statusPoints: [{ x: 1, status: "failed", row: null as never }]
      }
    ];

    const option = buildThroughputChartOption({
      metric: "decodeTps",
      series,
      selectedName: null
    });
    const lineSeries = option.series[0];
    const failedSeries = option.series[1];

    expect(failedSeries.name).toBe("web-rwkv");
    expect(failedSeries.symbol).toBe(FAILED_SYMBOL);
    expect(failedSeries.itemStyle.color).toBe(lineSeries.itemStyle.color);
    expect(String(option.tooltip.formatter([
      { name: "1", seriesName: "web-rwkv", seriesType: "line", value: null, marker: "line-marker" },
      { name: "1", seriesName: "web-rwkv", seriesType: "scatter", data: { value: 50, status: "failed" }, marker: "failed-marker" }
    ]))).toContain(">web-rwkv</span><strong");
  });

  it("keeps a numeric tooltip value when a same-name failed marker is also present", () => {
    const series: ChartSeries[] = [
      {
        name: "albatross",
        points: [{ x: 128, y: 3122, row: null as never }],
        statusPoints: [{ x: 128, status: "failed", row: null as never }]
      }
    ];

    const option = buildThroughputChartOption({
      metric: "decodeTps",
      series,
      selectedName: null
    });
    const tooltip = String(option.tooltip.formatter([
      { name: "128", seriesName: "albatross", seriesType: "line", value: 3122, marker: "line-marker" },
      { name: "128", seriesName: "albatross", seriesType: "scatter", data: { value: 50, status: "failed" }, marker: "failed-marker" }
    ]));

    expect(tooltip).toContain(">albatross</span><strong");
    expect(tooltip).toContain("3,122 TPS</strong>");
    expect(tooltip).not.toContain("failed</strong>");
  });

  it("uses fixed tooltip columns so backend names and values do not touch", () => {
    const option = buildThroughputChartOption({
      metric: "decodeTps",
      series: [],
      selectedName: null
    });
    const tooltip = String(option.tooltip.formatter([
      { name: "1024", seriesName: "nano-vllm", seriesType: "line", value: 30860, marker: "marker" }
    ]));

    expect(tooltip).toContain("grid-template-columns:14px minmax(120px, 1fr) auto");
    expect(tooltip).toContain("column-gap:8px");
  });
});
