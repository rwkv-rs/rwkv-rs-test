"use client";

import ReactECharts from "echarts-for-react";
import type { ChartSeries, MetricKey } from "@/lib/types";

const COLORS = ["#cc0000", "#1a1a1a", "#3a8a9a", "#b86030", "#5a8a3a", "#d4a017", "#6b7280", "#9333ea"];
const BSZ_TICKS = [1, 16, 64, 128, 256, 512, 1024];
export const FAILED_SYMBOL = "path://M-5,-5L5,5M5,-5L-5,5";

export function ThroughputChart({
  metric,
  series,
  selectedName,
  onSelectName
}: {
  metric: MetricKey;
  series: ChartSeries[];
  selectedName: string | null;
  onSelectName: (name: string) => void;
}) {
  const option = buildThroughputChartOption({ metric, series, selectedName });

  return (
    <div className="chartFrame">
      <div className="chartLegend" aria-label="Chart series">
        {series.map((item, index) => {
          const color = COLORS[index % COLORS.length];
          const isDimmed = Boolean(selectedName && selectedName !== item.name);
          return (
            <button
              key={item.name}
              type="button"
              className={isDimmed ? "dimmed" : ""}
              onClick={() => onSelectName(item.name)}
            >
              <span className="legendLine" style={{ backgroundColor: color }} />
              <span className="legendDot" style={{ backgroundColor: color }} />
              <span>{item.name}</span>
            </button>
          );
        })}
      </div>
      <ReactECharts
        option={option}
        notMerge
        lazyUpdate
        style={{ height: "100%", minHeight: 520 }}
        onEvents={{
          click: (params: { seriesName?: string }) => {
            if (params.seriesName && !params.seriesName.endsWith(" failed")) {
              onSelectName(params.seriesName);
            }
          }
        }}
      />
    </div>
  );
}

export function buildThroughputChartOption({
  metric,
  series,
  selectedName
}: {
  metric: MetricKey;
  series: ChartSeries[];
  selectedName: string | null;
}) {
  const xTicks = collectBszTicks(series);
  const statusY = statusMarkerY(series);
  return {
    color: COLORS,
    animationDuration: 260,
    grid: { left: 72, right: 28, top: 24, bottom: 76 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#fdfbf8",
      borderColor: "#1a1a1a",
      borderWidth: 2,
      textStyle: { color: "#1a1a1a", fontFamily: "Menlo, Consolas, monospace" },
      formatter: formatTooltip
    },
    legend: {
      show: false,
      data: series.map((item) => item.name),
      textStyle: { color: "#1a1a1a", fontWeight: 700 }
    },
    xAxis: {
      type: "category",
      data: xTicks,
      name: "bsz",
      nameLocation: "middle",
      nameGap: 42,
      axisLabel: {
        interval: 0,
        fontFamily: "Menlo, Consolas, monospace"
      },
      splitLine: { show: true, lineStyle: { color: "#e7dfd5" } }
    },
    yAxis: {
      type: "log",
      name: metric === "decodeTps" ? "decode TPS" : "prefill TPS",
      nameLocation: "middle",
      nameGap: 54,
      min: logAxisMin(statusY),
      axisLabel: { fontFamily: "Menlo, Consolas, monospace" },
      splitLine: { lineStyle: { color: "#e7dfd5" } }
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0 },
      { type: "slider", xAxisIndex: 0, bottom: 18, height: 22, borderColor: "#1a1a1a" }
    ],
    series: series.flatMap((item, index) => {
      const isDimmed = Boolean(selectedName && selectedName !== item.name);
      const color = COLORS[index % COLORS.length];
      const lineSeries = {
        name: item.name,
        type: "line",
        symbol: "circle",
        symbolSize: isDimmed ? 5 : 8,
        lineStyle: { width: isDimmed ? 1.2 : 3, opacity: isDimmed ? 0.25 : 1, color },
        itemStyle: { color, opacity: isDimmed ? 0.35 : 1 },
        emphasis: { focus: "series" },
        data: xTicks.map((bsz) => item.points.find((point) => point.x === bsz)?.y ?? null)
      };
      if (!statusY || item.statusPoints.length === 0) {
        return [lineSeries];
      }
      const failedSeries = {
        name: item.name,
        type: "scatter",
        symbol: FAILED_SYMBOL,
        symbolSize: isDimmed ? 11 : 15,
        showInLegend: false,
        itemStyle: { color, opacity: isDimmed ? 0.35 : 1 },
        data: xTicks.map((bsz) => {
          const point = item.statusPoints.find((candidate) => candidate.x === bsz && candidate.status === "failed");
          return point ? { value: statusY, status: "failed" } : null;
        })
      };
      return [lineSeries, failedSeries];
    })
  };
}

function formatTooltip(params: unknown): string {
  const items = Array.isArray(params) ? params : [params];
  const title = escapeHtml(String(readParam(items[0], "name") ?? ""));
  const byName = new Map<string, { marker: string; value: number | null; failed: boolean }>();
  for (const item of items) {
    const seriesName = String(readParam(item, "seriesName") ?? "");
    if (!seriesName) {
      continue;
    }
    const marker = String(readParam(item, "marker") ?? "");
    const value = readParam(item, "value");
    const data = readParam(item, "data");
    const failed = isFailedData(data);
    const current = byName.get(seriesName);
    if (failed) {
      if (current && current.value !== null) {
        continue;
      }
      byName.set(seriesName, { marker, value: null, failed: true });
    } else if (typeof value === "number" && Number.isFinite(value)) {
      byName.set(seriesName, { marker, value, failed: false });
    }
  }
  const lines = Array.from(byName.entries()).map(([name, item]) => {
    const value = item.failed ? "failed" : `${Math.round(item.value ?? 0).toLocaleString()} TPS`;
    return `<div style="display:grid;grid-template-columns:14px minmax(120px, 1fr) auto;column-gap:8px;align-items:center;margin:4px 0;">${item.marker}<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(name)}</span><strong style="text-align:right;white-space:nowrap;">${value}</strong></div>`;
  });
  return `<div style="min-width:230px;"><div style="margin-bottom:6px;font-weight:700;">${title}</div>${lines.join("")}</div>`;
}

function readParam(source: unknown, key: string): unknown {
  if (!source || typeof source !== "object") {
    return undefined;
  }
  return (source as Record<string, unknown>)[key];
}

function isFailedData(data: unknown): boolean {
  return Boolean(data && typeof data === "object" && (data as { status?: unknown }).status === "failed");
}

function escapeHtml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function collectBszTicks(series: ChartSeries[]): number[] {
  const values = new Set(BSZ_TICKS);
  for (const item of series) {
    for (const point of item.points) {
      values.add(point.x);
    }
    for (const point of item.statusPoints) {
      values.add(point.x);
    }
  }
  return Array.from(values).sort((a, b) => a - b);
}

function statusMarkerY(series: ChartSeries[]): number | null {
  const values = series.flatMap((item) => item.points.map((point) => point.y)).filter((value) => value > 0);
  if (values.length === 0) {
    return 1;
  }
  return Math.min(...values) / 2;
}

export function logAxisMin(value: number | null): number | undefined {
  if (!value || value <= 0) {
    return undefined;
  }
  return 10 ** Math.floor(Math.log10(value));
}
