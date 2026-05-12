import type { BenchmarkRow, ChartSeries, MetricKey, RaceSummary, StatusPoint } from "./types";

export function metricValue(row: BenchmarkRow, metric: MetricKey): number | null {
  if (row.status !== "ok") {
    return null;
  }
  return row[metric];
}

export function normalizedStatus(status: string): "ok" | "failed" | "other" {
  if (status === "ok") {
    return "ok";
  }
  if (isFailedStatus(status)) {
    return "failed";
  }
  return "other";
}

export function displayModelName(modelId: string, fallback = modelId): string {
  const source = modelId || fallback;
  const rwkvMatch = source.match(/^rwkv\d*-(g[^-]+)-(\d+(?:\.\d+)?b)(?:-|$)/i);
  if (rwkvMatch) {
    return `rwkv-${rwkvMatch[1].toLowerCase()}-${rwkvMatch[2].toLowerCase()}`;
  }
  return source
    .replace(/-\d{8}(?=-|$).*/, "")
    .replace(/-ctx\d+$/i, "");
}

export function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "--:--:--";
  }
  return [date.getUTCHours(), date.getUTCMinutes(), date.getUTCSeconds()]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

export function computeRaceSummary(rows: BenchmarkRow[], metric: MetricKey): RaceSummary {
  const statusCounts = { ok: 0, failed: 0, other: 0 };
  for (const row of rows) {
    statusCounts[normalizedStatus(row.status)] += 1;
  }

  const okRows = rows.filter((row) => metricValue(row, metric) !== null);
  const fastest = okRows.reduce<BenchmarkRow | null>((best, row) => {
    if (!best) {
      return row;
    }
    return (metricValue(row, metric) ?? 0) > (metricValue(best, metric) ?? 0) ? row : best;
  }, null);
  const albatross = okRows.find((row) => row.repo === "albatross") ?? null;
  const fastestValue = fastest ? metricValue(fastest, metric) : null;
  const albatrossValue = albatross ? metricValue(albatross, metric) : null;

  return {
    fastest,
    albatross,
    albatrossRatio: fastestValue && albatrossValue ? roundRatio(fastestValue / albatrossValue) : null,
    statusCounts
  };
}

export function makeSeries(rows: BenchmarkRow[], metric: MetricKey, keyFn: (row: BenchmarkRow) => string): ChartSeries[] {
  const byName = new Map<string, { name: string; valueBuckets: Map<number, SeriesValue[]>; statusBuckets: Map<string, StatusPoint> }>();
  for (const row of rows) {
    const name = keyFn(row);
    const series = byName.get(name) ?? { name, valueBuckets: new Map<number, SeriesValue[]>(), statusBuckets: new Map<string, StatusPoint>() };
    const value = metricValue(row, metric);
    if (value !== null) {
      const bucket = series.valueBuckets.get(row.bsz) ?? [];
      bucket.push({ value, row });
      series.valueBuckets.set(row.bsz, bucket);
    } else if (isFailedStatus(row.status)) {
      const key = String(row.bsz);
      if (!series.statusBuckets.has(key)) {
        series.statusBuckets.set(key, { x: row.bsz, status: "failed", row });
      }
    }
    byName.set(name, series);
  }
  return Array.from(byName.values())
    .map((series) => ({
      name: series.name,
      points: Array.from(series.valueBuckets.entries())
        .map(([x, values]) => medianPoint(x, values))
        .sort((a, b) => a.x - b.x),
      statusPoints: Array.from(series.statusBuckets.values())
        .filter((point) => !series.valueBuckets.has(point.x))
        .sort((a, b) => a.x - b.x)
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

export function seriesNameForRow(row: BenchmarkRow, rows: BenchmarkRow[], mode: "backend" | "model", groupBy: "model" | "backend"): string {
  const needsQuant = backendHasMultipleQuantizations(row.repo, rows);
  const quantSuffix = needsQuant ? ` · ${row.quantization || row.dtype || "unknown"}` : "";
  if (mode === "backend") {
    return `${row.repo}${quantSuffix}`;
  }
  const modelName = displayModelName(row.modelId, row.modelLabel);
  const base = groupBy === "model" ? `${modelName} · ${row.repo}` : `${row.repo} · ${modelName}`;
  return `${base}${quantSuffix}`;
}

export function filterRows(
  rows: BenchmarkRow[],
  filters: {
    metric: MetricKey;
    mode: "backend" | "model";
    modelId?: string;
    paramGroup?: string;
    promptLen?: number;
    quantization?: string;
    repo?: string;
    status?: string;
  }
): BenchmarkRow[] {
  return rows.filter((row) => {
    if (row.benchmarkKind !== "synthetic_throughput") return false;
    if (filters.modelId && row.modelId !== filters.modelId) return false;
    if (filters.paramGroup && row.paramGroup !== filters.paramGroup) return false;
    if (filters.promptLen && row.promptLen !== filters.promptLen) return false;
    if (filters.quantization && row.quantization !== filters.quantization) return false;
    if (filters.repo && row.repo !== filters.repo) return false;
    if (filters.status && normalizedStatus(row.status) !== filters.status) return false;
    return true;
  });
}

export function uniqueSorted<T>(values: T[], rank?: (value: T) => number): T[] {
  return Array.from(new Set(values)).sort((a, b) => {
    if (rank) {
      return rank(a) - rank(b);
    }
    return String(a).localeCompare(String(b), undefined, { numeric: true });
  });
}

function roundRatio(value: number): number {
  return Math.round(value * 100) / 100;
}

function isFailedStatus(status: string): boolean {
  return status === "failed" || status === "unsupported";
}

interface SeriesValue {
  value: number;
  row: BenchmarkRow;
}

function medianPoint(x: number, values: SeriesValue[]) {
  const sorted = [...values].sort((a, b) => a.value - b.value);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) {
    return { x, y: sorted[middle].value, row: sorted[middle].row };
  }
  return {
    x,
    y: (sorted[middle - 1].value + sorted[middle].value) / 2,
    row: sorted[middle - 1].row
  };
}

function backendHasMultipleQuantizations(repo: string, rows: BenchmarkRow[]): boolean {
  const quantizations = new Set(
    rows
      .filter((row) => row.repo === repo && row.status === "ok")
      .map((row) => row.quantization || row.dtype || "unknown")
  );
  return quantizations.size > 1;
}
