export type BenchmarkStatus = "ok" | "failed" | "unsupported" | string;

export type MetricKey = "decodeTps" | "prefillTps" | "e2eTps";

export interface BenchmarkRow {
  id: string;
  benchmarkKind: string;
  modelId: string;
  modelLabel: string;
  modelSize: string;
  modelSizeB: number | null;
  paramGroup: string;
  repo: string;
  backend: string;
  runner: string;
  dtype: string;
  quantization: string;
  bsz: number;
  promptLen: number;
  decodeLen: number;
  status: BenchmarkStatus;
  error: string;
  prefillTps: number | null;
  decodeTps: number | null;
  e2eTps: number | null;
  gpuName: string;
  gpuUuid: string;
  sourcePath: string;
  command: string;
}

export interface ModelGroupOverride {
  id: string;
  label: string;
  models: string[];
}

export interface ModelGroupConfig {
  groups: ModelGroupOverride[];
}

export interface ModelGroup {
  id: string;
  label: string;
  models: string[];
}

export interface Task5Dataset {
  generatedAt: string;
  sourceRoot: string;
  sourceRoots: string[];
  rows: BenchmarkRow[];
  groups: ModelGroup[];
}

export interface SeriesPoint {
  x: number;
  y: number;
  row: BenchmarkRow;
}

export interface StatusPoint {
  x: number;
  status: BenchmarkStatus;
  row: BenchmarkRow;
}

export interface ChartSeries {
  name: string;
  points: SeriesPoint[];
  statusPoints: StatusPoint[];
}

export interface RaceSummary {
  fastest: BenchmarkRow | null;
  albatross: BenchmarkRow | null;
  albatrossRatio: number | null;
  statusCounts: {
    ok: number;
    failed: number;
    other: number;
  };
}
