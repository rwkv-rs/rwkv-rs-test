export type BenchmarkStatus = "ok" | "failed" | "unsupported" | string;

export type MetricKey = "forwardSampleTps" | "p50Ms";
export type BenchmarkTask = "decode" | "prefill" | "batch_decode" | "batch_prefill" | string;

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
  task: BenchmarkTask;
  dtype: string;
  quantization: string;
  B: number;
  T: number;
  status: BenchmarkStatus;
  error: string;
  inputTokens: number;
  measuredTokens: number;
  totalTimeS: number | null;
  forwardTimeS: number | null;
  sampleTimeS: number | null;
  p50Ms: number | null;
  forwardSampleTps: number | null;
  entrypoint: string;
  measurementBoundary: string;
  gpuName: string;
  gpuUuid: string;
  startedAt?: string;
  endedAt?: string;
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
