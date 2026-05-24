import { describe, expect, it } from "vitest";
import { computeRaceSummary, displayModelName, filterRows, formatTimestamp, makeSeries, seriesNameForRow, taskAxisDescription, xAxisNameForTask, xValueForTask } from "./analytics";
import type { BenchmarkRow } from "./types";

const base: BenchmarkRow = {
  id: "base",
  benchmarkKind: "core_forward_sample_throughput",
  modelId: "rwkv7-g1f-1.5b",
  modelLabel: "RWKV7 G1F 1.5B",
  modelSize: "1.5B",
  modelSizeB: 1.5,
  paramGroup: "1-3B",
  repo: "albatross",
  backend: "albatross-direct",
  runner: "direct",
  task: "batch_decode",
  dtype: "fp16",
  quantization: "fp16",
  B: 16,
  T: 1,
  status: "ok",
  error: "",
  inputTokens: 16,
  measuredTokens: 16,
  totalTimeS: 0.01,
  forwardTimeS: null,
  sampleTimeS: null,
  p50Ms: 10,
  forwardSampleTps: 2000,
  entrypoint: "forward_batch+sampler_simple_batch",
  measurementBoundary: "forward+sampler; no scheduler",
  gpuName: "RTX 5090",
  gpuUuid: "GPU-a",
  sourcePath: "a.csv",
  command: ""
};

describe("benchmark analytics", () => {
  it("shortens rwkv model ids for display", () => {
    expect(displayModelName("rwkv7-g1f-13.3b-20260415-ctx8192")).toBe("rwkv-g1f-13.3b");
    expect(displayModelName("rwkv7-g1f-1.5b-20260419-ctx8192")).toBe("rwkv-g1f-1.5b");
    expect(displayModelName("rwkv7-g1f-0.1b")).toBe("rwkv-g1f-0.1b");
  });

  it("formats timestamps without locale-dependent hydration output", () => {
    expect(formatTimestamp("1970-01-01T00:00:00.000Z")).toBe("00:00:00");
    expect(formatTimestamp("2026-05-10T17:28:12.123Z")).toBe("17:28:12");
  });

  it("ranks fastest ok rows by throughput and compares to albatross", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "nano", repo: "nano-vllm", backend: "direct", forwardSampleTps: 5000, sourcePath: "n.csv" },
      { ...base, id: "fail", repo: "llama.cpp", backend: "bench", status: "failed", forwardSampleTps: null, error: "OOM" },
      { ...base, id: "unsupported", repo: "llama.cpp", backend: "bench", status: "unsupported", forwardSampleTps: null, error: "ctx" }
    ];

    const summary = computeRaceSummary(rows, "forwardSampleTps");

    expect(summary.fastest?.repo).toBe("nano-vllm");
    expect(summary.albatrossRatio).toBe(2.5);
    expect(summary.statusCounts).toEqual({ ok: 2, failed: 2, other: 0 });
  });

  it("ranks p50 by lower latency", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "fast", repo: "nano-vllm", p50Ms: 4, forwardSampleTps: 1000 }
    ];

    expect(computeRaceSummary(rows, "p50Ms").fastest?.id).toBe("fast");
  });

  it("uses one failed status bucket for failed and unsupported filters", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "failed", status: "failed", forwardSampleTps: null },
      { ...base, id: "unsupported", status: "unsupported", forwardSampleTps: null }
    ];

    const filtered = filterRows(rows, { metric: "forwardSampleTps", mode: "backend", status: "failed" });

    expect(filtered.map((row) => row.id)).toEqual(["failed", "unsupported"]);
  });

  it("uses task-aware x values", () => {
    expect(xValueForTask({ ...base, task: "decode", B: 1, T: 1 })).toBe(1);
    expect(xValueForTask({ ...base, task: "prefill", B: 1, T: 1024 })).toBe(1024);
    expect(xValueForTask({ ...base, task: "batch_decode", B: 64, T: 1 })).toBe(64);
    expect(xValueForTask({ ...base, task: "batch_prefill", B: 32, T: 32 })).toBe(1024);
    expect(xAxisNameForTask("batch_prefill")).toBe("B*T (BnTn tokens)");
    expect(taskAxisDescription("prefill")).toContain("B1Tn");
  });

  it("does not turn failed rows into zero-valued chart points", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "nano", repo: "nano-vllm", B: 64, forwardSampleTps: 6000 },
      { ...base, id: "bad", repo: "llama.cpp", B: 64, status: "unsupported", forwardSampleTps: null }
    ];

    const series = makeSeries(rows, "forwardSampleTps", (row) => row.repo);

    expect(series.find((item) => item.name === "llama.cpp")?.points).toEqual([]);
    expect(series.find((item) => item.name === "llama.cpp")?.statusPoints).toEqual([
      { x: 64, status: "failed", row: rows[2] }
    ]);
    expect(series.find((item) => item.name === "albatross")?.points).toEqual([{ x: 16, y: 2000, row: base }]);
  });

  it("aggregates duplicate points for the same series and task x value", () => {
    const rows: BenchmarkRow[] = [
      { ...base, id: "b-16", B: 16, forwardSampleTps: 100 },
      { ...base, id: "b-16b", B: 16, forwardSampleTps: 200 },
      { ...base, id: "b-16c", B: 16, forwardSampleTps: 900 },
      { ...base, id: "failed", B: 16, status: "failed", forwardSampleTps: null, error: "OOM" }
    ];

    const series = makeSeries(rows, "forwardSampleTps", (row) => row.repo);

    expect(series.find((item) => item.name === "albatross")?.points).toEqual([
      { x: 16, y: 200, row: rows[1] }
    ]);
    expect(series.find((item) => item.name === "albatross")?.statusPoints).toEqual([]);
  });

  it("splits backend lines by quantization when a backend has multiple precisions", () => {
    const fp16 = { ...base, id: "llama-fp16", repo: "llama.cpp", backend: "bench", quantization: "fp16", forwardSampleTps: 1000 };
    const q8 = { ...base, id: "llama-q8", repo: "llama.cpp", backend: "bench", quantization: "q8_0", forwardSampleTps: 1200 };
    const rows: BenchmarkRow[] = [base, fp16, q8];

    const series = makeSeries(rows, "forwardSampleTps", (row) => seriesNameForRow(row, rows, "backend", "model"));

    expect(series.map((item) => item.name).sort()).toEqual(["albatross-direct", "bench · fp16", "bench · q8_0"]);
    expect(series.find((item) => item.name === "llama.cpp")).toBeUndefined();
  });

  it("uses shortened model names and concrete backend names in model comparison series", () => {
    const row = {
      ...base,
      modelId: "rwkv7-g1f-13.3b-20260415-ctx8192",
      modelLabel: "rwkv7-g1f-13.3b-20260415-ctx8192"
    };

    expect(seriesNameForRow(row, [row], "model", "model")).toBe("rwkv-g1f-13.3b · albatross-direct");
  });
});
