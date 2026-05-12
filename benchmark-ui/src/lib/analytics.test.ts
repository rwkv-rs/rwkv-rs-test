import { describe, expect, it } from "vitest";
import { computeRaceSummary, displayModelName, filterRows, formatTimestamp, makeSeries, seriesNameForRow } from "./analytics";
import type { BenchmarkRow } from "./types";

const base: BenchmarkRow = {
  id: "base",
  benchmarkKind: "synthetic_throughput",
  modelId: "rwkv7-g1f-1.5b",
  modelLabel: "RWKV7 G1F 1.5B",
  modelSize: "1.5B",
  modelSizeB: 1.5,
  paramGroup: "1-3B",
  repo: "albatross",
  backend: "albatross-direct",
  runner: "direct",
  dtype: "fp16",
  quantization: "fp16",
  bsz: 16,
  promptLen: 1024,
  decodeLen: 16,
  status: "ok",
  error: "",
  prefillTps: 1000,
  decodeTps: 2000,
  e2eTps: 1500,
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

  it("ranks fastest ok rows by selected metric and compares to albatross", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "nano", repo: "nano-vllm", backend: "direct", decodeTps: 5000, sourcePath: "n.csv" },
      { ...base, id: "fail", repo: "llama.cpp", backend: "bench", status: "failed", decodeTps: null, error: "OOM" },
      { ...base, id: "unsupported", repo: "llama.cpp", backend: "bench", status: "unsupported", decodeTps: null, error: "ctx" }
    ];

    const summary = computeRaceSummary(rows, "decodeTps");

    expect(summary.fastest?.repo).toBe("nano-vllm");
    expect(summary.albatrossRatio).toBe(2.5);
    expect(summary.statusCounts).toEqual({ ok: 2, failed: 2, other: 0 });
  });

  it("uses one failed status bucket for failed and unsupported filters", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "failed", status: "failed", decodeTps: null },
      { ...base, id: "unsupported", status: "unsupported", decodeTps: null }
    ];

    const filtered = filterRows(rows, { metric: "decodeTps", mode: "backend", status: "failed" });

    expect(filtered.map((row) => row.id)).toEqual(["failed", "unsupported"]);
  });

  it("does not turn failed rows into zero-valued chart points", () => {
    const rows: BenchmarkRow[] = [
      base,
      { ...base, id: "nano", repo: "nano-vllm", bsz: 64, decodeTps: 6000 },
      { ...base, id: "bad", repo: "llama.cpp", bsz: 64, status: "unsupported", decodeTps: null }
    ];

    const series = makeSeries(rows, "decodeTps", (row) => row.repo);

    expect(series.find((item) => item.name === "llama.cpp")?.points).toEqual([]);
    expect(series.find((item) => item.name === "llama.cpp")?.statusPoints).toEqual([
      { x: 64, status: "failed", row: rows[2] }
    ]);
    expect(series.find((item) => item.name === "albatross")?.points).toEqual([{ x: 16, y: 2000, row: base }]);
  });

  it("collapses failed and unsupported rows into one failed chart marker per series and batch size", () => {
    const rows: BenchmarkRow[] = [
      { ...base, id: "failed", repo: "llama.cpp", status: "failed", decodeTps: null },
      { ...base, id: "unsupported", repo: "llama.cpp", status: "unsupported", decodeTps: null }
    ];

    const series = makeSeries(rows, "decodeTps", (row) => row.repo);

    expect(series.find((item) => item.name === "llama.cpp")?.statusPoints).toEqual([
      { x: 16, status: "failed", row: rows[0] }
    ]);
  });

  it("aggregates duplicate decode points for the same series and batch size across contexts", () => {
    const rows: BenchmarkRow[] = [
      { ...base, id: "ctx-16", promptLen: 16, decodeTps: 100 },
      { ...base, id: "ctx-512", promptLen: 512, decodeTps: 200 },
      { ...base, id: "ctx-1024", promptLen: 1024, decodeTps: 900 },
      { ...base, id: "ctx-4096", promptLen: 4096, status: "failed", decodeTps: null, error: "OOM" }
    ];

    const series = makeSeries(rows, "decodeTps", (row) => row.repo);

    expect(series.find((item) => item.name === "albatross")?.points).toEqual([
      { x: 16, y: 200, row: rows[1] }
    ]);
    expect(series.find((item) => item.name === "albatross")?.statusPoints).toEqual([]);
  });

  it("does not keep failed markers for a series and batch size that has an ok value", () => {
    const rows: BenchmarkRow[] = [
      { ...base, id: "ok-context", promptLen: 64, decodeTps: 1200 },
      { ...base, id: "failed-context", promptLen: 4096, status: "failed", decodeTps: null }
    ];

    const series = makeSeries(rows, "decodeTps", (row) => row.repo);

    expect(series.find((item) => item.name === "albatross")?.points).toEqual([
      { x: 16, y: 1200, row: rows[0] }
    ]);
    expect(series.find((item) => item.name === "albatross")?.statusPoints).toEqual([]);
  });

  it("splits backend lines by quantization when a backend has multiple precisions", () => {
    const fp16 = { ...base, id: "llama-fp16", repo: "llama.cpp", backend: "bench", quantization: "fp16", decodeTps: 1000 };
    const q8 = { ...base, id: "llama-q8", repo: "llama.cpp", backend: "bench", quantization: "q8_0", decodeTps: 1200 };
    const rows: BenchmarkRow[] = [base, fp16, q8];

    const series = makeSeries(rows, "decodeTps", (row) => seriesNameForRow(row, rows, "backend", "model"));

    expect(series.map((item) => item.name).sort()).toEqual(["albatross", "llama.cpp · fp16", "llama.cpp · q8_0"]);
    expect(series.find((item) => item.name === "llama.cpp")).toBeUndefined();
  });

  it("uses shortened model names in model comparison series", () => {
    const row = {
      ...base,
      modelId: "rwkv7-g1f-13.3b-20260415-ctx8192",
      modelLabel: "rwkv7-g1f-13.3b-20260415-ctx8192"
    };

    expect(seriesNameForRow(row, [row], "model", "model")).toBe("rwkv-g1f-13.3b · albatross");
  });
});
