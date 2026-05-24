import { describe, expect, it } from "vitest";
import { parseTask5Csv, normalizeTask5Rows, buildModelGroups, dedupeBenchmarkRows } from "./ingest";

const CSV = `run_id,repo,backend,runner,benchmark_kind,task,model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,B,T,warmup,repeat,seed,status,error,input_tokens,measured_tokens,total_time_s,forward_time_s,sample_time_s,p10_ms,p50_ms,p90_ms,forward_sample_tps,entrypoint,measurement_boundary,command
ok-1,albatross,albatross-direct,direct,core_forward_sample_throughput,decode,1.5B,/weights/rwkv7-g1f-1.5b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,1,1,3,10,0,ok,,1,1,0.001,,,0.9,1.0,1.2,1000,forward_one+sampler_simple,"forward+sampler; no scheduler","python task5.py --x 1,2"
bad-1,nano-vllm,direct_engine,direct,core_forward_sample_throughput,batch_decode,1.5B,/weights/rwkv7-g1f-1.5b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,64,1,3,10,0,unsupported,no true batch decode,64,,,,,,,,,model_runner,"unsupported","python task5.py"
lat-1,llama.cpp,llama.cpp,server,synthetic_latency,prefill,1.5B,/weights/rwkv7-g1f-1.5b.gguf,gguf,cuda0,RTX 5090,GPU-a,fp16,fp16,1,1024,3,10,0,ok,,1024,1024,1,,,,2,,1024,server,"server","server"
`;

describe("Task 5 ingestion", () => {
  it("parses quoted CSV commands and keeps only core forward+sample rows", () => {
    const parsed = parseTask5Csv(CSV);
    const rows = normalizeTask5Rows(parsed, "fixture.csv");

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      benchmarkKind: "core_forward_sample_throughput",
      repo: "albatross",
      modelSize: "1.5B",
      modelSizeB: 1.5,
      paramGroup: "1-3B",
      task: "decode",
      status: "ok",
      B: 1,
      T: 1,
      forwardSampleTps: 1000,
      p50Ms: 1,
      entrypoint: "forward_one+sampler_simple",
      sourcePath: "fixture.csv"
    });
    expect(rows[0].command).toContain("--x 1,2");
    expect(rows[1].status).toBe("unsupported");
    expect(rows[1].forwardSampleTps).toBeNull();
  });

  it("keeps row ids unique when a CSV repeats run ids", () => {
    const parsed = parseTask5Csv(`${CSV}ok-1,albatross,albatross-direct,direct,core_forward_sample_throughput,prefill,1.5B,/weights/rwkv7-g1f-1.5b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,1,1024,3,10,0,ok,,1024,1024,0.5,,,45,50,60,2048,forward_seq,"forward+sampler","python task5.py"\n`);
    const rows = normalizeTask5Rows(parsed, "fixture.csv");

    expect(new Set(rows.map((row) => row.id)).size).toBe(rows.length);
  });

  it("uses model group overrides before automatic parameter buckets", () => {
    const rows = normalizeTask5Rows(parseTask5Csv(CSV), "fixture.csv");
    const grouped = buildModelGroups(rows, {
      groups: [{ id: "rwkv-1b-class", label: "RWKV 1B Class", models: ["rwkv7-g1f-1.5b"] }]
    });

    expect(grouped.rows[0].paramGroup).toBe("rwkv-1b-class");
    expect(grouped.groups.find((group) => group.id === "rwkv-1b-class")?.label).toBe("RWKV 1B Class");
  });

  it("prefers the newest duplicate benchmark row over stale ok rows", () => {
    const [oldRow] = normalizeTask5Rows(parseTask5Csv(CSV), "old.csv");
    const newerFailed = { ...oldRow, id: "newer", status: "failed", forwardSampleTps: null, endedAt: "2026-05-17T05:00:00Z", sourcePath: "newer.csv" };
    const olderOk = { ...oldRow, id: "older", gpuUuid: "GPU-b", endedAt: "2026-05-17T04:00:00Z", sourcePath: "older.csv" };

    expect(dedupeBenchmarkRows([olderOk, newerFailed])).toEqual([newerFailed]);
  });

  it("does not ingest legacy albatross forward_batch rows as task-native batch prefill results", () => {
    const legacyBatchPrefill = `run_id,repo,backend,runner,benchmark_kind,task,model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,B,T,warmup,repeat,seed,status,error,input_tokens,measured_tokens,total_time_s,forward_time_s,sample_time_s,p10_ms,p50_ms,p90_ms,forward_sample_tps,entrypoint,measurement_boundary,command
legacy-bp,albatross,albatross-faster2_251201,direct,core_forward_sample_throughput,batch_prefill,7.2B,/weights/rwkv7-g1f-7.2b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,32,32,3,10,0,ok,CUDA illegal memory access,1024,1024,0.1,,,90,100,110,10240,RWKV_x070.forward_batch+sampler_simple_batch,"forward+sampler","python task5.py"
`;

    const [row] = normalizeTask5Rows(parseTask5Csv(legacyBatchPrefill), "legacy.csv");

    expect(row.status).toBe("unsupported");
    expect(row.forwardSampleTps).toBeNull();
    expect(row.error).toContain("not task-native BnTn");
    expect(row.error).not.toContain("illegal memory access");
  });
});
