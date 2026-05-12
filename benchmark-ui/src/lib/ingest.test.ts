import { describe, expect, it } from "vitest";
import { parseTask5Csv, normalizeTask5Rows, buildModelGroups } from "./ingest";

const CSV = `run_id,repo,backend,runner,benchmark_kind,model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,bsz,prompt_len,decode_len,status,error,prefill_tps,decode_tps,e2e_tps,command
ok-1,albatross,albatross-direct,direct,synthetic_throughput,1.5B,/weights/rwkv7-g1f-1.5b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,16,1024,16,ok,,1200,3400,2200,"python task5.py --x 1,2"
bad-1,nano-vllm,direct_engine,direct,synthetic_throughput,1.5B,/weights/rwkv7-g1f-1.5b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,16,1024,16,failed,OOM,0,0,0,"python task5.py"
lat-1,llama.cpp,llama.cpp,server,synthetic_latency,1.5B,/weights/rwkv7-g1f-1.5b.gguf,gguf,cuda0,RTX 5090,GPU-a,fp16,fp16,16,1024,16,ok,,1,2,3,"server"
`;

describe("Task 5 ingestion", () => {
  it("parses quoted CSV commands and keeps only synthetic throughput rows", () => {
    const parsed = parseTask5Csv(CSV);
    const rows = normalizeTask5Rows(parsed, "fixture.csv");

    expect(rows).toHaveLength(2);
    expect(rows[0]).toMatchObject({
      benchmarkKind: "synthetic_throughput",
      repo: "albatross",
      modelSize: "1.5B",
      modelSizeB: 1.5,
      paramGroup: "1-3B",
      status: "ok",
      bsz: 16,
      promptLen: 1024,
      decodeTps: 3400,
      sourcePath: "fixture.csv"
    });
    expect(rows[0].command).toContain("--x 1,2");
    expect(rows[1].status).toBe("failed");
    expect(rows[1].decodeTps).toBeNull();
  });

  it("keeps row ids unique when a CSV repeats run ids", () => {
    const parsed = parseTask5Csv(`${CSV}ok-1,albatross,albatross-direct,direct,synthetic_throughput,1.5B,/weights/rwkv7-g1f-1.5b.pth,pth,cuda0,RTX 5090,GPU-a,fp16,fp16,64,1024,16,ok,,1200,3400,2200,"python task5.py"\n`);
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
});
