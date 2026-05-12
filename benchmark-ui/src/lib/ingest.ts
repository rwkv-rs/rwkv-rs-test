import { parse } from "csv-parse/sync";
import type { BenchmarkRow, ModelGroup, ModelGroupConfig, ModelGroupOverride, Task5Dataset } from "./types";

type RawRow = Record<string, string>;

const DEFAULT_GROUPS: ModelGroup[] = [
  { id: "0-1B", label: "0-1B", models: [] },
  { id: "1-3B", label: "1-3B", models: [] },
  { id: "3-8B", label: "3-8B", models: [] },
  { id: "8-15B", label: "8-15B", models: [] },
  { id: "15B+", label: "15B+", models: [] }
];

export function parseTask5Csv(content: string): RawRow[] {
  return parse(content, {
    columns: true,
    skip_empty_lines: true,
    bom: true,
    relax_column_count: true,
    trim: false
  }) as RawRow[];
}

export function normalizeTask5Rows(rows: RawRow[], sourcePath: string): BenchmarkRow[] {
  return rows
    .filter((row) => row.benchmark_kind === "synthetic_throughput")
    .map((row, index) => {
      const modelSize = row.model_size || inferModelSize(row.model_path) || "unknown";
      const modelSizeB = parseModelSizeB(modelSize);
      const modelId = inferModelId(row.model_path, modelSize, row.repo);
      const status = row.status || "unknown";
      return {
        id: `${sourcePath}:${row.run_id || "row"}:${index}`,
        benchmarkKind: row.benchmark_kind,
        modelId,
        modelLabel: modelIdToLabel(modelId, modelSize),
        modelSize,
        modelSizeB,
        paramGroup: bucketForSize(modelSizeB),
        repo: row.repo || "unknown",
        backend: row.backend || row.repo || "unknown",
        runner: row.runner || "",
        dtype: row.dtype || "",
        quantization: row.quantization || "",
        bsz: intOrZero(row.bsz),
        promptLen: intOrZero(row.prompt_len),
        decodeLen: intOrZero(row.decode_len),
        status,
        error: row.error || "",
        prefillTps: metricOrNull(row.prefill_tps, status),
        decodeTps: metricOrNull(row.decode_tps, status),
        e2eTps: metricOrNull(row.e2e_tps, status),
        gpuName: row.gpu_name || "",
        gpuUuid: row.gpu_uuid || "",
        sourcePath,
        command: row.command || ""
      };
    });
}

export function buildModelGroups(rows: BenchmarkRow[], config?: ModelGroupConfig): { rows: BenchmarkRow[]; groups: ModelGroup[] } {
  const overrides = config?.groups ?? [];
  const overrideByModel = new Map<string, ModelGroupOverride>();
  for (const group of overrides) {
    for (const model of group.models) {
      overrideByModel.set(normalizeModelKey(model), group);
    }
  }

  const nextRows = rows.map((row) => {
    const override = overrideByModel.get(normalizeModelKey(row.modelId));
    return override ? { ...row, paramGroup: override.id } : row;
  });

  const groupsById = new Map<string, ModelGroup>();
  for (const group of DEFAULT_GROUPS) {
    groupsById.set(group.id, { ...group, models: [] });
  }
  for (const group of overrides) {
    groupsById.set(group.id, { id: group.id, label: group.label, models: [] });
  }
  for (const row of nextRows) {
    const group = groupsById.get(row.paramGroup) ?? { id: row.paramGroup, label: row.paramGroup, models: [] };
    if (!group.models.includes(row.modelId)) {
      group.models.push(row.modelId);
    }
    groupsById.set(group.id, group);
  }

  const groups = Array.from(groupsById.values())
    .filter((group) => group.models.length > 0)
    .sort((a, b) => groupRank(a.id) - groupRank(b.id) || a.label.localeCompare(b.label));
  return { rows: nextRows, groups };
}

export function buildDataset(rows: BenchmarkRow[], sourceRoot: string, config?: ModelGroupConfig): Task5Dataset {
  const grouped = buildModelGroups(rows, config);
  return {
    generatedAt: new Date().toISOString(),
    sourceRoot,
    sourceRoots: sourceRoot.split(",").map((value) => value.trim()).filter(Boolean),
    rows: grouped.rows,
    groups: grouped.groups
  };
}

export function parseModelSizeB(value: string): number | null {
  const match = value.match(/([\d.]+)\s*b/i);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) ? parsed : null;
}

export function bucketForSize(size: number | null): string {
  if (size === null) {
    return "unknown";
  }
  if (size < 1) {
    return "0-1B";
  }
  if (size < 3) {
    return "1-3B";
  }
  if (size < 8) {
    return "3-8B";
  }
  if (size < 15) {
    return "8-15B";
  }
  return "15B+";
}

function inferModelSize(modelPath?: string): string {
  const match = modelPath?.match(/([\d.]+b)/i);
  return match ? match[1].toUpperCase() : "";
}

function inferModelId(modelPath: string | undefined, modelSize: string, repo: string | undefined): string {
  const filename = modelPath?.split(/[\\/]/).pop() ?? "";
  const withoutExtension = filename.replace(/\.(pth|gguf|st|safetensors)$/i, "");
  const withoutQuant = withoutExtension.replace(/-(FP16|Q[4568]_K_M|Q[68]_K|Q8_0|Q4_K_M|Q5_K_M)$/i, "");
  if (withoutQuant) {
    return withoutQuant.toLowerCase();
  }
  return `${repo || "model"}-${modelSize}`.toLowerCase();
}

function modelIdToLabel(modelId: string, modelSize: string): string {
  if (modelId.includes(modelSize.toLowerCase())) {
    return modelId;
  }
  return `${modelId} (${modelSize})`;
}

function normalizeModelKey(value: string): string {
  return value.toLowerCase().replace(/\.(pth|gguf|st|safetensors)$/i, "");
}

function intOrZero(value: string | undefined): number {
  const parsed = Number.parseInt(value || "", 10);
  return Number.isFinite(parsed) ? parsed : 0;
}

function metricOrNull(value: string | undefined, status: string): number | null {
  if (status !== "ok") {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function groupRank(groupId: string): number {
  const index = DEFAULT_GROUPS.findIndex((group) => group.id === groupId);
  return index === -1 ? 999 : index;
}
