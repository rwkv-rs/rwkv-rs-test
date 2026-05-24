import { promises as fs } from "node:fs";
import path from "node:path";
import { buildDataset, normalizeTask5Rows, parseTask5Csv } from "./ingest";
import type { BenchmarkRow, ModelGroupConfig, Task5Dataset } from "./types";

const DEFAULT_SOURCE_ROOT = path.resolve(process.cwd(), "..", "results", "core-forward-sample", "merged-current");

export async function loadTask5Dataset(sourceRoot = process.env.TASK5_RESULTS_ROOT || process.env.TASK5_RESULTS_ROOTS || DEFAULT_SOURCE_ROOT): Promise<Task5Dataset> {
  const config = await readModelConfig();
  const roots = parseSourceRoots(sourceRoot);
  if (roots.length !== 1 && process.env.TASK5_ALLOW_MIXED_ROOTS !== "1") {
    throw new Error(`Task5 frontend data must come from exactly one result batch root; got ${roots.length}: ${roots.join(", ")}`);
  }
  const files = (await Promise.all(roots.map((root) => findTask5CsvFiles(root)))).flat();
  const rows: BenchmarkRow[] = [];
  for (const file of Array.from(new Set(files)).sort()) {
    if (path.basename(file).includes("probe") || file.split(path.sep).some((segment) => segment.includes("smoke"))) {
      continue;
    }
    const content = await fs.readFile(file, "utf8");
    if (!content.trim()) {
      continue;
    }
    const relative = path.relative(process.cwd(), file);
    const rawRows = parseTask5Csv(content);
    if (isLegacyUnwiredWebRwkvRows(rawRows)) {
      continue;
    }
    rows.push(...normalizeTask5Rows(rawRows, relative));
  }
  return buildDataset(rows, roots.join(","), config);
}

export async function findTask5CsvFiles(root: string): Promise<string[]> {
  const results: string[] = [];
  async function walk(dir: string, depth: number): Promise<void> {
    if (depth > 8) {
      return;
    }
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        await walk(fullPath, depth + 1);
      } else if (/task5.*\.csv$/i.test(entry.name)) {
        results.push(fullPath);
      }
    }
  }
  await walk(root, 0);
  return results.sort();
}

async function readModelConfig(): Promise<ModelGroupConfig | undefined> {
  const configPath = path.join(process.cwd(), "models.json");
  try {
    return JSON.parse(await fs.readFile(configPath, "utf8")) as ModelGroupConfig;
  } catch {
    return undefined;
  }
}

function parseSourceRoots(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => path.resolve(item));
}

function isLegacyUnwiredWebRwkvRows(rows: Record<string, string>[]): boolean {
  return rows.length > 0 && rows.every((row) => (
    row.repo === "web-rwkv"
    && row.backend === "web-rwkv"
    && row.status !== "ok"
    && row.error.includes("direct core forward+sample timing is not wired yet")
  ));
}
