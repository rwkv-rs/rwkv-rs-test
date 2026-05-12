import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { loadTask5Dataset } from "../src/lib/files";

const rootArg = process.argv.find((arg) => arg.startsWith("--root="));
const rootsArg = process.argv.find((arg) => arg.startsWith("--roots="));
const sourceRoot = rootsArg
  ? rootsArg.slice("--roots=".length).split(",").map((item) => path.resolve(item)).join(",")
  : rootArg
    ? path.resolve(rootArg.slice("--root=".length))
    : undefined;
const output = path.resolve(process.cwd(), "public", "data", "task5.json");

async function main() {
  const dataset = await loadTask5Dataset(sourceRoot);
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(dataset, null, 2), "utf8");
  console.log(`wrote ${output}`);
  console.log(`${dataset.rows.length} rows, ${dataset.groups.length} model groups`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
