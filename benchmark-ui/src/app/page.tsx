import { BenchmarkExplorer } from "@/components/benchmark-explorer";
import { loadTask5Dataset } from "@/lib/files";

export const dynamic = "force-dynamic";

export default async function Page() {
  const dataset = await loadTask5Dataset();
  return <BenchmarkExplorer dataset={dataset} />;
}
