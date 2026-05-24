"use client";

import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, BarChart3, CheckCircle2, Database, Gauge, RefreshCcw, Trophy } from "lucide-react";
import type { BenchmarkRow, BenchmarkTask, MetricKey, Task5Dataset } from "@/lib/types";
import { computeRaceSummary, displayModelName, filterRows, formatTimestamp, makeSeries, normalizedStatus, seriesNameForRow, taskAxisDescription, uniqueSorted } from "@/lib/analytics";
import { ThroughputChart } from "./throughput-chart";

type Mode = "backend" | "model";
type GroupBy = "model" | "backend";

const METRICS: { key: MetricKey; label: string }[] = [
  { key: "forwardSampleTps", label: "Forward+Sample TPS" },
  { key: "p50Ms", label: "p50 ms" }
];

const TASKS: { key: BenchmarkTask; label: string }[] = [
  { key: "decode", label: "Decode" },
  { key: "prefill", label: "Prefill" },
  { key: "batch_decode", label: "Batch Decode" },
  { key: "batch_prefill", label: "Batch Prefill" }
];

export function BenchmarkExplorer({ dataset }: { dataset: Task5Dataset | null }) {
  const [data, setData] = useState<Task5Dataset>(() => dataset ?? emptyDataset());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLoadingStatic, setIsLoadingStatic] = useState(!dataset);
  const [mode, setMode] = useState<Mode>("backend");
  const [metric, setMetric] = useState<MetricKey>("forwardSampleTps");
  const [task, setTask] = useState<BenchmarkTask>("decode");
  const [modelId, setModelId] = useState(() => chooseDefaultModel(data.rows));
  const [paramGroup, setParamGroup] = useState(() => data.groups[0]?.id ?? "");
  const [quantization, setQuantization] = useState("");
  const [repo, setRepo] = useState("");
  const [backend, setBackend] = useState("");
  const [status, setStatus] = useState("");
  const [groupBy, setGroupBy] = useState<GroupBy>("model");
  const [selectedName, setSelectedName] = useState<string | null>(null);

  const options = useMemo(() => buildOptions(data.rows), [data.rows]);
  const globalStatus = useMemo(() => countStatuses(data.rows), [data.rows]);

  useEffect(() => {
    if (dataset) {
      return;
    }
    let cancelled = false;
    setIsLoadingStatic(true);
    fetch("/data/task5.json", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`static dataset failed: ${response.status}`);
        }
        return response.json() as Promise<Task5Dataset>;
      })
      .then((nextData) => {
        if (cancelled) {
          return;
        }
        setData(nextData);
        setModelId(chooseDefaultModel(nextData.rows));
        setParamGroup(nextData.groups[0]?.id ?? "");
      })
      .catch(() => {
        if (!cancelled) {
          setData(emptyDataset());
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingStatic(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  const activeRows = useMemo(() => {
    return filterRows(data.rows, {
      metric,
      mode,
      modelId: mode === "backend" ? modelId : undefined,
      paramGroup: mode === "model" ? paramGroup : undefined,
      task,
      quantization: quantization || undefined,
      repo: repo || undefined,
      backend: backend || undefined,
      status: status || undefined
    });
  }, [data.rows, metric, mode, modelId, paramGroup, task, quantization, repo, backend, status]);

  const series = useMemo(() => {
    if (mode === "backend") {
      return makeSeries(activeRows, metric, (row) => seriesNameForRow(row, activeRows, "backend", groupBy));
    }
    return makeSeries(activeRows, metric, (row) => seriesNameForRow(row, activeRows, "model", groupBy));
  }, [activeRows, groupBy, metric, mode]);
  const summary = useMemo(() => computeRaceSummary(activeRows, metric), [activeRows, metric]);
  const displayedRows = useMemo(() => sortRows(activeRows, metric).slice(0, 300), [activeRows, metric]);

  const resetSelection = () => setSelectedName(null);
  const refreshData = async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch("/api/task5", { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`refresh failed: ${response.status}`);
      }
      const nextData = (await response.json()) as Task5Dataset;
      setData(nextData);
      if (!nextData.rows.some((row) => row.modelId === modelId)) {
        setModelId(chooseDefaultModel(nextData.rows));
      }
      if (!nextData.groups.some((group) => group.id === paramGroup)) {
        setParamGroup(nextData.groups[0]?.id ?? "");
      }
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <div className="eyebrow">RWKV TASK 5</div>
          <h1>Core Throughput</h1>
          <p className="topbarNote">Forward+sample rows from the DevPod 2 core benchmark matrix.</p>
        </div>
        <div className="topbarStats">
          <Stat icon={<Database size={16} />} label="Rows" value={data.rows.length.toLocaleString()} />
          <Stat icon={<CheckCircle2 size={16} />} label="OK" value={globalStatus.ok.toLocaleString()} />
          <Stat icon={<AlertTriangle size={16} />} label="Failed" value={globalStatus.failed.toLocaleString()} />
          <Stat icon={<BarChart3 size={16} />} label="Unsupported" value={globalStatus.unsupported.toLocaleString()} />
          <Stat icon={<RefreshCcw size={16} />} label="Updated" value={formatTimestamp(data.generatedAt)} />
        </div>
      </header>

      <section className="modebar" aria-label="View mode">
        <button className={mode === "backend" ? "active" : ""} onClick={() => { setMode("backend"); resetSelection(); }}>
          Backend Race
        </button>
        <button className={mode === "model" ? "active" : ""} onClick={() => { setMode("model"); resetSelection(); }}>
          Model Race
        </button>
        <div className="metricSwitch">
          {METRICS.map((item) => (
            <button key={item.key} className={metric === item.key ? "active" : ""} onClick={() => {
              setMetric(item.key);
            }}>
              {item.label}
            </button>
          ))}
        </div>
      </section>

      <section className="workspace">
        <aside className="filters">
          <PanelTitle icon={<Gauge size={16} />} title="Filters" />
          {mode === "backend" ? (
            <Select label="Model" value={modelId} onChange={setModelId} options={options.models} />
          ) : (
            <Select label="Param group" value={paramGroup} onChange={setParamGroup} options={data.groups.map((group) => ({ value: group.id, label: group.label }))} />
          )}
          <TaskSegmented value={task} onChange={setTask} />
          <Select label="Quant" value={quantization} onChange={setQuantization} options={[{ value: "", label: "all" }, ...options.quantizations]} />
          <Select label="Repo" value={repo} onChange={setRepo} options={[{ value: "", label: "all" }, ...options.repos]} />
          <Select label="Implementation" value={backend} onChange={setBackend} options={[{ value: "", label: "all" }, ...options.backends]} />
          <Select label="Status" value={status} onChange={setStatus} options={[{ value: "", label: "all" }, ...options.statuses]} />
          {mode === "model" ? (
            <Select label="Group by" value={groupBy} onChange={(value) => setGroupBy(value as GroupBy)} options={[{ value: "model", label: "model" }, { value: "backend", label: "backend" }]} />
          ) : null}
        </aside>

        <section className="chartPane">
          <div className="chartHeader">
            <div>
              <div className="eyebrow">{mode === "backend" ? "Same model backend comparison" : "Similar-size model comparison"}</div>
              <h2>{mode === "backend" ? displayModelName(modelId) : paramGroup}</h2>
              <p className="metricNote">{taskAxisDescription(task)}</p>
            </div>
            <button className="quietButton" onClick={refreshData} disabled={isRefreshing}>
              <RefreshCcw size={15} />
              {isRefreshing ? "refreshing" : "refresh data"}
            </button>
          </div>
          {isLoadingStatic ? <div className="loadingPanel">Loading cached Task 5 dataset...</div> : null}
          <ThroughputChart
            metric={metric}
            task={task}
            series={series}
            selectedName={selectedName}
            onSelectName={setSelectedName}
          />
        </section>

        <aside className="inspector">
          <PanelTitle icon={<Trophy size={16} />} title="Inspector" />
          <div className="winner">
            <span>Fastest</span>
            <strong>{summary.fastest ? fastestLabel(summary.fastest, activeRows, mode, groupBy) : "none"}</strong>
            <code>{summary.fastest ? formatMetric(summary.fastest[metric]) : "-"}</code>
          </div>
          <div className="kv">
            <span>Albatross ratio</span>
            <strong>{summary.albatrossRatio ? `${summary.albatrossRatio}x` : "-"}</strong>
          </div>
          <div className="kv">
            <span>GPU</span>
            <strong>{summary.fastest?.gpuName || "-"}</strong>
          </div>
          <div className="statusGrid">
            <StatusBlock label="ok" value={summary.statusCounts.ok} />
            <StatusBlock label="failed" value={summary.statusCounts.failed} />
            <StatusBlock label="other" value={summary.statusCounts.other} />
          </div>
          <div className="sourceList">
            <span>Sources</span>
            {uniqueSorted(activeRows.map((row) => row.sourcePath)).slice(0, 8).map((source) => (
              <code key={source}>{source}</code>
            ))}
          </div>
        </aside>
      </section>

      <section className="tablePane">
        <div className="tableHeader">
          <h2>Rows</h2>
          <span>{displayedRows.length} shown / {activeRows.length} selected</span>
        </div>
        <div className="tableWrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Repo</th>
                <th>Backend</th>
                <th>Task</th>
                <th>B</th>
                <th>T</th>
                <th>Quant</th>
                <th>Status</th>
                <th>Forward+Sample</th>
                <th>p50 ms</th>
                <th>Entrypoint</th>
                <th>Source</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody>
              {displayedRows.map((row) => (
                <tr key={row.id}>
                  <td>{displayModelName(row.modelId, row.modelLabel)}</td>
                  <td>{row.repo}</td>
                  <td>{row.backend}</td>
                  <td>{row.task}</td>
                  <td>{row.B}</td>
                  <td>{row.T}</td>
                  <td>{row.quantization || row.dtype || "-"}</td>
                  <td><span className={`status ${normalizedStatus(row.status)}`}>{normalizedStatus(row.status)}</span></td>
                  <td>{formatMetric(row.forwardSampleTps)}</td>
                  <td>{formatMetric(row.p50Ms)}</td>
                  <td>{row.entrypoint || "-"}</td>
                  <td><code>{row.sourcePath}</code></td>
                  <td className="errorCell">{row.error}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function emptyDataset(): Task5Dataset {
  return {
    generatedAt: new Date(0).toISOString(),
    sourceRoot: "",
    sourceRoots: [],
    rows: [],
    groups: []
  };
}

function Stat({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <div className="stat">{icon}<span>{label}</span><strong>{value}</strong></div>;
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return <div className="panelTitle">{icon}<span>{title}</span></div>;
}

function Select({ label, value, options, onChange }: { label: string; value: string; options: { value: string; label: string }[]; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => <option key={`${label}-${option.value}`} value={option.value}>{option.label}</option>)}
      </select>
    </label>
  );
}

function TaskSegmented({ value, onChange }: { value: BenchmarkTask; onChange: (value: BenchmarkTask) => void }) {
  return (
    <div className="field">
      <span>Task</span>
      <div className="taskTabs" role="tablist" aria-label="Task">
        {TASKS.map((item) => (
          <button
            key={item.key}
            type="button"
            role="tab"
            aria-selected={value === item.key}
            className={value === item.key ? "active" : ""}
            onClick={() => onChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function StatusBlock({ label, value }: { label: string; value: number }) {
  return <div className={`statusBlock ${label}`}><span>{label}</span><strong>{value}</strong></div>;
}

function countStatuses(rows: BenchmarkRow[]) {
  return rows.reduce(
    (counts, row) => {
      if (row.status === "ok") {
        counts.ok += 1;
      } else if (row.status === "unsupported") {
        counts.unsupported += 1;
      } else {
        counts.failed += 1;
      }
      return counts;
    },
    { ok: 0, failed: 0, unsupported: 0 }
  );
}

function buildOptions(rows: BenchmarkRow[]) {
  return {
    models: uniqueSorted(rows.map((row) => row.modelId)).map((model) => ({ value: model, label: displayModelName(model) })),
    quantizations: uniqueSorted(rows.map((row) => row.quantization).filter(Boolean)).map((value) => ({ value, label: value })),
    repos: uniqueSorted(rows.map((row) => row.repo)).map((value) => ({ value, label: value })),
    backends: uniqueSorted(rows.map((row) => row.backend)).map((value) => ({ value, label: value })),
    statuses: uniqueSorted(rows.map((row) => normalizedStatus(row.status))).map((value) => ({ value, label: value }))
  };
}

function chooseDefaultModel(rows: BenchmarkRow[]): string {
  const albatrossBackendsByModel = new Map<string, Set<string>>();
  const backendsByModel = new Map<string, Set<string>>();
  const okRowsByModel = new Map<string, number>();
  for (const row of rows) {
    if (row.status === "ok" || row.status === "failed" || row.status === "unsupported") {
      const backends = backendsByModel.get(row.modelId) ?? new Set<string>();
      backends.add(row.backend);
      backendsByModel.set(row.modelId, backends);
      if (row.repo === "albatross") {
        const albatrossBackends = albatrossBackendsByModel.get(row.modelId) ?? new Set<string>();
        albatrossBackends.add(row.backend);
        albatrossBackendsByModel.set(row.modelId, albatrossBackends);
      }
    }
    if (row.status === "ok") {
      okRowsByModel.set(row.modelId, (okRowsByModel.get(row.modelId) ?? 0) + 1);
    }
  }
  return Array.from(backendsByModel.entries())
    .sort(
      (a, b) =>
        (albatrossBackendsByModel.get(b[0])?.size ?? 0) - (albatrossBackendsByModel.get(a[0])?.size ?? 0) ||
        b[1].size - a[1].size ||
        (okRowsByModel.get(b[0]) ?? 0) - (okRowsByModel.get(a[0]) ?? 0) ||
        a[0].localeCompare(b[0], undefined, { numeric: true })
    )[0]?.[0] ?? rows[0]?.modelId ?? "";
}

function sortRows(rows: BenchmarkRow[], metric: MetricKey): BenchmarkRow[] {
  if (metric === "p50Ms") {
    return [...rows].sort((a, b) => (a[metric] ?? Number.POSITIVE_INFINITY) - (b[metric] ?? Number.POSITIVE_INFINITY) || a.backend.localeCompare(b.backend));
  }
  return [...rows].sort((a, b) => (b[metric] ?? -1) - (a[metric] ?? -1) || a.backend.localeCompare(b.backend));
}

function fastestLabel(row: BenchmarkRow, rows: BenchmarkRow[], mode: Mode, groupBy: GroupBy): string {
  return seriesNameForRow(row, rows, mode, groupBy);
}

function formatMetric(value: number | null | undefined): string {
  if (!value) {
    return "-";
  }
  return Math.round(value).toLocaleString();
}
