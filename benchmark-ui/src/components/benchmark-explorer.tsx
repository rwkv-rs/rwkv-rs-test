"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, Database, Gauge, RefreshCcw, Trophy } from "lucide-react";
import type { BenchmarkRow, MetricKey, Task5Dataset } from "@/lib/types";
import { computeRaceSummary, displayModelName, filterRows, formatTimestamp, makeSeries, normalizedStatus, seriesNameForRow, uniqueSorted } from "@/lib/analytics";
import { ThroughputChart } from "./throughput-chart";

type Mode = "backend" | "model";
type GroupBy = "model" | "backend";

const METRICS: { key: MetricKey; label: string }[] = [
  { key: "decodeTps", label: "Decode TPS" },
  { key: "prefillTps", label: "Prefill TPS" }
];

export function BenchmarkExplorer({ dataset }: { dataset: Task5Dataset | null }) {
  const [data, setData] = useState<Task5Dataset>(() => dataset ?? emptyDataset());
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isLoadingStatic, setIsLoadingStatic] = useState(!dataset);
  const [mode, setMode] = useState<Mode>("backend");
  const [metric, setMetric] = useState<MetricKey>("decodeTps");
  const [modelId, setModelId] = useState(() => chooseDefaultModel(data.rows));
  const [paramGroup, setParamGroup] = useState(() => data.groups[0]?.id ?? "");
  const [promptLen, setPromptLen] = useState("");
  const [quantization, setQuantization] = useState("");
  const [repo, setRepo] = useState("");
  const [status, setStatus] = useState("");
  const [groupBy, setGroupBy] = useState<GroupBy>("model");
  const [selectedName, setSelectedName] = useState<string | null>(null);

  const options = useMemo(() => buildOptions(data.rows), [data.rows]);
  const promptLenFilter = promptLen ? Number(promptLen) : undefined;
  const contextOptions = useMemo(() => {
    const contextValues = options.promptLens.map((value) => ({ value: String(value), label: String(value) }));
    if (metric === "decodeTps") {
      return [{ value: "", label: "all contexts (median)" }, ...contextValues];
    }
    return contextValues;
  }, [metric, options.promptLens]);

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

  useEffect(() => {
    if (metric === "prefillTps" && !promptLen) {
      setPromptLen(String(defaultPromptLen(options.promptLens)));
    }
  }, [metric, options.promptLens, promptLen]);

  const activeRows = useMemo(() => {
    return filterRows(data.rows, {
      metric,
      mode,
      modelId: mode === "backend" ? modelId : undefined,
      paramGroup: mode === "model" ? paramGroup : undefined,
      promptLen: promptLenFilter,
      quantization: quantization || undefined,
      repo: repo || undefined,
      status: status || undefined
    });
  }, [data.rows, metric, mode, modelId, paramGroup, promptLenFilter, quantization, repo, status]);

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
          <h1>Benchmark Explorer</h1>
        </div>
        <div className="topbarStats">
          <Stat icon={<Database size={16} />} label="Rows" value={data.rows.length.toLocaleString()} />
          <Stat icon={<BarChart3 size={16} />} label="Groups" value={data.groups.length.toLocaleString()} />
          <Stat icon={<Activity size={16} />} label="Kind" value="synthetic throughput" />
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
              if (item.key === "decodeTps") {
                setPromptLen("");
              } else if (!promptLen) {
                setPromptLen(String(defaultPromptLen(options.promptLens)));
              }
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
          <Select label="Context" value={promptLen} onChange={setPromptLen} options={contextOptions} />
          <Select label="Quant" value={quantization} onChange={setQuantization} options={[{ value: "", label: "all" }, ...options.quantizations]} />
          <Select label="Backend" value={repo} onChange={setRepo} options={[{ value: "", label: "all" }, ...options.repos]} />
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
              <p className="metricNote">{metric === "decodeTps" && !promptLen ? "Decode TPS is aggregated as the median across all contexts." : "Prefill TPS is shown for the selected context length."}</p>
            </div>
            <button className="quietButton" onClick={refreshData} disabled={isRefreshing}>
              <RefreshCcw size={15} />
              {isRefreshing ? "refreshing" : "refresh data"}
            </button>
          </div>
          {isLoadingStatic ? <div className="loadingPanel">Loading cached Task 5 dataset...</div> : null}
          <ThroughputChart
            metric={metric}
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
                <th>Backend</th>
                <th>Quant</th>
                <th>bsz</th>
                <th>ctx</th>
                <th>Status</th>
                <th>Decode</th>
                <th>Prefill</th>
                <th>Source</th>
                <th>Error</th>
              </tr>
            </thead>
            <tbody>
              {displayedRows.map((row) => (
                <tr key={row.id}>
                  <td>{displayModelName(row.modelId, row.modelLabel)}</td>
                  <td>{row.repo}</td>
                  <td>{row.quantization || row.dtype || "-"}</td>
                  <td>{row.bsz}</td>
                  <td>{row.promptLen}</td>
                  <td><span className={`status ${normalizedStatus(row.status)}`}>{normalizedStatus(row.status)}</span></td>
                  <td>{formatMetric(row.decodeTps)}</td>
                  <td>{formatMetric(row.prefillTps)}</td>
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

function StatusBlock({ label, value }: { label: string; value: number }) {
  return <div className={`statusBlock ${label}`}><span>{label}</span><strong>{value}</strong></div>;
}

function buildOptions(rows: BenchmarkRow[]) {
  return {
    models: uniqueSorted(rows.map((row) => row.modelId)).map((model) => ({ value: model, label: displayModelName(model) })),
    promptLens: uniqueSorted(rows.map((row) => row.promptLen), Number),
    quantizations: uniqueSorted(rows.map((row) => row.quantization).filter(Boolean)).map((value) => ({ value, label: value })),
    repos: uniqueSorted(rows.map((row) => row.repo)).map((value) => ({ value, label: value })),
    statuses: uniqueSorted(rows.map((row) => normalizedStatus(row.status))).map((value) => ({ value, label: value }))
  };
}

function chooseDefaultModel(rows: BenchmarkRow[]): string {
  const counts = new Map<string, number>();
  for (const row of rows) {
    if (row.status === "ok" && row.modelSize === "1.5B") {
      counts.set(row.modelId, (counts.get(row.modelId) ?? 0) + 1);
    }
  }
  return Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] ?? rows[0]?.modelId ?? "";
}

function defaultPromptLen(promptLens: number[]): number {
  return promptLens.includes(1024) ? 1024 : promptLens[0] ?? 1024;
}

function sortRows(rows: BenchmarkRow[], metric: MetricKey): BenchmarkRow[] {
  return [...rows].sort((a, b) => (b[metric] ?? -1) - (a[metric] ?? -1) || a.repo.localeCompare(b.repo));
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
