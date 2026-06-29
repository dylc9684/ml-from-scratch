import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  RotateCcw,
  Search,
  Sparkles,
  Upload,
} from "lucide-react";
import { algorithms, makeDefaultParams } from "./algorithms";
import {
  initialMapping,
  normalizeDataset,
  parseDatasetFile,
  type DatasetIssue,
} from "./data/datasets";
import { AlgorithmCatalog } from "./components/AlgorithmCatalog";
import { CodeViewer } from "./components/CodeViewer";
import { ConceptGraph } from "./components/ConceptGraph";
import { DatasetPanel } from "./components/DatasetPanel";
import { EducationPanel } from "./components/EducationPanel";
import { AlgorithmDeepDive } from "./components/AlgorithmDeepDive";
import { LessonGuide } from "./components/LessonGuide";
import { MetricsGrid } from "./components/MetricsGrid";
import { ParameterControls } from "./components/ParameterControls";
import { RealApplicationsPanel } from "./components/RealApplicationsPanel";
import { VisualizationCanvas } from "./components/VisualizationCanvas";
import type {
  EngineResult,
  DatasetMapping,
  NormalizedDataset,
  ParameterState,
  ParameterValue,
  RawDataset,
} from "./types/algorithm";

type SecondaryPanel = "map" | "applications" | "data" | "math" | "code" | "deepDive";

export default function App() {
  const [query, setQuery] = useState("");
  const [activeId, setActiveId] = useState(algorithms[0].id);
  const activeAlgorithm = useMemo(
    () => algorithms.find((algorithm) => algorithm.id === activeId) ?? algorithms[0],
    [activeId],
  );

  const [params, setParams] = useState<ParameterState>(() =>
    makeDefaultParams(activeAlgorithm),
  );
  const [activePanel, setActivePanel] = useState<SecondaryPanel>("map");
  const [recentIds, setRecentIds] = useState<string[]>(() => [algorithms[0].id]);
  const [rawDataset, setRawDataset] = useState<RawDataset | null>(null);
  const [uploadIssues, setUploadIssues] = useState<DatasetIssue[]>([]);
  const [mapping, setMapping] = useState<DatasetMapping>(() => ({
    x: "x",
    y: "y",
    label: "label",
  }));
  const [autoRun, setAutoRun] = useState(true);
  const [engineResult, setEngineResult] = useState<EngineResult>(() =>
    activeAlgorithm.engine(activeAlgorithm.makeSampleDataset(), params),
  );
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const normalizedDataset = useMemo(() => {
    if (!rawDataset) {
      return {
        dataset: activeAlgorithm.makeSampleDataset(),
        issues: uploadIssues,
      };
    }

    const result = normalizeDataset(rawDataset, mapping);
    return {
      dataset: result.dataset,
      issues: [
        ...uploadIssues,
        ...result.issues,
        ...(result.dataset.points.length === 0
          ? [
              {
                severity: "warning" as const,
                message: "The visualization is using the generated sample until the upload has usable X/Y rows.",
              },
            ]
          : []),
      ],
    };
  }, [activeAlgorithm, mapping, rawDataset, uploadIssues]);

  const displayDataset: NormalizedDataset = normalizedDataset.dataset;
  const engineDataset: NormalizedDataset = useMemo(
    () =>
      displayDataset.points.length > 0
        ? displayDataset
        : activeAlgorithm.makeSampleDataset(),
    [activeAlgorithm, displayDataset],
  );
  const datasetIssues = normalizedDataset.issues;

  useEffect(() => {
    setParams(makeDefaultParams(activeAlgorithm));
    setFrameIndex(0);
    setIsPlaying(false);
  }, [activeAlgorithm]);

  useEffect(() => {
    setRecentIds((current) => [activeAlgorithm.id, ...current.filter((id) => id !== activeAlgorithm.id)].slice(0, 6));
  }, [activeAlgorithm.id]);

  const runAlgorithm = useCallback(() => {
    const result = activeAlgorithm.engine(engineDataset, params);
    setEngineResult(result);
    setFrameIndex(0);
    setIsPlaying(activeAlgorithm.controller?.shouldAutoPlay?.(result) ?? result.frames.length > 1);
  }, [activeAlgorithm, engineDataset, params]);

  const runWithParams = useCallback(
    (nextParams: ParameterState) => {
      const result = activeAlgorithm.engine(engineDataset, nextParams);
      setEngineResult(result);
      setFrameIndex(0);
      setIsPlaying(activeAlgorithm.controller?.shouldAutoPlay?.(result) ?? result.frames.length > 1);
    },
    [activeAlgorithm, engineDataset],
  );

  useEffect(() => {
    if (!autoRun) {
      return;
    }

    const timeout = window.setTimeout(runAlgorithm, 120);
    return () => window.clearTimeout(timeout);
  }, [autoRun, runAlgorithm]);

  useEffect(() => {
    if (!isPlaying || engineResult.frames.length < 2) {
      return;
    }

    if (frameIndex >= engineResult.frames.length - 1) {
      setIsPlaying(false);
      return;
    }

    const timeout = window.setTimeout(() => {
      setFrameIndex((index) => Math.min(index + 1, engineResult.frames.length - 1));
    }, 45);

    return () => window.clearTimeout(timeout);
  }, [engineResult.frames.length, frameIndex, isPlaying]);

  const currentFrame =
    engineResult.frames[Math.min(frameIndex, engineResult.frames.length - 1)] ?? null;

  const handleAlgorithmSelect = (id: string) => {
    setActiveId(id);
    setActivePanel("map");
  };

  const handleParamChange = (key: string, value: ParameterValue) => {
    setParams((current) => ({
      ...current,
      [key]: value,
    }));
  };

  const handleRunClick = () => {
    const nextParams = activeAlgorithm.controller?.prepareRunParams?.(params) ?? params;

    if (nextParams !== params) {
      setParams(nextParams);
      runWithParams(nextParams);
      return;
    }

    runAlgorithm();
  };

  const handleReset = () => {
    const nextParams = makeDefaultParams(activeAlgorithm);
    setParams(nextParams);
    setFrameIndex(0);
    const result = activeAlgorithm.engine(engineDataset, nextParams);
    setEngineResult(result);
    setIsPlaying(activeAlgorithm.controller?.shouldAutoPlay?.(result) ?? result.frames.length > 1);
  };

  const handleUseSample = () => {
    setRawDataset(null);
    setUploadIssues([]);
    setMapping({ x: "x", y: "y", label: "label" });
    setActivePanel("data");
  };

  const handleFileChange = async (file: File | undefined) => {
    if (!file) {
      return;
    }

    let text = "";

    try {
      text = await file.text();
    } catch {
      setUploadIssues([
        {
          severity: "error",
          message: "The browser could not read that file. Try exporting it again as CSV or JSON.",
        },
      ]);
      setRawDataset(null);
      setActivePanel("data");
      return;
    }

    const parsed = parseDatasetFile(file.name, text);

    setUploadIssues(parsed.issues);
    if (!parsed.ok) {
      setRawDataset(null);
      setActivePanel("data");
      return;
    }

    setRawDataset(parsed.dataset);
    setMapping(initialMapping(parsed.dataset));
    setActivePanel("data");
  };

  return (
    <div className="app-shell">
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".csv,.json,application/json,text/csv"
        onChange={(event) => {
          void handleFileChange(event.target.files?.[0]);
          event.target.value = "";
        }}
      />

      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <h1>Algorithm Zoo</h1>
            <p>Interactive AI Playground</p>
          </div>
        </div>

        <label className="search-shell" htmlFor="algorithm-search">
          <Search size={18} aria-hidden="true" />
          <input
            id="algorithm-search"
            type="search"
            value={query}
            placeholder="Search algorithms"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>

        <div className="topbar-actions">
          <button
            className="button secondary"
            type="button"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={17} aria-hidden="true" />
            Upload
          </button>
          <button className="button secondary" type="button" onClick={handleUseSample}>
            <Sparkles size={17} aria-hidden="true" />
            Sample
          </button>
          <button className="button primary" type="button" onClick={handleRunClick}>
            <Play size={17} aria-hidden="true" />
            {activeAlgorithm.controller?.primaryActionLabel ?? "Run"}
          </button>
          <button
            className="icon-button"
            type="button"
            aria-label="Reset parameters"
            title="Reset parameters"
            onClick={handleReset}
          >
            <RotateCcw size={18} aria-hidden="true" />
          </button>
        </div>
      </header>

      <div className="layout">
        <AlgorithmCatalog
          algorithms={algorithms}
          activeId={activeAlgorithm.id}
          query={query}
          recentIds={recentIds}
          onSelect={handleAlgorithmSelect}
        />

        <main className="stage-panel">
          <div className="stage-header">
            <div>
              <p className="eyebrow">{activeAlgorithm.category}</p>
              <h2>{activeAlgorithm.name}</h2>
              <p className="stage-summary">{activeAlgorithm.summary}</p>
            </div>
            <div className="status-pills" aria-live="polite">
              <span className="pill">{engineResult.runtime}</span>
              <span className="pill">{engineDataset.points.length} points</span>
              <span className="pill">
                Frame {Math.min(frameIndex + 1, engineResult.frames.length)} /{" "}
                {engineResult.frames.length || 1}
              </span>
            </div>
          </div>

          <LessonGuide
            algorithm={activeAlgorithm}
            frame={currentFrame}
            metrics={engineResult.metrics}
            onOpenPanel={setActivePanel}
          />

          <section className="primary-workspace" aria-label="Primary lesson playground">
            <div className="visual-column">
              <VisualizationCanvas
                frame={currentFrame}
                algorithm={activeAlgorithm}
                params={params}
                onParamChange={handleParamChange}
              />
            </div>

            <aside className="primary-rail" aria-label="Controls and key metrics">
              <section className="primary-card controls-card" aria-label="Lesson controls">
                <div className="primary-card-heading">
                  <span className="eyebrow">Controls</span>
                  <h3>Change the model</h3>
                </div>
                <ParameterControls
                  algorithm={activeAlgorithm}
                  params={params}
                  autoRun={autoRun}
                  onAutoRunChange={setAutoRun}
                  onChange={handleParamChange}
                />
              </section>

              <section className="primary-card metrics-card" aria-label="Key metrics">
                <div className="primary-card-heading">
                  <span className="eyebrow">Key Metrics</span>
                  <h3>Watch these move</h3>
                </div>
                <MetricsGrid metrics={engineResult.metrics} />
              </section>
            </aside>
          </section>

          <section className="secondary-learning" aria-label="Secondary lesson materials">
            <div className="secondary-learning-header">
              <div>
                <span className="eyebrow">Study Tabs</span>
                <h3>Go deeper when the graph raises a question.</h3>
              </div>
              <div className="secondary-tabbar" role="tablist" aria-label="Lesson study panels">
                {(["map", "applications", "data", "math", "code", "deepDive"] as SecondaryPanel[]).map((panel) => (
                  <button
                    key={panel}
                    className={`tab ${activePanel === panel ? "active" : ""}`}
                    type="button"
                    role="tab"
                    aria-selected={activePanel === panel}
                    onClick={() => setActivePanel(panel)}
                  >
                    {panelLabel(panel)}
                  </button>
                ))}
              </div>
            </div>

            <div className="secondary-panel-body">
                {activePanel === "map" && (
                  <ConceptGraph
                    algorithms={algorithms}
                    activeId={activeAlgorithm.id}
                    onSelect={handleAlgorithmSelect}
                  />
                )}
                {activePanel === "applications" && (
                  <RealApplicationsPanel algorithm={activeAlgorithm} />
                )}
                {activePanel === "data" && (
                  <DatasetPanel
                    rawDataset={rawDataset}
                    dataset={displayDataset}
                    issues={datasetIssues}
                    mapping={mapping}
                    onMappingChange={setMapping}
                    onUploadClick={() => fileInputRef.current?.click()}
                    onUseSample={handleUseSample}
                  />
                )}
                {activePanel === "math" && <EducationPanel algorithm={activeAlgorithm} />}
                {activePanel === "code" && (
                  <CodeViewer algorithm={activeAlgorithm} params={params} />
                )}
                {activePanel === "deepDive" && (
                  <AlgorithmDeepDive
                    algorithm={activeAlgorithm}
                    frame={currentFrame}
                    metrics={engineResult.metrics}
                  />
                )}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

function panelLabel(panel: SecondaryPanel) {
  if (panel === "map") {
    return "Concept Map";
  }
  if (panel === "applications") {
    return "Applications";
  }
  if (panel === "data") {
    return "Data";
  }
  if (panel === "math") {
    return "Math";
  }
  if (panel === "code") {
    return "Code";
  }
  if (panel === "deepDive") {
    return "Deep Dive";
  }
  return "Panel";
}
