import { ExternalLink, RefreshCcw } from "lucide-react";
import { useMemo, useState } from "react";
import type {
  AlgorithmDefinition,
  NormalizedDataset,
  ParameterState,
} from "../types/algorithm";

type Props = {
  algorithm: AlgorithmDefinition;
  dataset: NormalizedDataset;
  params: ParameterState;
};

const DEFAULT_JUPYTERLITE_REPL_URL = "https://jupyterlite.github.io/demo/repl/index.html";
const MAX_NOTEBOOK_POINTS = 250;

export function NotebookPanel({ algorithm, dataset, params }: Props) {
  const [refreshKey, setRefreshKey] = useState(0);
  const notebookCode = useMemo(
    () => makeNotebookCode(algorithm, dataset, params),
    [algorithm, dataset, params, refreshKey],
  );
  const sourceUrl = useMemo(() => makeJupyterLiteUrl(notebookCode), [notebookCode]);

  return (
    <section className="notebook-panel" aria-label={`${algorithm.name} notebook`}>
      <div className="notebook-header">
        <div>
          <span className="eyebrow">Notebook</span>
          <h3>Run the Python side in JupyterLite</h3>
          <p>
            The visualization still runs through the browser JavaScript engine; this notebook gives
            you a Python workspace for the same lesson data and reference implementation.
          </p>
        </div>

        <div className="notebook-actions">
          <button
            className="button secondary compact-button"
            type="button"
            onClick={() => setRefreshKey((key) => key + 1)}
          >
            <RefreshCcw size={15} aria-hidden="true" />
            Refresh
          </button>
          <a
            className="button secondary compact-button"
            href={sourceUrl}
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={15} aria-hidden="true" />
            Open
          </a>
        </div>
      </div>

      <div className="notebook-frame-shell">
        <iframe
          key={sourceUrl}
          title={`${algorithm.name} JupyterLite notebook`}
          src={sourceUrl}
          sandbox="allow-downloads allow-forms allow-modals allow-popups allow-same-origin allow-scripts"
          allow="clipboard-read; clipboard-write"
        />
      </div>
    </section>
  );
}

function makeJupyterLiteUrl(code: string) {
  const baseUrl =
    typeof import.meta.env.VITE_JUPYTERLITE_REPL_URL === "string" &&
    import.meta.env.VITE_JUPYTERLITE_REPL_URL.length > 0
      ? import.meta.env.VITE_JUPYTERLITE_REPL_URL
      : DEFAULT_JUPYTERLITE_REPL_URL;
  const url = new URL(baseUrl, window.location.origin);
  url.searchParams.set("kernel", "python");
  url.searchParams.set("toolbar", "1");
  url.searchParams.set("execute", "0");
  url.searchParams.set("promptCellPosition", "left");
  url.searchParams.set("clearCellsOnExecute", "0");
  url.searchParams.set("clearCodeContentOnExecute", "0");
  url.searchParams.set("showBanner", "0");
  url.searchParams.set("code", code);
  return url.toString();
}

function makeNotebookCode(
  algorithm: AlgorithmDefinition,
  dataset: NormalizedDataset,
  params: ParameterState,
) {
  const pointRows = dataset.points.slice(0, MAX_NOTEBOOK_POINTS).map((point) => [
    roundForNotebook(point.x),
    roundForNotebook(point.y),
  ]);
  const labels = dataset.points
    .slice(0, MAX_NOTEBOOK_POINTS)
    .map((point) => point.label ?? "");
  const setupCode = algorithm.notebook?.setupCode?.(params).trim();
  const packages = algorithm.notebook?.packages?.join(", ") || "numpy, matplotlib";
  const referenceCode = commentBlock(algorithm.code.python(params).trim());
  const intro = algorithm.notebook?.intro ?? `Explore ${algorithm.name} with the same data used by the visual playground.`;

  return `# ${algorithm.name}
# ${intro}
# Suggested packages: ${packages}

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

points = np.array(${JSON.stringify(pointRows)}, dtype=float)
labels = ${JSON.stringify(labels)}

print("Lesson:", ${JSON.stringify(algorithm.name)})
print("Dataset:", ${JSON.stringify(dataset.name)})
print("Points:", len(points))
if len(points):
    print("First rows:")
    print(points[:5])

if plt is not None and len(points):
    plt.figure(figsize=(5, 3.5))
    plt.scatter(points[:, 0], points[:, 1], s=28)
    plt.title(${JSON.stringify(algorithm.name)})
    plt.xlabel(${JSON.stringify(dataset.mapping.x || "x")})
    plt.ylabel(${JSON.stringify(dataset.mapping.y || "y")})
    plt.grid(alpha=0.25)
    plt.show()

${setupCode ? `# Lesson-specific setup\n${setupCode}\n` : ""}# Python reference from the Algorithm Zoo code tab.
# It is commented so the notebook setup cell always runs; uncomment/adapt it as you experiment.
${referenceCode}
`;
}

function commentBlock(code: string) {
  if (!code) {
    return "# No Python reference code is available for this lesson yet.";
  }

  return code
    .split("\n")
    .map((line) => (line.trim().length > 0 ? `# ${line}` : "#"))
    .join("\n");
}

function roundForNotebook(value: number) {
  return Number.isFinite(value) ? Number(value.toFixed(6)) : 0;
}
