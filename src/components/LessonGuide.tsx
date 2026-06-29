import {
  BookOpen,
  Code2,
  Gauge,
  LineChart,
  SlidersHorizontal,
  TriangleAlert,
  WandSparkles,
  Workflow,
} from "lucide-react";
import {
  frameSummary,
  getDeepDiveContent,
} from "./AlgorithmDeepDive";
import type {
  AlgorithmDefinition,
  AlgorithmFrame,
  Metric,
} from "../types/algorithm";

type Props = {
  algorithm: AlgorithmDefinition;
  frame: AlgorithmFrame | null;
  metrics: Metric[];
  onOpenPanel: (panel: "map" | "applications" | "math" | "code" | "deepDive") => void;
};

export function LessonGuide({ algorithm, frame, metrics, onOpenPanel }: Props) {
  const content = getDeepDiveContent(algorithm);
  const firstMetric = metrics[0];
  const controlSummary =
    algorithm.parameters.length === 0
      ? "No tunable controls for this lesson."
      : algorithm.parameters
          .slice(0, 3)
          .map((parameter) => parameter.label)
          .join(", ");
  const formulaSummary =
    algorithm.formulas.length === 0
      ? "No formulas listed."
      : algorithm.formulas
          .slice(0, 2)
          .map((formula) => formula.title)
          .join(", ");
  const codeSummary = "Switch between synchronized Python and JavaScript implementations.";
  const failureMode = inferFailureMode(content.keyDetails, content.graphNotes);

  const cards = [
    {
      label: "What it does",
      value: algorithm.summary,
      icon: WandSparkles,
    },
    {
      label: "What to watch",
      value: content.graphNotes[0] ?? content.graphTitle,
      icon: LineChart,
      onClick: () => onOpenPanel("deepDive"),
    },
    {
      label: "Controls",
      value: controlSummary,
      icon: SlidersHorizontal,
    },
    {
      label: "Math",
      value: formulaSummary,
      icon: BookOpen,
      onClick: () => onOpenPanel("math"),
    },
    {
      label: "Runtime / Big-O",
      value: content.complexity.time,
      icon: Gauge,
      onClick: () => onOpenPanel("deepDive"),
    },
    {
      label: "Python / JS",
      value: codeSummary,
      icon: Code2,
      onClick: () => onOpenPanel("code"),
    },
    {
      label: "Real-world example",
      value: content.realWorld[0] ?? "Use this as a visual baseline before production modeling.",
      icon: Workflow,
      onClick: () => onOpenPanel("applications"),
    },
    {
      label: "Common failure mode",
      value: failureMode,
      icon: TriangleAlert,
    },
  ];

  return (
    <section className="lesson-guide" aria-label={`${algorithm.name} guided lesson structure`}>
      <div className="lesson-guide-header">
        <div>
          <span className="eyebrow">Guided Lesson</span>
          <h3>Learn by changing one thing, then watching the graph answer back.</h3>
        </div>
        <button className="button secondary compact-button" type="button" onClick={() => onOpenPanel("map")}>
          Concept Map
        </button>
      </div>

      <div className="lesson-guide-grid">
        {cards.map((card) => {
          const Icon = card.icon;
          const contentBlock = (
            <>
              <span className="guide-icon" aria-hidden="true">
                <Icon size={15} />
              </span>
              <span className="guide-copy">
                <strong>{card.label}</strong>
                <small>{card.value}</small>
              </span>
            </>
          );

          if (card.onClick) {
            return (
              <button className="lesson-guide-card interactive" key={card.label} type="button" onClick={card.onClick}>
                {contentBlock}
              </button>
            );
          }

          return (
            <div className="lesson-guide-card" key={card.label}>
              {contentBlock}
            </div>
          );
        })}
      </div>

      <div className="lesson-current-run">
        <span>Current run</span>
        <strong>{frameSummary(frame)}</strong>
        {firstMetric && (
          <em>
            {firstMetric.label}: {firstMetric.value}
          </em>
        )}
      </div>
    </section>
  );
}

function inferFailureMode(keyDetails: string[], graphNotes: string[]) {
  const source =
    [...keyDetails, ...graphNotes].find((item) =>
      /(too high|overfit|misleading|unstable|scale|outlier|leak|saturat|explode|slow|fails|struggles|sensitive)/i.test(item),
    ) ??
    keyDetails[0] ??
    graphNotes[0] ??
    "The visual can look stable even when validation behavior is poor.";

  return source;
}
