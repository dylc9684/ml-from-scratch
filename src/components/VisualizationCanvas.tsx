import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";
import type {
  AlgorithmDefinition,
  AlgorithmFrame,
  ConceptFrame,
  DataPoint,
  KMeansFrame,
  LinearRegressionFrame,
} from "../types/algorithm";

const colors = ["#0f766e", "#2f6fbe", "#b7791f", "#d34a43", "#6f58c9", "#258f66"];

type Props = {
  frame: AlgorithmFrame | null;
  algorithm: AlgorithmDefinition;
};

type Size = {
  width: number;
  height: number;
};

export function VisualizationCanvas({ frame, algorithm }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<Size>({ width: 820, height: 520 });
  const isBackpropLesson = frame?.type === "concept-demo" && Boolean(frame.backprop);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({
        width: Math.max(360, width),
        height: Math.max(320, height),
      });
    });

    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    draw(canvas, size, frame);
  }, [frame, size]);

  return (
    <section
      className={`visual-shell ${isBackpropLesson ? "backprop-shell" : ""}`}
      aria-label={`${algorithm.name} visualization`}
    >
      <canvas ref={canvasRef} />
      <div className="canvas-readout">
        {frame ? (
          <>
            <strong>Iteration {frame.iteration}</strong>
            <span>{readout(frame)}</span>
          </>
        ) : (
          <span>No usable data to visualize.</span>
        )}
      </div>
    </section>
  );
}

function draw(
  canvas: HTMLCanvasElement,
  size: Size,
  frame: AlgorithmFrame | null,
) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(size.width * ratio);
  canvas.height = Math.floor(size.height * ratio);
  canvas.style.width = `${size.width}px`;
  canvas.style.height = `${size.height}px`;

  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, size.width, size.height);
  paintBackground(context, size);

  if (!frame || frame.points.length === 0) {
    paintEmpty(context, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.network && frame.backprop) {
    paintBackpropLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.network && frame.heatmap) {
    paintNetworkBoundaryLesson(context, frame, size);
    return;
  }

  const scales = makeScales(size, collectPoints(frame));
  paintAxes(context, size, scales);

  if (frame.type === "linear-regression") {
    paintLinearRegression(context, frame, scales);
  } else if (frame.type === "k-means") {
    paintKMeans(context, frame, scales);
  } else {
    paintConceptDemo(context, frame, scales, size);
  }
}

function paintBackground(context: CanvasRenderingContext2D, size: Size) {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, size.width, size.height);

  context.strokeStyle = "rgba(23, 33, 43, 0.055)";
  context.lineWidth = 1;
  for (let x = 0; x <= size.width; x += 24) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, size.height);
    context.stroke();
  }

  for (let y = 0; y <= size.height; y += 24) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(size.width, y);
    context.stroke();
  }
}

function paintEmpty(context: CanvasRenderingContext2D, size: Size) {
  context.fillStyle = "#61707f";
  context.font = "600 15px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText("Upload or map two numeric columns to begin.", size.width / 2, size.height / 2);
}

function makeScales(size: Size, points: DataPoint[]) {
  const margin = { top: 30, right: 34, bottom: 42, left: 48 };
  const xExtent = d3.extent(points, (point) => point.x) as [number, number];
  const yExtent = d3.extent(points, (point) => point.y) as [number, number];
  const xPad = Math.max(0.5, (xExtent[1] - xExtent[0]) * 0.12);
  const yPad = Math.max(0.5, (yExtent[1] - yExtent[0]) * 0.12);

  return {
    margin,
    xDomain: [xExtent[0] - xPad, xExtent[1] + xPad] as [number, number],
    yDomain: [yExtent[0] - yPad, yExtent[1] + yPad] as [number, number],
    x: d3
      .scaleLinear()
      .domain([xExtent[0] - xPad, xExtent[1] + xPad])
      .range([margin.left, size.width - margin.right]),
    y: d3
      .scaleLinear()
      .domain([yExtent[0] - yPad, yExtent[1] + yPad])
      .range([size.height - margin.bottom, margin.top]),
  };
}

function paintAxes(
  context: CanvasRenderingContext2D,
  size: Size,
  scales: ReturnType<typeof makeScales>,
) {
  context.strokeStyle = "#c9d4da";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(scales.margin.left, size.height - scales.margin.bottom);
  context.lineTo(size.width - scales.margin.right, size.height - scales.margin.bottom);
  context.moveTo(scales.margin.left, scales.margin.top);
  context.lineTo(scales.margin.left, size.height - scales.margin.bottom);
  context.stroke();

  context.fillStyle = "#61707f";
  context.font = "600 11px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  scales.x.ticks(6).forEach((tick) => {
    const x = scales.x(tick);
    context.fillText(tick.toFixed(1), x, size.height - 15);
  });

  context.textAlign = "right";
  scales.y.ticks(5).forEach((tick) => {
    const y = scales.y(tick);
    context.fillText(tick.toFixed(1), scales.margin.left - 9, y + 4);
  });
}

function paintLinearRegression(
  context: CanvasRenderingContext2D,
  frame: LinearRegressionFrame,
  scales: ReturnType<typeof makeScales>,
) {
  frame.points.forEach((point) => {
    context.beginPath();
    context.fillStyle = "rgba(47, 111, 190, 0.78)";
    context.arc(scales.x(point.x), scales.y(point.y), 4.5, 0, Math.PI * 2);
    context.fill();
  });

  const [xMin, xMax] = scales.xDomain;
  const yMin = frame.slope * xMin + frame.intercept;
  const yMax = frame.slope * xMax + frame.intercept;

  context.strokeStyle = "#d34a43";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(scales.x(xMin), scales.y(yMin));
  context.lineTo(scales.x(xMax), scales.y(yMax));
  context.stroke();
}

function paintKMeans(
  context: CanvasRenderingContext2D,
  frame: KMeansFrame,
  scales: ReturnType<typeof makeScales>,
) {
  frame.points.forEach((point, index) => {
    const assignment = frame.assignments[index] ?? 0;
    context.beginPath();
    context.fillStyle = colors[assignment % colors.length];
    context.globalAlpha = 0.82;
    context.arc(scales.x(point.x), scales.y(point.y), 4.8, 0, Math.PI * 2);
    context.fill();
  });
  context.globalAlpha = 1;

  frame.centroids.forEach((centroid, index) => {
    const x = scales.x(centroid.x);
    const y = scales.y(centroid.y);

    context.fillStyle = "#ffffff";
    context.strokeStyle = colors[index % colors.length];
    context.lineWidth = 4;
    context.beginPath();
    context.arc(x, y, 12, 0, Math.PI * 2);
    context.fill();
    context.stroke();

    context.strokeStyle = "#17212b";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x - 6, y);
    context.lineTo(x + 6, y);
    context.moveTo(x, y - 6);
    context.lineTo(x, y + 6);
    context.stroke();
  });
}

function paintConceptDemo(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  scales: ReturnType<typeof makeScales>,
  size: Size,
) {
  if (frame.bars && frame.bars.length > 0) {
    paintBars(context, frame, scales, size);
    return;
  }

  paintSeries(context, frame, scales);
  frame.points.forEach((point) => {
    context.beginPath();
    context.fillStyle = pointColor(point);
    context.globalAlpha = point.label === "query" ? 1 : 0.82;
    context.arc(
      scales.x(point.x),
      scales.y(point.y),
      point.label === "query" ? 9 : point.label === "anomaly" ? 6.5 : 4.8,
      0,
      Math.PI * 2,
    );
    context.fill();

    if (point.label === "query") {
      context.strokeStyle = "#17212b";
      context.lineWidth = 2;
      context.stroke();
    }
  });
  context.globalAlpha = 1;

  paintMarkers(context, frame, scales, size);
}

function paintNetworkBoundaryLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const gap = 20;
  const header = 30;
  const footer = 42;
  const architectureWidth = Math.max(250, size.width * 0.43);
  const architecturePane = {
    x: 24,
    y: 28,
    width: architectureWidth - 36,
    height: size.height - footer - 10,
  };
  const plotPane = {
    x: architecturePane.x + architecturePane.width + gap + 22,
    y: 28,
    width: size.width - (architecturePane.x + architecturePane.width + gap + 54),
    height: size.height - footer - 10,
  };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Architecture graph", architecturePane.x, architecturePane.y);
  context.fillText("Decision boundary", plotPane.x, plotPane.y);

  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(architecturePane.x + architecturePane.width + gap / 2, 22);
  context.lineTo(architecturePane.x + architecturePane.width + gap / 2, size.height - 24);
  context.stroke();

  paintNetworkPane(context, frame, {
    ...architecturePane,
    y: architecturePane.y + header,
    height: architecturePane.height - header,
  });
  paintDecisionBoundaryPane(context, frame, {
    ...plotPane,
    y: plotPane.y + header,
    height: plotPane.height - header,
  });
}

function paintBackpropLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const narrow = size.width < 700;
  const padding = 24;
  const gap = 18;
  const networkPane = narrow
    ? {
        x: padding,
        y: 48,
        width: size.width - padding * 2,
        height: Math.max(150, size.height * 0.34),
      }
    : {
        x: padding,
        y: 58,
        width: Math.max(270, size.width * 0.54),
        height: size.height - 92,
      };
  const formulaPane = narrow
    ? {
        x: padding,
        y: networkPane.y + networkPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(150, size.height - networkPane.y - networkPane.height - gap - 28),
      }
    : {
        x: networkPane.x + networkPane.width + gap,
        y: 58,
        width: size.width - networkPane.x - networkPane.width - gap - padding,
        height: size.height - 92,
      };

  const phase = frame.backprop?.pulse.phase ?? "forward";
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Pulse animation", networkPane.x, networkPane.y - 22);
  context.fillText("Dynamic calculus sandbox", formulaPane.x, formulaPane.y - 22);

  paintBackpropNetworkPane(context, frame, networkPane);
  paintBackpropFormulaPane(context, frame, formulaPane);

  context.fillStyle = phase === "forward" ? "#2f6fbe" : "#d34a43";
  context.font = "800 11px Inter, system-ui, sans-serif";
  context.textAlign = "right";
  context.fillText(
    phase === "forward" ? "Forward activations" : "Backward gradients",
    networkPane.x + networkPane.width,
    networkPane.y - 22,
  );
}

function paintBackpropNetworkPane(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  pane: { x: number; y: number; width: number; height: number },
) {
  const graph = frame.network;
  const backprop = frame.backprop;
  if (!graph || !backprop) {
    return;
  }

  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.72)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  const positions = layoutNetworkPositions(frame, {
    x: pane.x + 26,
    y: pane.y + 22,
    width: Math.max(160, pane.width - 52),
    height: Math.max(120, pane.height - 54),
  });

  graph.weights.forEach((weight) => {
    const from = positions.get(weight.from);
    const to = positions.get(weight.to);
    if (!from || !to) {
      return;
    }

    const magnitude = Math.min(1, Math.abs(weight.value));
    const backward = backprop.pulse.phase === "backward";
    context.save();
    context.globalAlpha = backward ? 0.2 + magnitude * 0.72 : 0.18 + magnitude * 0.38;
    context.strokeStyle = backward
      ? weight.value >= 0
        ? "#d34a43"
        : "#b7791f"
      : weight.value >= 0
        ? "#2f6fbe"
        : "#0f766e";
    context.lineWidth = backward ? 1.4 + magnitude * 6.2 : 1.2 + magnitude * 3.2;
    if (backward) {
      context.shadowBlur = 12 * magnitude;
      context.shadowColor = context.strokeStyle;
    }
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.restore();
  });

  paintBackpropPulses(context, frame, positions);

  graph.layers.forEach((layer, layerIndex) => {
    const layerNodes = graph.nodes.filter((node) => node.layer === layerIndex);
    const first = positions.get(layerNodes[0]?.id ?? "");
    if (!first) {
      return;
    }

    context.fillStyle = "#61707f";
    context.font = "800 10px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(layer.label, first.x, pane.y + pane.height - 14);
  });

  graph.nodes.forEach((node) => {
    const point = positions.get(node.id);
    if (!point) {
      return;
    }

    const activeLayer =
      backprop.pulse.phase === "forward"
        ? Math.min(2, Math.floor(backprop.pulse.progress * 3))
        : Math.max(0, 2 - Math.floor(backprop.pulse.progress * 3));
    const isActive = node.layer === activeLayer;

    context.beginPath();
    context.fillStyle = "#ffffff";
    context.strokeStyle =
      node.layer === 0 ? "#2f6fbe" : node.layer === graph.layers.length - 1 ? "#d34a43" : "#0f766e";
    context.lineWidth = isActive ? 3 : 2;
    if (isActive) {
      context.save();
      context.shadowBlur = 14;
      context.shadowColor = context.strokeStyle;
      context.arc(point.x, point.y, 19, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.restore();
    } else {
      context.arc(point.x, point.y, 17, 0, Math.PI * 2);
      context.fill();
      context.stroke();
    }

    context.fillStyle = "#17212b";
    context.font = "900 11px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(node.label ?? node.id, point.x, point.y + 4);
  });

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    `x=[${backprop.sample.x.join(", ")}], y=${backprop.sample.y} -> yhat=${backprop.prediction.toFixed(4)}`,
    pane.x + 14,
    pane.y + 18,
  );
  context.restore();
}

function paintBackpropPulses(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  positions: Map<string, { x: number; y: number }>,
) {
  const graph = frame.network;
  const backprop = frame.backprop;
  if (!graph || !backprop) {
    return;
  }

  graph.weights.forEach((weight) => {
    const from = positions.get(weight.from);
    const to = positions.get(weight.to);
    const fromNode = graph.nodes.find((node) => node.id === weight.from);
    if (!from || !to || !fromNode) {
      return;
    }

    const phase = backprop.pulse.phase;
    const segmentStart =
      phase === "forward"
        ? fromNode.layer === 0
          ? 0
          : 0.46
        : fromNode.layer === 1
          ? 0
          : 0.46;
    const segmentEnd = segmentStart + 0.54;
    const local = (backprop.pulse.progress - segmentStart) / (segmentEnd - segmentStart);
    if (local < 0 || local > 1) {
      return;
    }

    const eased = 0.5 - Math.cos(local * Math.PI) / 2;
    const start = phase === "forward" ? from : to;
    const end = phase === "forward" ? to : from;
    const x = start.x + (end.x - start.x) * eased;
    const y = start.y + (end.y - start.y) * eased;
    const color = phase === "forward" ? "#2f6fbe" : "#d34a43";

    context.save();
    context.globalAlpha = 0.9;
    context.shadowBlur = 16;
    context.shadowColor = color;
    context.fillStyle = color;
    context.beginPath();
    context.arc(x, y, 5.8, 0, Math.PI * 2);
    context.fill();
    context.restore();
  });
}

function paintBackpropFormulaPane(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  pane: { x: number; y: number; width: number; height: number },
) {
  const backprop = frame.backprop;
  if (!backprop) {
    return;
  }

  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`loss = ${backprop.loss.toFixed(5)}`, pane.x + 14, pane.y + 22);

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    `delta2=${backprop.deltas.output.toFixed(4)} · delta1=[${backprop.deltas.hidden
      .map((value) => value.toFixed(4))
      .join(", ")}]`,
    pane.x + 14,
    pane.y + 42,
  );

  const formulas = backprop.formulas;
  const availableHeight = Math.max(80, pane.height - 60);
  const cardHeight = Math.max(68, Math.min(94, availableHeight / Math.max(1, formulas.length) - 6));
  let y = pane.y + 56;

  formulas.forEach((formula) => {
    context.fillStyle = "#f7fafc";
    context.strokeStyle = "#dce5ea";
    context.fillRect(pane.x + 12, y, pane.width - 24, cardHeight);
    context.strokeRect(pane.x + 12, y, pane.width - 24, cardHeight);

    context.fillStyle = "#17212b";
    context.font = "900 11px Inter, system-ui, sans-serif";
    context.fillText(formula.title, pane.x + 22, y + 17);

    context.fillStyle = "#61707f";
    context.font = "700 10px Inter, system-ui, sans-serif";
    wrapCanvasText(context, formula.expression, pane.x + 22, y + 33, pane.width - 44, 12, 1);
    wrapCanvasText(context, formula.substitution, pane.x + 22, y + 47, pane.width - 44, 12, 1);

    context.fillStyle = "#0f766e";
    context.font = "900 10px Inter, system-ui, sans-serif";
    wrapCanvasText(context, formula.value, pane.x + 22, y + cardHeight - 12, pane.width - 44, 12, 1);

    y += cardHeight + 6;
  });

  context.restore();
}

function layoutNetworkPositions(
  frame: ConceptFrame,
  pane: { x: number; y: number; width: number; height: number },
) {
  const graph = frame.network;
  const nodePositions = new Map<string, { x: number; y: number }>();
  if (!graph) {
    return nodePositions;
  }

  const layerCount = graph.layers.length;
  graph.layers.forEach((_, layerIndex) => {
    const x = pane.x + (layerIndex / Math.max(1, layerCount - 1)) * pane.width;
    const top = pane.y + 18;
    const bottom = pane.y + pane.height - 42;
    const nodes = graph.nodes.filter((node) => node.layer === layerIndex);

    nodes.forEach((node) => {
      const y =
        nodes.length === 1
          ? (top + bottom) / 2
          : top + (node.index / (nodes.length - 1)) * (bottom - top);
      nodePositions.set(node.id, { x, y });
    });
  });

  return nodePositions;
}

function wrapCanvasText(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
  maxLines: number,
) {
  const words = text.split(" ");
  let line = "";
  let lines = 0;

  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (context.measureText(candidate).width > maxWidth && line) {
      context.fillText(
        lines === maxLines - 1 ? fitCanvasText(context, line, maxWidth) : line,
        x,
        y + lines * lineHeight,
      );
      lines += 1;
      line = word;
      if (lines >= maxLines) {
        return;
      }
    } else if (!line && context.measureText(word).width > maxWidth) {
      context.fillText(fitCanvasText(context, word, maxWidth), x, y + lines * lineHeight);
      lines += 1;
      line = "";
      if (lines >= maxLines) {
        return;
      }
    } else {
      line = candidate;
    }
  }

  if (line && lines < maxLines) {
    context.fillText(fitCanvasText(context, line, maxWidth), x, y + lines * lineHeight);
  }
}

function fitCanvasText(
  context: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
) {
  if (context.measureText(text).width <= maxWidth) {
    return text;
  }

  let fitted = text;
  while (fitted.length > 1 && context.measureText(`${fitted}...`).width > maxWidth) {
    fitted = fitted.slice(0, -1);
  }

  return `${fitted}...`;
}

function paintNetworkPane(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  pane: { x: number; y: number; width: number; height: number },
) {
  const graph = frame.network;
  if (!graph) {
    return;
  }

  const layerCount = graph.layers.length;
  const nodePositions = new Map<string, { x: number; y: number }>();

  graph.layers.forEach((layer, layerIndex) => {
    const x = pane.x + (layerIndex / Math.max(1, layerCount - 1)) * pane.width;
    const top = pane.y + 18;
    const bottom = pane.y + pane.height - 40;
    const nodes = graph.nodes.filter((node) => node.layer === layerIndex);

    nodes.forEach((node) => {
      const y =
        nodes.length === 1
          ? (top + bottom) / 2
          : top + (node.index / (nodes.length - 1)) * (bottom - top);
      nodePositions.set(node.id, { x, y });
    });

    context.fillStyle = "#61707f";
    context.font = "800 10px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(layer.label, x, pane.y + pane.height - 12);
    context.fillText(String(layer.units), x, pane.y + pane.height);
  });

  graph.weights.forEach((weight) => {
    const from = nodePositions.get(weight.from);
    const to = nodePositions.get(weight.to);
    if (!from || !to) {
      return;
    }

    const magnitude = Math.min(1, Math.abs(weight.value) / 2.4);
    context.save();
    context.globalAlpha = 0.12 + magnitude * 0.55;
    context.strokeStyle = weight.value >= 0 ? "#2f6fbe" : "#d34a43";
    context.lineWidth = 0.4 + magnitude * 4.2;
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.restore();
  });

  graph.nodes.forEach((node) => {
    const point = nodePositions.get(node.id);
    if (!point) {
      return;
    }

    const layer = graph.layers[node.layer];
    const radius = Math.max(4.5, Math.min(8.5, 12 - layer.units * 0.28));
    context.beginPath();
    context.fillStyle = "#ffffff";
    context.strokeStyle =
      node.layer === 0 ? "#2f6fbe" : node.layer === graph.layers.length - 1 ? "#d34a43" : "#0f766e";
    context.lineWidth = 2;
    context.arc(point.x, point.y, radius, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
}

function paintDecisionBoundaryPane(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  pane: { x: number; y: number; width: number; height: number },
) {
  const plotMargin = { top: 8, right: 12, bottom: 28, left: 34 };
  const plot = {
    x: pane.x + plotMargin.left,
    y: pane.y + plotMargin.top,
    width: pane.width - plotMargin.left - plotMargin.right,
    height: pane.height - plotMargin.top - plotMargin.bottom,
  };
  const xScale = d3.scaleLinear().domain([-3, 3]).range([plot.x, plot.x + plot.width]);
  const yScale = d3.scaleLinear().domain([-3, 3]).range([plot.y + plot.height, plot.y]);
  const cells = frame.heatmap ?? [];
  const gridSize = Math.max(1, Math.round(Math.sqrt(cells.length)));
  const cellWidth = plot.width / gridSize + 1;
  const cellHeight = plot.height / gridSize + 1;

  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();

  cells.forEach((cell) => {
    context.fillStyle = heatmapColor(cell.value);
    context.fillRect(
      xScale(cell.x) - cellWidth / 2,
      yScale(cell.y) - cellHeight / 2,
      cellWidth,
      cellHeight,
    );
  });

  frame.points.forEach((point) => {
    context.beginPath();
    context.fillStyle = pointColor(point);
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1.4;
    context.arc(xScale(point.x), yScale(point.y), 4.5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
  context.restore();

  context.strokeStyle = "#c9d4da";
  context.lineWidth = 1;
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  [-2, 0, 2].forEach((tick) => {
    context.fillText(String(tick), xScale(tick), plot.y + plot.height + 18);
  });
  context.textAlign = "right";
  [-2, 0, 2].forEach((tick) => {
    context.fillText(String(tick), plot.x - 8, yScale(tick) + 3);
  });
}

function paintSeries(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  scales: ReturnType<typeof makeScales>,
) {
  frame.series?.forEach((series) => {
    if (series.points.length < 2) {
      return;
    }

    context.save();
    context.strokeStyle = series.color ?? "#0f766e";
    context.lineWidth = 2.5;
    if (series.dashed) {
      context.setLineDash([7, 7]);
    }
    context.beginPath();
    series.points.forEach((point, index) => {
      const x = scales.x(point.x);
      const y = scales.y(point.y);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
    context.restore();
  });
}

function paintBars(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  scales: ReturnType<typeof makeScales>,
  size: Size,
) {
  const bars = frame.bars ?? [];
  const step =
    bars.length > 1 ? Math.abs(scales.x(1) - scales.x(0)) : size.width * 0.32;
  const barWidth = Math.min(58, Math.max(18, step * 0.62));
  const yZero = scales.y(0);

  bars.forEach((bar, index) => {
    const x = scales.x(index) - barWidth / 2;
    const y = scales.y(bar.value);
    const height = Math.max(2, yZero - y);

    context.fillStyle = bar.color ?? colors[index % colors.length];
    context.fillRect(x, y, barWidth, height);
    context.fillStyle = "#17212b";
    context.font = "700 11px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(bar.label, scales.x(index), size.height - 18);
    context.fillStyle = "#61707f";
    context.fillText(`${Math.round(bar.value * 100)}%`, scales.x(index), y - 8);
  });
}

function paintMarkers(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  scales: ReturnType<typeof makeScales>,
  size: Size,
) {
  frame.markers?.forEach((marker) => {
    context.save();
    context.strokeStyle = marker.color ?? "#b7791f";
    context.fillStyle = marker.color ?? "#b7791f";
    context.lineWidth = 2;
    context.setLineDash([5, 5]);
    context.beginPath();
    if (marker.axis === "x") {
      const x = scales.x(marker.value);
      context.moveTo(x, scales.margin.top);
      context.lineTo(x, size.height - scales.margin.bottom);
      context.stroke();
      context.setLineDash([]);
      context.textAlign = "center";
      context.font = "800 11px Inter, system-ui, sans-serif";
      context.fillText(marker.label, x, scales.margin.top + 14);
    } else {
      const y = scales.y(marker.value);
      context.moveTo(scales.margin.left, y);
      context.lineTo(size.width - scales.margin.right, y);
      context.stroke();
      context.setLineDash([]);
      context.textAlign = "right";
      context.font = "800 11px Inter, system-ui, sans-serif";
      context.fillText(marker.label, size.width - scales.margin.right, y - 8);
    }
    context.restore();
  });
}

function collectPoints(frame: AlgorithmFrame): DataPoint[] {
  if (frame.type === "k-means") {
    return [...frame.points, ...frame.centroids];
  }

  if (frame.type === "concept-demo") {
    const seriesPoints = frame.series?.flatMap((series) => series.points) ?? [];
    const markerPoints =
      frame.markers?.map((marker) =>
        marker.axis === "x"
          ? { x: marker.value, y: frame.points[0]?.y ?? 0 }
          : { x: frame.points[0]?.x ?? 0, y: marker.value },
      ) ?? [];

    return [...frame.points, ...seriesPoints, ...markerPoints, { x: 0, y: 0 }];
  }

  const [minX, maxX] = d3.extent(frame.points, (point) => point.x) as [number, number];
  return [
    ...frame.points,
    { x: minX, y: frame.slope * minX + frame.intercept },
    { x: maxX, y: frame.slope * maxX + frame.intercept },
  ];
}

function readout(frame: AlgorithmFrame) {
  if (frame.type === "linear-regression") {
    return `MSE ${frame.loss.toFixed(4)} · m ${frame.slope.toFixed(3)} · b ${frame.intercept.toFixed(3)}`;
  }

  if (frame.type === "concept-demo") {
    return frame.summary;
  }

  return `Inertia ${frame.inertia.toFixed(3)} · ${frame.centroids.length} centroids`;
}

function pointColor(point: DataPoint) {
  if (point.label === "positive") {
    return "#2f6fbe";
  }

  if (point.label === "negative") {
    return "#0f766e";
  }

  if (point.label === "query") {
    return "#ffffff";
  }

  if (point.label === "anomaly") {
    return "#d34a43";
  }

  if (point.label === "normal") {
    return "#2f6fbe";
  }

  const label = String(point.label ?? "default");
  const index = [...label].reduce((sum, char) => sum + char.charCodeAt(0), 0);
  return colors[index % colors.length];
}

function heatmapColor(value: number) {
  const clamped = Math.max(0, Math.min(1, value));
  const negative = { r: 15, g: 118, b: 110 };
  const positive = { r: 211, g: 74, b: 67 };
  const r = Math.round(negative.r + (positive.r - negative.r) * clamped);
  const g = Math.round(negative.g + (positive.g - negative.g) * clamped);
  const b = Math.round(negative.b + (positive.b - negative.b) * clamped);

  return `rgba(${r}, ${g}, ${b}, 0.28)`;
}
