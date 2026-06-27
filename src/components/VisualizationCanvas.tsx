import * as d3 from "d3";
import { type MouseEvent, useEffect, useRef, useState } from "react";
import type {
  ActivationCurve,
  ActivationFunctionKey,
  ActivationNeuronState,
  AlgorithmDefinition,
  AlgorithmFrame,
  ConceptFrame,
  ContrastivePair,
  DataPoint,
  DynamicProgrammingAction,
  GridWorldCell,
  GridWorldValue,
  KMeansFrame,
  LinearRegressionFrame,
  LossFunctionKey,
  LossSurfaceKey,
  OptimizerRunnerState,
  ParameterState,
  ParameterValue,
} from "../types/algorithm";

const colors = ["#0f766e", "#2f6fbe", "#b7791f", "#d34a43", "#6f58c9", "#258f66"];

type Props = {
  frame: AlgorithmFrame | null;
  algorithm: AlgorithmDefinition;
  params?: ParameterState;
  onParamChange?: (key: string, value: ParameterValue) => void;
};

type Size = {
  width: number;
  height: number;
};

type CanvasPane = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export function VisualizationCanvas({ frame, algorithm, params, onParamChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<Size>({ width: 820, height: 420 });
  const isBackpropLesson = frame?.type === "concept-demo" && Boolean(frame.backprop);
  const isActivationLesson = frame?.type === "concept-demo" && Boolean(frame.activation);
  const isLossLesson = frame?.type === "concept-demo" && Boolean(frame.loss);
  const isOptimizerLesson = frame?.type === "concept-demo" && Boolean(frame.optimizer);
  const isFrameworkLesson = frame?.type === "concept-demo" && Boolean(frame.framework);
  const isStochasticLesson = frame?.type === "concept-demo" && Boolean(frame.stochastic);
  const isSvdLesson = frame?.type === "concept-demo" && Boolean(frame.svd);
  const isConvexLesson = frame?.type === "concept-demo" && Boolean(frame.convex);
  const isConvolutionLesson = frame?.type === "concept-demo" && Boolean(frame.convolution);
  const isDynamicProgrammingLesson = frame?.type === "concept-demo" && Boolean(frame.dynamicProgramming);

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

  const handleCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (
      frame?.type !== "concept-demo" ||
      !frame.dynamicProgramming ||
      !params ||
      !onParamChange
    ) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const bounds = canvas.getBoundingClientRect();
    const point = {
      x: ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * size.width,
      y: ((event.clientY - bounds.top) / Math.max(1, bounds.height)) * size.height,
    };
    const hit = hitTestDynamicProgrammingGrid(frame.dynamicProgramming, size, point);
    if (!hit) {
      return;
    }

    const tool = gridWorldPaintTool(params.paintTool);
    const nextGrid = paintGridWorldCell(frame.dynamicProgramming.grid, hit.row, hit.column, tool);
    onParamChange("gridWorld", nextGrid);
  };

  return (
    <section
      className={`visual-shell ${isBackpropLesson ? "backprop-shell" : ""} ${
        isActivationLesson ? "activation-shell" : ""
      } ${isLossLesson ? "loss-shell" : ""} ${isOptimizerLesson ? "optimizer-shell" : ""} ${
        isFrameworkLesson ? "framework-shell" : ""
      } ${
        isStochasticLesson ? "stochastic-shell" : ""
      } ${
        isSvdLesson ? "svd-shell" : ""
      } ${
        isConvexLesson ? "convex-shell" : ""
      } ${
        isConvolutionLesson ? "convolution-shell" : ""
      } ${
        isDynamicProgrammingLesson ? "dynamic-programming-shell" : ""
      }`}
      aria-label={`${algorithm.name} visualization`}
    >
      <canvas
        key={isConvexLesson ? "webgl-convex" : "canvas-2d"}
        ref={canvasRef}
        onClick={handleCanvasClick}
      />
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

  if (frame?.type === "concept-demo" && frame.convex) {
    paintConvexLessonWebgl(canvas, size, frame);
    return;
  }

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

  if (frame.type === "concept-demo" && frame.activation) {
    paintActivationLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.loss) {
    paintLossLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.optimizer) {
    paintOptimizerLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.framework) {
    paintFrameworkLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.stochastic) {
    paintStochasticLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.svd) {
    paintSvdLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.dynamicProgramming) {
    paintDynamicProgrammingLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.convolution) {
    paintConvolutionLesson(context, frame, size);
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

function paintActivationLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const activation = frame.activation;
  if (!activation) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  const plotWidth = narrow ? size.width - padding * 2 : Math.max(360, size.width * 0.6);
  const plotHeight = narrow ? 190 : Math.max(185, (size.height - 112) / 2);
  const functionPane = {
    x: padding,
    y: 52,
    width: plotWidth,
    height: plotHeight,
  };
  const derivativePane = {
    x: padding,
    y: functionPane.y + functionPane.height + gap + 26,
    width: plotWidth,
    height: plotHeight,
  };
  const neuronPane = narrow
    ? {
        x: padding,
        y: derivativePane.y + derivativePane.height + gap + 26,
        width: size.width - padding * 2,
        height: Math.max(190, size.height - derivativePane.y - derivativePane.height - gap - 48),
      }
    : {
        x: functionPane.x + functionPane.width + gap,
        y: 52,
        width: size.width - functionPane.x - functionPane.width - gap - padding,
        height: size.height - 92,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Interactive function plotter", functionPane.x, functionPane.y - 24);
  context.fillText("Derivative curve", derivativePane.x, derivativePane.y - 24);
  context.fillText("Neuron death lab", neuronPane.x, neuronPane.y - 24);

  paintActivationPlot(context, activation.curves, activation.selected, activation.inputX, "value", functionPane);
  paintActivationPlot(context, activation.curves, activation.selected, activation.inputX, "derivative", derivativePane);
  paintActivationNeuronPane(context, activation.neurons, activation.selected, activation, neuronPane);
}

function paintActivationPlot(
  context: CanvasRenderingContext2D,
  curves: ActivationCurve[],
  selected: ActivationFunctionKey,
  inputX: number,
  mode: "value" | "derivative",
  pane: { x: number; y: number; width: number; height: number },
) {
  const activeCurve = curves.find((curve) => curve.id === selected) ?? curves[0];
  const yDomain = mode === "value" ? [-1.05, 6.15] : [-0.2, 1.25];
  const margin = { top: 18, right: 14, bottom: 30, left: 40 };
  const plot = {
    x: pane.x + margin.left,
    y: pane.y + margin.top,
    width: Math.max(120, pane.width - margin.left - margin.right),
    height: Math.max(90, pane.height - margin.top - margin.bottom),
  };
  const xScale = d3.scaleLinear().domain([-6, 6]).range([plot.x, plot.x + plot.width]);
  const yScale = d3.scaleLinear().domain(yDomain).range([plot.y + plot.height, plot.y]);

  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.strokeStyle = "rgba(97, 112, 127, 0.18)";
  context.lineWidth = 1;
  [-4, -2, 0, 2, 4].forEach((tick) => {
    context.beginPath();
    context.moveTo(xScale(tick), plot.y);
    context.lineTo(xScale(tick), plot.y + plot.height);
    context.stroke();
  });
  yScale.ticks(4).forEach((tick) => {
    context.beginPath();
    context.moveTo(plot.x, yScale(tick));
    context.lineTo(plot.x + plot.width, yScale(tick));
    context.stroke();
  });

  context.strokeStyle = "#aebbc5";
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(plot.x, yScale(0));
  context.lineTo(plot.x + plot.width, yScale(0));
  context.moveTo(xScale(0), plot.y);
  context.lineTo(xScale(0), plot.y + plot.height);
  context.stroke();

  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();

  curves.forEach((curve) => {
    const points = mode === "value" ? curve.points : curve.derivativePoints;
    context.save();
    context.globalAlpha = curve.id === selected ? 0.95 : 0.28;
    context.strokeStyle = curve.color;
    context.lineWidth = curve.id === selected ? 3 : 1.7;
    context.beginPath();
    points.forEach((point, index) => {
      const x = xScale(point.x);
      const y = yScale(point.y);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
    context.restore();
  });

  const activeY = mode === "value" ? activeCurve.value : activeCurve.derivative;
  const dotX = xScale(inputX);
  const dotY = yScale(activeY);
  context.strokeStyle = "rgba(23, 33, 43, 0.26)";
  context.lineWidth = 1;
  context.setLineDash([5, 5]);
  context.beginPath();
  context.moveTo(dotX, plot.y);
  context.lineTo(dotX, plot.y + plot.height);
  context.stroke();
  context.setLineDash([]);

  context.save();
  context.shadowBlur = 14;
  context.shadowColor = activeCurve.color;
  context.fillStyle = activeCurve.color;
  context.beginPath();
  context.arc(dotX, dotY, 6.5, 0, Math.PI * 2);
  context.fill();
  context.restore();
  context.restore();

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  [-6, -3, 0, 3, 6].forEach((tick) => {
    context.fillText(String(tick), xScale(tick), plot.y + plot.height + 18);
  });
  context.textAlign = "right";
  yScale.ticks(4).forEach((tick) => {
    context.fillText(tick.toFixed(mode === "value" ? 0 : 1), plot.x - 8, yScale(tick) + 3);
  });

  context.textAlign = "left";
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  const valueLabel = mode === "value" ? "f" : "f'";
  context.fillText(
    `${activeCurve.label} ${valueLabel}(${inputX.toFixed(2)})=${activeY.toFixed(3)}`,
    pane.x + 12,
    pane.y + 16,
  );

  let legendX = pane.x + 12;
  const legendY = pane.y + pane.height - 8;
  curves.forEach((curve) => {
    context.fillStyle = curve.color;
    context.fillRect(legendX, legendY - 8, 10, 3);
    context.fillStyle = curve.id === selected ? "#17212b" : "#61707f";
    context.font = curve.id === selected ? "900 10px Inter, system-ui, sans-serif" : "700 10px Inter, system-ui, sans-serif";
    context.fillText(curve.label, legendX + 14, legendY - 5);
    legendX += Math.min(92, Math.max(64, context.measureText(curve.label).width + 28));
  });
}

function paintActivationNeuronPane(
  context: CanvasRenderingContext2D,
  neurons: ActivationNeuronState[],
  selected: ActivationFunctionKey,
  activation: NonNullable<ConceptFrame["activation"]>,
  pane: { x: number; y: number; width: number; height: number },
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  const label = activationLabel(selected);
  const subtitle =
    selected === "relu"
      ? `${activation.deadCount} dead zero-gradient units`
      : selected === "sigmoid"
        ? `${activation.saturatedCount} saturated low-gradient units`
        : `${activation.recoveryCount} negative units still learning`;
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(label, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, `${subtitle} · shift ${activation.negativeShift.toFixed(1)} · lr ${activation.learningRate.toFixed(2)}`, pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  const graph = {
    x: pane.x + 20,
    y: pane.y + 66,
    width: pane.width - 40,
    height: Math.max(120, pane.height - 88),
  };
  const inputX = graph.x + 14;
  const hiddenX = graph.x + graph.width * 0.52;
  const outputX = graph.x + graph.width - 18;
  const inputYs = [graph.y + graph.height * 0.24, graph.y + graph.height * 0.5, graph.y + graph.height * 0.76];
  const outputY = graph.y + graph.height * 0.5;
  const hiddenYs = neurons.map((_, index) =>
    neurons.length === 1
      ? graph.y + graph.height / 2
      : graph.y + (index / (neurons.length - 1)) * graph.height,
  );

  context.strokeStyle = "rgba(97, 112, 127, 0.17)";
  context.lineWidth = 1;
  inputYs.forEach((y) => {
    hiddenYs.forEach((hiddenY) => {
      context.beginPath();
      context.moveTo(inputX, y);
      context.lineTo(hiddenX, hiddenY);
      context.stroke();
    });
  });
  hiddenYs.forEach((hiddenY) => {
    context.beginPath();
    context.moveTo(hiddenX, hiddenY);
    context.lineTo(outputX, outputY);
    context.stroke();
  });

  inputYs.forEach((y, index) => {
    paintActivationNode(context, inputX, y, `x${index + 1}`, "#2f6fbe", "#ffffff");
  });
  paintActivationNode(context, outputX, outputY, "y", "#17212b", "#ffffff");

  neurons.forEach((neuron, index) => {
    const y = hiddenYs[index];
    const color = neuronStatusColor(neuron.status);
    paintActivationNode(context, hiddenX, y, neuron.label, color, neuron.status === "dead" ? "#fff5f5" : "#ffffff");

    const textX = hiddenX + 22;
    context.fillStyle = color;
    context.font = "900 10px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(neuron.status, textX, y - 3);
    context.fillStyle = "#61707f";
    context.font = "700 9px Inter, system-ui, sans-serif";
    context.fillText(`z=${neuron.preActivation.toFixed(2)}  grad=${neuron.derivative.toFixed(3)}`, textX, y + 10);
  });

  context.restore();
}

function paintActivationNode(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  label: string,
  color: string,
  fill: string,
) {
  context.beginPath();
  context.fillStyle = fill;
  context.strokeStyle = color;
  context.lineWidth = 2.2;
  context.arc(x, y, 13, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "#17212b";
  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(label, x, y + 3);
}

function neuronStatusColor(status: ActivationNeuronState["status"]) {
  if (status === "dead") {
    return "#d34a43";
  }

  if (status === "recovering") {
    return "#0f766e";
  }

  if (status === "saturated") {
    return "#b7791f";
  }

  if (status === "inactive") {
    return "#8a98a5";
  }

  return "#2f6fbe";
}

function activationLabel(selected: ActivationFunctionKey) {
  if (selected === "leaky-relu") {
    return "LeakyReLU";
  }

  if (selected === "relu") {
    return "ReLU";
  }

  if (selected === "gelu") {
    return "GELU";
  }

  return "Sigmoid";
}

function paintLossLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const loss = frame.loss;
  if (!loss) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  const curvePane = narrow
    ? {
        x: padding,
        y: 52,
        width: size.width - padding * 2,
        height: Math.max(260, size.height * 0.44),
      }
    : {
        x: padding,
        y: 52,
        width: Math.max(390, size.width * 0.58),
        height: size.height - 92,
      };
  const vectorPane = narrow
    ? {
        x: padding,
        y: curvePane.y + curvePane.height + gap + 28,
        width: size.width - padding * 2,
        height: Math.max(230, size.height - curvePane.y - curvePane.height - gap - 58),
      }
    : {
        x: curvePane.x + curvePane.width + gap,
        y: 52,
        width: size.width - curvePane.x - curvePane.width - gap - padding,
        height: size.height - 92,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Real-time loss curve monitor", curvePane.x, curvePane.y - 24);
  context.fillText("Contrastive vector space", vectorPane.x, vectorPane.y - 24);

  paintLossCurvePane(context, loss, curvePane);
  paintContrastivePane(context, loss.contrastivePairs, loss.selected, loss.margin, vectorPane);
}

function paintLossCurvePane(
  context: CanvasRenderingContext2D,
  loss: NonNullable<ConceptFrame["loss"]>,
  pane: { x: number; y: number; width: number; height: number },
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  const margin = { top: 46, right: 22, bottom: 92, left: 54 };
  const plot = {
    x: pane.x + margin.left,
    y: pane.y + margin.top,
    width: Math.max(160, pane.width - margin.left - margin.right),
    height: Math.max(135, pane.height - margin.top - margin.bottom),
  };
  const history = loss.lossHistory;
  const maxEpoch = Math.max(12, history[history.length - 1]?.x ?? 12);
  const maxLoss = Math.max(0.1, ...history.map((point) => point.y)) * 1.18;
  const xScale = d3.scaleLinear().domain([1, maxEpoch]).range([plot.x, plot.x + plot.width]);
  const yScale = d3.scaleLinear().domain([0, maxLoss]).range([plot.y + plot.height, plot.y]);
  const color = lossStatusColor(loss.status);

  context.fillStyle =
    loss.status === "diverging"
      ? "rgba(211, 74, 67, 0.06)"
      : loss.status === "oscillating"
        ? "rgba(183, 121, 31, 0.07)"
        : "rgba(15, 118, 110, 0.055)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);

  context.strokeStyle = "rgba(97, 112, 127, 0.2)";
  context.lineWidth = 1;
  xScale.ticks(5).forEach((tick) => {
    context.beginPath();
    context.moveTo(xScale(tick), plot.y);
    context.lineTo(xScale(tick), plot.y + plot.height);
    context.stroke();
  });
  yScale.ticks(5).forEach((tick) => {
    context.beginPath();
    context.moveTo(plot.x, yScale(tick));
    context.lineTo(plot.x + plot.width, yScale(tick));
    context.stroke();
  });

  context.strokeStyle = "#aebbc5";
  context.lineWidth = 1.2;
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  if (history.length > 0) {
    context.save();
    context.beginPath();
    context.rect(plot.x, plot.y, plot.width, plot.height);
    context.clip();
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    history.forEach((point, index) => {
      const x = xScale(point.x);
      const y = yScale(point.y);
      if (index === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();

    const current = history[history.length - 1];
    context.shadowBlur = 14;
    context.shadowColor = color;
    context.fillStyle = color;
    context.beginPath();
    context.arc(xScale(current.x), yScale(current.y), 6.5, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  xScale.ticks(5).forEach((tick) => {
    context.fillText(String(Math.round(tick)), xScale(tick), plot.y + plot.height + 18);
  });
  context.fillText("Epochs", plot.x + plot.width / 2, plot.y + plot.height + 38);

  context.textAlign = "right";
  yScale.ticks(5).forEach((tick) => {
    context.fillText(tick.toFixed(2), plot.x - 8, yScale(tick) + 3);
  });

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    `${lossDisplayName(loss.selected)} · epoch ${loss.epoch} · loss ${loss.currentLoss.toFixed(4)}`,
    pane.x + 14,
    pane.y + 22,
  );
  context.fillStyle = color;
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText(`trend: ${loss.status}`, pane.x + 14, pane.y + 41);

  paintLossPredictionStrip(context, loss, {
    x: pane.x + 14,
    y: pane.y + pane.height - 55,
    width: pane.width - 28,
    height: 40,
  });

  context.restore();
}

function paintLossPredictionStrip(
  context: CanvasRenderingContext2D,
  loss: NonNullable<ConceptFrame["loss"]>,
  pane: { x: number; y: number; width: number; height: number },
) {
  const items = loss.predictions.slice(0, 4);
  if (items.length === 0) {
    return;
  }

  const gap = 8;
  const width = (pane.width - gap * (items.length - 1)) / items.length;
  items.forEach((item, index) => {
    const x = pane.x + index * (width + gap);
    const prediction = loss.selected === "contrastive" ? Math.min(1, item.prediction / Math.max(0.1, loss.margin)) : item.prediction;
    context.fillStyle = "#f7fafc";
    context.strokeStyle = "#dce5ea";
    context.fillRect(x, pane.y, width, pane.height);
    context.strokeRect(x, pane.y, width, pane.height);
    context.fillStyle = item.target >= 0.5 ? "#2f6fbe" : "#0f766e";
    context.fillRect(x + 8, pane.y + pane.height - 11, Math.max(2, (width - 16) * prediction), 4);
    context.fillStyle = "#17212b";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(item.label, x + 8, pane.y + 13);
    context.fillStyle = "#61707f";
    context.font = "700 9px Inter, system-ui, sans-serif";
    context.fillText(
      loss.selected === "contrastive" ? `d=${item.prediction.toFixed(2)}` : `p=${item.prediction.toFixed(2)}`,
      x + 8,
      pane.y + 26,
    );
  });
}

function paintContrastivePane(
  context: CanvasRenderingContext2D,
  pairs: ContrastivePair[],
  selected: LossFunctionKey,
  margin: number,
  pane: { x: number; y: number; width: number; height: number },
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  const points = uniqueContrastivePoints(pairs);
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const xPad = Math.max(0.35, (Math.max(...xs) - Math.min(...xs)) * 0.16);
  const yPad = Math.max(0.35, (Math.max(...ys) - Math.min(...ys)) * 0.16);
  const marginBox = { top: 54, right: 22, bottom: 34, left: 34 };
  const plot = {
    x: pane.x + marginBox.left,
    y: pane.y + marginBox.top,
    width: Math.max(130, pane.width - marginBox.left - marginBox.right),
    height: Math.max(130, pane.height - marginBox.top - marginBox.bottom),
  };
  const xScale = d3
    .scaleLinear()
    .domain([Math.min(...xs) - xPad, Math.max(...xs) + xPad])
    .range([plot.x, plot.x + plot.width]);
  const yScale = d3
    .scaleLinear()
    .domain([Math.min(...ys) - yPad, Math.max(...ys) + yPad])
    .range([plot.y + plot.height, plot.y]);
  const active = selected === "contrastive";

  context.fillStyle = active ? "rgba(47, 111, 190, 0.045)" : "rgba(97, 112, 127, 0.045)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#c9d4da";
  context.lineWidth = 1;
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  pairs.forEach((pair) => {
    const fromX = xScale(pair.from.x);
    const fromY = yScale(pair.from.y);
    const toX = xScale(pair.to.x);
    const toY = yScale(pair.to.y);
    const color = pair.relation === "similar" ? "#0f766e" : "#d34a43";
    const magnitude = Math.min(1, pair.gradientMagnitude / 3);

    context.save();
    context.globalAlpha = active ? 0.34 + magnitude * 0.44 : 0.18;
    context.strokeStyle = color;
    context.lineWidth = 1.2 + magnitude * 4;
    if (pair.relation === "different") {
      context.setLineDash([6, 5]);
    }
    context.beginPath();
    context.moveTo(fromX, fromY);
    context.lineTo(toX, toY);
    context.stroke();
    context.restore();

    paintContrastiveForce(context, fromX, fromY, toX, toY, pair.relation, color, active);
  });

  points.forEach((point) => {
    const color = point.id.startsWith("a") ? "#2f6fbe" : "#b7791f";
    const x = xScale(point.x);
    const y = yScale(point.y);
    context.beginPath();
    context.fillStyle = "#ffffff";
    context.strokeStyle = color;
    context.lineWidth = 2.5;
    context.arc(x, y, 8.5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = "#17212b";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(point.label, x, y - 12);
  });

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(active ? "Active contrastive step" : "Contrastive reference", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, `similar pairs pull together · different pairs push until margin ${margin.toFixed(2)}`, pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  context.restore();
}

function paintContrastiveForce(
  context: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  relation: "similar" | "different",
  color: string,
  active: boolean,
) {
  if (!active) {
    return;
  }

  const midX = (fromX + toX) / 2;
  const midY = (fromY + toY) / 2;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const length = Math.max(1, Math.hypot(dx, dy));
  const unitX = dx / length;
  const unitY = dy / length;
  const offset = relation === "similar" ? 14 : -14;

  context.save();
  context.fillStyle = color;
  context.globalAlpha = 0.75;
  [-1, 1].forEach((side) => {
    const x = midX + unitX * offset * side;
    const y = midY + unitY * offset * side;
    context.beginPath();
    context.arc(x, y, 3.2, 0, Math.PI * 2);
    context.fill();
  });
  context.restore();
}

function uniqueContrastivePoints(pairs: ContrastivePair[]) {
  const map = new Map<string, ContrastivePair["from"]>();
  pairs.forEach((pair) => {
    map.set(pair.from.id, pair.from);
    map.set(pair.to.id, pair.to);
  });
  return [...map.values()];
}

function lossStatusColor(status: NonNullable<ConceptFrame["loss"]>["status"]) {
  if (status === "diverging") {
    return "#d34a43";
  }

  if (status === "oscillating") {
    return "#b7791f";
  }

  return "#0f766e";
}

function lossDisplayName(loss: LossFunctionKey) {
  if (loss === "mse") {
    return "MSE";
  }

  if (loss === "contrastive") {
    return "Contrastive";
  }

  return "Cross-Entropy";
}

function paintOptimizerLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const optimizer = frame.optimizer;
  if (!optimizer) {
    return;
  }

  const narrow = size.width < 780;
  const padding = 24;
  const gap = 18;
  const surfacePane = narrow
    ? {
        x: padding,
        y: 52,
        width: size.width - padding * 2,
        height: Math.max(310, size.height * 0.52),
      }
    : {
        x: padding,
        y: 52,
        width: Math.max(430, size.width * 0.62),
        height: size.height - 92,
      };
  const adaptivePane = narrow
    ? {
        x: padding,
        y: surfacePane.y + surfacePane.height + gap + 28,
        width: size.width - padding * 2,
        height: Math.max(230, size.height - surfacePane.y - surfacePane.height - gap - 58),
      }
    : {
        x: surfacePane.x + surfacePane.width + gap,
        y: 52,
        width: size.width - surfacePane.x - surfacePane.width - gap - padding,
        height: size.height - 92,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Loss surface optimization race", surfacePane.x, surfacePane.y - 24);
  context.fillText("Adam adaptive learning rates", adaptivePane.x, adaptivePane.y - 24);

  paintOptimizerSurfacePane(context, optimizer, surfacePane);
  paintAdaptiveOptimizerPane(context, optimizer, adaptivePane);
}

function paintOptimizerSurfacePane(
  context: CanvasRenderingContext2D,
  optimizer: NonNullable<ConceptFrame["optimizer"]>,
  pane: { x: number; y: number; width: number; height: number },
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${optimizerSurfaceLabel(optimizer.surface)} · step ${optimizer.step}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `eta ${optimizer.learningRate.toFixed(3)} · beta1 ${optimizer.beta1.toFixed(2)} · beta2 ${optimizer.beta2.toFixed(3)} · lambda ${optimizer.weightDecay.toFixed(3)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const plot = {
    x: pane.x + 20,
    y: pane.y + 58,
    width: pane.width - 40,
    height: pane.height - 82,
  };

  paintOptimizerMesh(context, optimizer.surface, plot);
  paintOptimizerMinimum(context, optimizer.surface, plot);
  optimizer.runners.forEach((runner) => {
    paintOptimizerTrace(context, optimizer.surface, runner, plot);
  });

  context.restore();
}

function paintOptimizerMesh(
  context: CanvasRenderingContext2D,
  surface: LossSurfaceKey,
  pane: { x: number; y: number; width: number; height: number },
) {
  const bounds = optimizerSurfaceBounds(surface);
  const lines = 24;

  context.save();
  context.beginPath();
  context.rect(pane.x, pane.y, pane.width, pane.height);
  context.clip();
  context.fillStyle = "rgba(247, 250, 252, 0.86)";
  context.fillRect(pane.x, pane.y, pane.width, pane.height);

  for (let row = 0; row <= lines; row += 1) {
    const y = bounds.y[0] + (row / lines) * (bounds.y[1] - bounds.y[0]);
    drawOptimizerSurfaceLine(
      context,
      surface,
      pane,
      Array.from({ length: lines + 1 }, (_, column) => ({
        x: bounds.x[0] + (column / lines) * (bounds.x[1] - bounds.x[0]),
        y,
      })),
    );
  }

  for (let column = 0; column <= lines; column += 1) {
    const x = bounds.x[0] + (column / lines) * (bounds.x[1] - bounds.x[0]);
    drawOptimizerSurfaceLine(
      context,
      surface,
      pane,
      Array.from({ length: lines + 1 }, (_, row) => ({
        x,
        y: bounds.y[0] + (row / lines) * (bounds.y[1] - bounds.y[0]),
      })),
    );
  }

  context.restore();
}

function drawOptimizerSurfaceLine(
  context: CanvasRenderingContext2D,
  surface: LossSurfaceKey,
  pane: { x: number; y: number; width: number; height: number },
  points: Array<{ x: number; y: number }>,
) {
  context.beginPath();
  points.forEach((point, index) => {
    const projected = projectOptimizerPoint(surface, point.x, point.y, optimizerSurfaceValue(surface, point.x, point.y), pane);
    if (index === 0) {
      context.moveTo(projected.x, projected.y);
    } else {
      context.lineTo(projected.x, projected.y);
    }
  });
  context.strokeStyle = "rgba(97, 112, 127, 0.22)";
  context.lineWidth = 0.8;
  context.stroke();
}

function paintOptimizerTrace(
  context: CanvasRenderingContext2D,
  surface: LossSurfaceKey,
  runner: OptimizerRunnerState,
  pane: { x: number; y: number; width: number; height: number },
) {
  if (runner.history.length === 0) {
    return;
  }

  context.save();
  context.beginPath();
  context.rect(pane.x, pane.y, pane.width, pane.height);
  context.clip();
  context.strokeStyle = runner.color;
  context.lineWidth = 3;
  context.globalAlpha = 0.9;
  context.beginPath();
  runner.history.forEach((point, index) => {
    const projected = projectOptimizerPoint(surface, point.x, point.y, point.z, pane);
    if (index === 0) {
      context.moveTo(projected.x, projected.y);
    } else {
      context.lineTo(projected.x, projected.y);
    }
  });
  context.stroke();

  const current = runner.position;
  const projected = projectOptimizerPoint(surface, current.x, current.y, current.z, pane);
  context.shadowBlur = 16;
  context.shadowColor = runner.color;
  context.fillStyle = runner.color;
  context.beginPath();
  context.arc(projected.x, projected.y, 6.8, 0, Math.PI * 2);
  context.fill();
  context.shadowBlur = 0;
  context.fillStyle = "#17212b";
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(runner.label, projected.x + 9, projected.y - 7);
  context.fillStyle = "#61707f";
  context.font = "700 9px Inter, system-ui, sans-serif";
  context.fillText(`L=${current.z.toFixed(current.z > 99 ? 0 : 2)}`, projected.x + 9, projected.y + 6);
  context.restore();
}

function paintOptimizerMinimum(
  context: CanvasRenderingContext2D,
  surface: LossSurfaceKey,
  pane: { x: number; y: number; width: number; height: number },
) {
  const minimum = surface === "beale" ? { x: 3, y: 0.5, z: 0 } : { x: 1, y: 1, z: 0 };
  const point = projectOptimizerPoint(surface, minimum.x, minimum.y, minimum.z, pane);
  context.save();
  context.fillStyle = "#17212b";
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("minimum", point.x + 8, point.y + 3);
  context.restore();
}

function projectOptimizerPoint(
  surface: LossSurfaceKey,
  x: number,
  y: number,
  z: number,
  pane: { x: number; y: number; width: number; height: number },
) {
  const bounds = optimizerSurfaceBounds(surface);
  const nx = (x - (bounds.x[0] + bounds.x[1]) / 2) / (bounds.x[1] - bounds.x[0]);
  const ny = (y - (bounds.y[0] + bounds.y[1]) / 2) / (bounds.y[1] - bounds.y[0]);
  const zNorm = Math.min(1, Math.log1p(Math.max(0, z)) / Math.log1p(optimizerSurfaceMax(surface)));

  return {
    x: pane.x + pane.width * 0.5 + (nx - ny) * pane.width * 0.72,
    y: pane.y + pane.height * 0.68 + (nx + ny) * pane.height * 0.34 - zNorm * pane.height * 0.5,
  };
}

function optimizerSurfaceValue(surface: LossSurfaceKey, x: number, y: number) {
  if (surface === "beale") {
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2;
  }

  return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2;
}

function optimizerSurfaceBounds(surface: LossSurfaceKey) {
  if (surface === "beale") {
    return { x: [-4.5, 4.5] as [number, number], y: [-4.5, 4.5] as [number, number] };
  }

  return { x: [-2.2, 2.2] as [number, number], y: [-1.2, 2.8] as [number, number] };
}

function optimizerSurfaceMax(surface: LossSurfaceKey) {
  const bounds = optimizerSurfaceBounds(surface);
  const sampleCount = 18;
  let max = 1;
  for (let row = 0; row <= sampleCount; row += 1) {
    for (let column = 0; column <= sampleCount; column += 1) {
      const x = bounds.x[0] + (column / sampleCount) * (bounds.x[1] - bounds.x[0]);
      const y = bounds.y[0] + (row / sampleCount) * (bounds.y[1] - bounds.y[0]);
      max = Math.max(max, optimizerSurfaceValue(surface, x, y));
    }
  }
  return max;
}

function optimizerSurfaceLabel(surface: LossSurfaceKey) {
  return surface === "beale" ? "Beale function" : "Rosenbrock banana";
}

function paintAdaptiveOptimizerPane(
  context: CanvasRenderingContext2D,
  optimizer: NonNullable<ConceptFrame["optimizer"]>,
  pane: { x: number; y: number; width: number; height: number },
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Adam denominator sqrt(vhat)", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, "thicker red lines mean big gradients; compressed eta_i keeps the update stable", pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  const graph = {
    x: pane.x + 24,
    y: pane.y + 70,
    width: pane.width - 48,
    height: pane.height - 112,
  };
  const positions = new Map<string, { x: number; y: number }>();
  const layers = [
    { ids: ["x1", "x2"], x: graph.x + 12 },
    { ids: ["h1", "h2", "h3"], x: graph.x + graph.width * 0.52 },
    { ids: ["y"], x: graph.x + graph.width - 12 },
  ];

  layers.forEach((layer) => {
    layer.ids.forEach((id, index) => {
      const y =
        layer.ids.length === 1
          ? graph.y + graph.height / 2
          : graph.y + (index / (layer.ids.length - 1)) * graph.height;
      positions.set(id, { x: layer.x, y });
    });
  });

  const maxGradient = Math.max(0.01, ...optimizer.adaptiveWeights.map((weight) => weight.gradient));
  const maxRate = Math.max(0.001, ...optimizer.adaptiveWeights.map((weight) => weight.adaptiveRate));
  optimizer.adaptiveWeights.forEach((weight) => {
    const from = positions.get(weight.from);
    const to = positions.get(weight.to);
    if (!from || !to) {
      return;
    }

    const gradientRatio = weight.gradient / maxGradient;
    const rateRatio = weight.adaptiveRate / maxRate;
    context.save();
    context.strokeStyle = adaptiveRateColor(rateRatio);
    context.globalAlpha = 0.24 + gradientRatio * 0.62;
    context.lineWidth = 1 + gradientRatio * 6;
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.restore();
  });

  positions.forEach((point, id) => {
    context.beginPath();
    context.fillStyle = "#ffffff";
    context.strokeStyle = id.startsWith("x") ? "#2f6fbe" : id.startsWith("h") ? "#0f766e" : "#17212b";
    context.lineWidth = 2.2;
    context.arc(point.x, point.y, 13, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = "#17212b";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(id, point.x, point.y + 3);
  });

  const largest = [...optimizer.adaptiveWeights].sort((a, b) => b.gradient - a.gradient)[0];
  if (largest) {
    context.fillStyle = "#61707f";
    context.font = "700 10px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(
      `largest gradient ${largest.from}->${largest.to}: |g|=${largest.gradient.toFixed(3)}, sqrt(vhat)=${largest.denominator.toFixed(3)}, eta_i=${largest.adaptiveRate.toFixed(4)}`,
      pane.x + 14,
      pane.y + pane.height - 24,
    );
  }

  context.restore();
}

function adaptiveRateColor(rateRatio: number) {
  const clamped = Math.max(0, Math.min(1, rateRatio));
  const compressed = { r: 211, g: 74, b: 67 };
  const free = { r: 47, g: 111, b: 190 };
  const r = Math.round(compressed.r + (free.r - compressed.r) * clamped);
  const g = Math.round(compressed.g + (free.g - compressed.g) * clamped);
  const b = Math.round(compressed.b + (free.b - compressed.b) * clamped);

  return `rgb(${r}, ${g}, ${b})`;
}

function paintDynamicProgrammingLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const dynamicProgramming = frame.dynamicProgramming;
  if (!dynamicProgramming) {
    return;
  }

  const layout = dynamicProgrammingLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Gridworld value matrix", layout.gridPane.x, layout.gridPane.y - 24);
  context.fillText("Bellman update loop", layout.infoPane.x, layout.infoPane.y - 24);

  paintDynamicProgrammingGrid(context, dynamicProgramming, layout.gridPane);
  paintDynamicProgrammingBellmanPane(context, dynamicProgramming, layout.infoPane, frame.iteration);
}

function dynamicProgrammingLayout(size: Size) {
  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  if (narrow) {
    const gridHeight = Math.min(460, Math.max(330, size.height * 0.52));
    return {
      gridPane: {
        x: padding,
        y: 52,
        width: size.width - padding * 2,
        height: gridHeight,
      },
      infoPane: {
        x: padding,
        y: 52 + gridHeight + gap + 26,
        width: size.width - padding * 2,
        height: Math.max(260, size.height - 52 - gridHeight - gap - 82),
      },
    };
  }

  const gridWidth = Math.max(440, size.width * 0.62);
  return {
    gridPane: {
      x: padding,
      y: 58,
      width: gridWidth,
      height: size.height - 96,
    },
    infoPane: {
      x: padding + gridWidth + gap,
      y: 58,
      width: size.width - padding * 2 - gridWidth - gap,
      height: size.height - 96,
    },
  };
}

function dynamicProgrammingGridBox(
  dynamicProgramming: NonNullable<ConceptFrame["dynamicProgramming"]>,
  pane: CanvasPane,
) {
  const rows = dynamicProgramming.grid.rows;
  const columns = dynamicProgramming.grid.columns;
  const available = {
    x: pane.x + 18,
    y: pane.y + 62,
    width: pane.width - 36,
    height: pane.height - 84,
  };
  const cellSize = Math.min(available.width / columns, available.height / rows);
  return {
    x: available.x + (available.width - columns * cellSize) / 2,
    y: available.y + (available.height - rows * cellSize) / 2,
    width: columns * cellSize,
    height: rows * cellSize,
    cellSize,
  };
}

function paintDynamicProgrammingGrid(
  context: CanvasRenderingContext2D,
  dynamicProgramming: NonNullable<ConceptFrame["dynamicProgramming"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${dynamicProgrammingMethodLabel(dynamicProgramming.method)} sweep ${dynamicProgramming.sweep}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `gamma ${dynamicProgramming.gamma.toFixed(2)} · delta ${dynamicProgramming.delta.toFixed(4)} · click cells with the selected paint tool`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const box = dynamicProgrammingGridBox(dynamicProgramming, pane);
  const values = dynamicProgramming.values.flatMap((row, rowIndex) =>
    row.filter((_, columnIndex) => dynamicProgramming.grid.cells[rowIndex][columnIndex] !== "wall"),
  );
  const positiveMax = Math.max(0.001, ...values.filter((value) => value > 0));
  const negativeMin = Math.min(-0.001, ...values.filter((value) => value < 0));

  for (let row = 0; row < dynamicProgramming.grid.rows; row += 1) {
    for (let column = 0; column < dynamicProgramming.grid.columns; column += 1) {
      const cell = dynamicProgramming.grid.cells[row][column];
      const value = dynamicProgramming.values[row][column];
      const x = box.x + column * box.cellSize;
      const y = box.y + row * box.cellSize;
      context.fillStyle = dynamicProgrammingCellColor(cell, value, positiveMax, negativeMin);
      context.fillRect(x, y, box.cellSize, box.cellSize);
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.4;
      context.strokeRect(x, y, box.cellSize, box.cellSize);

      const active = dynamicProgramming.activeCell.row === row && dynamicProgramming.activeCell.column === column;
      if (active) {
        context.strokeStyle = "#2f6fbe";
        context.lineWidth = Math.max(2, box.cellSize * 0.08);
        context.strokeRect(x + 2, y + 2, box.cellSize - 4, box.cellSize - 4);
      }

      paintDynamicProgrammingCellText(context, cell, value, x, y, box.cellSize);
    }
  }

  if (dynamicProgramming.stable) {
    dynamicProgramming.policy.forEach((policyCell) => {
      const x = box.x + policyCell.column * box.cellSize;
      const y = box.y + policyCell.row * box.cellSize;
      paintPolicyArrow(context, policyCell.action, x, y, box.cellSize, dynamicProgramming.sweep);
    });
  }

  context.restore();
}

function paintDynamicProgrammingCellText(
  context: CanvasRenderingContext2D,
  cell: GridWorldCell,
  value: number,
  x: number,
  y: number,
  cellSize: number,
) {
  const label = cell === "wall" ? "WALL" : cell === "gold" ? "GOLD" : cell === "fire" ? "FIRE" : cell === "start" ? "START" : "";
  context.textAlign = "center";
  context.fillStyle = cell === "wall" ? "#f7fafc" : "#17212b";
  const valueFontSize = Math.max(9, Math.min(17, cellSize * 0.24));
  context.font = `900 ${valueFontSize}px Inter, system-ui, sans-serif`;
  context.fillText(value.toFixed(Math.abs(value) >= 10 ? 0 : 1), x + cellSize / 2, y + cellSize / 2 + 5);
  if (label && cellSize > 42) {
    context.fillStyle = cell === "wall" ? "#cfd8df" : "#4d5c68";
    context.font = "900 8px Inter, system-ui, sans-serif";
    context.fillText(label, x + cellSize / 2, y + 13);
  }
}

function paintPolicyArrow(
  context: CanvasRenderingContext2D,
  action: DynamicProgrammingAction,
  x: number,
  y: number,
  cellSize: number,
  iteration: number,
) {
  const direction: Record<DynamicProgrammingAction, [number, number]> = {
    up: [0, -1],
    right: [1, 0],
    down: [0, 1],
    left: [-1, 0],
  };
  const [dx, dy] = direction[action];
  const pulse = 0.82 + Math.sin(iteration * 0.42) * 0.1;
  const center = { x: x + cellSize / 2, y: y + cellSize / 2 };
  const length = cellSize * 0.28 * pulse;
  const end = { x: center.x + dx * length, y: center.y + dy * length };
  const angle = Math.atan2(dy, dx);

  context.save();
  context.strokeStyle = "rgba(23, 33, 43, 0.76)";
  context.fillStyle = "rgba(23, 33, 43, 0.76)";
  context.lineWidth = Math.max(1.8, cellSize * 0.045);
  context.beginPath();
  context.moveTo(center.x - dx * cellSize * 0.12, center.y - dy * cellSize * 0.12);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.translate(end.x, end.y);
  context.rotate(angle);
  const arrow = Math.max(5, cellSize * 0.11);
  context.beginPath();
  context.moveTo(arrow, 0);
  context.lineTo(-arrow * 0.72, -arrow * 0.56);
  context.lineTo(-arrow * 0.72, arrow * 0.56);
  context.closePath();
  context.fill();
  context.restore();
}

function paintDynamicProgrammingBellmanPane(
  context: CanvasRenderingContext2D,
  dynamicProgramming: NonNullable<ConceptFrame["dynamicProgramming"]>,
  pane: CanvasPane,
  iteration: number,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.82)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`State (${dynamicProgramming.activeCell.row}, ${dynamicProgramming.activeCell.column})`, pane.x + 14, pane.y + 22);
  context.fillStyle = dynamicProgramming.stable ? "#0f766e" : "#b7791f";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText(dynamicProgramming.stable ? "Policy arrows unlocked" : "Values still propagating", pane.x + 14, pane.y + 42);

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  wrapCanvasText(
    context,
    "V(s) is overwritten by scanning rows, columns, and four actions with a stochastic transition model.",
    pane.x + 14,
    pane.y + 64,
    pane.width - 28,
    14,
    3,
  );

  const chart = {
    x: pane.x + 14,
    y: pane.y + 114,
    width: pane.width - 28,
    height: Math.min(210, pane.height - 190),
  };
  const minValue = Math.min(-0.1, ...dynamicProgramming.actionValues.map((item) => item.value));
  const maxValue = Math.max(0.1, ...dynamicProgramming.actionValues.map((item) => item.value));
  const zeroY = d3.scaleLinear().domain([minValue, maxValue]).range([chart.y + chart.height, chart.y])(0);

  context.strokeStyle = "#dce5ea";
  context.strokeRect(chart.x, chart.y, chart.width, chart.height);
  context.strokeStyle = "rgba(97, 112, 127, 0.32)";
  context.beginPath();
  context.moveTo(chart.x, zeroY);
  context.lineTo(chart.x + chart.width, zeroY);
  context.stroke();

  const barGap = 8;
  const barWidth = (chart.width - barGap * 3) / 4;
  dynamicProgramming.actionValues.forEach((item, index) => {
    const heightScale = d3.scaleLinear().domain([minValue, maxValue]).range([chart.y + chart.height, chart.y]);
    const yValue = heightScale(item.value);
    const x = chart.x + index * (barWidth + barGap);
    const y = Math.min(yValue, zeroY);
    const height = Math.max(2, Math.abs(zeroY - yValue));
    const best = item.value === Math.max(...dynamicProgramming.actionValues.map((candidate) => candidate.value));
    context.fillStyle = best ? "#0f766e" : item.value < 0 ? "#d34a43" : "#2f6fbe";
    context.globalAlpha = best ? 0.92 : 0.58;
    context.fillRect(x, y, barWidth, height);
    context.globalAlpha = 1;
    context.fillStyle = "#17212b";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(item.label, x + barWidth / 2, chart.y + chart.height + 16);
    context.fillStyle = "#61707f";
    context.font = "700 9px Inter, system-ui, sans-serif";
    context.fillText(item.value.toFixed(2), x + barWidth / 2, y - 5);
  });

  const legendY = pane.y + pane.height - 48;
  const legend = [
    { label: "open", color: "#f7fafc" },
    { label: "wall", color: "#17212b" },
    { label: "fire -10", color: "#ffd9d5" },
    { label: "gold +10", color: "#ffe8a7" },
  ];
  legend.forEach((item, index) => {
    const chipWidth = (pane.width - 28 - 18) / 4;
    const x = pane.x + 14 + index * (chipWidth + 6);
    context.fillStyle = item.color;
    context.strokeStyle = "#dce5ea";
    context.fillRect(x, legendY, chipWidth, 24);
    context.strokeRect(x, legendY, chipWidth, 24);
    context.fillStyle = item.label === "wall" ? "#f7fafc" : "#17212b";
    context.font = "900 8px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(fitCanvasText(context, item.label, chipWidth - 8), x + chipWidth / 2, legendY + 16);
  });

  if (dynamicProgramming.stable) {
    context.fillStyle = `rgba(15, 118, 110, ${0.16 + 0.08 * Math.sin(iteration * 0.6)})`;
    context.fillRect(pane.x + 14, pane.y + 48, pane.width - 28, 2);
  }

  context.restore();
}

function dynamicProgrammingCellColor(
  cell: GridWorldCell,
  value: number,
  positiveMax: number,
  negativeMin: number,
) {
  if (cell === "wall") return "#17212b";
  if (cell === "gold") return "#ffe8a7";
  if (cell === "fire") return "#ffd9d5";
  if (cell === "start") return "#d8edff";
  if (value >= 0) {
    return d3.interpolateRgb("#ffffff", "#b9ede3")(Math.min(1, value / positiveMax));
  }
  return d3.interpolateRgb("#ffffff", "#ffc6bd")(Math.min(1, value / negativeMin));
}

function dynamicProgrammingMethodLabel(method: NonNullable<ConceptFrame["dynamicProgramming"]>["method"]) {
  return method === "policy-iteration" ? "Policy Iteration" : "Value Iteration";
}

function hitTestDynamicProgrammingGrid(
  dynamicProgramming: NonNullable<ConceptFrame["dynamicProgramming"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const layout = dynamicProgrammingLayout(size);
  const box = dynamicProgrammingGridBox(dynamicProgramming, layout.gridPane);
  if (
    point.x < box.x ||
    point.y < box.y ||
    point.x > box.x + box.width ||
    point.y > box.y + box.height
  ) {
    return null;
  }
  return {
    row: Math.min(dynamicProgramming.grid.rows - 1, Math.floor((point.y - box.y) / box.cellSize)),
    column: Math.min(dynamicProgramming.grid.columns - 1, Math.floor((point.x - box.x) / box.cellSize)),
  };
}

function gridWorldPaintTool(value: unknown): GridWorldCell {
  return value === "empty" || value === "wall" || value === "fire" || value === "gold" || value === "start"
    ? value
    : "wall";
}

function paintGridWorldCell(grid: GridWorldValue, row: number, column: number, tool: GridWorldCell): GridWorldValue {
  const cells = grid.cells.map((gridRow) => [...gridRow]);
  if (tool === "start") {
    for (let r = 0; r < cells.length; r += 1) {
      for (let c = 0; c < cells[r].length; c += 1) {
        if (cells[r][c] === "start") {
          cells[r][c] = "empty";
        }
      }
    }
  }
  cells[row][column] = tool;
  return {
    ...grid,
    cells,
  };
}

function paintConvolutionLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const convolution = frame.convolution;
  if (!convolution) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  let sourcePane: CanvasPane;
  let outputPane: CanvasPane;
  let mathPane: CanvasPane;

  if (narrow) {
    const sourceHeight = Math.min(310, Math.max(260, size.height * 0.34));
    const outputHeight = Math.min(270, Math.max(230, size.height * 0.27));
    sourcePane = {
      x: padding,
      y: 52,
      width: size.width - padding * 2,
      height: sourceHeight,
    };
    outputPane = {
      x: padding,
      y: sourcePane.y + sourcePane.height + gap + 26,
      width: size.width - padding * 2,
      height: outputHeight,
    };
    mathPane = {
      x: padding,
      y: outputPane.y + outputPane.height + gap + 26,
      width: size.width - padding * 2,
      height: Math.max(180, size.height - outputPane.y - outputPane.height - gap - 80),
    };
  } else {
    const mathHeight = 124;
    const mathY = size.height - mathHeight - 54;
    const topHeight = Math.max(330, mathY - 96);
    const paneWidth = (size.width - padding * 2 - gap) / 2;
    sourcePane = {
      x: padding,
      y: 58,
      width: paneWidth,
      height: topHeight,
    };
    outputPane = {
      x: sourcePane.x + sourcePane.width + gap,
      y: 58,
      width: paneWidth,
      height: topHeight,
    };
    mathPane = {
      x: padding,
      y: mathY,
      width: size.width - padding * 2,
      height: mathHeight,
    };
  }

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Interactive filter sandbox", sourcePane.x, sourcePane.y - 24);
  context.fillText("Filtered feature map", outputPane.x, outputPane.y - 24);
  context.fillText("Element-wise calculation", mathPane.x, mathPane.y - 16);

  paintConvolutionSourcePane(context, convolution, sourcePane);
  paintConvolutionOutputPane(context, convolution, outputPane);
  paintConvolutionMathPane(context, convolution, mathPane);
}

function paintConvolutionSourcePane(
  context: CanvasRenderingContext2D,
  convolution: NonNullable<ConceptFrame["convolution"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(fitCanvasText(context, convolution.source.name, pane.width - 28), pane.x + 14, pane.y + 22);

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `${convolution.source.height}x${convolution.source.width} input · ${convolution.padding ? "zero padding visible" : "valid convolution"} · stride ${convolution.stride}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const imageBox = {
    x: pane.x + 14,
    y: pane.y + 58,
    width: pane.width - 28,
    height: pane.height - 72,
  };
  paintConvolutionMatrixImage(context, convolution.padded, imageBox, {
    highlight: {
      row: convolution.cursor.inputRow,
      column: convolution.cursor.inputColumn,
      size: 3,
      color: "#f59e0b",
    },
    paddingCells: convolution.padding ? 1 : 0,
  });
  context.restore();
}

function paintConvolutionOutputPane(
  context: CanvasRenderingContext2D,
  convolution: NonNullable<ConceptFrame["convolution"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    `${convolution.outputShape.height}x${convolution.outputShape.width} feature map`,
    pane.x + 14,
    pane.y + 22,
  );

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `current Y[${convolution.cursor.outputRow}, ${convolution.cursor.outputColumn}] = ${formatConvolutionCanvasNumber(convolution.currentValue)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const imageBox = {
    x: pane.x + 14,
    y: pane.y + 58,
    width: pane.width - 28,
    height: pane.height - 72,
  };
  paintConvolutionMatrixImage(context, convolution.normalizedOutput, imageBox, {
    highlight: {
      row: convolution.cursor.outputRow,
      column: convolution.cursor.outputColumn,
      size: 1,
      color: "#2f6fbe",
    },
  });
  context.restore();
}

function paintConvolutionMatrixImage(
  context: CanvasRenderingContext2D,
  matrix: number[][],
  box: CanvasPane,
  options: {
    highlight?: { row: number; column: number; size: number; color: string };
    paddingCells?: number;
  } = {},
) {
  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;
  if (rows === 0 || columns === 0) {
    return;
  }

  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(box.x, box.y, box.width, box.height);
  context.strokeRect(box.x, box.y, box.width, box.height);

  const cellSize = Math.min((box.width - 12) / columns, (box.height - 12) / rows);
  const imageWidth = columns * cellSize;
  const imageHeight = rows * cellSize;
  const imageX = box.x + (box.width - imageWidth) / 2;
  const imageY = box.y + (box.height - imageHeight) / 2;
  const paddingCells = options.paddingCells ?? 0;

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = Math.max(0, Math.min(1, matrix[row][column]));
      const shade = Math.round(value * 255);
      const isPadding =
        paddingCells > 0 &&
        (row < paddingCells ||
          column < paddingCells ||
          row >= rows - paddingCells ||
          column >= columns - paddingCells);
      context.fillStyle = isPadding ? "#cfd8df" : `rgb(${shade}, ${shade}, ${shade})`;
      context.fillRect(
        imageX + column * cellSize,
        imageY + row * cellSize,
        Math.ceil(cellSize),
        Math.ceil(cellSize),
      );
      if (cellSize >= 8) {
        context.strokeStyle = isPadding ? "rgba(97, 112, 127, 0.22)" : "rgba(23, 33, 43, 0.08)";
        context.lineWidth = 0.5;
        context.strokeRect(imageX + column * cellSize, imageY + row * cellSize, cellSize, cellSize);
      }
    }
  }

  const highlight = options.highlight;
  if (highlight) {
    const x = imageX + highlight.column * cellSize;
    const y = imageY + highlight.row * cellSize;
    const size = highlight.size * cellSize;
    context.save();
    context.fillStyle = "rgba(245, 158, 11, 0.12)";
    context.strokeStyle = highlight.color;
    context.shadowColor = "rgba(245, 158, 11, 0.38)";
    context.shadowBlur = 10;
    context.lineWidth = Math.max(2, Math.min(5, cellSize * 0.18));
    context.fillRect(x, y, size, size);
    context.strokeRect(x + 1, y + 1, size - 2, size - 2);
    context.restore();
  }
}

function paintConvolutionMathPane(
  context: CanvasRenderingContext2D,
  convolution: NonNullable<ConceptFrame["convolution"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.84)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  const gridSize = Math.min(128, Math.max(78, pane.height - 42), pane.width * 0.34);
  const cell = gridSize / 3;
  const gridX = pane.x + 14;
  const gridY = pane.y + 24;

  convolution.terms.forEach((term) => {
    const x = gridX + term.column * cell;
    const y = gridY + term.row * cell;
    context.fillStyle = term.product >= 0 ? "#e7f6f2" : "#fff1ed";
    context.strokeStyle = term.product >= 0 ? "rgba(15, 118, 110, 0.28)" : "rgba(211, 74, 67, 0.28)";
    context.fillRect(x, y, cell, cell);
    context.strokeRect(x, y, cell, cell);

    context.fillStyle = "#17212b";
    context.font = "800 8px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(
      fitCanvasText(
        context,
        `${formatConvolutionCanvasNumber(term.imageValue)}x${formatConvolutionCanvasNumber(term.kernelValue)}`,
        cell - 4,
      ),
      x + cell / 2,
      y + cell / 2 - 2,
    );
    context.fillStyle = term.product >= 0 ? "#0b5f59" : "#b13a34";
    context.fillText(
      fitCanvasText(context, `=${formatConvolutionCanvasNumber(term.product)}`, cell - 4),
      x + cell / 2,
      y + cell / 2 + 10,
    );
  });

  const textX = gridX + gridSize + 18;
  const textWidth = pane.x + pane.width - textX - 14;
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(
      context,
      `Y[${convolution.cursor.outputRow}, ${convolution.cursor.outputColumn}] = sum(patch * kernel)`,
      textWidth,
    ),
    textX,
    pane.y + 28,
  );

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  const products = convolution.terms
    .map((term) => formatConvolutionCanvasNumber(term.product))
    .join(" + ")
    .replace(/\+ -/g, "- ");
  wrapCanvasText(context, products, textX, pane.y + 49, textWidth, 14, 3);

  context.fillStyle = "#0f766e";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.fillText(`= ${formatConvolutionCanvasNumber(convolution.currentValue)}`, textX, pane.y + pane.height - 18);
  context.restore();
}

function formatConvolutionCanvasNumber(value: number) {
  if (Math.abs(value) < 0.005) {
    return "0";
  }
  const digits = Math.abs(value) >= 10 ? 1 : 2;
  return value.toFixed(digits).replace(/\.?0+$/, "");
}

function paintConvexLessonWebgl(
  canvas: HTMLCanvasElement,
  size: Size,
  frame: ConceptFrame,
) {
  const convex = frame.convex;
  if (!convex) {
    return;
  }

  const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
  if (!gl) {
    const context = canvas.getContext("2d");
    if (context) {
      context.clearRect(0, 0, size.width, size.height);
      paintBackground(context, size);
      paintConvexFallback2d(context, convex, size);
    }
    return;
  }

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(1, 1, 1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const program = createConvexProgram(gl);
  if (!program) {
    return;
  }
  gl.useProgram(program.program);

  const pixelRatio = canvas.width / Math.max(1, size.width);
  const leftPane = {
    x: 24,
    y: 54,
    width: Math.max(400, size.width * 0.66),
    height: size.height - 94,
  };
  const rightPane = {
    x: leftPane.x + leftPane.width + 18,
    y: 54,
    width: size.width - leftPane.x - leftPane.width - 42,
    height: size.height - 94,
  };
  const compact = rightPane.width < 180;
  const surfacePane = compact
    ? { x: 24, y: 54, width: size.width - 48, height: size.height * 0.64 }
    : leftPane;
  const morphPane = compact
    ? { x: 24, y: surfacePane.y + surfacePane.height + 16, width: size.width - 48, height: size.height - surfacePane.y - surfacePane.height - 56 }
    : rightPane;

  const surface = makeConvexSurfaceVertices(convex, surfacePane, size);
  drawConvexPrimitive(gl, program, surface.triangles, gl.TRIANGLES, pixelRatio);
  drawConvexPrimitive(gl, program, makeConvexConstraintWalls(convex, surfacePane, size), gl.TRIANGLES, pixelRatio);
  drawConvexPrimitive(gl, program, makeConvexConstraintBoxLines(convex, surfacePane, size), gl.LINES, pixelRatio);
  drawConvexPrimitive(gl, program, makeConvexPathLines(convex, surfacePane, size), gl.LINE_STRIP, pixelRatio);
  drawConvexPrimitive(gl, program, makeConvexMarkerPoints(convex, surfacePane, size), gl.POINTS, pixelRatio);
  drawConvexPrimitive(gl, program, makeConvexMorphPanel(convex, morphPane, size), gl.LINES, pixelRatio);
  drawConvexPrimitive(gl, program, makeConvexStatusPanel(convex, morphPane, size), gl.TRIANGLES, pixelRatio);
}

function createConvexProgram(gl: WebGLRenderingContext) {
  const vertexShader = compileConvexShader(
    gl,
    gl.VERTEX_SHADER,
    `
      attribute vec2 a_position;
      attribute vec4 a_color;
      attribute float a_pointSize;
      varying vec4 v_color;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
        gl_PointSize = a_pointSize;
        v_color = a_color;
      }
    `,
  );
  const fragmentShader = compileConvexShader(
    gl,
    gl.FRAGMENT_SHADER,
    `
      precision mediump float;
      varying vec4 v_color;
      void main() {
        gl_FragColor = v_color;
      }
    `,
  );
  if (!vertexShader || !fragmentShader) {
    return null;
  }

  const program = gl.createProgram();
  if (!program) {
    return null;
  }

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    return null;
  }

  return {
    program,
    position: gl.getAttribLocation(program, "a_position"),
    color: gl.getAttribLocation(program, "a_color"),
    pointSize: gl.getAttribLocation(program, "a_pointSize"),
  };
}

function compileConvexShader(
  gl: WebGLRenderingContext,
  type: number,
  source: string,
) {
  const shader = gl.createShader(type);
  if (!shader) {
    return null;
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    return null;
  }
  return shader;
}

function drawConvexPrimitive(
  gl: WebGLRenderingContext,
  program: {
    position: number;
    color: number;
    pointSize: number;
  },
  vertices: number[],
  mode: number,
  pixelRatio: number,
) {
  if (vertices.length === 0) {
    return;
  }

  const stride = 7 * Float32Array.BYTES_PER_ELEMENT;
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(program.position);
  gl.vertexAttribPointer(program.position, 2, gl.FLOAT, false, stride, 0);
  gl.enableVertexAttribArray(program.color);
  gl.vertexAttribPointer(program.color, 4, gl.FLOAT, false, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
  gl.enableVertexAttribArray(program.pointSize);
  gl.vertexAttribPointer(program.pointSize, 1, gl.FLOAT, false, stride, 6 * Float32Array.BYTES_PER_ELEMENT);
  gl.drawArrays(mode, 0, vertices.length / 7);
  gl.deleteBuffer(buffer);

  if (pixelRatio !== 1) {
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  }
}

function makeConvexSurfaceVertices(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const domain = 1.7;
  const grid = 32;
  const zValues: number[] = [];
  for (let row = 0; row <= grid; row += 1) {
    for (let column = 0; column <= grid; column += 1) {
      const x = -domain + (column / grid) * domain * 2;
      const y = -domain + (row / grid) * domain * 2;
      zValues.push(convexWebglObjective(convex, x, y));
    }
  }
  const minZ = Math.min(...zValues);
  const maxZ = Math.max(...zValues);
  const triangles: number[] = [];

  for (let row = 0; row < grid; row += 1) {
    for (let column = 0; column < grid; column += 1) {
      const points = [
        {
          x: -domain + (column / grid) * domain * 2,
          y: -domain + (row / grid) * domain * 2,
        },
        {
          x: -domain + ((column + 1) / grid) * domain * 2,
          y: -domain + (row / grid) * domain * 2,
        },
        {
          x: -domain + ((column + 1) / grid) * domain * 2,
          y: -domain + ((row + 1) / grid) * domain * 2,
        },
        {
          x: -domain + (column / grid) * domain * 2,
          y: -domain + ((row + 1) / grid) * domain * 2,
        },
      ].map((point) => ({ ...point, z: convexWebglObjective(convex, point.x, point.y) }));
      [points[0], points[1], points[2], points[0], points[2], points[3]].forEach((point) => {
        const shade = (point.z - minZ) / Math.max(0.001, maxZ - minZ);
        const color = interpolateConvexColor(shade, convex.status);
        pushConvexVertex(triangles, projectConvexPoint(point, pane, minZ, maxZ), color, size, 1);
      });
    }
  }

  return { triangles, minZ, maxZ };
}

function makeConvexConstraintWalls(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const vertices: number[] = [];
  const zMin = 0;
  const zMax = 5.6;
  const { minX, maxX, minY, maxY } = convex.constraint;
  const walls = [
    [
      { x: minX, y: minY, z: zMin },
      { x: minX, y: maxY, z: zMin },
      { x: minX, y: maxY, z: zMax },
      { x: minX, y: minY, z: zMax },
    ],
    [
      { x: maxX, y: minY, z: zMin },
      { x: maxX, y: maxY, z: zMin },
      { x: maxX, y: maxY, z: zMax },
      { x: maxX, y: minY, z: zMax },
    ],
    [
      { x: minX, y: minY, z: zMin },
      { x: maxX, y: minY, z: zMin },
      { x: maxX, y: minY, z: zMax },
      { x: minX, y: minY, z: zMax },
    ],
    [
      { x: minX, y: maxY, z: zMin },
      { x: maxX, y: maxY, z: zMin },
      { x: maxX, y: maxY, z: zMax },
      { x: minX, y: maxY, z: zMax },
    ],
  ];
  walls.forEach((wall, index) => {
    const color = index < 2 ? [0.82, 0.29, 0.26, 0.16] : [0.18, 0.44, 0.75, 0.15];
    [wall[0], wall[1], wall[2], wall[0], wall[2], wall[3]].forEach((point) => {
      pushConvexVertex(vertices, projectConvexPoint(point, pane, 0, 6), color, size, 1);
    });
  });
  return vertices;
}

function makeConvexConstraintBoxLines(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const vertices: number[] = [];
  const z = 0.16;
  const corners = [
    { x: convex.constraint.minX, y: convex.constraint.minY, z },
    { x: convex.constraint.maxX, y: convex.constraint.minY, z },
    { x: convex.constraint.maxX, y: convex.constraint.maxY, z },
    { x: convex.constraint.minX, y: convex.constraint.maxY, z },
  ];
  const color = [0.09, 0.13, 0.17, 0.85];
  [[0, 1], [1, 2], [2, 3], [3, 0]].forEach(([from, to]) => {
    pushConvexVertex(vertices, projectConvexPoint(corners[from], pane, 0, 6), color, size, 1);
    pushConvexVertex(vertices, projectConvexPoint(corners[to], pane, 0, 6), color, size, 1);
  });
  return vertices;
}

function makeConvexPathLines(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const vertices: number[] = [];
  const color = convex.status === "convex" ? [0.05, 0.46, 0.43, 0.96] : [0.82, 0.29, 0.26, 0.96];
  convex.path.forEach((point) => {
    pushConvexVertex(vertices, projectConvexPoint(point, pane, 0, 6), color, size, 1);
  });
  return vertices;
}

function makeConvexMarkerPoints(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const vertices: number[] = [];
  const points = [
    { point: convex.constrainedOptimum, color: [0.09, 0.13, 0.17, 0.9], size: 9 },
    { point: convex.current, color: [0.82, 0.29, 0.26, 1], size: 14 },
  ];
  points.forEach((item) => {
    pushConvexVertex(vertices, projectConvexPoint(item.point, pane, 0, 6), item.color, size, item.size);
  });
  return vertices;
}

function makeConvexMorphPanel(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const vertices: number[] = [];
  const plot = {
    x: pane.x + 14,
    y: pane.y + 74,
    width: Math.max(80, pane.width - 28),
    height: Math.max(130, pane.height - 142),
  };
  const axisColor = [0.62, 0.68, 0.73, 0.75];
  pushPanelLine(vertices, { x: plot.x, y: plot.y + plot.height / 2 }, { x: plot.x + plot.width, y: plot.y + plot.height / 2 }, axisColor, size);
  pushPanelLine(vertices, { x: plot.x + plot.width / 2, y: plot.y }, { x: plot.x + plot.width / 2, y: plot.y + plot.height }, axisColor, size);

  const curveColor = convex.status === "convex" ? [0.05, 0.46, 0.43, 1] : [0.82, 0.29, 0.26, 1];
  const samples = 80;
  for (let index = 0; index < samples - 1; index += 1) {
    const a = -1.8 + (index / (samples - 1)) * 3.6;
    const b = -1.8 + ((index + 1) / (samples - 1)) * 3.6;
    const pa = morphCurvePoint(convex, a, plot);
    const pb = morphCurvePoint(convex, b, plot);
    pushPanelLine(vertices, pa, pb, curveColor, size);
  }
  return vertices;
}

function makeConvexStatusPanel(
  convex: NonNullable<ConceptFrame["convex"]>,
  pane: CanvasPane,
  size: Size,
) {
  const color = convex.status === "convex" ? [0.05, 0.46, 0.43, 0.2] : [0.82, 0.29, 0.26, 0.2];
  const rect = {
    x: pane.x + 14,
    y: pane.y + 18,
    width: Math.max(80, pane.width - 28),
    height: 34,
  };
  return rectToConvexTriangles(rect, color, size);
}

function morphCurvePoint(
  convex: NonNullable<ConceptFrame["convex"]>,
  x: number,
  plot: CanvasPane,
) {
  const y =
    0.42 * convex.objective.curvatureX * x ** 2 +
    convex.objective.sCurve * (0.18 * x ** 4 - 1.1 * x ** 2 + 0.18 * x ** 3) +
    Math.min(0, convex.objective.curvatureY) * x ** 2;
  const normalizedX = (x + 1.8) / 3.6;
  const normalizedY = Math.max(-1.6, Math.min(1.8, y)) / 3.4 + 0.47;
  return {
    x: plot.x + normalizedX * plot.width,
    y: plot.y + plot.height - normalizedY * plot.height,
  };
}

function convexWebglObjective(
  convex: NonNullable<ConceptFrame["convex"]>,
  x: number,
  y: number,
) {
  const o = convex.objective;
  return (
    0.5 * o.curvatureX * x ** 2 +
    o.crossTerm * x * y +
    0.5 * o.curvatureY * y ** 2 +
    o.tiltX * x +
    o.tiltY * y +
    o.sCurve * (0.18 * x ** 4 - 0.7 * x ** 2 + 0.14 * Math.sin(3 * y)) +
    2.6
  );
}

function projectConvexPoint(
  point: { x: number; y: number; z: number },
  pane: CanvasPane,
  minZ: number,
  maxZ: number,
) {
  const zNorm = (point.z - minZ) / Math.max(0.001, maxZ - minZ);
  const isoX = (point.x - point.y) * 0.58;
  const isoY = (point.x + point.y) * 0.25 - zNorm * 1.72;
  const scale = Math.min(pane.width / 3.9, pane.height / 2.8);
  return {
    x: pane.x + pane.width / 2 + isoX * scale,
    y: pane.y + pane.height * 0.72 + isoY * scale,
  };
}

function interpolateConvexColor(value: number, status: "convex" | "non-convex") {
  const low = status === "convex" ? [0.86, 0.96, 0.94] : [0.99, 0.91, 0.89];
  const high = status === "convex" ? [0.18, 0.44, 0.75] : [0.82, 0.29, 0.26];
  return [
    low[0] + (high[0] - low[0]) * value,
    low[1] + (high[1] - low[1]) * value,
    low[2] + (high[2] - low[2]) * value,
    0.72,
  ];
}

function pushConvexVertex(
  vertices: number[],
  point: { x: number; y: number },
  color: number[],
  size: Size,
  pointSize: number,
) {
  vertices.push((point.x / size.width) * 2 - 1, 1 - (point.y / size.height) * 2, color[0], color[1], color[2], color[3], pointSize);
}

function pushPanelLine(
  vertices: number[],
  from: { x: number; y: number },
  to: { x: number; y: number },
  color: number[],
  size: Size,
) {
  pushConvexVertex(vertices, from, color, size, 1);
  pushConvexVertex(vertices, to, color, size, 1);
}

function rectToConvexTriangles(
  rect: CanvasPane,
  color: number[],
  size: Size,
) {
  const vertices: number[] = [];
  const corners = [
    { x: rect.x, y: rect.y },
    { x: rect.x + rect.width, y: rect.y },
    { x: rect.x + rect.width, y: rect.y + rect.height },
    { x: rect.x, y: rect.y + rect.height },
  ];
  [corners[0], corners[1], corners[2], corners[0], corners[2], corners[3]].forEach((point) => {
    pushConvexVertex(vertices, point, color, size, 1);
  });
  return vertices;
}

function paintConvexFallback2d(
  context: CanvasRenderingContext2D,
  convex: NonNullable<ConceptFrame["convex"]>,
  size: Size,
) {
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.fillText("Convex constraint sandbox", 24, 32);
  context.fillStyle = convex.status === "convex" ? "#0f766e" : "#d34a43";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.fillText(`Status: ${convex.status === "convex" ? "Convex" : "Non-Convex"}`, 24, 54);
  const scales = d3
    .scaleLinear()
    .domain([-1.8, 1.8])
    .range([64, size.width - 64]);
  const yScale = d3
    .scaleLinear()
    .domain([-1.8, 1.8])
    .range([size.height - 64, 84]);
  context.strokeStyle = "#d8e0e5";
  context.strokeRect(
    scales(convex.constraint.minX),
    yScale(convex.constraint.maxY),
    scales(convex.constraint.maxX) - scales(convex.constraint.minX),
    yScale(convex.constraint.minY) - yScale(convex.constraint.maxY),
  );
  context.strokeStyle = "#d34a43";
  context.lineWidth = 2;
  context.beginPath();
  convex.path.forEach((point, index) => {
    const x = scales(point.x);
    const y = yScale(point.y);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function paintSvdLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const svd = frame.svd;
  if (!svd) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  const compressionPane = narrow
    ? {
        x: padding,
        y: 52,
        width: size.width - padding * 2,
        height: Math.max(360, Math.min(420, size.height * 0.52)),
      }
    : {
        x: padding,
        y: 58,
        width: Math.max(430, size.width * 0.62),
        height: size.height - 96,
      };
  const geometryPane = narrow
    ? {
        x: padding,
        y: compressionPane.y + compressionPane.height + gap + 26,
        width: size.width - padding * 2,
        height: Math.max(310, size.height - compressionPane.y - compressionPane.height - gap - 54),
      }
    : {
        x: compressionPane.x + compressionPane.width + gap,
        y: 58,
        width: size.width - compressionPane.x - compressionPane.width - gap - padding,
        height: size.height - 96,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Low-rank image compression", compressionPane.x, compressionPane.y - 24);
  context.fillText("Vector space factorization", geometryPane.x, geometryPane.y - 24);

  paintSvdCompressionPane(context, svd, compressionPane);
  paintSvdGeometryPane(context, svd, geometryPane);
}

function paintSvdCompressionPane(
  context: CanvasRenderingContext2D,
  svd: NonNullable<ConceptFrame["svd"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${svd.source.name}`, pane.x + 14, pane.y + 22);

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `rank ${svd.rank}/${svd.maxRank} · retained energy ${(svd.retainedEnergy * 100).toFixed(1)}% · error ${svd.reconstructionError.toFixed(3)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const imageGap = 14;
  const imageTop = pane.y + 64;
  const imageHeight = Math.min(pane.height * 0.5, pane.width * 0.42, 230);
  const imageWidth = (pane.width - 28 - imageGap) / 2;
  const originalBox = {
    x: pane.x + 14,
    y: imageTop,
    width: imageWidth,
    height: imageHeight,
  };
  const approxBox = {
    x: originalBox.x + imageWidth + imageGap,
    y: imageTop,
    width: imageWidth,
    height: imageHeight,
  };

  paintSvdImageBox(context, "original A", svd.original, originalBox);
  paintSvdImageBox(context, `truncated A_k`, svd.approximation, approxBox);

  const spectrumPane = {
    x: pane.x + 14,
    y: imageTop + imageHeight + 42,
    width: pane.width - 28,
    height: Math.max(110, pane.y + pane.height - imageTop - imageHeight - 62),
  };
  paintSvdSpectrum(context, svd, spectrumPane);

  context.restore();
}

function paintSvdImageBox(
  context: CanvasRenderingContext2D,
  label: string,
  matrix: number[][],
  box: CanvasPane,
) {
  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(box.x, box.y, box.width, box.height);
  context.strokeRect(box.x, box.y, box.width, box.height);

  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;
  const scale = Math.min((box.width - 14) / Math.max(1, columns), (box.height - 24) / Math.max(1, rows));
  const imageWidth = columns * scale;
  const imageHeight = rows * scale;
  const imageX = box.x + (box.width - imageWidth) / 2;
  const imageY = box.y + Math.max(8, (box.height - 18 - imageHeight) / 2);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = Math.max(0, Math.min(1, matrix[row][column]));
      const shade = Math.round(value * 255);
      context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      context.fillRect(imageX + column * scale, imageY + row * scale, Math.ceil(scale), Math.ceil(scale));
    }
  }

  context.fillStyle = "#17212b";
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(label, box.x + box.width / 2, box.y + box.height - 8);
}

function paintSvdSpectrum(
  context: CanvasRenderingContext2D,
  svd: NonNullable<ConceptFrame["svd"]>,
  pane: CanvasPane,
) {
  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("singular value spectrum", pane.x + 12, pane.y + 18);

  const values = svd.singularValues.slice(0, Math.min(18, svd.singularValues.length));
  const maxValue = Math.max(1e-8, ...values);
  const chart = {
    x: pane.x + 16,
    y: pane.y + 34,
    width: pane.width - 32,
    height: pane.height - 48,
  };
  const gap = 3;
  const barWidth = Math.max(4, (chart.width - gap * Math.max(0, values.length - 1)) / Math.max(1, values.length));

  values.forEach((value, index) => {
    const height = (value / maxValue) * chart.height;
    const x = chart.x + index * (barWidth + gap);
    const y = chart.y + chart.height - height;
    context.fillStyle = index < svd.rank ? "#2f6fbe" : "#c9d4da";
    context.fillRect(x, y, barWidth, height);
  });

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  context.textAlign = "right";
  context.fillText(`sigma1=${(svd.singularValues[0] ?? 0).toFixed(2)}`, pane.x + pane.width - 12, pane.y + 18);
}

function paintSvdGeometryPane(
  context: CanvasRenderingContext2D,
  svd: NonNullable<ConceptFrame["svd"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${svd.geometry.phase}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `V^T rotates · Sigma scales (${svd.geometry.scaleX.toFixed(2)}, ${svd.geometry.scaleY.toFixed(2)}) · U rotates`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const plot = {
    x: pane.x + 28,
    y: pane.y + 68,
    width: pane.width - 56,
    height: Math.max(170, pane.height - 148),
  };
  const center = { x: plot.x + plot.width / 2, y: plot.y + plot.height / 2 };
  const scale = Math.min(plot.width, plot.height) / 4.2;

  context.fillStyle = "rgba(247, 250, 252, 0.88)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#dce5ea";
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.strokeStyle = "#c9d4da";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(plot.x + 12, center.y);
  context.lineTo(plot.x + plot.width - 12, center.y);
  context.moveTo(center.x, plot.y + 12);
  context.lineTo(center.x, plot.y + plot.height - 12);
  context.stroke();

  context.setLineDash([4, 5]);
  context.strokeStyle = "rgba(97, 112, 127, 0.34)";
  context.beginPath();
  context.arc(center.x, center.y, scale, 0, Math.PI * 2);
  context.stroke();
  context.setLineDash([]);

  svd.geometry.inputVectors.forEach((vector) => {
    paintSvdArrow(context, center, scale, vector, "rgba(97, 112, 127, 0.28)", true);
  });
  svd.geometry.currentVectors.forEach((vector) => {
    paintSvdArrow(context, center, scale, vector, vector.color, false);
  });

  const chips = ["V^T rotation", "Sigma scaling", "U rotation"];
  const chipWidth = (pane.width - 28 - 12) / 3;
  const chipY = pane.y + pane.height - 52;
  chips.forEach((chip, index) => {
    const x = pane.x + 14 + index * (chipWidth + 6);
    const active = chip === svd.geometry.phase;
    context.fillStyle = active ? "#e7f6f2" : "#f7fafc";
    context.strokeStyle = active ? "#0f766e" : "#dce5ea";
    context.fillRect(x, chipY, chipWidth, 30);
    context.strokeRect(x, chipY, chipWidth, 30);
    context.fillStyle = active ? "#0b5f59" : "#61707f";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(fitCanvasText(context, chip, chipWidth - 8), x + chipWidth / 2, chipY + 19);
  });

  context.restore();
}

function paintSvdArrow(
  context: CanvasRenderingContext2D,
  center: { x: number; y: number },
  scale: number,
  vector: { x: number; y: number; color: string; label: string },
  color: string,
  ghost: boolean,
) {
  const end = {
    x: center.x + vector.x * scale,
    y: center.y - vector.y * scale,
  };
  const angle = Math.atan2(end.y - center.y, end.x - center.x);

  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = ghost ? 1.3 : 2.8;
  context.globalAlpha = ghost ? 0.7 : 0.94;
  if (ghost) {
    context.setLineDash([5, 4]);
  }
  context.beginPath();
  context.moveTo(center.x, center.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.setLineDash([]);
  context.translate(end.x, end.y);
  context.rotate(angle);
  context.beginPath();
  context.moveTo(8, 0);
  context.lineTo(-5, -5);
  context.lineTo(-5, 5);
  context.closePath();
  context.fill();
  context.rotate(-angle);
  context.fillStyle = ghost ? "#61707f" : vector.color;
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(vector.label, 8, -8);
  context.restore();
}

function paintStochasticLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const stochastic = frame.stochastic;
  if (!stochastic) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  const walkPane = narrow
    ? {
        x: padding,
        y: 52,
        width: size.width - padding * 2,
        height: Math.max(330, Math.min(390, size.height * 0.47)),
      }
    : {
        x: padding,
        y: 58,
        width: Math.max(430, size.width * 0.62),
        height: size.height - 96,
      };
  const markovPane = narrow
    ? {
        x: padding,
        y: walkPane.y + walkPane.height + gap + 26,
        width: size.width - padding * 2,
        height: Math.max(340, size.height - walkPane.y - walkPane.height - gap - 54),
      }
    : {
        x: walkPane.x + walkPane.width + gap,
        y: 58,
        width: size.width - walkPane.x - walkPane.width - gap - padding,
        height: size.height - 96,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Multi-path trajectory visualizer", walkPane.x, walkPane.y - 24);
  context.fillText("Markov transition matrix", markovPane.x, markovPane.y - 24);

  paintStochasticWalkPane(context, stochastic, walkPane);
  paintMarkovTransitionPane(context, stochastic, markovPane);
}

function paintStochasticWalkPane(
  context: CanvasRenderingContext2D,
  stochastic: NonNullable<ConceptFrame["stochastic"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`Random walk t=${stochastic.visibleStep}/${stochastic.timeSteps}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `mu ${stochastic.drift.toFixed(3)} · sigma ${stochastic.volatility.toFixed(3)} · ${stochastic.pathCount} particles`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const plot = {
    x: pane.x + 48,
    y: pane.y + 64,
    width: pane.width - 70,
    height: pane.height - 112,
  };
  const values = stochastic.paths.flatMap((path) => path.points.map((point) => point.y));
  const extent = d3.extent(values.length ? values : [0]) as [number, number];
  const yPad = Math.max(0.5, (extent[1] - extent[0]) * 0.18, stochastic.terminalStd * 0.18);
  const yDomain = [extent[0] - yPad, extent[1] + yPad] as [number, number];
  const xScale = d3.scaleLinear().domain([0, stochastic.timeSteps]).range([plot.x, plot.x + plot.width]);
  const yScale = d3.scaleLinear().domain(yDomain).range([plot.y + plot.height, plot.y]);

  context.fillStyle = "rgba(247, 250, 252, 0.88)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#c9d4da";
  context.lineWidth = 1;
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.strokeStyle = "rgba(97, 112, 127, 0.16)";
  context.lineWidth = 1;
  xScale.ticks(6).forEach((tick) => {
    const x = xScale(tick);
    context.beginPath();
    context.moveTo(x, plot.y);
    context.lineTo(x, plot.y + plot.height);
    context.stroke();
  });
  yScale.ticks(5).forEach((tick) => {
    const y = yScale(tick);
    context.beginPath();
    context.moveTo(plot.x, y);
    context.lineTo(plot.x + plot.width, y);
    context.stroke();
  });

  stochastic.paths.forEach((path, pathIndex) => {
    if (path.points.length < 2) {
      return;
    }

    context.save();
    context.strokeStyle = path.color;
    context.globalAlpha = 0.26 + (pathIndex % 5) * 0.045;
    context.lineWidth = 1.2;
    context.beginPath();
    path.points.forEach((point, pointIndex) => {
      const x = xScale(point.t);
      const y = yScale(point.y);
      if (pointIndex === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();

    const current = path.current;
    context.globalAlpha = 0.86;
    context.fillStyle = path.color;
    context.beginPath();
    context.arc(xScale(current.t), yScale(current.y), 3.2, 0, Math.PI * 2);
    context.fill();
    context.restore();
  });

  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  xScale.ticks(5).forEach((tick) => {
    context.fillText(String(Math.round(tick)), xScale(tick), plot.y + plot.height + 18);
  });
  context.fillText("time", plot.x + plot.width / 2, plot.y + plot.height + 38);

  context.textAlign = "right";
  yScale.ticks(5).forEach((tick) => {
    context.fillText(tick.toFixed(1), plot.x - 8, yScale(tick) + 3);
  });

  const bandY = pane.y + pane.height - 34;
  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(pane.x + 12, bandY, pane.width - 24, 22);
  context.strokeRect(pane.x + 12, bandY, pane.width - 24, 22);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(
      context,
      `terminal mean ${stochastic.terminalMean.toFixed(2)} · terminal spread ${stochastic.terminalStd.toFixed(2)}`,
      pane.width - 44,
    ),
    pane.x + 22,
    bandY + 15,
  );

  context.restore();
}

function paintMarkovTransitionPane(
  context: CanvasRenderingContext2D,
  stochastic: NonNullable<ConceptFrame["stochastic"]>,
  pane: CanvasPane,
) {
  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  const current = stochastic.states[stochastic.currentState] ?? stochastic.states[0];
  const next = stochastic.states[stochastic.nextState] ?? current;
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`State hop ${stochastic.markovStep}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, `${current.label} -> ${next.label} · entropy ${stochastic.entropy.toFixed(2)}`, pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  const graph = {
    x: pane.x + 24,
    y: pane.y + 58,
    width: pane.width - 48,
    height: Math.max(156, pane.height * 0.48),
  };
  const matrixPane = {
    x: pane.x + 14,
    y: graph.y + graph.height + 16,
    width: pane.width - 28,
    height: Math.max(112, pane.y + pane.height - graph.y - graph.height - 30),
  };

  context.fillStyle = "rgba(247, 250, 252, 0.88)";
  context.fillRect(graph.x, graph.y, graph.width, graph.height);
  context.strokeStyle = "#dce5ea";
  context.strokeRect(graph.x, graph.y, graph.width, graph.height);

  const positions = markovNodePositions(stochastic, graph);
  stochastic.normalizedTransitionMatrix.forEach((row, fromIndex) => {
    row.forEach((probability, toIndex) => {
      if (probability <= 0.015) {
        return;
      }

      const active = fromIndex === stochastic.currentState && toIndex === stochastic.nextState;
      paintMarkovEdge(
        context,
        positions[fromIndex],
        positions[toIndex],
        probability,
        stochastic.states[fromIndex]?.color ?? "#61707f",
        active,
        fromIndex,
        toIndex,
      );
    });
  });

  paintMarkovPulse(context, stochastic, positions);

  stochastic.states.forEach((state, index) => {
    const point = positions[index];
    const isCurrent = index === stochastic.currentState;
    const isNext = index === stochastic.nextState;
    context.save();
    if (isCurrent || isNext) {
      context.shadowBlur = 14;
      context.shadowColor = isCurrent ? state.color : "#d34a43";
    }
    context.fillStyle = "#ffffff";
    context.strokeStyle = state.color;
    context.lineWidth = isCurrent || isNext ? 3 : 2;
    context.beginPath();
    context.arc(point.x, point.y, 23, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.shadowBlur = 0;
    context.fillStyle = "#17212b";
    context.font = "900 10px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(state.label, point.x, point.y + 4);
    context.restore();
  });

  paintMarkovMatrixReadout(context, stochastic, matrixPane);
  context.restore();
}

function markovNodePositions(
  stochastic: NonNullable<ConceptFrame["stochastic"]>,
  pane: CanvasPane,
) {
  const centerX = pane.x + pane.width / 2;
  const top = pane.y + 34;
  const bottom = pane.y + pane.height - 38;
  const spread = Math.min(pane.width * 0.32, 96);

  return stochastic.states.map((_, index) => {
    if (index === 0) {
      return { x: centerX, y: top };
    }

    if (index === 1) {
      return { x: centerX - spread, y: bottom };
    }

    return { x: centerX + spread, y: bottom };
  });
}

function paintMarkovEdge(
  context: CanvasRenderingContext2D,
  from: { x: number; y: number },
  to: { x: number; y: number },
  probability: number,
  color: string,
  active: boolean,
  fromIndex: number,
  toIndex: number,
) {
  context.save();
  context.strokeStyle = active ? "#d34a43" : color;
  context.fillStyle = active ? "#d34a43" : color;
  context.globalAlpha = active ? 0.86 : 0.18 + probability * 0.42;
  context.lineWidth = active ? 2.5 + probability * 3 : 0.9 + probability * 3.4;

  if (fromIndex === toIndex) {
    context.beginPath();
    context.arc(from.x + 20, from.y - 13, 18, -0.2, Math.PI * 1.65);
    context.stroke();
    context.beginPath();
    context.moveTo(from.x + 16, from.y - 30);
    context.lineTo(from.x + 25, from.y - 30);
    context.lineTo(from.x + 22, from.y - 21);
    context.closePath();
    context.fill();
    context.restore();
    return;
  }

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const unitX = dx / length;
  const unitY = dy / length;
  const normalX = -unitY;
  const normalY = unitX;
  const bend = fromIndex < toIndex ? 24 : -24;
  const start = { x: from.x + unitX * 27, y: from.y + unitY * 27 };
  const end = { x: to.x - unitX * 27, y: to.y - unitY * 27 };
  const control = {
    x: (start.x + end.x) / 2 + normalX * bend,
    y: (start.y + end.y) / 2 + normalY * bend,
  };

  context.beginPath();
  context.moveTo(start.x, start.y);
  context.quadraticCurveTo(control.x, control.y, end.x, end.y);
  context.stroke();

  const arrow = quadraticPoint(start, control, end, 0.88);
  const before = quadraticPoint(start, control, end, 0.78);
  const angle = Math.atan2(arrow.y - before.y, arrow.x - before.x);
  context.translate(arrow.x, arrow.y);
  context.rotate(angle);
  context.beginPath();
  context.moveTo(7, 0);
  context.lineTo(-4, -4);
  context.lineTo(-4, 4);
  context.closePath();
  context.fill();
  context.restore();
}

function paintMarkovPulse(
  context: CanvasRenderingContext2D,
  stochastic: NonNullable<ConceptFrame["stochastic"]>,
  positions: Array<{ x: number; y: number }>,
) {
  const from = positions[stochastic.currentState];
  const to = positions[stochastic.nextState];
  if (!from || !to) {
    return;
  }

  const progress = 0.5 - Math.cos(stochastic.pulseProgress * Math.PI) / 2;
  let point = from;
  if (stochastic.currentState === stochastic.nextState) {
    const angle = progress * Math.PI * 2 - Math.PI / 2;
    point = {
      x: from.x + Math.cos(angle) * 31,
      y: from.y + Math.sin(angle) * 24,
    };
  } else {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const length = Math.max(1, Math.hypot(dx, dy));
    const normalX = -dy / length;
    const normalY = dx / length;
    const bend = stochastic.currentState < stochastic.nextState ? 24 : -24;
    point = quadraticPoint(
      from,
      {
        x: (from.x + to.x) / 2 + normalX * bend,
        y: (from.y + to.y) / 2 + normalY * bend,
      },
      to,
      progress,
    );
  }

  context.save();
  context.shadowBlur = 18;
  context.shadowColor = "#d34a43";
  context.fillStyle = "#d34a43";
  context.beginPath();
  context.arc(point.x, point.y, 6.4, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2;
  context.stroke();
  context.restore();
}

function quadraticPoint(
  start: { x: number; y: number },
  control: { x: number; y: number },
  end: { x: number; y: number },
  t: number,
) {
  const a = (1 - t) ** 2;
  const b = 2 * (1 - t) * t;
  const c = t ** 2;
  return {
    x: a * start.x + b * control.x + c * end.x,
    y: a * start.y + b * control.y + c * end.y,
  };
}

function paintMarkovMatrixReadout(
  context: CanvasRenderingContext2D,
  stochastic: NonNullable<ConceptFrame["stochastic"]>,
  pane: CanvasPane,
) {
  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("P normalized", pane.x + 10, pane.y + 16);

  const labels = ["A", "B", "C"];
  const startX = pane.x + 36;
  const startY = pane.y + 28;
  const cellWidth = Math.max(36, (pane.width - 48) / 4);
  const cellHeight = Math.max(20, Math.min(30, (pane.height - 36) / 4));

  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  labels.forEach((label, index) => {
    context.fillStyle = "#61707f";
    context.fillText(label, startX + (index + 1) * cellWidth + cellWidth / 2, startY - 6);
    context.fillText(label, startX - 12, startY + index * cellHeight + cellHeight / 2 + 3);
  });

  stochastic.normalizedTransitionMatrix.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      const x = startX + (columnIndex + 1) * cellWidth;
      const y = startY + rowIndex * cellHeight;
      const active = rowIndex === stochastic.currentState && columnIndex === stochastic.nextState;
      context.fillStyle = active ? "rgba(211, 74, 67, 0.12)" : "#ffffff";
      context.strokeStyle = active ? "#d34a43" : "#dce5ea";
      context.lineWidth = active ? 1.8 : 1;
      context.fillRect(x, y, cellWidth - 5, cellHeight - 5);
      context.strokeRect(x, y, cellWidth - 5, cellHeight - 5);
      context.fillStyle = active ? "#d34a43" : "#17212b";
      context.font = "900 9px Inter, system-ui, sans-serif";
      context.fillText(value.toFixed(2), x + (cellWidth - 5) / 2, y + cellHeight / 2 + 2);
    });
  });
}

function paintFrameworkLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const framework = frame.framework;
  if (!framework) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  const builderHeight = narrow
    ? Math.max(390, Math.min(430, size.height * 0.52))
    : size.height - 96;
  const builderPane = narrow
    ? {
        x: padding,
        y: 52,
        width: size.width - padding * 2,
        height: builderHeight,
      }
    : {
        x: padding,
        y: 58,
        width: Math.max(430, size.width * 0.62),
        height: builderHeight,
      };
  const trainerPane = narrow
    ? {
        x: padding,
        y: builderPane.y + builderPane.height + gap + 26,
        width: size.width - padding * 2,
        height: Math.max(280, size.height - builderPane.y - builderPane.height - gap - 54),
      }
    : {
        x: builderPane.x + builderPane.width + gap,
        y: 58,
        width: size.width - builderPane.x - builderPane.width - gap - padding,
        height: size.height - 96,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Custom network builder", builderPane.x, builderPane.y - 24);
  context.fillText("Trainer and class contract", trainerPane.x, trainerPane.y - 24);

  paintFrameworkPipelinePane(context, framework, builderPane);
  paintFrameworkTrainerPane(context, framework, trainerPane);
}

function paintFrameworkPipelinePane(
  context: CanvasRenderingContext2D,
  framework: NonNullable<ConceptFrame["framework"]>,
  pane: CanvasPane,
) {
  const { layers, event } = framework;
  const activeColor = event.phase === "forward" ? "#2f6fbe" : "#d34a43";
  const rects = layoutFrameworkLayerRects(framework, pane);

  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("network.layers[]", pane.x + 14, pane.y + 22);

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `${layers.length} Layer instances · batch ${framework.batchSize} · lr ${framework.learningRate.toFixed(3)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  context.fillStyle = activeColor;
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText(`${event.phase.toUpperCase()} HOOK`, pane.x + 14, pane.y + 61);
  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, event.description, pane.width - 120),
    pane.x + 118,
    pane.y + 61,
  );

  for (let index = 0; index < rects.length - 1; index += 1) {
    const isActive =
      (event.phase === "forward" && event.layerIndex === index + 1) ||
      (event.phase === "backward" && event.layerIndex === index);
    paintFrameworkConnector(context, rects[index], rects[index + 1], event.phase, isActive);
  }

  paintFrameworkPulse(context, framework, rects, pane);

  layers.forEach((layer, index) => {
    paintFrameworkLayerCard(context, layer, rects[index], index === event.layerIndex, event.phase);
  });

  const footerY = pane.y + pane.height - 36;
  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(pane.x + 12, footerY, pane.width - 24, 24);
  context.strokeRect(pane.x + 12, footerY, pane.width - 24, 24);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(context, "visual hooks fire inside every layer.forward() and layer.backward()", pane.width - 44),
    pane.x + 22,
    footerY + 16,
  );

  context.restore();
}

function layoutFrameworkLayerRects(
  framework: NonNullable<ConceptFrame["framework"]>,
  pane: CanvasPane,
) {
  const grid = {
    x: pane.x + 14,
    y: pane.y + 76,
    width: pane.width - 28,
    height: pane.height - 124,
  };
  const narrow = pane.width < 540;
  const targetWidth = narrow ? 94 : 126;
  const maxColumns = narrow ? 3 : 6;
  const columns = Math.max(2, Math.min(maxColumns, Math.floor(grid.width / targetWidth) || 2));
  const rows = Math.ceil(framework.layers.length / columns);
  const gap = 10;
  const cardWidth = (grid.width - gap * (columns - 1)) / columns;
  const cardHeight = Math.max(34, Math.min(74, (grid.height - gap * (rows - 1)) / rows));

  return framework.layers.map((_, index) => {
    const row = Math.floor(index / columns);
    const column = index % columns;

    return {
      x: grid.x + column * (cardWidth + gap),
      y: grid.y + row * (cardHeight + gap),
      width: cardWidth,
      height: cardHeight,
    };
  });
}

function paintFrameworkConnector(
  context: CanvasRenderingContext2D,
  from: CanvasPane,
  to: CanvasPane,
  phase: NonNullable<ConceptFrame["framework"]>["event"]["phase"],
  active: boolean,
) {
  const color = active ? (phase === "forward" ? "#2f6fbe" : "#d34a43") : "#c9d4da";
  const start = { x: from.x + from.width, y: from.y + from.height / 2 };
  const end = { x: to.x, y: to.y + to.height / 2 };

  context.save();
  context.strokeStyle = color;
  context.lineWidth = active ? 2.6 : 1.2;
  context.globalAlpha = active ? 0.9 : 0.55;
  context.beginPath();
  context.moveTo(start.x, start.y);

  if (Math.abs(start.y - end.y) < 2) {
    context.lineTo(end.x, end.y);
  } else {
    const laneY = (from.y + from.height + to.y) / 2;
    context.lineTo(start.x + 8, laneY);
    context.lineTo(end.x - 8, laneY);
    context.lineTo(end.x, end.y);
  }

  context.stroke();

  const arrowX = active && phase === "backward" ? start.x : end.x;
  const arrowY = active && phase === "backward" ? start.y : end.y;
  const direction = active && phase === "backward" ? -1 : 1;
  context.fillStyle = color;
  context.beginPath();
  context.moveTo(arrowX + direction * 5, arrowY);
  context.lineTo(arrowX - direction * 4, arrowY - 4);
  context.lineTo(arrowX - direction * 4, arrowY + 4);
  context.closePath();
  context.fill();
  context.restore();
}

function paintFrameworkPulse(
  context: CanvasRenderingContext2D,
  framework: NonNullable<ConceptFrame["framework"]>,
  rects: CanvasPane[],
  pane: CanvasPane,
) {
  const event = framework.event;
  const active = rects[event.layerIndex];
  if (!active) {
    return;
  }

  const activeCenter = {
    x: active.x + active.width / 2,
    y: active.y + active.height / 2,
  };
  let start = activeCenter;
  let end = activeCenter;

  if (event.phase === "forward") {
    const previous = rects[event.layerIndex - 1];
    start = previous
      ? { x: previous.x + previous.width / 2, y: previous.y + previous.height / 2 }
      : { x: pane.x + 16, y: activeCenter.y };
    end = activeCenter;
  } else {
    const next = rects[event.layerIndex + 1];
    start = next
      ? { x: next.x + next.width / 2, y: next.y + next.height / 2 }
      : { x: pane.x + pane.width - 16, y: activeCenter.y };
    end = activeCenter;
  }

  const progress = 0.5 - Math.cos((event.progress % 1) * Math.PI) / 2;
  const x = start.x + (end.x - start.x) * progress;
  const y = start.y + (end.y - start.y) * progress;
  const color = event.phase === "forward" ? "#2f6fbe" : "#d34a43";

  context.save();
  context.shadowBlur = 18;
  context.shadowColor = color;
  context.fillStyle = color;
  context.globalAlpha = 0.92;
  context.beginPath();
  context.arc(x, y, 6.2, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = "#ffffff";
  context.lineWidth = 2;
  context.stroke();
  context.restore();
}

function paintFrameworkLayerCard(
  context: CanvasRenderingContext2D,
  layer: NonNullable<ConceptFrame["framework"]>["layers"][number],
  rect: CanvasPane,
  active: boolean,
  phase: NonNullable<ConceptFrame["framework"]>["event"]["phase"],
) {
  const color = frameworkLayerColor(layer.kind);
  const activeColor = phase === "forward" ? "#2f6fbe" : "#d34a43";
  const compact = rect.height < 48 || rect.width < 106;

  context.save();
  if (active) {
    context.shadowBlur = 12;
    context.shadowColor = activeColor;
  }
  context.fillStyle = active
    ? phase === "forward"
      ? "rgba(47, 111, 190, 0.1)"
      : "rgba(211, 74, 67, 0.1)"
    : "#ffffff";
  context.strokeStyle = active ? activeColor : "#d8e0e5";
  context.lineWidth = active ? 2.2 : 1;
  context.fillRect(rect.x, rect.y, rect.width, rect.height);
  context.strokeRect(rect.x, rect.y, rect.width, rect.height);
  context.shadowBlur = 0;

  context.fillStyle = color;
  context.fillRect(rect.x, rect.y, 4, rect.height);

  context.fillStyle = "#17212b";
  context.font = `${compact ? "800 9px" : "900 11px"} Inter, system-ui, sans-serif`;
  context.textAlign = "left";
  context.fillText(fitCanvasText(context, layer.label, rect.width - 16), rect.x + 10, rect.y + (compact ? 15 : 18));

  context.fillStyle = "#61707f";
  context.font = `${compact ? "700 8px" : "700 9px"} Inter, system-ui, sans-serif`;
  context.fillText(
    fitCanvasText(context, `${frameworkKindLabel(layer.kind)} · ${layer.units} unit${layer.units === 1 ? "" : "s"}`, rect.width - 16),
    rect.x + 10,
    rect.y + (compact ? 28 : 34),
  );

  if (!compact) {
    context.fillText(
      fitCanvasText(context, `cache ${layer.cacheShape ?? "-"}`, rect.width - 16),
      rect.x + 10,
      rect.y + 48,
    );
  }

  if (rect.height > 62) {
    context.fillStyle = layer.parameters > 0 ? "#0f766e" : "#61707f";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.fillText(
      layer.parameters > 0 ? `${layer.parameters.toLocaleString()} params` : layer.gradientShape ?? "stateless",
      rect.x + 10,
      rect.y + rect.height - 10,
    );
  }

  context.restore();
}

function paintFrameworkTrainerPane(
  context: CanvasRenderingContext2D,
  framework: NonNullable<ConceptFrame["framework"]>,
  pane: CanvasPane,
) {
  const activeColor = framework.event.phase === "forward" ? "#2f6fbe" : "#d34a43";

  context.save();
  context.fillStyle = "rgba(255, 255, 255, 0.78)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Trainer.fit(model, loss, optimizer)", pane.x + 14, pane.y + 22);

  context.fillStyle = activeColor;
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText(`loss ${framework.loss.toFixed(4)}`, pane.x + 14, pane.y + 42);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(
    `${framework.totalParameters.toLocaleString()} params · batch ${framework.batchSize}`,
    pane.x + Math.min(128, pane.width * 0.42),
    pane.y + 42,
  );

  const contractY = pane.y + pane.height - 76;
  const loopTop = pane.y + 60;
  const loopHeight = Math.max(132, contractY - loopTop - 14);
  const stepGap = 6;
  const stepHeight = Math.max(26, Math.min(38, (loopHeight - stepGap * 4) / 5));
  const steps = [
    {
      label: "Batch",
      detail: `X: ${framework.batchSize} x 2`,
      active: framework.event.phase === "forward" && framework.event.layerIndex === 0,
    },
    {
      label: "Sequential.forward",
      detail: "loop layers left to right",
      active: framework.event.phase === "forward" && framework.event.layerIndex > 0,
    },
    {
      label: "Loss",
      detail: "value and final gradient",
      active: framework.event.layerIndex === framework.layers.length - 1,
    },
    {
      label: "Sequential.backward",
      detail: "reversed(layers)",
      active: framework.event.phase === "backward",
    },
    {
      label: "Optimizer.step",
      detail: "apply stored dW and db",
      active: framework.event.phase === "backward" && framework.event.layerIndex <= 1,
    },
  ];

  steps.forEach((step, index) => {
    const y = loopTop + index * (stepHeight + stepGap);
    paintFrameworkLoopStep(context, step.label, step.detail, step.active, activeColor, {
      x: pane.x + 14,
      y,
      width: pane.width - 28,
      height: stepHeight,
    });

    if (index < steps.length - 1) {
      context.strokeStyle = "#c9d4da";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(pane.x + pane.width / 2, y + stepHeight);
      context.lineTo(pane.x + pane.width / 2, y + stepHeight + stepGap);
      context.stroke();
    }
  });

  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(pane.x + 12, contractY, pane.width - 24, 62);
  context.strokeRect(pane.x + 12, contractY, pane.width - 24, 62);

  context.fillStyle = "#17212b";
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.fillText("Layer interface", pane.x + 22, contractY + 17);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, "forward(input_data): cache X, return Y", pane.width - 44),
    pane.x + 22,
    contractY + 35,
  );
  context.fillText(
    fitCanvasText(context, "backward(output_gradient): update grads, return dL/dX", pane.width - 44),
    pane.x + 22,
    contractY + 51,
  );

  context.restore();
}

function paintFrameworkLoopStep(
  context: CanvasRenderingContext2D,
  label: string,
  detail: string,
  active: boolean,
  activeColor: string,
  rect: CanvasPane,
) {
  context.fillStyle = active ? "rgba(47, 111, 190, 0.08)" : "#ffffff";
  if (activeColor === "#d34a43" && active) {
    context.fillStyle = "rgba(211, 74, 67, 0.08)";
  }
  context.strokeStyle = active ? activeColor : "#dce5ea";
  context.lineWidth = active ? 2 : 1;
  context.fillRect(rect.x, rect.y, rect.width, rect.height);
  context.strokeRect(rect.x, rect.y, rect.width, rect.height);

  context.fillStyle = active ? activeColor : "#17212b";
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(fitCanvasText(context, label, rect.width - 18), rect.x + 9, rect.y + 12);

  if (rect.height > 30) {
    context.fillStyle = "#61707f";
    context.font = "700 9px Inter, system-ui, sans-serif";
    context.fillText(fitCanvasText(context, detail, rect.width - 18), rect.x + 9, rect.y + 26);
  }
}

function frameworkLayerColor(kind: NonNullable<ConceptFrame["framework"]>["layers"][number]["kind"]) {
  if (kind === "input") {
    return "#2f6fbe";
  }

  if (kind === "linear") {
    return "#0f766e";
  }

  if (kind === "activation") {
    return "#6f58c9";
  }

  if (kind === "dropout") {
    return "#b7791f";
  }

  return "#d34a43";
}

function frameworkKindLabel(kind: NonNullable<ConceptFrame["framework"]>["layers"][number]["kind"]) {
  if (kind === "input") {
    return "input";
  }

  if (kind === "linear") {
    return "Linear";
  }

  if (kind === "activation") {
    return "activation";
  }

  if (kind === "dropout") {
    return "Dropout";
  }

  return "loss";
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
