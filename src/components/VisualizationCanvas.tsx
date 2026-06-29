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
  const [polynomialHoverIndex, setPolynomialHoverIndex] = useState<number | null>(null);
  const [determinantDrag, setDeterminantDrag] = useState<"basisI" | "basisJ" | null>(null);
  const [eigenDirectionDrag, setEigenDirectionDrag] = useState(false);
  const isBackpropLesson = frame?.type === "concept-demo" && Boolean(frame.backprop);
  const isActivationLesson = frame?.type === "concept-demo" && Boolean(frame.activation);
  const isLossLesson = frame?.type === "concept-demo" && Boolean(frame.loss);
  const isOptimizerLesson = frame?.type === "concept-demo" && Boolean(frame.optimizer);
  const isFrameworkLesson = frame?.type === "concept-demo" && Boolean(frame.framework);
  const isStochasticLesson = frame?.type === "concept-demo" && Boolean(frame.stochastic);
  const isSvdLesson = frame?.type === "concept-demo" && Boolean(frame.svd);
  const isNmfLesson = frame?.type === "concept-demo" && Boolean(frame.nmf);
  const isPolynomialLesson = frame?.type === "concept-demo" && Boolean(frame.polynomial);
  const isBayesianRegressionLesson = frame?.type === "concept-demo" && Boolean(frame.bayesianRegression);
  const isBayesRuleLesson = frame?.type === "concept-demo" && Boolean(frame.bayesRule);
  const isGdaLesson = frame?.type === "concept-demo" && Boolean(frame.gda);
  const isDeterminantLesson = frame?.type === "concept-demo" && Boolean(frame.determinant);
  const isEigenDirectionLesson = frame?.type === "concept-demo" && Boolean(frame.eigenDirection);
  const isRegularizationLesson = frame?.type === "concept-demo" && Boolean(frame.regularization);
  const isConvexLesson = frame?.type === "concept-demo" && Boolean(frame.convex);
  const isConvolutionLesson = frame?.type === "concept-demo" && Boolean(frame.convolution);
  const isDynamicProgrammingLesson = frame?.type === "concept-demo" && Boolean(frame.dynamicProgramming);
  const isTokenizerLesson = frame?.type === "concept-demo" && Boolean(frame.tokenizer);

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

    draw(canvas, size, frame, { polynomialHoverIndex });
  }, [frame, size, polynomialHoverIndex]);

  const handleCanvasClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (
      frame?.type !== "concept-demo" ||
      !params ||
      !onParamChange ||
      (!frame.dynamicProgramming && !frame.polynomial && !frame.bayesianRegression && !frame.gda)
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

    if (frame.polynomial) {
      const nextPoint = canvasPointToPolynomialPoint(frame.polynomial, size, point);
      if (!nextPoint) {
        return;
      }

      const currentPointSet =
        typeof params.polynomialPoints === "object" &&
        params.polynomialPoints !== null &&
        "kind" in params.polynomialPoints &&
        params.polynomialPoints.kind === "point-set"
          ? params.polynomialPoints
          : { kind: "point-set" as const, points: frame.polynomial.points };
      const nextPoints = [...currentPointSet.points, { ...nextPoint, label: "custom" }].slice(-42);
      onParamChange("polynomialPoints", { kind: "point-set", points: nextPoints });
      setPolynomialHoverIndex(nextPoints.length - 1);
      return;
    }

    if (frame.bayesianRegression) {
      const nextPoint = canvasPointToBayesianPoint(frame.bayesianRegression, size, point);
      if (!nextPoint) {
        return;
      }

      const currentPointSet =
        typeof params.bayesianPoints === "object" &&
        params.bayesianPoints !== null &&
        "kind" in params.bayesianPoints &&
        params.bayesianPoints.kind === "point-set"
          ? params.bayesianPoints
          : { kind: "point-set" as const, points: frame.bayesianRegression.points };
      const nextPoints = [...currentPointSet.points, { ...nextPoint, label: "custom" }].slice(-42);
      onParamChange("bayesianPoints", { kind: "point-set", points: nextPoints });
      return;
    }

    if (frame.gda) {
      const nextPoint = canvasPointToGdaPoint(frame.gda, size, point);
      if (!nextPoint) {
        return;
      }

      const nextClass = params.gdaClass === "blue" ? "blue" : "red";
      const currentPointSet =
        typeof params.gdaPoints === "object" &&
        params.gdaPoints !== null &&
        "kind" in params.gdaPoints &&
        params.gdaPoints.kind === "point-set"
          ? params.gdaPoints
          : { kind: "point-set" as const, points: frame.gda.points };
      const nextPoints = [...currentPointSet.points, { ...nextPoint, label: nextClass }].slice(-80);
      onParamChange("gdaPoints", { kind: "point-set", points: nextPoints });
      return;
    }

    if (!frame.dynamicProgramming) {
      return;
    }

    const hit = hitTestDynamicProgrammingGrid(frame.dynamicProgramming, size, point);
    if (!hit) {
      return;
    }

    const tool = gridWorldPaintTool(params.paintTool);
    const nextGrid = paintGridWorldCell(frame.dynamicProgramming.grid, hit.row, hit.column, tool);
    onParamChange("gridWorld", nextGrid);
  };

  const handleCanvasMove = (event: MouseEvent<HTMLCanvasElement>) => {
    if (frame?.type === "concept-demo" && frame.determinant && determinantDrag && onParamChange) {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const bounds = canvas.getBoundingClientRect();
      const point = {
        x: ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * size.width,
        y: ((event.clientY - bounds.top) / Math.max(1, bounds.height)) * size.height,
      };
      const vector = canvasPointToDeterminantVector(frame.determinant, size, point);
      if (!vector) {
        return;
      }

      if (determinantDrag === "basisI") {
        onParamChange("matrixA", vector.x);
        onParamChange("matrixC", vector.y);
      } else {
        onParamChange("matrixB", vector.x);
        onParamChange("matrixD", vector.y);
      }
      return;
    }

    if (frame?.type === "concept-demo" && frame.eigenDirection && eigenDirectionDrag && onParamChange) {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      const bounds = canvas.getBoundingClientRect();
      const point = {
        x: ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * size.width,
        y: ((event.clientY - bounds.top) / Math.max(1, bounds.height)) * size.height,
      };
      const angle = canvasPointToEigenDirectionAngle(frame.eigenDirection, size, point);
      if (angle !== null) {
        onParamChange("magicAngle", angle);
      }
      return;
    }

    if (frame?.type !== "concept-demo" || !frame.polynomial) {
      if (polynomialHoverIndex !== null) {
        setPolynomialHoverIndex(null);
      }
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
    const nextIndex = hitTestPolynomialPoint(frame.polynomial, size, point);
    if (nextIndex !== polynomialHoverIndex) {
      setPolynomialHoverIndex(nextIndex);
    }
  };

  const handleCanvasMouseDown = (event: MouseEvent<HTMLCanvasElement>) => {
    if (frame?.type !== "concept-demo" || !params || !onParamChange) {
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

    if (frame.determinant) {
      const handle = hitTestDeterminantHandle(frame.determinant, size, point);
      if (handle) {
        setDeterminantDrag(handle);
      }
      return;
    }

    if (frame.eigenDirection) {
      const angle = canvasPointToEigenDirectionAngle(frame.eigenDirection, size, point);
      if (angle !== null) {
        setEigenDirectionDrag(true);
        onParamChange("magicAngle", angle);
      }
    }
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
        isNmfLesson ? "nmf-shell" : ""
      } ${
        isPolynomialLesson ? "polynomial-shell" : ""
      } ${
        isBayesianRegressionLesson ? "bayesian-regression-shell" : ""
      } ${
        isBayesRuleLesson ? "bayes-rule-shell" : ""
      } ${
        isGdaLesson ? "gda-shell" : ""
      } ${
        isDeterminantLesson ? "determinant-shell" : ""
      } ${
        isEigenDirectionLesson ? "eigen-direction-shell" : ""
      } ${
        isRegularizationLesson ? "regularization-shell" : ""
      } ${
        isConvexLesson ? "convex-shell" : ""
      } ${
        isConvolutionLesson ? "convolution-shell" : ""
      } ${
        isDynamicProgrammingLesson ? "dynamic-programming-shell" : ""
      } ${
        isTokenizerLesson ? "tokenizer-shell" : ""
      }`}
      aria-label={`${algorithm.name} visualization`}
    >
      <canvas
        key={isConvexLesson ? "webgl-convex" : "canvas-2d"}
        ref={canvasRef}
        onClick={handleCanvasClick}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMove}
        onMouseUp={() => {
          setDeterminantDrag(null);
          setEigenDirectionDrag(false);
        }}
        onMouseLeave={() => {
          setPolynomialHoverIndex(null);
          setDeterminantDrag(null);
          setEigenDirectionDrag(false);
        }}
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
  interaction: { polynomialHoverIndex?: number | null } = {},
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

  if (frame.type === "concept-demo" && frame.nmf) {
    paintNmfLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.polynomial) {
    paintPolynomialLesson(context, frame, size, interaction.polynomialHoverIndex ?? null);
    return;
  }

  if (frame.type === "concept-demo" && frame.bayesianRegression) {
    paintBayesianRegressionLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.bayesRule) {
    paintBayesRuleLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.gda) {
    paintGdaLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.determinant) {
    paintDeterminantLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.eigenDirection) {
    paintEigenDirectionLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.regularization) {
    paintRegularizationLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.dynamicProgramming) {
    paintDynamicProgrammingLesson(context, frame, size);
    return;
  }

  if (frame.type === "concept-demo" && frame.tokenizer) {
    paintTokenizerLesson(context, frame, size);
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

const tokenizerChipColors = ["#d9f2ee", "#e4ecff", "#fff0d8", "#f8dedc", "#eee9ff", "#dcf5e8"];

function paintTokenizerLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const tokenizer = frame.tokenizer;
  if (!tokenizer) {
    return;
  }

  const narrow = size.width < 760;
  const padding = 24;
  const gap = 18;
  const leftPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(250, size.height * 0.46) }
    : { x: padding, y: 58, width: Math.max(420, size.width * 0.58), height: size.height - 96 };
  const rightPane: CanvasPane = narrow
    ? {
        x: padding,
        y: leftPane.y + leftPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(250, size.height - leftPane.height - 120),
      }
    : {
        x: leftPane.x + leftPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - leftPane.width - gap,
        height: size.height - 96,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Subword colorizer sandbox", leftPane.x, leftPane.y - 24);
  context.fillText("Token IDs and merge pressure", rightPane.x, rightPane.y - 24);

  paintTokenizerPieces(context, tokenizer, leftPane);
  paintTokenizerDetails(context, tokenizer, rightPane);
}

function paintTokenizerPieces(
  context: CanvasRenderingContext2D,
  tokenizer: NonNullable<ConceptFrame["tokenizer"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Live BPE segmentation", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `${tokenizer.characterLength} characters · ${tokenizer.byteLength} UTF-8 bytes · ${tokenizer.mergeCount} trained merges active`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  let x = pane.x + 16;
  let y = pane.y + 72;
  const maxX = pane.x + pane.width - 16;
  const maxY = pane.y + pane.height - 66;
  const chipHeight = 34;
  const rowGap = 10;

  tokenizer.pieces.forEach((piece, index) => {
    const label = tokenizerPieceLabel(piece.text);
    context.font = piece.byteFallback
      ? "900 11px ui-monospace, SFMono-Regular, Menlo, monospace"
      : "900 13px Inter, system-ui, sans-serif";
    const chipWidth = Math.min(maxX - (pane.x + 16), Math.max(42, context.measureText(label).width + 22));

    if (x + chipWidth > maxX) {
      x = pane.x + 16;
      y += chipHeight + rowGap;
    }

    if (y + chipHeight > maxY) {
      if (index < tokenizer.pieces.length) {
        context.fillStyle = "#61707f";
        context.font = "800 11px Inter, system-ui, sans-serif";
        context.fillText(`+${tokenizer.pieces.length - index} more token(s)`, x, y + 18);
      }
      return;
    }

    const color = piece.byteFallback ? "#fff7f6" : tokenizerChipColors[piece.colorIndex % tokenizerChipColors.length];
    paintRoundedRect(context, x, y, chipWidth, chipHeight, 8, color, piece.byteFallback ? "#d34a43" : "#d8e0e5");
    context.fillStyle = piece.byteFallback ? "#9f2d28" : "#17212b";
    context.textAlign = "center";
    context.fillText(fitCanvasText(context, label, chipWidth - 14), x + chipWidth / 2, y + 21);

    x += chipWidth + 8;
  });

  const baselineY = pane.y + pane.height - 48;
  paintTokenizerStatStrip(context, tokenizer, {
    x: pane.x + 14,
    y: baselineY,
    width: pane.width - 28,
    height: 32,
  });

  context.restore();
}

function paintTokenizerStatStrip(
  context: CanvasRenderingContext2D,
  tokenizer: NonNullable<ConceptFrame["tokenizer"]>,
  pane: CanvasPane,
) {
  const stats = [
    { label: "tokens", value: tokenizer.pieces.length.toString(), color: "#0f766e" },
    { label: "byte fallback", value: tokenizer.byteFallbackCount.toString(), color: "#d34a43" },
    { label: "overhead", value: `${tokenizer.compressionRatio.toFixed(2)}x`, color: "#2f6fbe" },
  ];
  const gap = 8;
  const width = (pane.width - gap * (stats.length - 1)) / stats.length;

  stats.forEach((stat, index) => {
    const x = pane.x + index * (width + gap);
    paintRoundedRect(context, x, pane.y, width, pane.height, 8, "#ffffff", "#d8e0e5");
    context.fillStyle = stat.color;
    context.font = "900 11px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(stat.value, x + width / 2, pane.y + 13);
    context.fillStyle = "#61707f";
    context.font = "800 8px Inter, system-ui, sans-serif";
    context.fillText(stat.label, x + width / 2, pane.y + 25);
  });
}

function paintTokenizerDetails(
  context: CanvasRenderingContext2D,
  tokenizer: NonNullable<ConceptFrame["tokenizer"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`Raw token ID array [${tokenizer.tokenIds.length}]`, pane.x + 14, pane.y + 22);

  const idsText = `[${tokenizer.tokenIds.join(", ")}]`;
  context.fillStyle = "#26323f";
  context.font = "800 11px ui-monospace, SFMono-Regular, Menlo, monospace";
  wrapCanvasText(context, idsText, pane.x + 14, pane.y + 48, pane.width - 28, 16, 7);

  const byteBoxY = pane.y + Math.min(170, pane.height * 0.34);
  paintRoundedRect(context, pane.x + 14, byteBoxY, pane.width - 28, 72, 8, "#fff7f6", "rgba(211, 74, 67, 0.3)");
  context.fillStyle = "#9f2d28";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("Byte-level fallback", pane.x + 28, byteBoxY + 21);
  context.fillStyle = "#61707f";
  context.font = "700 10px Inter, system-ui, sans-serif";
  wrapCanvasText(
    context,
    tokenizer.byteFallbackCount > 0
      ? `${tokenizer.byteFallbackCount} token(s) are raw UTF-8 bytes. Unmerged emoji or underrepresented scripts cost more context.`
      : "No byte fallbacks in this input. Common ASCII subwords are being merged into larger chunks.",
    pane.x + 28,
    byteBoxY + 42,
    pane.width - 56,
    14,
    2,
  );

  const mergeY = byteBoxY + 100;
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("Recent learned merges", pane.x + 14, mergeY);

  tokenizer.topMerges.forEach((merge, index) => {
    const rowY = mergeY + 18 + index * 28;
    if (rowY + 22 > pane.y + pane.height - 14) {
      return;
    }
    paintRoundedRect(context, pane.x + 14, rowY, pane.width - 28, 22, 7, "#ffffff", "#d8e0e5");
    context.fillStyle = "#17212b";
    context.font = "800 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.textAlign = "left";
    context.fillText(
      fitCanvasText(context, `${tokenizerPieceLabel(merge.left)} + ${tokenizerPieceLabel(merge.right)} -> ${tokenizerPieceLabel(merge.merged)}`, pane.width - 96),
      pane.x + 24,
      rowY + 15,
    );
    context.fillStyle = "#61707f";
    context.textAlign = "right";
    context.fillText(`x${merge.count}`, pane.x + pane.width - 24, rowY + 15);
  });

  context.restore();
}

function tokenizerPieceLabel(text: string) {
  if (text === " ") {
    return "space";
  }

  return text.replace(/\s/g, "·");
}

function paintNmfLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const nmf = frame.nmf;
  if (!nmf) {
    return;
  }

  const narrow = size.width < 820;
  const padding = 24;
  const gap = 18;
  const topicPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(320, size.height * 0.48) }
    : { x: padding, y: 58, width: Math.max(460, size.width * 0.55), height: size.height - 96 };
  const facePane: CanvasPane = narrow
    ? {
        x: padding,
        y: topicPane.y + topicPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(330, size.height - topicPane.height - 120),
      }
    : {
        x: topicPane.x + topicPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - topicPane.width - gap,
        height: size.height - 96,
      };

  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Topic extraction workbench", topicPane.x, topicPane.y - 24);
  context.fillText("Face decomposition slider", facePane.x, facePane.y - 24);

  paintNmfTopicPane(context, nmf, topicPane);
  paintNmfFacePane(context, nmf, facePane);
}

function paintNmfTopicPane(
  context: CanvasRenderingContext2D,
  nmf: NonNullable<ConceptFrame["nmf"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${nmf.documents.length} documents -> W x H`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `${nmf.vocabulary.length} terms · ${nmf.iterations} multiplicative updates · sparsity ${Math.round(nmf.sparsity * 100)}%`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const matrixTop = pane.y + 64;
  const matrixHeight = Math.min(98, pane.height * 0.18);
  const matrixGap = 10;
  const matrixWidth = (pane.width - 28 - matrixGap * 2) / 3;
  paintNmfMiniMatrix(context, "X docs x terms", nmf.documents.length, nmf.vocabulary.length, {
    x: pane.x + 14,
    y: matrixTop,
    width: matrixWidth,
    height: matrixHeight,
  }, "#64748b");
  paintNmfMiniMatrix(context, "W docs x topics", nmf.documentTopicMatrix.length, nmf.topicCount, {
    x: pane.x + 14 + matrixWidth + matrixGap,
    y: matrixTop,
    width: matrixWidth,
    height: matrixHeight,
  }, "#2f6fbe", nmf.documentTopicMatrix);
  paintNmfMiniMatrix(context, "H topics x words", nmf.topicCount, nmf.vocabulary.length, {
    x: pane.x + 14 + (matrixWidth + matrixGap) * 2,
    y: matrixTop,
    width: matrixWidth,
    height: matrixHeight,
  }, "#0f766e", nmf.topicWordMatrix);

  const topicTop = matrixTop + matrixHeight + 24;
  const topicGap = 10;
  const columns = pane.width > 620 ? 2 : 1;
  const rows = Math.ceil(nmf.topics.length / columns);
  const cardWidth = (pane.width - 28 - topicGap * (columns - 1)) / columns;
  const cardHeight = Math.max(104, Math.min(150, (pane.y + pane.height - topicTop - 16 - topicGap * (rows - 1)) / rows));

  nmf.topics.forEach((topic, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    paintNmfTopicCard(context, topic, index, {
      x: pane.x + 14 + column * (cardWidth + topicGap),
      y: topicTop + row * (cardHeight + topicGap),
      width: cardWidth,
      height: cardHeight,
    });
  });

  context.restore();
}

function paintNmfTopicCard(
  context: CanvasRenderingContext2D,
  topic: NonNullable<ConceptFrame["nmf"]>["topics"][number],
  topicIndex: number,
  box: CanvasPane,
) {
  const color = colors[topicIndex % colors.length];
  paintRoundedRect(context, box.x, box.y, box.width, box.height, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = color;
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(topic.label, box.x + 12, box.y + 18);

  const topWeight = Math.max(1e-8, ...topic.words.map((word) => word.weight));
  const barX = box.x + 12;
  const barY = box.y + 32;
  const barWidth = box.width - 24;
  const barHeight = 12;
  topic.words.slice(0, 5).forEach((word, index) => {
    const y = barY + index * 18;
    const width = (word.weight / topWeight) * barWidth;
    context.fillStyle = "#eef3f6";
    context.fillRect(barX, y, barWidth, barHeight);
    context.fillStyle = color;
    context.fillRect(barX, y, width, barHeight);
    context.fillStyle = "#17212b";
    context.font = "800 9px Inter, system-ui, sans-serif";
    context.fillText(fitCanvasText(context, word.term, barWidth - 8), barX + 5, y + 9);
  });

  context.fillStyle = "#61707f";
  context.font = "700 9px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  wrapCanvasText(context, topic.topDocument, box.x + 12, box.y + box.height - 26, box.width - 24, 12, 2);
}

function paintNmfMiniMatrix(
  context: CanvasRenderingContext2D,
  label: string,
  rows: number,
  columns: number,
  box: CanvasPane,
  color: string,
  matrix?: number[][],
) {
  paintRoundedRect(context, box.x, box.y, box.width, box.height, 8, "#f7fafc", "#d8e0e5");
  const grid = {
    x: box.x + 10,
    y: box.y + 24,
    width: box.width - 20,
    height: box.height - 44,
  };
  const visibleRows = Math.min(rows, 10);
  const visibleColumns = Math.min(columns, 16);
  const cellWidth = grid.width / Math.max(1, visibleColumns);
  const cellHeight = grid.height / Math.max(1, visibleRows);
  const values = matrix?.flat() ?? [];
  const maxValue = Math.max(1e-8, ...values);

  for (let row = 0; row < visibleRows; row += 1) {
    for (let column = 0; column < visibleColumns; column += 1) {
      const value = matrix ? matrix[row]?.[column] ?? 0 : ((row + column) % 4) / 4;
      const alpha = 0.16 + Math.min(1, value / maxValue) * 0.7;
      context.fillStyle = `${color}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`;
      context.fillRect(
        grid.x + column * cellWidth,
        grid.y + row * cellHeight,
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }
  }

  context.fillStyle = "#17212b";
  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(fitCanvasText(context, label, box.width - 12), box.x + box.width / 2, box.y + 15);
  context.fillStyle = "#61707f";
  context.font = "800 8px Inter, system-ui, sans-serif";
  context.fillText(`${rows}x${columns}`, box.x + box.width / 2, box.y + box.height - 10);
}

function paintNmfFacePane(
  context: CanvasRenderingContext2D,
  nmf: NonNullable<ConceptFrame["nmf"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Parts add without cancellation", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, `Move one W column weight: only its localized facial part changes. Error ${nmf.faceReconstructionError.toFixed(3)}`, pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  const top = pane.y + 64;
  const imageGap = 12;
  const imageHeight = Math.min(210, Math.max(150, pane.height * 0.34));
  const imageWidth = (pane.width - 28 - imageGap) / 2;
  paintSvdImageBox(context, "canonical face", nmf.faceOriginal, {
    x: pane.x + 14,
    y: top,
    width: imageWidth,
    height: imageHeight,
  });
  paintSvdImageBox(context, "W-weighted face", nmf.faceReconstruction, {
    x: pane.x + 14 + imageWidth + imageGap,
    y: top,
    width: imageWidth,
    height: imageHeight,
  });

  const partTop = top + imageHeight + 34;
  const partGap = 10;
  const columns = pane.width > 420 ? 3 : 2;
  const partWidth = (pane.width - 28 - partGap * (columns - 1)) / columns;
  const partHeight = Math.max(78, Math.min(112, (pane.y + pane.height - partTop - 16) / 2));

  nmf.faceParts.forEach((part, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const box = {
      x: pane.x + 14 + column * (partWidth + partGap),
      y: partTop + row * (partHeight + partGap),
      width: partWidth,
      height: partHeight,
    };
    paintNmfFacePart(context, part, box);
  });

  context.restore();
}

function paintNmfFacePart(
  context: CanvasRenderingContext2D,
  part: NonNullable<ConceptFrame["nmf"]>["faceParts"][number],
  box: CanvasPane,
) {
  paintRoundedRect(context, box.x, box.y, box.width, box.height, 8, "#ffffff", "#d8e0e5");
  const imageSize = Math.min(box.height - 22, box.width * 0.42);
  const imageBox = {
    x: box.x + 8,
    y: box.y + 10,
    width: imageSize,
    height: imageSize,
  };
  paintSvdImageBox(context, "", part.matrix, imageBox);

  context.fillStyle = part.color;
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(fitCanvasText(context, part.label, box.width - imageSize - 24), imageBox.x + imageSize + 8, box.y + 22);
  context.fillStyle = "#eef3f6";
  context.fillRect(imageBox.x + imageSize + 8, box.y + 34, box.width - imageSize - 24, 10);
  context.fillStyle = part.color;
  context.fillRect(imageBox.x + imageSize + 8, box.y + 34, (box.width - imageSize - 24) * Math.min(1, part.weight / 1.6), 10);
  context.fillStyle = "#61707f";
  context.font = "800 9px Inter, system-ui, sans-serif";
  context.fillText(`W=${part.weight.toFixed(2)}`, imageBox.x + imageSize + 8, box.y + 58);
}

function paintPolynomialLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
  hoverIndex: number | null,
) {
  const polynomial = frame.polynomial;
  if (!polynomial) {
    return;
  }

  const layout = polynomialLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Live curve-fitting sandbox", layout.plotPane.x, layout.plotPane.y - 24);
  context.fillText("Feature matrix expansion", layout.matrixPane.x, layout.matrixPane.y - 24);

  paintPolynomialPlot(context, polynomial, layout.plotPane, hoverIndex);
  paintPolynomialFeaturePanel(context, polynomial, layout.matrixPane, hoverIndex);
}

function polynomialLayout(size: Size) {
  const narrow = size.width < 820;
  const padding = 24;
  const gap = 18;
  const plotPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(330, size.height * 0.5) }
    : { x: padding, y: 58, width: Math.max(470, size.width * 0.58), height: size.height - 96 };
  const matrixPane: CanvasPane = narrow
    ? {
        x: padding,
        y: plotPane.y + plotPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(310, size.height - plotPane.height - 120),
      }
    : {
        x: plotPane.x + plotPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - plotPane.width - gap,
        height: size.height - 96,
      };

  return { plotPane, matrixPane };
}

function polynomialPlotArea(pane: CanvasPane): CanvasPane {
  return {
    x: pane.x + 52,
    y: pane.y + 72,
    width: pane.width - 78,
    height: pane.height - 126,
  };
}

function polynomialScales(polynomial: NonNullable<ConceptFrame["polynomial"]>, pane: CanvasPane) {
  const plot = polynomialPlotArea(pane);
  return {
    plot,
    x: d3.scaleLinear().domain(polynomial.xDomain).range([plot.x, plot.x + plot.width]),
    y: d3.scaleLinear().domain(polynomial.yDomain).range([plot.y + plot.height, plot.y]),
  };
}

function paintPolynomialPlot(
  context: CanvasRenderingContext2D,
  polynomial: NonNullable<ConceptFrame["polynomial"]>,
  pane: CanvasPane,
  hoverIndex: number | null,
) {
  context.save();
  paintCanvasPanel(context, pane);

  const overfit = polynomial.degree >= 8 || polynomial.overfitScore > 0.58;
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`Degree ${polynomial.degree}: ${polynomial.degree === 1 ? "straight line" : overfit ? "overfit risk" : "curved fit"}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, `${polynomial.points.length} points · ${polynomial.degree + 1} power columns · MSE ${polynomial.mse.toFixed(3)}`, pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  const { plot, x, y } = polynomialScales(polynomial, pane);
  context.fillStyle = "rgba(247, 250, 252, 0.9)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#d8e0e5";
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.strokeStyle = "rgba(100, 116, 139, 0.18)";
  context.lineWidth = 1;
  x.ticks(7).forEach((tick) => {
    const tx = x(tick);
    context.beginPath();
    context.moveTo(tx, plot.y);
    context.lineTo(tx, plot.y + plot.height);
    context.stroke();
  });
  y.ticks(6).forEach((tick) => {
    const ty = y(tick);
    context.beginPath();
    context.moveTo(plot.x, ty);
    context.lineTo(plot.x + plot.width, ty);
    context.stroke();
  });

  context.strokeStyle = "#c9d4da";
  context.beginPath();
  context.moveTo(plot.x, y(0));
  context.lineTo(plot.x + plot.width, y(0));
  context.moveTo(x(0), plot.y);
  context.lineTo(x(0), plot.y + plot.height);
  context.stroke();

  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  context.strokeStyle = overfit ? "#d34a43" : polynomial.degree === 1 ? "#64748b" : "#0f766e";
  context.lineWidth = overfit ? 3.2 : 2.8;
  context.beginPath();
  polynomial.curve.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      return;
    }
    if (index === 0) {
      context.moveTo(px, py);
    } else {
      context.lineTo(px, py);
    }
  });
  context.stroke();
  context.restore();

  polynomial.points.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    const active = index === hoverIndex;
    context.fillStyle = active ? "#fff7ed" : "#2f6fbe";
    context.strokeStyle = active ? "#b7791f" : "#ffffff";
    context.lineWidth = active ? 3 : 1.6;
    context.beginPath();
    context.arc(px, py, active ? 7 : 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });

  const statusBox = {
    x: pane.x + 14,
    y: pane.y + pane.height - 42,
    width: pane.width - 28,
    height: 26,
  };
  paintRoundedRect(context, statusBox.x, statusBox.y, statusBox.width, statusBox.height, 8, overfit ? "#fff7f6" : "#f7fafc", overfit ? "rgba(211, 74, 67, 0.3)" : "#d8e0e5");
  context.fillStyle = overfit ? "#9f2d28" : "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(
      context,
      overfit
        ? "High-degree polynomial is starting to chase individual points. Try degree 2 or 3 for a smoother general trend."
        : "Click inside the plot to add a point. Hover a point to inspect its expanded feature row.",
      statusBox.width - 16,
    ),
    statusBox.x + 8,
    statusBox.y + 17,
  );

  context.restore();
}

function paintPolynomialFeaturePanel(
  context: CanvasRenderingContext2D,
  polynomial: NonNullable<ConceptFrame["polynomial"]>,
  pane: CanvasPane,
  hoverIndex: number | null,
) {
  context.save();
  paintCanvasPanel(context, pane);

  const selectedIndex = hoverIndex ?? 0;
  const selectedPoint = polynomial.points[selectedIndex] ?? polynomial.points[0];
  const featureRow = polynomial.featureRows[selectedIndex] ?? polynomial.featureRows[0] ?? [];
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("One x becomes many columns", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      selectedPoint
        ? `Selected x=${selectedPoint.x.toFixed(2)}, y=${selectedPoint.y.toFixed(2)}`
        : "Hover a point to inspect its row.",
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const rowY = pane.y + 72;
  const chipGap = 8;
  const chipColumns = pane.width > 440 ? 3 : 2;
  const chipWidth = (pane.width - 28 - chipGap * (chipColumns - 1)) / chipColumns;
  const chipHeight = 42;
  featureRow.forEach((cell, index) => {
    const column = index % chipColumns;
    const row = Math.floor(index / chipColumns);
    const x = pane.x + 14 + column * (chipWidth + chipGap);
    const y = rowY + row * (chipHeight + chipGap);
    if (y + chipHeight > pane.y + pane.height - 176) {
      return;
    }
    paintRoundedRect(context, x, y, chipWidth, chipHeight, 8, "#ffffff", "#d8e0e5");
    context.fillStyle = index === 0 ? "#64748b" : colors[(index - 1) % colors.length];
    context.font = "900 10px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(polynomialPowerLabel(cell.power), x + 9, y + 15);
    context.fillStyle = "#17212b";
    context.font = "800 10px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillText(fitCanvasText(context, formatPolynomialNumber(cell.value), chipWidth - 18), x + 9, y + 31);
  });

  const coefficientTop = pane.y + Math.max(250, pane.height * 0.54);
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("Learned coefficients theta", pane.x + 14, coefficientTop);
  const coeffValues = polynomial.coefficients.slice(0, Math.min(11, polynomial.coefficients.length));
  const maxCoeff = Math.max(1e-8, ...coeffValues.map((value) => Math.abs(value)));
  coeffValues.forEach((coefficient, index) => {
    const y = coefficientTop + 18 + index * 18;
    if (y + 12 > pane.y + pane.height - 18) {
      return;
    }
    const labelWidth = 34;
    const barWidth = pane.width - 28 - labelWidth - 48;
    const centerX = pane.x + 14 + labelWidth + barWidth / 2;
    context.fillStyle = "#61707f";
    context.font = "800 9px Inter, system-ui, sans-serif";
    context.textAlign = "right";
    context.fillText(`θ${index}`, pane.x + 14 + labelWidth - 6, y + 9);
    context.fillStyle = "#eef3f6";
    context.fillRect(centerX - barWidth / 2, y, barWidth, 10);
    context.fillStyle = coefficient >= 0 ? "#0f766e" : "#d34a43";
    const width = (Math.abs(coefficient) / maxCoeff) * (barWidth / 2);
    context.fillRect(coefficient >= 0 ? centerX : centerX - width, y, width, 10);
    context.fillStyle = "#61707f";
    context.textAlign = "left";
    context.fillText(formatPolynomialNumber(coefficient), centerX + barWidth / 2 + 7, y + 9);
  });

  context.restore();
}

function polynomialPowerLabel(power: number) {
  if (power === 0) {
    return "bias 1";
  }
  if (power === 1) {
    return "x";
  }
  return `x^${power}`;
}

function formatPolynomialNumber(value: number) {
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2);
  }
  return value.toFixed(Math.abs(value) >= 10 ? 1 : 3);
}

function hitTestPolynomialPoint(
  polynomial: NonNullable<ConceptFrame["polynomial"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const { plotPane } = polynomialLayout(size);
  const { x, y } = polynomialScales(polynomial, plotPane);
  let nearestIndex: number | null = null;
  let nearestDistance = Number.POSITIVE_INFINITY;

  polynomial.points.forEach((dataPoint, index) => {
    const distance = Math.hypot(point.x - x(dataPoint.x), point.y - y(dataPoint.y));
    if (distance < nearestDistance) {
      nearestIndex = index;
      nearestDistance = distance;
    }
  });

  return nearestIndex !== null && nearestDistance <= 14 ? nearestIndex : null;
}

function canvasPointToPolynomialPoint(
  polynomial: NonNullable<ConceptFrame["polynomial"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const { plotPane } = polynomialLayout(size);
  const { plot, x, y } = polynomialScales(polynomial, plotPane);
  if (
    point.x < plot.x ||
    point.x > plot.x + plot.width ||
    point.y < plot.y ||
    point.y > plot.y + plot.height
  ) {
    return null;
  }

  return {
    x: Number(x.invert(point.x).toFixed(3)),
    y: Number(y.invert(point.y).toFixed(3)),
  };
}

function paintBayesRuleLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const bayesRule = frame.bayesRule;
  if (!bayesRule) {
    return;
  }

  const layout = bayesRuleLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Frequency filtering", layout.gridPane.x, layout.gridPane.y - 24);
  context.fillText("Formula and probability tree", layout.treePane.x, layout.treePane.y - 24);

  paintBayesRuleGrid(context, bayesRule, layout.gridPane);
  paintBayesRuleTreePanel(context, bayesRule, layout.treePane);
}

function bayesRuleLayout(size: Size) {
  const narrow = size.width < 820;
  const padding = 24;
  const gap = 18;
  const gridPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(380, size.height * 0.56) }
    : { x: padding, y: 58, width: Math.max(500, size.width * 0.6), height: size.height - 96 };
  const treePane: CanvasPane = narrow
    ? {
        x: padding,
        y: gridPane.y + gridPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(330, size.height - gridPane.height - 120),
      }
    : {
        x: gridPane.x + gridPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - gridPane.width - gap,
        height: size.height - 96,
      };

  return { gridPane, treePane };
}

function paintBayesRuleGrid(
  context: CanvasRenderingContext2D,
  bayesRule: NonNullable<ConceptFrame["bayesRule"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("1,000 people: where the positive tests come from", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `${bayesRule.counts.truePositive} true positives + ${bayesRule.counts.falsePositive} false positives = ${bayesRule.counts.positiveTests} positive tests`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const grid = bayesRuleGridArea(pane);
  const columns = 40;
  const rows = 25;
  const gap = Math.max(1, Math.min(3, grid.width / 260));
  const cellSize = Math.min((grid.width - gap * (columns - 1)) / columns, (grid.height - gap * (rows - 1)) / rows);
  const offsetX = grid.x + (grid.width - (cellSize * columns + gap * (columns - 1))) / 2;
  const offsetY = grid.y + (grid.height - (cellSize * rows + gap * (rows - 1))) / 2;

  context.fillStyle = "#f7fafc";
  context.fillRect(grid.x, grid.y, grid.width, grid.height);
  context.strokeStyle = "#d8e0e5";
  context.strokeRect(grid.x, grid.y, grid.width, grid.height);

  bayesRule.cells.forEach((cell, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = offsetX + column * (cellSize + gap);
    const y = offsetY + row * (cellSize + gap);
    const visual = bayesRuleCellVisual(cell.status, bayesRule.phase, bayesRule.phaseProgress);
    context.globalAlpha = visual.alpha;
    context.fillStyle = visual.color;
    context.fillRect(x, y, cellSize, cellSize);
  });
  context.globalAlpha = 1;

  paintBayesRuleLegend(context, pane, grid.y + grid.height + 14);
  paintBayesRulePipeline(context, bayesRule, pane);
  context.restore();
}

function bayesRuleGridArea(pane: CanvasPane): CanvasPane {
  return {
    x: pane.x + 16,
    y: pane.y + 68,
    width: pane.width - 32,
    height: Math.max(250, pane.height - 250),
  };
}

function bayesRuleCellVisual(
  status: NonNullable<ConceptFrame["bayesRule"]>["cells"][number]["status"],
  phase: NonNullable<ConceptFrame["bayesRule"]>["phase"],
  phaseProgress: number,
) {
  const testedPositive = status === "true-positive" || status === "false-positive";
  const actualPositive = status === "true-positive" || status === "false-negative";
  const colorsByStatus = {
    "true-positive": "#0f766e",
    "false-positive": "#b7791f",
    "false-negative": "#7fb6ad",
    "true-negative": "#d8e0e5",
  };

  if (phase === "population") {
    return {
      color: actualPositive ? "#0f766e" : "#cfd8df",
      alpha: 0.35 + phaseProgress * (actualPositive ? 0.55 : 0.25),
    };
  }

  if (phase === "positive-tests") {
    return {
      color: colorsByStatus[status],
      alpha: testedPositive ? 0.32 + phaseProgress * 0.68 : 0.14,
    };
  }

  return {
    color: colorsByStatus[status],
    alpha: status === "true-positive" ? 1 : status === "false-positive" ? 0.75 : 0.12,
  };
}

function paintBayesRuleLegend(
  context: CanvasRenderingContext2D,
  pane: CanvasPane,
  y: number,
) {
  const items = [
    { label: "true positive", color: "#0f766e" },
    { label: "false positive", color: "#b7791f" },
    { label: "missed actual A", color: "#7fb6ad" },
    { label: "true negative", color: "#d8e0e5" },
  ];
  let x = pane.x + 16;
  items.forEach((item) => {
    context.fillStyle = item.color;
    context.fillRect(x, y, 10, 10);
    context.fillStyle = "#61707f";
    context.font = "800 9px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(item.label, x + 14, y + 9);
    x += Math.min(132, Math.max(92, context.measureText(item.label).width + 30));
  });
}

function paintBayesRulePipeline(
  context: CanvasRenderingContext2D,
  bayesRule: NonNullable<ConceptFrame["bayesRule"]>,
  pane: CanvasPane,
) {
  const top = pane.y + pane.height - 82;
  const gap = 10;
  const boxWidth = (pane.width - 32 - gap * 2) / 3;
  const boxes = [
    {
      key: "population",
      title: "Population",
      value: `${bayesRule.population}`,
      detail: `${bayesRule.counts.actualPositive} have A`,
    },
    {
      key: "positive-tests",
      title: "Positive tests",
      value: `${bayesRule.counts.positiveTests}`,
      detail: `${bayesRule.counts.falsePositive} false alarms`,
    },
    {
      key: "actual-positives",
      title: "Actual positives",
      value: `${bayesRule.counts.truePositive}`,
      detail: `${(bayesRule.posterior * 100).toFixed(1)}% of positives`,
    },
  ];

  boxes.forEach((box, index) => {
    const x = pane.x + 16 + index * (boxWidth + gap);
    const active = box.key === bayesRule.phase;
    paintRoundedRect(context, x, top, boxWidth, 58, 8, active ? "#e8f5f2" : "#ffffff", active ? "#0f766e" : "#d8e0e5");
    context.fillStyle = active ? "#0f766e" : "#61707f";
    context.font = "900 9px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(fitCanvasText(context, box.title, boxWidth - 16), x + 8, top + 16);
    context.fillStyle = "#17212b";
    context.font = "900 16px Inter, system-ui, sans-serif";
    context.fillText(box.value, x + 8, top + 36);
    context.fillStyle = "#61707f";
    context.font = "800 9px Inter, system-ui, sans-serif";
    context.fillText(fitCanvasText(context, box.detail, boxWidth - 16), x + 8, top + 50);

    if (index < boxes.length - 1) {
      const arrowX = x + boxWidth + 2;
      context.strokeStyle = "#9aa8b4";
      context.fillStyle = "#9aa8b4";
      context.lineWidth = 1.5;
      context.beginPath();
      context.moveTo(arrowX, top + 29);
      context.lineTo(arrowX + gap - 4, top + 29);
      context.stroke();
      context.beginPath();
      context.moveTo(arrowX + gap - 4, top + 29);
      context.lineTo(arrowX + gap - 9, top + 25);
      context.lineTo(arrowX + gap - 9, top + 33);
      context.closePath();
      context.fill();
    }
  });
}

function paintBayesRuleTreePanel(
  context: CanvasRenderingContext2D,
  bayesRule: NonNullable<ConceptFrame["bayesRule"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Posterior = useful evidence / total evidence", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, bayesRuleFormulaText(bayesRule), pane.width - 28),
    pane.x + 14,
    pane.y + 43,
  );

  const chipTop = pane.y + 66;
  const chipGap = 8;
  const chipColumns = pane.width > 430 ? 3 : 2;
  const chipWidth = (pane.width - 28 - chipGap * (chipColumns - 1)) / chipColumns;
  const chips = [
    { label: "P(A)", value: probabilityLabel(bayesRule.prior), color: "#2f6fbe" },
    { label: "P(B|A)", value: probabilityLabel(bayesRule.sensitivity), color: "#0f766e" },
    { label: "P(B|not A)", value: probabilityLabel(bayesRule.falsePositiveRate), color: "#b7791f" },
    { label: "P(B)", value: probabilityLabel(bayesRule.evidence), color: "#6f58c9" },
    { label: "P(B|A)P(A)", value: probabilityLabel(bayesRule.usefulEvidence), color: "#0f766e" },
    { label: "P(A|B)", value: probabilityLabel(bayesRule.posterior), color: "#d34a43" },
  ];
  chips.forEach((chip, index) => {
    const column = index % chipColumns;
    const row = Math.floor(index / chipColumns);
    const x = pane.x + 14 + column * (chipWidth + chipGap);
    const y = chipTop + row * 48;
    paintBayesRuleChip(context, x, y, chipWidth, chip.label, chip.value, chip.color);
  });

  const treeTop = chipTop + Math.ceil(chips.length / chipColumns) * 48 + 20;
  paintBayesRuleTree(context, bayesRule, {
    x: pane.x + 14,
    y: treeTop,
    width: pane.width - 28,
    height: Math.max(190, pane.y + pane.height - treeTop - 18),
  });

  context.restore();
}

function bayesRuleFormulaText(bayesRule: NonNullable<ConceptFrame["bayesRule"]>) {
  return `P(A|B) = (${bayesRule.sensitivity.toFixed(2)} * ${bayesRule.prior.toFixed(2)}) / ${bayesRule.evidence.toFixed(2)} = ${bayesRule.posterior.toFixed(2)}`;
}

function paintBayesRuleChip(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  label: string,
  value: string,
  color: string,
) {
  paintRoundedRect(context, x, y, width, 38, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = color;
  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(label, x + 8, y + 14);
  context.fillStyle = "#17212b";
  context.font = "900 13px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillText(value, x + 8, y + 30);
}

function paintBayesRuleTree(
  context: CanvasRenderingContext2D,
  bayesRule: NonNullable<ConceptFrame["bayesRule"]>,
  pane: CanvasPane,
) {
  paintRoundedRect(context, pane.x, pane.y, pane.width, pane.height, 8, "#f7fafc", "#d8e0e5");
  const root = { x: pane.x + 24, y: pane.y + pane.height * 0.5 };
  const middleX = pane.x + pane.width * 0.46;
  const leafX = pane.x + pane.width - 34;
  const a = { x: middleX, y: pane.y + pane.height * 0.28 };
  const notA = { x: middleX, y: pane.y + pane.height * 0.72 };
  const bGivenA = { x: leafX, y: pane.y + pane.height * 0.18 };
  const notBGivenA = { x: leafX, y: pane.y + pane.height * 0.38 };
  const bGivenNotA = { x: leafX, y: pane.y + pane.height * 0.62 };
  const notBGivenNotA = { x: leafX, y: pane.y + pane.height * 0.82 };

  paintBayesBranch(context, root, a, bayesRule.prior, "#2f6fbe", `A ${probabilityLabel(bayesRule.prior)}`);
  paintBayesBranch(context, root, notA, 1 - bayesRule.prior, "#94a3b8", `not A ${probabilityLabel(1 - bayesRule.prior)}`);
  paintBayesBranch(context, a, bGivenA, bayesRule.sensitivity, "#0f766e", `B ${probabilityLabel(bayesRule.sensitivity)}`);
  paintBayesBranch(context, a, notBGivenA, 1 - bayesRule.sensitivity, "#7fb6ad", `not B ${probabilityLabel(1 - bayesRule.sensitivity)}`);
  paintBayesBranch(context, notA, bGivenNotA, bayesRule.falsePositiveRate, "#b7791f", `B ${probabilityLabel(bayesRule.falsePositiveRate)}`);
  paintBayesBranch(context, notA, notBGivenNotA, 1 - bayesRule.falsePositiveRate, "#cbd5df", `not B ${probabilityLabel(1 - bayesRule.falsePositiveRate)}`);

  paintBayesNode(context, root.x, root.y, "Start", "#17212b");
  paintBayesNode(context, a.x, a.y, "A", "#2f6fbe");
  paintBayesNode(context, notA.x, notA.y, "not A", "#64748b");
  paintBayesLeaf(context, bGivenA.x, bGivenA.y, `${bayesRule.counts.truePositive}`, "true +", "#0f766e");
  paintBayesLeaf(context, notBGivenA.x, notBGivenA.y, `${bayesRule.counts.falseNegative}`, "missed", "#7fb6ad");
  paintBayesLeaf(context, bGivenNotA.x, bGivenNotA.y, `${bayesRule.counts.falsePositive}`, "false +", "#b7791f");
  paintBayesLeaf(context, notBGivenNotA.x, notBGivenNotA.y, `${bayesRule.counts.trueNegative}`, "true -", "#64748b");
}

function paintBayesBranch(
  context: CanvasRenderingContext2D,
  start: { x: number; y: number },
  end: { x: number; y: number },
  probability: number,
  color: string,
  label: string,
) {
  context.strokeStyle = color;
  context.lineWidth = 1.5 + probability * 5;
  context.globalAlpha = 0.75;
  context.beginPath();
  context.moveTo(start.x, start.y);
  context.lineTo(end.x, end.y);
  context.stroke();
  context.globalAlpha = 1;

  const labelX = start.x + (end.x - start.x) * 0.48;
  const labelY = start.y + (end.y - start.y) * 0.48 - 5;
  context.fillStyle = color;
  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(label, labelX, labelY);
}

function paintBayesNode(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  label: string,
  color: string,
) {
  context.fillStyle = "#ffffff";
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.beginPath();
  context.arc(x, y, 18, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = color;
  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(label, x, y + 3);
}

function paintBayesLeaf(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  value: string,
  label: string,
  color: string,
) {
  paintRoundedRect(context, x - 24, y - 18, 48, 36, 8, "#ffffff", color);
  context.fillStyle = color;
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(value, x, y - 2);
  context.fillStyle = "#61707f";
  context.font = "800 8px Inter, system-ui, sans-serif";
  context.fillText(label, x, y + 12);
}

function probabilityLabel(value: number) {
  return value < 0.1 ? value.toFixed(3) : value.toFixed(2);
}

function paintGdaLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const gda = frame.gda;
  if (!gda) {
    return;
  }

  const layout = gdaLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Generative profile topography", layout.plotPane.x, layout.plotPane.y - 24);
  context.fillText("Gaussian discriminant state", layout.infoPane.x, layout.infoPane.y - 24);

  paintGdaPlot(context, gda, layout.plotPane);
  paintGdaInfoPanel(context, gda, layout.infoPane);
}

function gdaLayout(size: Size) {
  const narrow = size.width < 820;
  const padding = 24;
  const gap = 18;
  const plotPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(390, size.height * 0.56) }
    : { x: padding, y: 58, width: Math.max(500, size.width * 0.62), height: size.height - 96 };
  const infoPane: CanvasPane = narrow
    ? {
        x: padding,
        y: plotPane.y + plotPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(330, size.height - plotPane.height - 120),
      }
    : {
        x: plotPane.x + plotPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - plotPane.width - gap,
        height: size.height - 96,
      };

  return { plotPane, infoPane };
}

function gdaPlotArea(pane: CanvasPane): CanvasPane {
  return {
    x: pane.x + 52,
    y: pane.y + 72,
    width: pane.width - 80,
    height: pane.height - 126,
  };
}

function gdaScales(gda: NonNullable<ConceptFrame["gda"]>, pane: CanvasPane) {
  const plot = gdaPlotArea(pane);
  return {
    plot,
    x: d3.scaleLinear().domain(gda.xDomain).range([plot.x, plot.x + plot.width]),
    y: d3.scaleLinear().domain(gda.yDomain).range([plot.y + plot.height, plot.y]),
  };
}

function paintGdaPlot(
  context: CanvasRenderingContext2D,
  gda: NonNullable<ConceptFrame["gda"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${gda.mode.toUpperCase()} likelihood contours`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `Click inside the plot to add a ${gda.activeClass} point · training size ${gda.trainingSize} · accuracy ${metricPercent(gda.accuracy)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const { plot, x, y } = gdaScales(gda, pane);
  context.fillStyle = "#f7fafc";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#d8e0e5";
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  paintGdaDecisionBackdrop(context, gda, plot, x, y);
  paintGdaGrid(context, plot, x, y);

  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  gda.profiles.forEach((profile) => paintGdaContours(context, gda, profile, x, y));
  gda.profiles.forEach((profile) => {
    context.strokeStyle = profile.color;
    context.lineWidth = 2.4;
    context.fillStyle = "#ffffff";
    context.beginPath();
    context.arc(x(profile.mean.x), y(profile.mean.y), 6, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.beginPath();
    context.moveTo(x(profile.mean.x) - 8, y(profile.mean.y));
    context.lineTo(x(profile.mean.x) + 8, y(profile.mean.y));
    context.moveTo(x(profile.mean.x), y(profile.mean.y) - 8);
    context.lineTo(x(profile.mean.x), y(profile.mean.y) + 8);
    context.stroke();
  });

  gda.points.forEach((point) => {
    const classKey = point.label === "blue" ? "blue" : "red";
    const color = classKey === "blue" ? "#2f6fbe" : "#d34a43";
    context.fillStyle = color;
    context.strokeStyle = point.label === gda.activeClass ? "#17212b" : "#ffffff";
    context.lineWidth = point.label === gda.activeClass ? 2.2 : 1.4;
    context.beginPath();
    context.arc(x(point.x), y(point.y), point.label === gda.activeClass ? 5.6 : 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
  context.restore();

  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  x.ticks(5).forEach((tick) => {
    context.fillText(tick.toFixed(1), x(tick), plot.y + plot.height + 18);
  });
  context.textAlign = "right";
  y.ticks(5).forEach((tick) => {
    context.fillText(tick.toFixed(1), plot.x - 8, y(tick) + 4);
  });

  const statusBox = {
    x: pane.x + 14,
    y: pane.y + pane.height - 42,
    width: pane.width - 28,
    height: 26,
  };
  paintRoundedRect(context, statusBox.x, statusBox.y, statusBox.width, statusBox.height, 8, "#f7fafc", "#d8e0e5");
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(context, "Low training-size constraints inflate and fade the Gaussian profiles; higher values sharpen the rings.", statusBox.width - 16),
    statusBox.x + 8,
    statusBox.y + 17,
  );

  context.restore();
}

function paintGdaDecisionBackdrop(
  context: CanvasRenderingContext2D,
  gda: NonNullable<ConceptFrame["gda"]>,
  plot: CanvasPane,
  x: (value: number) => number,
  y: (value: number) => number,
) {
  const columns = 38;
  const rows = 28;
  const cellWidth = plot.width / columns;
  const cellHeight = plot.height / rows;
  gda.decisionGrid.forEach((cell, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const color = cell.predicted === "blue" ? "47, 111, 190" : "211, 74, 67";
    context.fillStyle = `rgba(${color}, ${0.04 + cell.margin * 0.12})`;
    context.fillRect(plot.x + column * cellWidth, plot.y + row * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
    if (cell.margin < 0.045) {
      context.fillStyle = "rgba(23, 33, 43, 0.28)";
      context.fillRect(x(cell.x) - 1, y(cell.y) - 1, 2, 2);
    }
  });
}

function paintGdaGrid(
  context: CanvasRenderingContext2D,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  context.strokeStyle = "rgba(100, 116, 139, 0.18)";
  context.lineWidth = 1;
  x.ticks(7).forEach((tick) => {
    const tx = x(tick);
    context.beginPath();
    context.moveTo(tx, plot.y);
    context.lineTo(tx, plot.y + plot.height);
    context.stroke();
  });
  y.ticks(6).forEach((tick) => {
    const ty = y(tick);
    context.beginPath();
    context.moveTo(plot.x, ty);
    context.lineTo(plot.x + plot.width, ty);
    context.stroke();
  });
}

function paintGdaContours(
  context: CanvasRenderingContext2D,
  gda: NonNullable<ConceptFrame["gda"]>,
  profile: NonNullable<ConceptFrame["gda"]>["profiles"][number],
  x: (value: number) => number,
  y: (value: number) => number,
) {
  context.save();
  context.shadowBlur = 3 + gda.uncertainty * 14;
  context.shadowColor = profile.color;
  profile.contours.forEach((contour, index) => {
    context.globalAlpha = (0.72 - index * 0.12) * (0.42 + gda.confidence * 0.58);
    context.strokeStyle = profile.color;
    context.lineWidth = 2.6 - index * 0.25;
    context.beginPath();
    contour.forEach((point, pointIndex) => {
      const px = x(point.x);
      const py = y(point.y);
      if (pointIndex === 0) {
        context.moveTo(px, py);
      } else {
        context.lineTo(px, py);
      }
    });
    context.closePath();
    context.stroke();
  });
  context.restore();
}

function paintGdaInfoPanel(
  context: CanvasRenderingContext2D,
  gda: NonNullable<ConceptFrame["gda"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Class likelihood profiles", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      gda.mode === "lda"
        ? "LDA forces both classes to share one covariance shape."
        : "QDA lets each class keep its own covariance shape.",
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const chipTop = pane.y + 68;
  const chipGap = 8;
  const chipWidth = (pane.width - 28 - chipGap) / 2;
  paintGdaStatChip(context, pane.x + 14, chipTop, chipWidth, "mode", gda.mode.toUpperCase(), "#6f58c9");
  paintGdaStatChip(context, pane.x + 14 + chipWidth + chipGap, chipTop, chipWidth, "confidence", `${Math.round(gda.confidence * 100)}%`, "#0f766e");

  const meterTop = chipTop + 74;
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("Uncertainty blur", pane.x + 14, meterTop);
  context.fillStyle = "#eef3f6";
  context.fillRect(pane.x + 14, meterTop + 16, pane.width - 28, 12);
  context.fillStyle = "#b7791f";
  context.fillRect(pane.x + 14, meterTop + 16, (pane.width - 28) * gda.uncertainty, 12);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(
    `${gda.trainingSize} effective examples · ${gda.uncertainty > 0.5 ? "wide, translucent profiles" : "tight, vivid profiles"}`,
    pane.x + 14,
    meterTop + 46,
  );

  const profileTop = meterTop + 78;
  gda.profiles.forEach((profile, index) => {
    const top = profileTop + index * 142;
    if (top + 124 > pane.y + pane.height - 18) {
      return;
    }
    paintGdaProfileCard(context, profile, pane.x + 14, top, pane.width - 28);
  });

  context.restore();
}

function paintGdaStatChip(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  label: string,
  value: string,
  color: string,
) {
  paintRoundedRect(context, x, y, width, 54, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = color;
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(label, x + 10, y + 18);
  context.fillStyle = "#17212b";
  context.font = "900 15px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillText(fitCanvasText(context, value, width - 20), x + 10, y + 39);
}

function paintGdaProfileCard(
  context: CanvasRenderingContext2D,
  profile: NonNullable<ConceptFrame["gda"]>["profiles"][number],
  x: number,
  y: number,
  width: number,
) {
  paintRoundedRect(context, x, y, width, 124, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = profile.color;
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${profile.label} profile`, x + 10, y + 18);
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(`n=${profile.count} · prior=${probabilityLabel(profile.prior)} · det=${profile.determinant.toFixed(3)}`, x + 10, y + 38);
  context.fillText(`mean=(${profile.mean.x.toFixed(2)}, ${profile.mean.y.toFixed(2)})`, x + 10, y + 57);

  const matrixTop = y + 72;
  const cellWidth = (width - 28) / 2;
  const values = [
    profile.covariance[0][0],
    profile.covariance[0][1],
    profile.covariance[1][0],
    profile.covariance[1][1],
  ];
  values.forEach((value, index) => {
    const column = index % 2;
    const row = Math.floor(index / 2);
    const cellX = x + 10 + column * cellWidth;
    const cellY = matrixTop + row * 20;
    context.fillStyle = index % 3 === 0 ? "rgba(15, 118, 110, 0.08)" : "rgba(183, 121, 31, 0.08)";
    context.fillRect(cellX, cellY, cellWidth - 6, 16);
    context.fillStyle = "#17212b";
    context.font = "900 9px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillText(value.toFixed(3), cellX + 5, cellY + 12);
  });
}

function metricPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function canvasPointToGdaPoint(
  gda: NonNullable<ConceptFrame["gda"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const { plotPane } = gdaLayout(size);
  const { plot, x, y } = gdaScales(gda, plotPane);
  if (
    point.x < plot.x ||
    point.x > plot.x + plot.width ||
    point.y < plot.y ||
    point.y > plot.y + plot.height
  ) {
    return null;
  }

  return {
    x: Number(x.invert(point.x).toFixed(3)),
    y: Number(y.invert(point.y).toFixed(3)),
  };
}

function paintBayesianRegressionLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const bayesian = frame.bayesianRegression;
  if (!bayesian) {
    return;
  }

  const layout = bayesianRegressionLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Interactive uncertainty sandbox", layout.plotPane.x, layout.plotPane.y - 24);
  context.fillText("Posterior state", layout.infoPane.x, layout.infoPane.y - 24);

  paintBayesianRegressionPlot(context, bayesian, layout.plotPane);
  paintBayesianPosteriorPanel(context, bayesian, layout.infoPane);
}

function bayesianRegressionLayout(size: Size) {
  const narrow = size.width < 820;
  const padding = 24;
  const gap = 18;
  const plotPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(350, size.height * 0.53) }
    : { x: padding, y: 58, width: Math.max(500, size.width * 0.62), height: size.height - 96 };
  const infoPane: CanvasPane = narrow
    ? {
        x: padding,
        y: plotPane.y + plotPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(280, size.height - plotPane.height - 120),
      }
    : {
        x: plotPane.x + plotPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - plotPane.width - gap,
        height: size.height - 96,
      };

  return { plotPane, infoPane };
}

function bayesianRegressionPlotArea(pane: CanvasPane): CanvasPane {
  return {
    x: pane.x + 54,
    y: pane.y + 72,
    width: pane.width - 82,
    height: pane.height - 128,
  };
}

function bayesianRegressionScales(
  bayesian: NonNullable<ConceptFrame["bayesianRegression"]>,
  pane: CanvasPane,
) {
  const plot = bayesianRegressionPlotArea(pane);

  return {
    plot,
    x: d3.scaleLinear().domain(bayesian.xDomain).range([plot.x, plot.x + plot.width]),
    y: d3.scaleLinear().domain(bayesian.yDomain).range([plot.y + plot.height, plot.y]),
  };
}

function paintBayesianRegressionPlot(
  context: CanvasRenderingContext2D,
  bayesian: NonNullable<ConceptFrame["bayesianRegression"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Posterior mean plus uncertainty band", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `${bayesian.points.length} evidence points · slope ${formatPolynomialNumber(bayesian.meanWeights[1])} · avg std ${bayesian.averageUncertainty.toFixed(2)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const { plot, x, y } = bayesianRegressionScales(bayesian, pane);
  context.fillStyle = "rgba(247, 250, 252, 0.95)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#d8e0e5";
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.strokeStyle = "rgba(100, 116, 139, 0.18)";
  context.lineWidth = 1;
  x.ticks(7).forEach((tick) => {
    const tx = x(tick);
    context.beginPath();
    context.moveTo(tx, plot.y);
    context.lineTo(tx, plot.y + plot.height);
    context.stroke();
  });
  y.ticks(6).forEach((tick) => {
    const ty = y(tick);
    context.beginPath();
    context.moveTo(plot.x, ty);
    context.lineTo(plot.x + plot.width, ty);
    context.stroke();
  });

  context.strokeStyle = "#c9d4da";
  context.beginPath();
  context.moveTo(plot.x, y(0));
  context.lineTo(plot.x + plot.width, y(0));
  context.moveTo(x(0), plot.y);
  context.lineTo(x(0), plot.y + plot.height);
  context.stroke();

  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();

  paintBayesianBand(context, bayesian, x, y);

  bayesian.sampleLines.forEach((line, index) => {
    context.strokeStyle = line.color;
    context.globalAlpha = 0.18 + (index % 4) * 0.035;
    context.lineWidth = 1.2;
    drawBayesianPolyline(context, line.points, x, y);
  });
  context.globalAlpha = 1;

  context.strokeStyle = "#0f766e";
  context.lineWidth = 3;
  drawBayesianPolyline(context, bayesian.meanLine, x, y);

  bayesian.points.forEach((point) => {
    const px = x(point.x);
    const py = y(point.y);
    context.fillStyle = point.label === "custom" ? "#fff7ed" : "#2f6fbe";
    context.strokeStyle = point.label === "custom" ? "#b7791f" : "#ffffff";
    context.lineWidth = point.label === "custom" ? 2.4 : 1.6;
    context.beginPath();
    context.arc(px, py, point.label === "custom" ? 6 : 5, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });
  context.restore();

  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  x.ticks(5).forEach((tick) => {
    context.fillText(tick.toFixed(1), x(tick), plot.y + plot.height + 18);
  });
  context.textAlign = "right";
  y.ticks(5).forEach((tick) => {
    context.fillText(tick.toFixed(1), plot.x - 8, y(tick) + 4);
  });

  const statusBox = {
    x: pane.x + 14,
    y: pane.y + pane.height - 42,
    width: pane.width - 28,
    height: 26,
  };
  paintRoundedRect(context, statusBox.x, statusBox.y, statusBox.width, statusBox.height, 8, "#f7fafc", "#d8e0e5");
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(context, "Click a wide shaded region to add evidence and watch uncertainty shrink around it.", statusBox.width - 16),
    statusBox.x + 8,
    statusBox.y + 17,
  );

  context.restore();
}

function paintBayesianBand(
  context: CanvasRenderingContext2D,
  bayesian: NonNullable<ConceptFrame["bayesianRegression"]>,
  x: (value: number) => number,
  y: (value: number) => number,
) {
  if (bayesian.upperBand.length === 0 || bayesian.lowerBand.length === 0) {
    return;
  }

  context.beginPath();
  bayesian.upperBand.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    if (index === 0) {
      context.moveTo(px, py);
    } else {
      context.lineTo(px, py);
    }
  });
  [...bayesian.lowerBand].reverse().forEach((point) => {
    context.lineTo(x(point.x), y(point.y));
  });
  context.closePath();
  context.fillStyle = "rgba(47, 111, 190, 0.16)";
  context.fill();

  context.strokeStyle = "rgba(47, 111, 190, 0.45)";
  context.lineWidth = 1.4;
  drawBayesianPolyline(context, bayesian.upperBand, x, y);
  drawBayesianPolyline(context, bayesian.lowerBand, x, y);
}

function drawBayesianPolyline(
  context: CanvasRenderingContext2D,
  points: DataPoint[],
  x: (value: number) => number,
  y: (value: number) => number,
) {
  context.beginPath();
  points.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      return;
    }
    if (index === 0) {
      context.moveTo(px, py);
    } else {
      context.lineTo(px, py);
    }
  });
  context.stroke();
}

function paintBayesianPosteriorPanel(
  context: CanvasRenderingContext2D,
  bayesian: NonNullable<ConceptFrame["bayesianRegression"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Distribution over plausible lines", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      bayesian.sampleLines.length > 0
        ? `${bayesian.sampleLines.length} posterior draws are overlaid on the plot.`
        : "Press the sample button to draw 20 possible lines from the posterior.",
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const chipTop = pane.y + 72;
  const chipGap = 10;
  const chipWidth = (pane.width - 28 - chipGap) / 2;
  paintBayesianStatChip(context, pane.x + 14, chipTop, chipWidth, "intercept", formatPolynomialNumber(bayesian.meanWeights[0]), "#2f6fbe");
  paintBayesianStatChip(context, pane.x + 14 + chipWidth + chipGap, chipTop, chipWidth, "slope", formatPolynomialNumber(bayesian.meanWeights[1]), "#0f766e");

  const matrixTop = chipTop + 76;
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("Posterior covariance S_N", pane.x + 14, matrixTop);
  const cellGap = 8;
  const cellWidth = (pane.width - 28 - cellGap) / 2;
  const covarianceValues = [
    bayesian.covariance[0][0],
    bayesian.covariance[0][1],
    bayesian.covariance[1][0],
    bayesian.covariance[1][1],
  ];
  covarianceValues.forEach((value, index) => {
    const column = index % 2;
    const row = Math.floor(index / 2);
    const xPos = pane.x + 14 + column * (cellWidth + cellGap);
    const yPos = matrixTop + 16 + row * 42;
    paintRoundedRect(context, xPos, yPos, cellWidth, 34, 8, "#ffffff", "#d8e0e5");
    context.fillStyle = "#17212b";
    context.font = "900 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillText(formatPolynomialNumber(value), xPos + 9, yPos + 21);
  });

  const meterTop = matrixTop + 116;
  const meterWidth = pane.width - 28;
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("Average predictive uncertainty", pane.x + 14, meterTop);
  context.fillStyle = "#eef3f6";
  context.fillRect(pane.x + 14, meterTop + 16, meterWidth, 12);
  context.fillStyle = "#2f6fbe";
  context.fillRect(
    pane.x + 14,
    meterTop + 16,
    Math.min(1, bayesian.averageUncertainty / 3) * meterWidth,
    12,
  );
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(
    `std ${bayesian.averageUncertainty.toFixed(2)} · noise ${Math.sqrt(bayesian.noiseVariance).toFixed(2)} · prior ${Math.sqrt(bayesian.priorVariance).toFixed(2)}`,
    pane.x + 14,
    meterTop + 46,
  );

  const notesTop = meterTop + 78;
  const notes = [
    "Few points leave many lines plausible, so the blue band fans outward.",
    "A new point updates both the mean line and the covariance matrix.",
    "The band grows far from evidence because extrapolation is less certain.",
  ];
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.fillText("What to watch", pane.x + 14, notesTop);
  notes.forEach((note, index) => {
    const yPos = notesTop + 22 + index * 46;
    if (yPos + 28 > pane.y + pane.height - 18) {
      return;
    }
    context.fillStyle = "#f7fafc";
    paintRoundedRect(context, pane.x + 14, yPos, pane.width - 28, 34, 8, "#f7fafc", "#d8e0e5");
    context.fillStyle = "#61707f";
    context.font = "800 9px Inter, system-ui, sans-serif";
    wrapCanvasText(context, note, pane.x + 24, yPos + 14, pane.width - 48, 12, 2);
  });

  context.restore();
}

function paintBayesianStatChip(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  label: string,
  value: string,
  color: string,
) {
  paintRoundedRect(context, x, y, width, 54, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = color;
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(label, x + 10, y + 18);
  context.fillStyle = "#17212b";
  context.font = "900 16px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillText(fitCanvasText(context, value, width - 20), x + 10, y + 39);
}

function canvasPointToBayesianPoint(
  bayesian: NonNullable<ConceptFrame["bayesianRegression"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const { plotPane } = bayesianRegressionLayout(size);
  const { plot, x, y } = bayesianRegressionScales(bayesian, plotPane);
  if (
    point.x < plot.x ||
    point.x > plot.x + plot.width ||
    point.y < plot.y ||
    point.y > plot.y + plot.height
  ) {
    return null;
  }

  return {
    x: Number(x.invert(point.x).toFixed(3)),
    y: Number(y.invert(point.y).toFixed(3)),
  };
}

function paintDeterminantLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const determinant = frame.determinant;
  if (!determinant) {
    return;
  }

  const layout = determinantLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Area and grid transformation", layout.plotPane.x, layout.plotPane.y - 24);
  context.fillText("Eigenvalue connection", layout.infoPane.x, layout.infoPane.y - 24);

  paintDeterminantPlot(context, determinant, layout.plotPane);
  paintDeterminantInfo(context, determinant, layout.infoPane);
}

function determinantLayout(size: Size) {
  const narrow = size.width < 820;
  const padding = 24;
  const gap = 18;
  const plotPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(350, size.height * 0.54) }
    : { x: padding, y: 58, width: Math.max(470, size.width * 0.62), height: size.height - 96 };
  const infoPane: CanvasPane = narrow
    ? {
        x: padding,
        y: plotPane.y + plotPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(280, size.height - plotPane.height - 120),
      }
    : {
        x: plotPane.x + plotPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - plotPane.width - gap,
        height: size.height - 96,
      };

  return { plotPane, infoPane };
}

function determinantPlotArea(pane: CanvasPane): CanvasPane {
  return {
    x: pane.x + 44,
    y: pane.y + 64,
    width: pane.width - 74,
    height: pane.height - 112,
  };
}

function determinantPlotScales(
  determinant: NonNullable<ConceptFrame["determinant"]>,
  pane: CanvasPane,
) {
  const plot = determinantPlotArea(pane);
  const coordinates = [
    ...determinant.transformedSquare,
    determinant.basisI,
    determinant.basisJ,
    { x: 1, y: 1 },
    { x: -1, y: -1 },
  ];
  const limit = Math.max(
    3.2,
    ...coordinates.flatMap((point) => [Math.abs(point.x), Math.abs(point.y)]),
  ) + 0.35;
  return {
    plot,
    limit,
    x: d3.scaleLinear().domain([-limit, limit]).range([plot.x, plot.x + plot.width]),
    y: d3.scaleLinear().domain([-limit, limit]).range([plot.y + plot.height, plot.y]),
  };
}

function paintDeterminantPlot(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  const signColor = determinantColor(determinant.orientation);
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`det(A) = ${formatDeterminantNumber(determinant.determinant)}`, pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `Area scale ${determinant.areaScale.toFixed(2)}x · ${determinant.orientation === "negative" ? "orientation flips" : determinant.orientation === "zero" ? "collapsed onto a line" : "orientation preserved"}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const { plot, x, y } = determinantPlotScales(determinant, pane);
  context.fillStyle = "rgba(247, 250, 252, 0.92)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#d8e0e5";
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  paintDeterminantOriginalGrid(context, determinant, plot, x, y);
  paintDeterminantTransformedGrid(context, determinant, plot, x, y);
  paintDeterminantAxes(context, plot, x, y);
  paintDeterminantSquares(context, determinant, x, y, signColor);
  paintDeterminantEigenvectors(context, determinant, plot, x, y);
  paintDeterminantBasisArrows(context, determinant, x, y);

  const chip = {
    x: plot.x + 12,
    y: plot.y + 12,
    width: Math.min(230, plot.width - 24),
    height: 40,
  };
  paintRoundedRect(context, chip.x, chip.y, chip.width, chip.height, 8, "#ffffff", signColor);
  context.fillStyle = signColor;
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`Area Scale = ${determinant.areaScale.toFixed(2)}x`, chip.x + 12, chip.y + 17);
  context.fillStyle = "#61707f";
  context.font = "800 9px Inter, system-ui, sans-serif";
  context.fillText(`${determinant.orientation.toUpperCase()} determinant`, chip.x + 12, chip.y + 31);

  context.restore();
}

function paintDeterminantOriginalGrid(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  const ticks = d3.range(-4, 4.1, 1);
  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  context.strokeStyle = "rgba(100, 116, 139, 0.12)";
  context.lineWidth = 1;
  ticks.forEach((tick) => {
    context.beginPath();
    context.moveTo(x(tick), plot.y);
    context.lineTo(x(tick), plot.y + plot.height);
    context.moveTo(plot.x, y(tick));
    context.lineTo(plot.x + plot.width, y(tick));
    context.stroke();
  });

  context.setLineDash([5, 4]);
  context.strokeStyle = "rgba(100, 116, 139, 0.45)";
  context.lineWidth = 1.4;
  drawDeterminantPolygon(context, determinant.unitSquare, x, y, false);
  context.setLineDash([]);
  context.restore();
}

function paintDeterminantTransformedGrid(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  const [a, b] = determinant.matrix[0];
  const [c, d] = determinant.matrix[1];
  const ticks = d3.range(-5, 5.1, 1);
  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  context.strokeStyle = determinant.orientation === "negative" ? "rgba(211, 74, 67, 0.22)" : "rgba(47, 111, 190, 0.24)";
  context.lineWidth = 1.15;
  ticks.forEach((tick) => {
    const verticalA = determinantTransformForCanvas(a, b, c, d, tick, -5);
    const verticalB = determinantTransformForCanvas(a, b, c, d, tick, 5);
    const horizontalA = determinantTransformForCanvas(a, b, c, d, -5, tick);
    const horizontalB = determinantTransformForCanvas(a, b, c, d, 5, tick);
    context.beginPath();
    context.moveTo(x(verticalA.x), y(verticalA.y));
    context.lineTo(x(verticalB.x), y(verticalB.y));
    context.moveTo(x(horizontalA.x), y(horizontalA.y));
    context.lineTo(x(horizontalB.x), y(horizontalB.y));
    context.stroke();
  });
  context.restore();
}

function paintDeterminantAxes(
  context: CanvasRenderingContext2D,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  context.strokeStyle = "#aebbc4";
  context.lineWidth = 1.3;
  context.beginPath();
  context.moveTo(plot.x, y(0));
  context.lineTo(plot.x + plot.width, y(0));
  context.moveTo(x(0), plot.y);
  context.lineTo(x(0), plot.y + plot.height);
  context.stroke();
}

function paintDeterminantSquares(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
  signColor: string,
) {
  context.save();
  context.fillStyle = `${signColor}2b`;
  context.strokeStyle = signColor;
  context.lineWidth = 2.6;
  drawDeterminantPolygon(context, determinant.transformedSquare, x, y, true);
  context.restore();
}

function paintDeterminantEigenvectors(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  determinant.eigenvalues.forEach((eigenvalue) => {
    if (!eigenvalue.vector || Math.abs(eigenvalue.imaginary) > 1e-7) {
      return;
    }
    const span = 5;
    context.strokeStyle = eigenvalue.color;
    context.lineWidth = 2;
    context.setLineDash([7, 5]);
    context.beginPath();
    context.moveTo(x(-eigenvalue.vector.x * span), y(-eigenvalue.vector.y * span));
    context.lineTo(x(eigenvalue.vector.x * span), y(eigenvalue.vector.y * span));
    context.stroke();
    context.setLineDash([]);
  });
  context.restore();
}

function paintDeterminantBasisArrows(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  paintDeterminantArrow(context, x(0), y(0), x(determinant.basisI.x), y(determinant.basisI.y), "#2f6fbe", "col 1");
  paintDeterminantArrow(context, x(0), y(0), x(determinant.basisJ.x), y(determinant.basisJ.y), "#0f766e", "col 2");
}

function paintDeterminantArrow(
  context: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  color: string,
  label: string,
) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(fromX, fromY);
  context.lineTo(toX, toY);
  context.stroke();
  context.translate(toX, toY);
  context.rotate(angle);
  context.beginPath();
  context.moveTo(0, 0);
  context.lineTo(-11, -5);
  context.lineTo(-11, 5);
  context.closePath();
  context.fill();
  context.rotate(-angle);
  context.fillStyle = "#ffffff";
  context.strokeStyle = color;
  context.lineWidth = 2;
  context.beginPath();
  context.arc(0, 0, 8, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = color;
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(label, 10, -10);
  context.restore();
}

function paintDeterminantInfo(
  context: CanvasRenderingContext2D,
  determinant: NonNullable<ConceptFrame["determinant"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);
  const signColor = determinantColor(determinant.orientation);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Matrix A", pane.x + 14, pane.y + 22);

  const matrixBox = {
    x: pane.x + 14,
    y: pane.y + 42,
    width: pane.width - 28,
    height: 104,
  };
  paintRoundedRect(context, matrixBox.x, matrixBox.y, matrixBox.width, matrixBox.height, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = "#17212b";
  context.font = "900 22px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  const cellWidth = matrixBox.width / 2;
  const rows = determinant.matrix;
  rows.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      context.fillText(
        formatDeterminantNumber(value),
        matrixBox.x + cellWidth * columnIndex + cellWidth / 2,
        matrixBox.y + 42 + rowIndex * 36,
      );
    });
  });
  context.fillStyle = "#61707f";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.fillText("[ a  b ]", matrixBox.x + matrixBox.width / 2, matrixBox.y + matrixBox.height - 10);

  const formulaY = matrixBox.y + matrixBox.height + 28;
  paintRoundedRect(context, pane.x + 14, formulaY, pane.width - 28, 78, 8, determinant.orientation === "negative" ? "#fff7f6" : "#f7fafc", `${signColor}66`);
  context.fillStyle = signColor;
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Signed area", pane.x + 28, formulaY + 22);
  context.fillStyle = "#17212b";
  context.font = "800 12px ui-monospace, SFMono-Regular, Menlo, monospace";
  const [[a, b], [c, d]] = determinant.matrix;
  context.fillText(
    `ad - bc = (${formatDeterminantNumber(a)})( ${formatDeterminantNumber(d)} ) - (${formatDeterminantNumber(b)})( ${formatDeterminantNumber(c)} )`,
    pane.x + 28,
    formulaY + 46,
  );
  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.fillText(`det(A) = ${formatDeterminantNumber(determinant.determinant)}`, pane.x + 28, formulaY + 64);

  const eigenY = formulaY + 104;
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.fillText("Eigenvalues", pane.x + 14, eigenY);

  determinant.eigenvalues.forEach((eigenvalue, index) => {
    const rowY = eigenY + 18 + index * 52;
    paintRoundedRect(context, pane.x + 14, rowY, pane.width - 28, 40, 8, "#ffffff", eigenvalue.color);
    context.fillStyle = eigenvalue.color;
    context.font = "900 11px Inter, system-ui, sans-serif";
    context.fillText(eigenvalue.label, pane.x + 28, rowY + 16);
    context.fillStyle = "#17212b";
    context.font = "800 13px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillText(formatEigenvalue(eigenvalue.real, eigenvalue.imaginary), pane.x + 28, rowY + 32);
  });

  const productY = eigenY + 132;
  paintRoundedRect(context, pane.x + 14, productY, pane.width - 28, 78, 8, "#f7fafc", "#d8e0e5");
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.fillText("Product connection", pane.x + 28, productY + 22);
  context.fillStyle = "#61707f";
  context.font = "800 11px Inter, system-ui, sans-serif";
  wrapCanvasText(
    context,
    determinant.eigenvalues.some((value) => Math.abs(value.imaginary) > 1e-7)
      ? "Complex conjugate eigenvalues have no real direction line, but their product still equals the determinant."
      : "Real eigenvector lines are directions that the transformation only stretches or flips.",
    pane.x + 28,
    productY + 42,
    pane.width - 56,
    14,
    2,
  );
  context.fillStyle = signColor;
  context.font = "900 12px ui-monospace, SFMono-Regular, Menlo, monospace";
  const lambdaProduct = determinantEigenvalueProduct(determinant.eigenvalues);
  context.fillText(
    `det = λ1 λ2 = ${formatDeterminantNumber(lambdaProduct)}`,
    pane.x + 28,
    productY + 68,
  );

  context.restore();
}

function drawDeterminantPolygon(
  context: CanvasRenderingContext2D,
  points: Array<{ x: number; y: number }>,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
  fill: boolean,
) {
  context.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(x(point.x), y(point.y));
    } else {
      context.lineTo(x(point.x), y(point.y));
    }
  });
  context.closePath();
  if (fill) {
    context.fill();
  }
  context.stroke();
}

function determinantTransformForCanvas(
  a: number,
  b: number,
  c: number,
  d: number,
  x: number,
  y: number,
) {
  return {
    x: a * x + b * y,
    y: c * x + d * y,
  };
}

function determinantColor(orientation: NonNullable<ConceptFrame["determinant"]>["orientation"]) {
  if (orientation === "negative") {
    return "#d34a43";
  }
  if (orientation === "zero") {
    return "#b7791f";
  }
  return "#2f6fbe";
}

function formatDeterminantNumber(value: number) {
  if (Math.abs(value) < 0.001) {
    return "0.000";
  }
  return value.toFixed(Math.abs(value) >= 10 ? 1 : 3);
}

function formatEigenvalue(real: number, imaginary: number) {
  if (Math.abs(imaginary) < 1e-7) {
    return formatDeterminantNumber(real);
  }
  return `${formatDeterminantNumber(real)} ${imaginary >= 0 ? "+" : "-"} ${formatDeterminantNumber(Math.abs(imaginary))}i`;
}

function determinantEigenvalueProduct(
  eigenvalues: NonNullable<ConceptFrame["determinant"]>["eigenvalues"],
) {
  if (eigenvalues.length < 2) {
    return eigenvalues[0]?.real ?? 0;
  }
  if (eigenvalues.some((value) => Math.abs(value.imaginary) > 1e-7)) {
    const first = eigenvalues[0];
    return first.real ** 2 + first.imaginary ** 2;
  }
  return eigenvalues[0].real * eigenvalues[1].real;
}

function paintRegularizationLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const regularization = frame.regularization;
  if (!regularization) {
    return;
  }

  const layout = regularizationLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Regularization tug-of-war", layout.plotPane.x, layout.plotPane.y - 24);
  context.fillText("Weight elimination ledger", layout.ledgerPane.x, layout.ledgerPane.y - 24);

  paintRegularizationPlot(context, regularization, layout.plotPane);
  paintRegularizationLedger(context, regularization, layout.ledgerPane);
}

function regularizationLayout(size: Size) {
  const narrow = size.width < 840;
  const padding = 24;
  const gap = 18;
  const plotPane: CanvasPane = narrow
    ? { x: padding, y: 56, width: size.width - padding * 2, height: Math.max(360, size.height * 0.52) }
    : { x: padding, y: 58, width: Math.max(500, size.width * 0.62), height: size.height - 96 };
  const ledgerPane: CanvasPane = narrow
    ? {
        x: padding,
        y: plotPane.y + plotPane.height + gap,
        width: size.width - padding * 2,
        height: Math.max(300, size.height - plotPane.height - 120),
      }
    : {
        x: plotPane.x + plotPane.width + gap,
        y: 58,
        width: size.width - padding * 2 - plotPane.width - gap,
        height: size.height - 96,
      };

  return { plotPane, ledgerPane };
}

function regularizationPlotArea(pane: CanvasPane): CanvasPane {
  const legendY = pane.y + 56;
  const legendHeight = regularizationLegendHeight(pane);
  const y = legendY + legendHeight + 14;
  const bottomReserve = 66;

  return {
    x: pane.x + 52,
    y,
    width: pane.width - 78,
    height: Math.max(150, pane.y + pane.height - bottomReserve - y),
  };
}

function regularizationLegendHeight(pane: CanvasPane) {
  return pane.width < 560 ? 54 : 30;
}

function regularizationScales(
  regularization: NonNullable<ConceptFrame["regularization"]>,
  pane: CanvasPane,
) {
  const plot = regularizationPlotArea(pane);
  return {
    plot,
    x: d3.scaleLinear().domain(regularization.xDomain).range([plot.x, plot.x + plot.width]),
    y: d3.scaleLinear().domain(regularization.yDomain).range([plot.y + plot.height, plot.y]),
  };
}

function paintRegularizationPlot(
  context: CanvasRenderingContext2D,
  regularization: NonNullable<ConceptFrame["regularization"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  const methodColor = regularization.method === "lasso" ? "#6f58c9" : "#0f766e";
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    `${regularization.method === "lasso" ? "Lasso" : "Ridge"} on a degree-${regularization.degree} polynomial`,
    pane.x + 14,
    pane.y + 22,
  );
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `alpha ${regularization.alpha.toFixed(2)} · noise ${regularization.noise.toFixed(2)} · MSE ${regularization.mse.toFixed(3)} vs alpha-0 ${regularization.unregularizedMse.toFixed(3)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const legendItems = [
    { label: "true signal", color: "#64748b", dashed: true },
    { label: "alpha 0 chases noise", color: "#d34a43", dashed: true },
    { label: regularization.method === "lasso" ? "lasso fit" : "ridge fit", color: methodColor },
  ];
  paintRegularizationLegend(context, legendItems, pane);

  const { plot, x, y } = regularizationScales(regularization, pane);
  context.fillStyle = "rgba(247, 250, 252, 0.92)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.strokeStyle = "rgba(100, 116, 139, 0.16)";
  x.ticks(7).forEach((tick) => {
    const tx = x(tick);
    context.beginPath();
    context.moveTo(tx, plot.y);
    context.lineTo(tx, plot.y + plot.height);
    context.stroke();
  });
  y.ticks(6).forEach((tick) => {
    const ty = y(tick);
    context.beginPath();
    context.moveTo(plot.x, ty);
    context.lineTo(plot.x + plot.width, ty);
    context.stroke();
  });

  context.strokeStyle = "#c9d4da";
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(plot.x, y(0));
  context.lineTo(plot.x + plot.width, y(0));
  context.moveTo(x(0), plot.y);
  context.lineTo(x(0), plot.y + plot.height);
  context.stroke();

  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  drawRegularizationCurve(context, regularization.signalCurve, x, y, "#64748b", 2.2, [6, 6]);
  drawRegularizationCurve(context, regularization.unregularizedCurve, x, y, "#d34a43", 2.2, [3, 5]);
  drawRegularizationCurve(context, regularization.regularizedCurve, x, y, methodColor, 3.2);
  context.restore();

  regularization.points.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    context.fillStyle = colors[index % colors.length];
    context.globalAlpha = 0.86;
    context.beginPath();
    context.arc(px, py, 4.8, 0, Math.PI * 2);
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 1.3;
    context.stroke();
  });

  const note = regularization.method === "lasso"
    ? "Lasso pays for absolute weight size, so some polynomial columns drop exactly to zero."
    : "Ridge pays for squared weight size, so the whole high-degree curve smooths out together.";
  paintRoundedRect(context, pane.x + 14, pane.y + pane.height - 44, pane.width - 28, 28, 8, "#f7fafc", `${methodColor}55`);
  context.fillStyle = methodColor;
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(fitCanvasText(context, note, pane.width - 48), pane.x + 24, pane.y + pane.height - 26);
  context.restore();
}

function paintRegularizationLegend(
  context: CanvasRenderingContext2D,
  legendItems: Array<{ label: string; color: string; dashed?: boolean }>,
  pane: CanvasPane,
) {
  const x = pane.x + 14;
  const y = pane.y + 56;
  const width = pane.width - 28;
  const height = regularizationLegendHeight(pane);
  const compact = height > 34;
  const columns = compact ? 2 : 3;
  const gap = 8;
  const chipWidth = (width - gap * (columns - 1)) / columns;
  const chipHeight = 20;

  paintRoundedRect(context, x, y, width, height, 8, "#ffffff", "#d8e0e5");
  legendItems.forEach((item, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const chipX = x + column * (chipWidth + gap) + 8;
    const chipY = y + 7 + row * 22;
    context.strokeStyle = item.color;
    context.lineWidth = 2.4;
    if (item.dashed) {
      context.setLineDash([5, 4]);
    }
    context.beginPath();
    context.moveTo(chipX, chipY + chipHeight / 2);
    context.lineTo(chipX + 24, chipY + chipHeight / 2);
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = "#17212b";
    context.font = "800 10px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(fitCanvasText(context, item.label, chipWidth - 40), chipX + 32, chipY + 14);
  });
}

function drawRegularizationCurve(
  context: CanvasRenderingContext2D,
  points: DataPoint[],
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
  color: string,
  width: number,
  dash?: number[],
) {
  if (points.length < 2) {
    return;
  }

  context.save();
  context.strokeStyle = color;
  context.lineWidth = width;
  context.lineJoin = "round";
  context.lineCap = "round";
  if (dash) {
    context.setLineDash(dash);
  }
  context.beginPath();
  points.forEach((point, index) => {
    const px = x(point.x);
    const py = y(point.y);
    if (!Number.isFinite(px) || !Number.isFinite(py)) {
      return;
    }
    if (index === 0) {
      context.moveTo(px, py);
    } else {
      context.lineTo(px, py);
    }
  });
  context.stroke();
  context.restore();
}

function paintRegularizationLedger(
  context: CanvasRenderingContext2D,
  regularization: NonNullable<ConceptFrame["regularization"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  const methodColor = regularization.method === "lasso" ? "#6f58c9" : "#0f766e";
  const eliminatedCount = regularization.weights.filter((weight) => weight.eliminated).length;
  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Coefficient sizes beta", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      regularization.method === "lasso"
        ? `${eliminatedCount} weights eliminated · ${regularization.activeWeights} active columns`
        : `${regularization.activeWeights} active columns · weights shrink without hard deletion`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const chart = {
    x: pane.x + 38,
    y: pane.y + 84,
    width: pane.width - 68,
    height: Math.max(154, pane.height - 252),
  };
  paintRoundedRect(context, chart.x, chart.y, chart.width, chart.height, 8, "#f7fafc", "#d8e0e5");
  const maxAbs = Math.max(
    0.1,
    ...regularization.weights.flatMap((weight) => [Math.abs(weight.value), Math.abs(weight.unregularizedValue)]),
  );
  const yScale = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([chart.y + chart.height - 22, chart.y + 20]);
  const baseline = yScale(0);
  context.strokeStyle = "#aebbc4";
  context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(chart.x + 8, baseline);
  context.lineTo(chart.x + chart.width - 8, baseline);
  context.stroke();

  const step = chart.width / Math.max(1, regularization.weights.length);
  const barWidth = Math.min(22, Math.max(9, step * 0.42));
  regularization.weights.forEach((weight, index) => {
    const centerX = chart.x + step * index + step / 2;
    drawRegularizationWeightBar(context, centerX, barWidth + 8, baseline, yScale(weight.unregularizedValue), "rgba(100, 116, 139, 0.18)");
    drawRegularizationWeightBar(
      context,
      centerX,
      barWidth,
      baseline,
      yScale(weight.value),
      weight.eliminated ? "rgba(211, 74, 67, 0.18)" : weight.value >= 0 ? methodColor : "#d34a43",
    );

    if (weight.eliminated) {
      context.strokeStyle = "#d34a43";
      context.lineWidth = 1.8;
      context.beginPath();
      context.moveTo(centerX - barWidth / 2, baseline - 5);
      context.lineTo(centerX + barWidth / 2, baseline + 5);
      context.moveTo(centerX + barWidth / 2, baseline - 5);
      context.lineTo(centerX - barWidth / 2, baseline + 5);
      context.stroke();
    }

    context.fillStyle = weight.eliminated ? "#d34a43" : "#61707f";
    context.font = "800 9px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(weight.label, centerX, chart.y + chart.height - 7);
  });

  context.fillStyle = "#61707f";
  context.font = "800 9px Inter, system-ui, sans-serif";
  context.textAlign = "right";
  context.fillText(formatPolynomialNumber(maxAbs), chart.x - 8, chart.y + 24);
  context.fillText("0", chart.x - 8, baseline + 3);
  context.fillText(formatPolynomialNumber(-maxAbs), chart.x - 8, chart.y + chart.height - 24);

  const cardTop = chart.y + chart.height + 16;
  const cards = [
    {
      title: regularization.method === "lasso" ? "L1 penalty" : "L2 penalty",
      text: regularization.method === "lasso"
        ? "Sparse result: some high-power columns vanish."
        : "Smooth result: weights shrink together.",
    },
    {
      title: "Big-O lens",
      text: "Ridge solve: O(d^3 + n d^2). Lasso loops: O(t n d).",
    },
    {
      title: "Real use",
      text: "Stabilizes noisy tabular models, forecasting curves, and high-dimensional features.",
    },
  ];
  const visibleCards = cards.slice(0, pane.height > 560 ? 3 : 2);
  const gap = 8;
  const cardHeight = Math.max(48, Math.min(62, (pane.y + pane.height - cardTop - 16 - gap * (visibleCards.length - 1)) / visibleCards.length));
  visibleCards.forEach((card, index) => {
    const y = cardTop + index * (cardHeight + gap);
    if (y + cardHeight > pane.y + pane.height - 8) {
      return;
    }
    paintRoundedRect(context, pane.x + 14, y, pane.width - 28, cardHeight, 8, "#ffffff", index === 0 ? methodColor : "#d8e0e5");
    context.fillStyle = index === 0 ? methodColor : "#17212b";
    context.font = "900 10px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(card.title, pane.x + 26, y + 17);
    context.fillStyle = "#61707f";
    context.font = "700 10px Inter, system-ui, sans-serif";
    wrapCanvasText(context, card.text, pane.x + 26, y + 34, pane.width - 52, 13, 2);
  });

  context.restore();
}

function drawRegularizationWeightBar(
  context: CanvasRenderingContext2D,
  centerX: number,
  width: number,
  baseline: number,
  valueY: number,
  color: string,
) {
  const top = Math.min(baseline, valueY);
  const height = Math.max(2, Math.abs(valueY - baseline));
  context.fillStyle = color;
  context.fillRect(centerX - width / 2, top, width, height);
}

function hitTestDeterminantHandle(
  determinant: NonNullable<ConceptFrame["determinant"]>,
  size: Size,
  point: { x: number; y: number },
): "basisI" | "basisJ" | null {
  const { plotPane } = determinantLayout(size);
  const { x, y } = determinantPlotScales(determinant, plotPane);
  const handles = [
    { id: "basisI" as const, point: determinant.basisI },
    { id: "basisJ" as const, point: determinant.basisJ },
  ];
  const hit = handles.find((handle) => Math.hypot(point.x - x(handle.point.x), point.y - y(handle.point.y)) <= 16);
  return hit?.id ?? null;
}

function canvasPointToDeterminantVector(
  determinant: NonNullable<ConceptFrame["determinant"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const { plotPane } = determinantLayout(size);
  const { plot, x, y } = determinantPlotScales(determinant, plotPane);
  if (
    point.x < plot.x ||
    point.x > plot.x + plot.width ||
    point.y < plot.y ||
    point.y > plot.y + plot.height
  ) {
    return null;
  }

  return {
    x: Math.max(-3, Math.min(3, Number(x.invert(point.x).toFixed(2)))),
    y: Math.max(-3, Math.min(3, Number(y.invert(point.y).toFixed(2)))),
  };
}

function paintEigenDirectionLesson(
  context: CanvasRenderingContext2D,
  frame: ConceptFrame,
  size: Size,
) {
  const eigenDirection = frame.eigenDirection;
  if (!eigenDirection) {
    return;
  }

  const layout = eigenDirectionLayout(size);
  context.fillStyle = "#17212b";
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Find the magic directions", layout.headerPane.x, layout.headerPane.y - 24);
  context.fillText("Grid and transformation", layout.gridPane.x, layout.gridPane.y - 24);
  context.fillText("Direction wheel", layout.wheelPane.x, layout.wheelPane.y - 24);

  paintEigenDirectionHeader(context, eigenDirection, layout.headerPane);
  paintEigenDirectionGrid(context, eigenDirection, layout.gridPane);
  paintEigenDirectionWheel(context, eigenDirection, layout.wheelPane);
  paintEigenDirectionMetrics(context, eigenDirection, layout.metricsPane);
}

function eigenDirectionLayout(size: Size) {
  const narrow = size.width < 860;
  const padding = 24;
  const gap = 18;
  const headerPane: CanvasPane = { x: padding, y: 58, width: size.width - padding * 2, height: narrow ? 142 : 116 };
  const top = headerPane.y + headerPane.height + 56;
  const metricsHeight = narrow ? 154 : 98;
  const availableHeight = Math.max(300, size.height - top - metricsHeight - 42);
  const gridPane: CanvasPane = narrow
    ? { x: padding, y: top, width: size.width - padding * 2, height: Math.max(320, availableHeight * 0.54) }
    : { x: padding, y: top, width: Math.max(500, size.width * 0.58), height: availableHeight };
  const wheelPane: CanvasPane = narrow
    ? {
        x: padding,
        y: gridPane.y + gridPane.height + gap + 32,
        width: size.width - padding * 2,
        height: Math.max(280, availableHeight * 0.46),
      }
    : {
        x: gridPane.x + gridPane.width + gap,
        y: top,
        width: size.width - padding * 2 - gridPane.width - gap,
        height: availableHeight,
      };
  const metricsPane: CanvasPane = narrow
    ? {
        x: padding,
        y: wheelPane.y + wheelPane.height + gap,
        width: size.width - padding * 2,
        height: metricsHeight,
      }
    : {
        x: padding,
        y: gridPane.y + gridPane.height + gap,
        width: size.width - padding * 2,
        height: metricsHeight,
      };

  return { headerPane, gridPane, wheelPane, metricsPane };
}

function paintEigenDirectionHeader(
  context: CanvasRenderingContext2D,
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);

  const matrixBox = {
    x: pane.x + 14,
    y: pane.y + 18,
    width: Math.min(156, pane.width * 0.3),
    height: pane.height - 36,
  };
  paintRoundedRect(context, matrixBox.x, matrixBox.y, matrixBox.width, matrixBox.height, 8, "#ffffff", "#d8e0e5");
  context.fillStyle = "#17212b";
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Matrix editor", matrixBox.x + 12, matrixBox.y + 18);
  context.font = "900 17px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  eigenDirection.matrix.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      context.fillText(
        formatEigenDirectionNumber(value),
        matrixBox.x + matrixBox.width * (columnIndex === 0 ? 0.34 : 0.68),
        matrixBox.y + 48 + rowIndex * 26,
      );
    });
  });

  const activeIndex = eigenDirectionPhaseIndex(eigenDirection.phase);
  const flow = [
    "Matrix",
    "Apply Transformation",
    "Compare Directions",
    "Direction Preserved?",
    eigenDirection.isEigenvector ? "Yes: Eigenvector" : "No: Ordinary Vector",
  ];
  const flowX = matrixBox.x + matrixBox.width + 18;
  const flowY = pane.y + 24;
  const flowWidth = pane.x + pane.width - flowX - 14;
  const compact = flowWidth < 560;
  const nodeWidth = compact ? Math.max(104, (flowWidth - 20) / 3) : (flowWidth - 52) / flow.length;
  const nodeHeight = 36;

  flow.forEach((label, index) => {
    const row = compact ? Math.floor(index / 3) : 0;
    const column = compact ? index % 3 : index;
    const x = flowX + column * (nodeWidth + 10);
    const y = flowY + row * 48;
    const active = index === activeIndex;
    paintRoundedRect(
      context,
      x,
      y,
      nodeWidth,
      nodeHeight,
      8,
      active ? "#eef7f5" : "#ffffff",
      active ? "#0f766e" : "#d8e0e5",
    );
    context.fillStyle = active ? "#0f766e" : "#61707f";
    context.font = "900 10px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.fillText(fitCanvasText(context, label, nodeWidth - 12), x + nodeWidth / 2, y + 22);

    if (!compact && index < flow.length - 1) {
      context.strokeStyle = "#aebbc4";
      context.lineWidth = 1.6;
      context.beginPath();
      context.moveTo(x + nodeWidth + 3, y + nodeHeight / 2);
      context.lineTo(x + nodeWidth + 9, y + nodeHeight / 2);
      context.stroke();
    }
  });

  context.fillStyle = "#61707f";
  context.font = "800 10px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(
    fitCanvasText(context, eigenDirectionPhaseCopy(eigenDirection), flowWidth),
    flowX,
    pane.y + pane.height - 18,
  );
  context.restore();
}

function paintEigenDirectionGrid(
  context: CanvasRenderingContext2D,
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);
  const { plot, x, y } = eigenDirectionScales(eigenDirection, pane);
  const statusColor = eigenDirectionStatusColor(eigenDirection);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Drag the blue vector", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, "Most directions rotate. Magic directions stay on the same line after A is applied.", pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  context.fillStyle = "rgba(247, 250, 252, 0.94)";
  context.fillRect(plot.x, plot.y, plot.width, plot.height);
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.strokeRect(plot.x, plot.y, plot.width, plot.height);

  context.save();
  context.beginPath();
  context.rect(plot.x, plot.y, plot.width, plot.height);
  context.clip();
  paintEigenOriginalGrid(context, plot, x, y);
  paintEigenTransformedGrid(context, eigenDirection, x, y);
  paintDeterminantAxes(context, plot, x, y);

  if (eigenDirection.showAll) {
    paintEigenDirectionLines(context, eigenDirection, plot, x, y);
  }

  if (eigenDirection.powerPath.length > 1) {
    context.strokeStyle = "rgba(183, 121, 31, 0.78)";
    context.lineWidth = 2;
    context.beginPath();
    eigenDirection.powerPath.forEach((point, index) => {
      const px = x(point.x * 1.5);
      const py = y(point.y * 1.5);
      if (index === 0) {
        context.moveTo(px, py);
      } else {
        context.lineTo(px, py);
      }
    });
    context.stroke();
    eigenDirection.powerPath.forEach((point, index) => {
      context.fillStyle = index === eigenDirection.powerPath.length - 1 ? "#b7791f" : "rgba(183, 121, 31, 0.45)";
      context.beginPath();
      context.arc(x(point.x * 1.5), y(point.y * 1.5), index === eigenDirection.powerPath.length - 1 ? 5 : 3, 0, Math.PI * 2);
      context.fill();
    });
  }

  drawEigenDirectionArrow(context, x(0), y(0), x(eigenDirection.testVector.x * 1.55), y(eigenDirection.testVector.y * 1.55), "#2f6fbe", "v");
  drawEigenDirectionArrow(context, x(0), y(0), x(eigenDirection.transformedVector.x), y(eigenDirection.transformedVector.y), statusColor, "Av");
  context.restore();

  const handleX = x(eigenDirection.testVector.x * 1.55);
  const handleY = y(eigenDirection.testVector.y * 1.55);
  context.fillStyle = "#ffffff";
  context.strokeStyle = "#2f6fbe";
  context.lineWidth = 3;
  context.beginPath();
  context.arc(handleX, handleY, 9, 0, Math.PI * 2);
  context.fill();
  context.stroke();

  context.restore();
}

function paintEigenDirectionWheel(
  context: CanvasRenderingContext2D,
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);
  const statusColor = eigenDirectionStatusColor(eigenDirection);
  const wheel = eigenDirectionWheelGeometry(pane);

  context.fillStyle = "#17212b";
  context.font = "900 12px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText("Before and after directions", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(context, "If blue and after-A land on the same line, the vector is an eigenvector.", pane.width - 28),
    pane.x + 14,
    pane.y + 42,
  );

  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1.4;
  context.beginPath();
  context.arc(wheel.cx, wheel.cy, wheel.radius, 0, Math.PI * 2);
  context.stroke();
  context.strokeStyle = "rgba(100, 116, 139, 0.26)";
  context.beginPath();
  context.moveTo(wheel.cx - wheel.radius, wheel.cy);
  context.lineTo(wheel.cx + wheel.radius, wheel.cy);
  context.moveTo(wheel.cx, wheel.cy - wheel.radius);
  context.lineTo(wheel.cx, wheel.cy + wheel.radius);
  context.stroke();

  if (eigenDirection.showAll) {
    eigenDirection.eigenvalues.forEach((eigenvalue) => {
      if (!eigenvalue.vector || Math.abs(eigenvalue.imaginary) > 1e-7) {
        return;
      }
      context.strokeStyle = eigenvalue.color;
      context.lineWidth = 2.2;
      context.setLineDash([7, 5]);
      context.beginPath();
      context.moveTo(wheel.cx - eigenvalue.vector.x * wheel.radius, wheel.cy + eigenvalue.vector.y * wheel.radius);
      context.lineTo(wheel.cx + eigenvalue.vector.x * wheel.radius, wheel.cy - eigenvalue.vector.y * wheel.radius);
      context.stroke();
      context.setLineDash([]);
    });
  }

  const beforeAngle = Math.atan2(eigenDirection.testVector.y, eigenDirection.testVector.x);
  const afterAngle = Math.atan2(eigenDirection.normalizedTransformedVector.y, eigenDirection.normalizedTransformedVector.x);
  context.strokeStyle = "rgba(183, 121, 31, 0.46)";
  context.lineWidth = 4;
  context.beginPath();
  context.arc(wheel.cx, wheel.cy, Math.max(18, wheel.radius * 0.34), -beforeAngle, -afterAngle, afterAngle > beforeAngle);
  context.stroke();

  drawEigenWheelVector(context, wheel, eigenDirection.testVector, "#2f6fbe", "v");
  drawEigenWheelVector(context, wheel, eigenDirection.normalizedTransformedVector, statusColor, "Av");

  const statusText = eigenDirection.isEigenvector
    ? eigenDirection.isReversed
      ? "Status: Eigenvector, reversed"
      : "Status: Eigenvector"
    : "Status: Ordinary vector";
  const cardY = pane.y + pane.height - 86;
  paintRoundedRect(context, pane.x + 14, cardY, pane.width - 28, 68, 8, eigenDirection.isEigenvector ? "#eef7f5" : "#fff7f6", `${statusColor}66`);
  context.fillStyle = statusColor;
  context.font = "900 13px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(statusText, pane.x + 28, cardY + 22);
  context.fillStyle = "#61707f";
  context.font = "800 11px Inter, system-ui, sans-serif";
  wrapCanvasText(
    context,
    `Angle difference ${eigenDirection.angleDifference.toFixed(2)} deg · length scale ${eigenDirection.lengthScale.toFixed(2)}x · lambda estimate ${eigenDirection.eigenvalueEstimate.toFixed(3)}`,
    pane.x + 28,
    cardY + 42,
    pane.width - 56,
    14,
    2,
  );
  context.restore();
}

function paintEigenDirectionMetrics(
  context: CanvasRenderingContext2D,
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  pane: CanvasPane,
) {
  context.save();
  paintCanvasPanel(context, pane);
  const statusColor = eigenDirectionStatusColor(eigenDirection);
  const cards = [
    { label: "Angle Difference", value: `${eigenDirection.angleDifference.toFixed(2)} deg` },
    { label: "Length Scale", value: `${eigenDirection.lengthScale.toFixed(2)}x` },
    { label: "Status", value: eigenDirection.isEigenvector ? "Eigenvector" : "Ordinary Vector" },
    { label: "Eigenvalue", value: eigenDirection.isEigenvector ? eigenDirection.eigenvalueEstimate.toFixed(3) : "not aligned" },
  ];
  const compact = pane.width < 720;
  const columns = compact ? 2 : 4;
  const gap = 10;
  const cardWidth = (pane.width - 28 - gap * (columns - 1)) / columns;
  cards.forEach((card, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = pane.x + 14 + column * (cardWidth + gap);
    const y = pane.y + 14 + row * 52;
    paintRoundedRect(context, x, y, cardWidth, 42, 8, "#ffffff", index === 2 ? statusColor : "#d8e0e5");
    context.fillStyle = "#61707f";
    context.font = "800 9px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(card.label, x + 10, y + 15);
    context.fillStyle = index === 2 ? statusColor : "#17212b";
    context.font = "900 12px Inter, system-ui, sans-serif";
    context.fillText(fitCanvasText(context, card.value, cardWidth - 20), x + 10, y + 32);
  });

  if (!compact) {
    const buttonY = pane.y + pane.height - 30;
    const labels = ["Animate", "Power Iteration", "Show All Eigenvectors", "Reset"];
    let x = pane.x + 14;
    labels.forEach((label, index) => {
      const width = Math.min(150, Math.max(70, context.measureText(label).width + 24));
      paintRoundedRect(context, x, buttonY, width, 20, 8, index === 2 && eigenDirection.showAll ? "#eef7f5" : "#f7fafc", index === 2 && eigenDirection.showAll ? "#0f766e" : "#d8e0e5");
      context.fillStyle = index === 2 && eigenDirection.showAll ? "#0f766e" : "#61707f";
      context.font = "900 9px Inter, system-ui, sans-serif";
      context.textAlign = "center";
      context.fillText(label, x + width / 2, buttonY + 14);
      x += width + 8;
    });
  }

  context.restore();
}

function eigenDirectionScales(
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  pane: CanvasPane,
) {
  const plot = {
    x: pane.x + 44,
    y: pane.y + 64,
    width: pane.width - 74,
    height: pane.height - 96,
  };
  const coordinates = [
    eigenDirection.testVector,
    eigenDirection.transformedVector,
    ...eigenDirection.powerPath.map((point) => ({ x: point.x * 1.5, y: point.y * 1.5 })),
    { x: eigenDirection.matrix[0][0], y: eigenDirection.matrix[1][0] },
    { x: eigenDirection.matrix[0][1], y: eigenDirection.matrix[1][1] },
  ];
  const limit = Math.max(3.1, ...coordinates.flatMap((point) => [Math.abs(point.x), Math.abs(point.y)])) + 0.35;
  return {
    plot,
    limit,
    x: d3.scaleLinear().domain([-limit, limit]).range([plot.x, plot.x + plot.width]),
    y: d3.scaleLinear().domain([-limit, limit]).range([plot.y + plot.height, plot.y]),
  };
}

function paintEigenOriginalGrid(
  context: CanvasRenderingContext2D,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  context.strokeStyle = "rgba(100, 116, 139, 0.16)";
  context.lineWidth = 1;
  d3.range(-4, 4.1, 1).forEach((tick) => {
    context.beginPath();
    context.moveTo(x(tick), plot.y);
    context.lineTo(x(tick), plot.y + plot.height);
    context.moveTo(plot.x, y(tick));
    context.lineTo(plot.x + plot.width, y(tick));
    context.stroke();
  });
}

function paintEigenTransformedGrid(
  context: CanvasRenderingContext2D,
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  const [[a, b], [c, d]] = eigenDirection.matrix;
  context.strokeStyle = "rgba(47, 111, 190, 0.24)";
  context.lineWidth = 1.2;
  d3.range(-5, 5.1, 1).forEach((tick) => {
    const verticalA = determinantTransformForCanvas(a, b, c, d, tick, -5);
    const verticalB = determinantTransformForCanvas(a, b, c, d, tick, 5);
    const horizontalA = determinantTransformForCanvas(a, b, c, d, -5, tick);
    const horizontalB = determinantTransformForCanvas(a, b, c, d, 5, tick);
    context.beginPath();
    context.moveTo(x(verticalA.x), y(verticalA.y));
    context.lineTo(x(verticalB.x), y(verticalB.y));
    context.moveTo(x(horizontalA.x), y(horizontalA.y));
    context.lineTo(x(horizontalB.x), y(horizontalB.y));
    context.stroke();
  });
}

function paintEigenDirectionLines(
  context: CanvasRenderingContext2D,
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  plot: CanvasPane,
  x: d3.ScaleLinear<number, number>,
  y: d3.ScaleLinear<number, number>,
) {
  eigenDirection.eigenvalues.forEach((eigenvalue) => {
    if (!eigenvalue.vector || Math.abs(eigenvalue.imaginary) > 1e-7) {
      return;
    }
    const span = 5;
    context.strokeStyle = eigenvalue.color;
    context.lineWidth = 2.2;
    context.setLineDash([8, 5]);
    context.beginPath();
    context.moveTo(x(-eigenvalue.vector.x * span), y(-eigenvalue.vector.y * span));
    context.lineTo(x(eigenvalue.vector.x * span), y(eigenvalue.vector.y * span));
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = eigenvalue.color;
    context.font = "900 10px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.fillText(eigenvalue.label, Math.min(plot.x + plot.width - 54, x(eigenvalue.vector.x * 2.5) + 6), y(eigenvalue.vector.y * 2.5) - 6);
  });
}

function drawEigenDirectionArrow(
  context: CanvasRenderingContext2D,
  fromX: number,
  fromY: number,
  toX: number,
  toY: number,
  color: string,
  label: string,
) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  context.save();
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(fromX, fromY);
  context.lineTo(toX, toY);
  context.stroke();
  context.translate(toX, toY);
  context.rotate(angle);
  context.beginPath();
  context.moveTo(0, 0);
  context.lineTo(-11, -5);
  context.lineTo(-11, 5);
  context.closePath();
  context.fill();
  context.rotate(-angle);
  context.font = "900 11px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(label, 10, -8);
  context.restore();
}

function eigenDirectionWheelGeometry(pane: CanvasPane) {
  const radius = Math.max(72, Math.min(pane.width, pane.height - 150) * 0.32);
  return {
    cx: pane.x + pane.width / 2,
    cy: pane.y + Math.max(142, pane.height * 0.48),
    radius,
  };
}

function drawEigenWheelVector(
  context: CanvasRenderingContext2D,
  wheel: { cx: number; cy: number; radius: number },
  vector: DataPoint,
  color: string,
  label: string,
) {
  const toX = wheel.cx + vector.x * wheel.radius;
  const toY = wheel.cy - vector.y * wheel.radius;
  drawEigenDirectionArrow(context, wheel.cx, wheel.cy, toX, toY, color, label);
}

function canvasPointToEigenDirectionAngle(
  eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>,
  size: Size,
  point: { x: number; y: number },
) {
  const layout = eigenDirectionLayout(size);
  const { plot, x, y } = eigenDirectionScales(eigenDirection, layout.gridPane);
  if (
    point.x >= plot.x &&
    point.x <= plot.x + plot.width &&
    point.y >= plot.y &&
    point.y <= plot.y + plot.height
  ) {
    const dataX = x.invert(point.x);
    const dataY = y.invert(point.y);
    if (Math.hypot(dataX, dataY) > 0.18) {
      return normalizeAngleDegrees((Math.atan2(dataY, dataX) * 180) / Math.PI);
    }
  }

  const wheel = eigenDirectionWheelGeometry(layout.wheelPane);
  const dx = point.x - wheel.cx;
  const dy = wheel.cy - point.y;
  if (Math.hypot(dx, dy) <= wheel.radius + 24 && Math.hypot(dx, dy) >= 18) {
    return normalizeAngleDegrees((Math.atan2(dy, dx) * 180) / Math.PI);
  }

  return null;
}

function normalizeAngleDegrees(angle: number) {
  const normalized = ((angle + 180) % 360 + 360) % 360 - 180;
  return Math.round(normalized);
}

function eigenDirectionPhaseIndex(phase: NonNullable<ConceptFrame["eigenDirection"]>["phase"]) {
  return {
    matrix: 0,
    transform: 1,
    compare: 2,
    decision: 3,
    result: 4,
  }[phase];
}

function eigenDirectionPhaseCopy(eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>) {
  if (eigenDirection.phase === "matrix") {
    return "Start with a matrix A and a draggable test vector v.";
  }
  if (eigenDirection.phase === "transform") {
    return "Apply the matrix: A sends v to a new vector Av.";
  }
  if (eigenDirection.phase === "compare") {
    return "Compare the before and after directions on the wheel.";
  }
  if (eigenDirection.phase === "decision") {
    return eigenDirection.isEigenvector
      ? "Direction preserved: v stays on the same line."
      : "Direction changed: this vector is ordinary.";
  }
  return eigenDirection.isEigenvector
    ? "Yes: Av = lambda v, so this is an eigenvector."
    : "No: drag closer to a dashed eigen-line to find a magic direction.";
}

function eigenDirectionStatusColor(eigenDirection: NonNullable<ConceptFrame["eigenDirection"]>) {
  return eigenDirection.isEigenvector ? "#0f766e" : "#d34a43";
}

function formatEigenDirectionNumber(value: number) {
  if (Math.abs(value) < 0.001) {
    return "0.00";
  }
  return value.toFixed(Math.abs(value) >= 10 ? 1 : 2);
}

function paintCanvasPanel(
  context: CanvasRenderingContext2D,
  pane: CanvasPane,
) {
  context.fillStyle = "rgba(255, 255, 255, 0.82)";
  context.strokeStyle = "#d8e0e5";
  context.lineWidth = 1;
  context.fillRect(pane.x, pane.y, pane.width, pane.height);
  context.strokeRect(pane.x, pane.y, pane.width, pane.height);
}

function paintRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
  fill: string,
  stroke?: string,
) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.lineTo(x + width - r, y);
  context.quadraticCurveTo(x + width, y, x + width, y + r);
  context.lineTo(x + width, y + height - r);
  context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  context.lineTo(x + r, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - r);
  context.lineTo(x, y + r);
  context.quadraticCurveTo(x, y, x + r, y);
  context.closePath();
  context.fillStyle = fill;
  context.fill();
  if (stroke) {
    context.strokeStyle = stroke;
    context.stroke();
  }
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
  context.fillText("Matrix split animation", compressionPane.x, compressionPane.y - 24);
  context.fillText("What it does: background lift", geometryPane.x, geometryPane.y - 24);

  paintSvdCompressionPane(context, svd, compressionPane);
  paintSvdUseCasePane(context, svd, geometryPane);
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
  context.fillText(`${svd.method === "nmf" ? "NMF additive parts" : "SVD U Sigma V^T"}: ${svd.source.name}`, pane.x + 14, pane.y + 22);

  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      `rank ${svd.rank}/${svd.maxRank} · structure retained ${(svd.retainedEnergy * 100).toFixed(1)}% · error ${svd.reconstructionError.toFixed(3)}`,
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const compactSplit = pane.width < 560;
  const splitTop = pane.y + 64;
  const splitHeight = compactSplit
    ? Math.min(280, Math.max(220, pane.height * 0.44))
    : Math.min(210, Math.max(150, pane.height * 0.38));
  const sourceSize = compactSplit
    ? Math.min(118, pane.width * 0.34, splitHeight * 0.46)
    : Math.min(splitHeight, Math.max(104, pane.width * 0.2));
  const sourceBox = {
    x: compactSplit ? pane.x + (pane.width - sourceSize) / 2 : pane.x + 14,
    y: compactSplit ? splitTop : splitTop + (splitHeight - sourceSize) / 2,
    width: sourceSize,
    height: sourceSize,
  };
  const factorAreaX = compactSplit ? pane.x + 14 : sourceBox.x + sourceBox.width + 48;
  const factorAreaWidth = compactSplit
    ? pane.width - 28
    : Math.max(120, pane.x + pane.width - 14 - factorAreaX);
  const factorGap = 10;
  const factorWidth = Math.max(54, (factorAreaWidth - factorGap * 2 - 52) / 3);
  const factorHeight = compactSplit
    ? Math.max(76, splitTop + splitHeight - sourceBox.y - sourceSize - 34)
    : Math.min(splitHeight, sourceSize);
  const factorY = compactSplit ? sourceBox.y + sourceSize + 30 : splitTop + (splitHeight - factorHeight) / 2;
  const splitEase = 0.5 - Math.cos(Math.max(0, Math.min(1, svd.splitProgress)) * Math.PI) / 2;

  paintSvdImageBox(context, "dense matrix A", svd.original, sourceBox);

  context.save();
  context.strokeStyle = `rgba(47, 111, 190, ${0.24 + splitEase * 0.5})`;
  context.lineWidth = 2;
  context.setLineDash([5, 5]);
  context.beginPath();
  if (compactSplit) {
    context.moveTo(sourceBox.x + sourceBox.width / 2, sourceBox.y + sourceBox.height + 8);
    context.lineTo(sourceBox.x + sourceBox.width / 2, factorY - 10);
  } else {
    context.moveTo(sourceBox.x + sourceBox.width + 10, sourceBox.y + sourceBox.height / 2);
    context.lineTo(factorAreaX - 14, sourceBox.y + sourceBox.height / 2);
  }
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = "#17212b";
  context.font = "900 17px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(
    "≈",
    compactSplit ? sourceBox.x + sourceBox.width / 2 + 18 : sourceBox.x + sourceBox.width + 30,
    compactSplit ? factorY - 12 : sourceBox.y + sourceBox.height / 2 + 6,
  );
  context.restore();

  svd.factors.forEach((factor, index) => {
    const target = {
      x: factorAreaX + index * (factorWidth + factorGap + 18),
      y: factorY,
      width: factorWidth,
      height: factorHeight,
    };
    const collapsed = {
      x: sourceBox.x + sourceBox.width * 0.38,
      y: sourceBox.y + sourceBox.height * 0.18,
      width: sourceBox.width * 0.52,
      height: sourceBox.height * 0.64,
    };
    const box = interpolateCanvasPane(collapsed, target, splitEase);
    context.save();
    context.globalAlpha = 0.2 + splitEase * 0.8;
    paintSvdFactorBlock(context, factor, box);
    context.restore();

    if (index < svd.factors.length - 1) {
      context.fillStyle = "#61707f";
      context.font = "900 13px Inter, system-ui, sans-serif";
      context.textAlign = "center";
      context.fillText("x", target.x + target.width + factorGap / 2 + 9, target.y + target.height / 2 + 5);
    }
  });

  const spectrumPane = {
    x: pane.x + 14,
    y: splitTop + splitHeight + 28,
    width: pane.width - 28,
    height: Math.max(118, pane.y + pane.height - splitTop - splitHeight - 48),
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
  context.fillText(svd.method === "nmf" ? "component strength spectrum" : "singular value spectrum", pane.x + 12, pane.y + 18);

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
  context.fillText(
    `${svd.method === "nmf" ? "c1" : "sigma1"}=${(svd.singularValues[0] ?? 0).toFixed(2)}`,
    pane.x + pane.width - 12,
    pane.y + 18,
  );
}

function paintSvdFactorBlock(
  context: CanvasRenderingContext2D,
  factor: NonNullable<ConceptFrame["svd"]>["factors"][number],
  box: CanvasPane,
) {
  context.fillStyle = "#f7fafc";
  context.strokeStyle = factor.color;
  context.lineWidth = 1.5;
  context.fillRect(box.x, box.y, box.width, box.height);
  context.strokeRect(box.x, box.y, box.width, box.height);

  paintSvdMatrixCells(context, factor.matrix, {
    x: box.x + 8,
    y: box.y + 26,
    width: box.width - 16,
    height: Math.max(34, box.height - 56),
  }, factor.color, Boolean(factor.signed));

  context.fillStyle = "#17212b";
  context.font = "900 10px Inter, system-ui, sans-serif";
  context.textAlign = "center";
  context.fillText(factor.label, box.x + box.width / 2, box.y + 16);

  context.fillStyle = "#61707f";
  context.font = "800 8px Inter, system-ui, sans-serif";
  context.fillText(fitCanvasText(context, factor.role, box.width - 10), box.x + box.width / 2, box.y + box.height - 22);
  context.fillText(`${factor.matrix.length}x${factor.matrix[0]?.length ?? 0}`, box.x + box.width / 2, box.y + box.height - 9);
}

function paintSvdMatrixCells(
  context: CanvasRenderingContext2D,
  matrix: number[][],
  box: CanvasPane,
  accent: string,
  signed: boolean,
) {
  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;
  const maxAbs = Math.max(1e-8, ...matrix.flat().map((value) => Math.abs(value)));
  const cellWidth = box.width / Math.max(1, columns);
  const cellHeight = box.height / Math.max(1, rows);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = matrix[row][column] ?? 0;
      const normalized = Math.min(1, Math.abs(value) / maxAbs);
      if (signed) {
        const positive = value >= 0;
        context.fillStyle = positive
          ? `rgba(47, 111, 190, ${0.12 + normalized * 0.82})`
          : `rgba(211, 74, 67, ${0.12 + normalized * 0.82})`;
      } else {
        context.fillStyle = `${accent}${Math.round((0.16 + normalized * 0.74) * 255)
          .toString(16)
          .padStart(2, "0")}`;
      }
      context.fillRect(
        box.x + column * cellWidth,
        box.y + row * cellHeight,
        Math.ceil(cellWidth),
        Math.ceil(cellHeight),
      );
    }
  }
}

function interpolateCanvasPane(from: CanvasPane, to: CanvasPane, progress: number): CanvasPane {
  return {
    x: from.x + (to.x - from.x) * progress,
    y: from.y + (to.y - from.y) * progress,
    width: from.width + (to.width - from.width) * progress,
    height: from.height + (to.height - from.height) * progress,
  };
}

function paintSvdUseCasePane(
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
  context.fillText("Low-rank background extraction", pane.x + 14, pane.y + 22);
  context.fillStyle = "#61707f";
  context.font = "700 11px Inter, system-ui, sans-serif";
  context.fillText(
    fitCanvasText(
      context,
      "The low-rank reconstruction models stable background; the residual exposes detail or motion.",
      pane.width - 28,
    ),
    pane.x + 14,
    pane.y + 42,
  );

  const mainBox = {
    x: pane.x + 14,
    y: pane.y + 66,
    width: pane.width - 28,
    height: Math.min(Math.max(145, pane.width * 0.62), pane.height * 0.34),
  };
  paintSvdImageBox(context, "uploaded frame", svd.useCase.frame, mainBox);

  const bottomTop = mainBox.y + mainBox.height + 34;
  const bottomHeight = Math.max(112, pane.y + pane.height - bottomTop - 78);
  const bottomGap = 12;
  const bottomWidth = (pane.width - 28 - bottomGap) / 2;
  paintSvdImageBox(context, "low-rank background", svd.useCase.background, {
    x: pane.x + 14,
    y: bottomTop,
    width: bottomWidth,
    height: bottomHeight,
  });
  paintSvdImageBox(context, "foreground residual", svd.useCase.foreground, {
    x: pane.x + 14 + bottomWidth + bottomGap,
    y: bottomTop,
    width: bottomWidth,
    height: bottomHeight,
  });

  const meterY = pane.y + pane.height - 54;
  paintSvdMeter(
    context,
    "background modeled",
    svd.useCase.backgroundEnergy,
    { x: pane.x + 14, y: meterY, width: pane.width - 28, height: 18 },
    "#2f6fbe",
  );
  paintSvdMeter(
    context,
    "foreground residual",
    svd.useCase.foregroundEnergy,
    { x: pane.x + 14, y: meterY + 25, width: pane.width - 28, height: 18 },
    "#d34a43",
  );

  context.restore();
}

function paintSvdMeter(
  context: CanvasRenderingContext2D,
  label: string,
  value: number,
  box: CanvasPane,
  color: string,
) {
  const normalized = Math.max(0, Math.min(1, value));
  context.fillStyle = "#f7fafc";
  context.strokeStyle = "#dce5ea";
  context.fillRect(box.x, box.y, box.width, box.height);
  context.strokeRect(box.x, box.y, box.width, box.height);
  context.fillStyle = color;
  context.fillRect(box.x, box.y, box.width * normalized, box.height);
  context.fillStyle = "#17212b";
  context.font = "900 9px Inter, system-ui, sans-serif";
  context.textAlign = "left";
  context.fillText(`${label}: ${Math.round(normalized * 100)}%`, box.x + 7, box.y + 13);
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
