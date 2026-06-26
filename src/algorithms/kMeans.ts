import { makeClusterDataset } from "../data/datasets";
import type {
  AlgorithmDefinition,
  DataPoint,
  EngineResult,
  KMeansFrame,
  NormalizedDataset,
  ParameterState,
} from "../types/algorithm";

const palette = ["#0f766e", "#2f6fbe", "#b7791f", "#d34a43", "#6f58c9", "#258f66"];

const numberParam = (
  params: ParameterState,
  key: string,
  fallback: number,
) => {
  const value = Number(params[key]);
  return Number.isFinite(value) ? value : fallback;
};

const stringParam = (
  params: ParameterState,
  key: string,
  fallback: string,
) => (typeof params[key] === "string" ? String(params[key]) : fallback);

export const kMeans: AlgorithmDefinition = {
  id: "k-means",
  name: "K-Means",
  category: "Clustering",
  summary: "Groups points by alternating centroid assignment and centroid updates.",
  parameters: [
    {
      kind: "range",
      id: "clusters",
      label: "Clusters k",
      min: 2,
      max: 6,
      step: 1,
      defaultValue: 3,
      format: "integer",
    },
    {
      kind: "range",
      id: "iterations",
      label: "Max iterations",
      min: 2,
      max: 30,
      step: 1,
      defaultValue: 12,
      format: "integer",
    },
    {
      kind: "select",
      id: "initialization",
      label: "Initialization",
      defaultValue: "spread",
      options: [
        { label: "Spread across x-axis", value: "spread" },
        { label: "Deterministic shuffle", value: "shuffle" },
      ],
    },
  ],
  makeSampleDataset: makeClusterDataset,
  engine: runKMeans,
  formulas: [
    {
      title: "Assignment",
      expression: "c_i=\\arg\\min_j\\lVert x_i-\\mu_j \\rVert^2",
    },
    {
      title: "Centroid update",
      expression: "\\mu_j=\\frac{1}{|C_j|}\\sum_{x_i\\in C_j}x_i",
    },
    {
      title: "Inertia",
      expression: "\\sum_{j=1}^{k}\\sum_{x_i\\in C_j}\\lVert x_i-\\mu_j \\rVert^2",
    },
  ],
  explanation: [
    "K-Means is an unsupervised clustering algorithm. It repeatedly assigns each point to the nearest centroid, then moves each centroid to the mean of its assigned points.",
    "The animation exposes the alternating structure directly: colors show current assignments and larger markers show centroid movement.",
    "The Python example uses scikit-learn for the practical notebook version, while this app runs a transparent JavaScript implementation so every intermediate frame can be drawn.",
  ],
  code: {
    python: (params) => {
      const clusters = Math.round(numberParam(params, "clusters", 3));
      const iterations = Math.round(numberParam(params, "iterations", 12));

      return `import numpy as np
from sklearn.cluster import KMeans

X = dataset[["x", "y"]].to_numpy(dtype=float)

model = KMeans(
    n_clusters=${clusters},
    max_iter=${iterations},
    n_init=1,
    random_state=7,
)
labels = model.fit_predict(X)

print({
    "centroids": model.cluster_centers_,
    "inertia": model.inertia_,
    "labels": labels,
})`;
    },
    javascript: (params) => {
      const clusters = Math.round(numberParam(params, "clusters", 3));
      const iterations = Math.round(numberParam(params, "iterations", 12));
      const initialization = stringParam(params, "initialization", "spread");

      return `const k = ${clusters};
const maxIterations = ${iterations};
const initialization = "${initialization}";

let centroids = initializeCentroids(points, k, initialization);
let assignments = new Array(points.length).fill(-1);

for (let iteration = 0; iteration < maxIterations; iteration += 1) {
  assignments = points.map((point) =>
    nearestCentroidIndex(point, centroids),
  );

  centroids = centroids.map((centroid, index) => {
    const members = points.filter((_, pointIndex) => assignments[pointIndex] === index);
    return members.length > 0 ? meanPoint(members) : centroid;
  });
}`;
    },
  },
};

function runKMeans(
  dataset: NormalizedDataset,
  params: ParameterState,
): EngineResult {
  const points = dataset.points;

  if (points.length === 0) {
    return {
      frames: [],
      metrics: [{ label: "Rows", value: "0 usable" }],
      runtime: "JavaScript",
    };
  }

  const k = Math.min(
    Math.round(numberParam(params, "clusters", 3)),
    points.length,
  );
  const maxIterations = Math.round(numberParam(params, "iterations", 12));
  const initialization = stringParam(params, "initialization", "spread");
  const frames: KMeansFrame[] = [];
  let centroids = initializeCentroids(points, k, initialization);
  let assignments = new Array<number>(points.length).fill(-1);

  for (let iteration = 0; iteration <= maxIterations; iteration += 1) {
    const nextAssignments = points.map((point) =>
      nearestCentroidIndex(point, centroids),
    );
    const inertia = calculateInertia(points, centroids, nextAssignments);

    frames.push({
      type: "k-means",
      iteration,
      points,
      centroids,
      assignments: nextAssignments,
      inertia,
    });

    if (sameAssignments(assignments, nextAssignments) && iteration > 0) {
      assignments = nextAssignments;
      break;
    }

    assignments = nextAssignments;
    centroids = centroids.map((centroid, index) => {
      const members = points.filter((_, pointIndex) => assignments[pointIndex] === index);
      return members.length > 0 ? meanPoint(members) : centroid;
    });
  }

  const finalFrame = frames[frames.length - 1];

  return {
    frames,
    metrics: [
      { label: "Inertia", value: finalFrame.inertia.toFixed(3) },
      { label: "Clusters", value: String(k) },
      { label: "Iterations", value: String(finalFrame.iteration) },
      { label: "Palette", value: palette.slice(0, k).join(" ") },
    ],
    runtime: "JavaScript",
  };
}

function initializeCentroids(
  points: DataPoint[],
  k: number,
  mode: string,
): DataPoint[] {
  if (mode === "shuffle") {
    return deterministicShuffle(points).slice(0, k).map(copyPoint);
  }

  const sorted = [...points].sort((a, b) => a.x - b.x);
  return Array.from({ length: k }, (_, index) => {
    const pointIndex = Math.round((index / Math.max(1, k - 1)) * (sorted.length - 1));
    return copyPoint(sorted[pointIndex]);
  });
}

function deterministicShuffle(points: DataPoint[]) {
  return [...points].sort((a, b) => {
    const hashA = Math.sin(a.x * 12.9898 + a.y * 78.233) * 43758.5453;
    const hashB = Math.sin(b.x * 12.9898 + b.y * 78.233) * 43758.5453;
    return fractional(hashA) - fractional(hashB);
  });
}

function nearestCentroidIndex(point: DataPoint, centroids: DataPoint[]) {
  return centroids.reduce(
    (best, centroid, index) => {
      const distance = squaredDistance(point, centroid);
      return distance < best.distance ? { index, distance } : best;
    },
    { index: 0, distance: Number.POSITIVE_INFINITY },
  ).index;
}

function meanPoint(points: DataPoint[]): DataPoint {
  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
    { x: 0, y: 0 },
  );

  return {
    x: total.x / points.length,
    y: total.y / points.length,
  };
}

function calculateInertia(
  points: DataPoint[],
  centroids: DataPoint[],
  assignments: number[],
) {
  return points.reduce((total, point, index) => {
    const centroid = centroids[assignments[index]];
    return total + squaredDistance(point, centroid);
  }, 0);
}

function sameAssignments(a: number[], b: number[]) {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function squaredDistance(a: DataPoint, b: DataPoint) {
  return (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
}

function fractional(value: number) {
  return value - Math.floor(value);
}

function copyPoint(point: DataPoint): DataPoint {
  return { x: point.x, y: point.y };
}
