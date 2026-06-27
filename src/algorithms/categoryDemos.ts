import type {
  ActivationCurve,
  ActivationFunctionKey,
  ActivationNeuronState,
  AlgorithmCategory,
  AlgorithmDefinition,
  BackpropFormulaValue,
  ConceptBar,
  ConceptFrame,
  ConceptSeries,
  ConvolutionKernelPreset,
  ConvexOptimizerKey,
  ContrastivePair,
  ContrastivePoint,
  DataPoint,
  DynamicProgrammingAction,
  DynamicProgrammingMethod,
  EngineResult,
  GridWorldCell,
  GridWorldValue,
  ImageMatrixParameterValue,
  LossFunctionKey,
  LossSurfaceKey,
  LossPrediction,
  NormalizedDataset,
  OptimizerKey,
  ParameterState,
  FrameworkLayerState,
} from "../types/algorithm";

const colors = ["#0f766e", "#2f6fbe", "#b7791f", "#d34a43", "#6f58c9", "#258f66"];

const numberParam = (params: ParameterState, key: string, fallback: number) => {
  const value = Number(params[key]);
  return Number.isFinite(value) ? value : fallback;
};

const stringParam = (params: ParameterState, key: string, fallback: string) =>
  typeof params[key] === "string" ? String(params[key]) : fallback;

function matrixParam(params: ParameterState, key: string, fallback: number[][]) {
  const value = params[key];
  const source = Array.isArray(value) ? value : fallback;

  return fallback.map((fallbackRow, rowIndex) => {
    const row = Array.isArray(source[rowIndex]) ? source[rowIndex] : fallbackRow;
    return fallbackRow.map((fallbackCell, columnIndex) => {
      const cell = Number(row[columnIndex]);
      return Number.isFinite(cell) ? Math.max(0, cell) : fallbackCell;
    });
  });
}

function signedMatrixParam(params: ParameterState, key: string, fallback: number[][]) {
  const value = params[key];
  const source = Array.isArray(value) ? value : fallback;

  return fallback.map((fallbackRow, rowIndex) => {
    const row = Array.isArray(source[rowIndex]) ? source[rowIndex] : fallbackRow;
    return fallbackRow.map((fallbackCell, columnIndex) => {
      const cell = Number(row[columnIndex]);
      return Number.isFinite(cell) ? cell : fallbackCell;
    });
  });
}

function imageMatrixParam(
  params: ParameterState,
  key: string,
  fallback: ImageMatrixParameterValue,
) {
  const value = params[key];
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    "kind" in value &&
    value.kind === "image-matrix"
  ) {
    return value;
  }

  return fallback;
}

function gridWorldParam(
  params: ParameterState,
  key: string,
  fallback: GridWorldValue,
) {
  const value = params[key];
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    "kind" in value &&
    value.kind === "gridworld" &&
    Array.isArray(value.cells)
  ) {
    return {
      ...value,
      rows: value.rows,
      columns: value.columns,
      cells: value.cells.map((row) =>
        row.map((cell) => (isGridWorldCell(cell) ? cell : "empty")),
      ),
    };
  }

  return {
    ...fallback,
    cells: fallback.cells.map((row) => [...row]),
  };
}

function isGridWorldCell(value: unknown): value is GridWorldCell {
  return value === "empty" || value === "wall" || value === "fire" || value === "gold" || value === "start";
}

function makeDataset(name: string, points: DataPoint[]): NormalizedDataset {
  return {
    name,
    points,
    columns: [
      { name: "x", type: "number" },
      { name: "y", type: "number" },
      { name: "label", type: "string" },
    ],
    mapping: { x: "x", y: "y", label: "label" },
    source: "sample",
  };
}

function makeClassificationSample(name = "Generated classification sample") {
  const points = Array.from({ length: 68 }, (_, index) => {
    const ring = index % 2 === 0 ? -1 : 1;
    const angle = index * 1.37;
    const radius = 1.15 + ((index * 11) % 17) * 0.08;
    const x = ring * 1.9 + Math.cos(angle) * radius;
    const y = ring * 1.25 + Math.sin(angle * 0.9) * radius;
    return {
      x: round(x),
      y: round(y),
      label: ring > 0 ? "positive" : "negative",
    };
  });

  return makeDataset(name, points);
}

function makeBarsDataset(name: string, bars: ConceptBar[]) {
  return makeDataset(
    name,
    bars.map((bar, index) => ({ x: index, y: bar.value, label: bar.label })),
  );
}

function round(value: number, digits = 3) {
  return Number.parseFloat(value.toFixed(digits));
}

function classValue(point: DataPoint) {
  if (point.label === "positive" || point.label === 1 || point.label === "1") {
    return 1;
  }

  if (point.label === "negative" || point.label === 0 || point.label === "0") {
    return 0;
  }

  return point.x + point.y > 0 ? 1 : 0;
}

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
}

function boundarySeries(
  points: DataPoint[],
  w1: number,
  w2: number,
  bias: number,
  label = "decision boundary",
): ConceptSeries | undefined {
  if (Math.abs(w2) < 0.001) {
    return undefined;
  }

  const xs = points.map((point) => point.x);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);

  return {
    label,
    color: "#d34a43",
    points: [
      { x: minX, y: -(w1 * minX + bias) / w2 },
      { x: maxX, y: -(w1 * maxX + bias) / w2 },
    ],
  };
}

function metricValue(value: number) {
  return `${Math.round(value * 100)}%`;
}

function conceptResult(frame: ConceptFrame, metrics: Array<{ label: string; value: string }>): EngineResult {
  return {
    frames: [frame],
    metrics,
    runtime: "JavaScript",
  };
}

const logisticRegression: AlgorithmDefinition = {
  id: "logistic-regression",
  name: "Logistic Regression",
  category: "Logistic Regression & Classification",
  summary: "Learns a sigmoid decision boundary for binary classification.",
  parameters: [
    {
      kind: "range",
      id: "learningRate",
      label: "Learning rate",
      min: 0.01,
      max: 0.4,
      step: 0.01,
      defaultValue: 0.12,
      format: "decimal",
    },
    {
      kind: "range",
      id: "epochs",
      label: "Epochs",
      min: 20,
      max: 400,
      step: 10,
      defaultValue: 160,
      format: "integer",
    },
    {
      kind: "range",
      id: "regularization",
      label: "L2 strength",
      min: 0,
      max: 0.2,
      step: 0.01,
      defaultValue: 0.02,
      format: "decimal",
    },
  ],
  makeSampleDataset: () => makeClassificationSample(),
  engine: runLogisticRegression,
  formulas: [
    { title: "Probability", expression: "p(y=1|x)=\\sigma(w^Tx+b)" },
    { title: "Sigmoid", expression: "\\sigma(z)=\\frac{1}{1+e^{-z}}" },
    { title: "Loss", expression: "-y\\log(p)-(1-y)\\log(1-p)+\\lambda\\lVert w\\rVert^2" },
  ],
  explanation: [
    "Logistic regression turns a linear score into a probability with the sigmoid function.",
    "The decision boundary is the set of points where the predicted probability is 0.5.",
    "Changing learning rate, epochs, and regularization updates the JavaScript-trained boundary immediately.",
  ],
  code: {
    python: (params) => `from sklearn.linear_model import LogisticRegression

X = dataset[["x", "y"]].to_numpy()
y = dataset["label"].map({"negative": 0, "positive": 1}).to_numpy()

model = LogisticRegression(
    C=${round(1 / Math.max(0.001, numberParam(params, "regularization", 0.02)), 2)},
    max_iter=${Math.round(numberParam(params, "epochs", 160))},
)
model.fit(X, y)
print(model.score(X, y))`,
    javascript: (params) => `let w1 = 0;
let w2 = 0;
let bias = 0;
const learningRate = ${numberParam(params, "learningRate", 0.12)};
const regularization = ${numberParam(params, "regularization", 0.02)};

for (let epoch = 0; epoch < ${Math.round(numberParam(params, "epochs", 160))}; epoch += 1) {
  for (const point of points) {
    const y = point.label === "positive" ? 1 : 0;
    const p = sigmoid(w1 * point.x + w2 * point.y + bias);
    const error = p - y;
    w1 -= learningRate * (error * point.x + regularization * w1);
    w2 -= learningRate * (error * point.y + regularization * w2);
    bias -= learningRate * error;
  }
}`,
  },
};

function runLogisticRegression(dataset: NormalizedDataset, params: ParameterState) {
  const points = dataset.points.length > 0 ? dataset.points : makeClassificationSample().points;
  const learningRate = numberParam(params, "learningRate", 0.12);
  const epochs = Math.round(numberParam(params, "epochs", 160));
  const regularization = numberParam(params, "regularization", 0.02);
  let w1 = 0;
  let w2 = 0;
  let bias = 0;

  for (let epoch = 0; epoch < epochs; epoch += 1) {
    points.forEach((point) => {
      const y = classValue(point);
      const probability = sigmoid(w1 * point.x + w2 * point.y + bias);
      const error = probability - y;
      w1 -= learningRate * (error * point.x + regularization * w1);
      w2 -= learningRate * (error * point.y + regularization * w2);
      bias -= learningRate * error;
    });
  }

  const predictions = points.map((point) => sigmoid(w1 * point.x + w2 * point.y + bias));
  const accuracy =
    predictions.filter((prediction, index) => Number(prediction >= 0.5) === classValue(points[index])).length /
    points.length;
  const loss =
    predictions.reduce((total, prediction, index) => {
      const y = classValue(points[index]);
      const clipped = Math.min(0.999, Math.max(0.001, prediction));
      return total - (y * Math.log(clipped) + (1 - y) * Math.log(1 - clipped));
    }, 0) / points.length;
  const boundary = boundarySeries(points, w1, w2, bias);

  return conceptResult(
    {
      type: "concept-demo",
      iteration: epochs,
      points,
      series: boundary ? [boundary] : [],
      summary: `Accuracy ${metricValue(accuracy)} · log loss ${loss.toFixed(3)}`,
    },
    [
      { label: "Accuracy", value: metricValue(accuracy) },
      { label: "Log loss", value: loss.toFixed(3) },
      { label: "Weight x", value: w1.toFixed(2) },
      { label: "Weight y", value: w2.toFixed(2) },
    ],
  );
}

const decisionTreeSplit = makeConceptAlgorithm({
  id: "decision-tree-split",
  name: "Decision Tree Split",
  category: "Decision Trees & Random Forests",
  summary: "Chooses a threshold that reduces class impurity.",
  parameters: [
    { kind: "range", id: "maxDepth", label: "Max depth", min: 1, max: 8, step: 1, defaultValue: 3, format: "integer" },
    { kind: "range", id: "minLeaf", label: "Min leaf size", min: 2, max: 18, step: 1, defaultValue: 6, format: "integer" },
  ],
  sample: () => makeClassificationSample("Generated tree split sample"),
  formulas: [
    { title: "Gini impurity", expression: "G=1-\\sum_k p_k^2" },
    { title: "Split gain", expression: "\\Delta G=G(parent)-\\sum_j \\frac{n_j}{n}G(child_j)" },
  ],
  explanation: [
    "A decision tree asks feature-threshold questions such as x <= t.",
    "This module scans possible x thresholds and marks the split with the best Gini reduction.",
  ],
  engine: (dataset, params) => {
    const points = dataset.points;
    const minLeaf = Math.round(numberParam(params, "minLeaf", 6));
    const candidates = [...points].sort((a, b) => a.x - b.x);
    let best = { threshold: candidates[0]?.x ?? 0, gain: -Infinity, left: 0, right: 0 };
    const parentGini = gini(points);

    for (let index = minLeaf; index < candidates.length - minLeaf; index += 1) {
      const threshold = (candidates[index - 1].x + candidates[index].x) / 2;
      const left = points.filter((point) => point.x <= threshold);
      const right = points.filter((point) => point.x > threshold);
      const weighted = (left.length / points.length) * gini(left) + (right.length / points.length) * gini(right);
      const gain = parentGini - weighted;
      if (gain > best.gain) {
        best = { threshold, gain, left: left.length, right: right.length };
      }
    }

    return conceptResult(
      {
        type: "concept-demo",
        iteration: Math.round(numberParam(params, "maxDepth", 3)),
        points,
        markers: [{ axis: "x", value: best.threshold, label: "best split", color: "#d34a43" }],
        summary: `Best split x <= ${best.threshold.toFixed(2)} · gain ${best.gain.toFixed(3)}`,
      },
      [
        { label: "Gini gain", value: best.gain.toFixed(3) },
        { label: "Threshold", value: best.threshold.toFixed(2) },
        { label: "Left leaf", value: String(best.left) },
        { label: "Right leaf", value: String(best.right) },
      ],
    );
  },
  python: (params) => `from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=${Math.round(numberParam(params, "maxDepth", 3))},
    min_samples_leaf=${Math.round(numberParam(params, "minLeaf", 6))},
)
model.fit(X, y)`,
  javascript: (params) => `const maxDepth = ${Math.round(numberParam(params, "maxDepth", 3))};
const minLeaf = ${Math.round(numberParam(params, "minLeaf", 6))};
const bestSplit = scanThresholds(points, "x", giniImpurity, minLeaf);`,
});

const supportVectorMachine = makeConceptAlgorithm({
  id: "support-vector-machine",
  name: "Linear SVM Margin",
  category: "Support Vector Machines",
  summary: "Finds a separating hyperplane with the widest useful margin.",
  parameters: [
    { kind: "range", id: "c", label: "Penalty C", min: 0.1, max: 10, step: 0.1, defaultValue: 2.4, format: "decimal" },
    { kind: "range", id: "margin", label: "Margin width", min: 0.2, max: 2.2, step: 0.1, defaultValue: 0.9, format: "decimal" },
  ],
  sample: () => makeClassificationSample("Generated SVM sample"),
  formulas: [
    { title: "Margin objective", expression: "\\min_w \\frac{1}{2}\\lVert w\\rVert^2+C\\sum_i \\xi_i" },
    { title: "Constraint", expression: "y_i(w^Tx_i+b)\\ge 1-\\xi_i" },
  ],
  explanation: [
    "Support vector machines care most about points near the separating boundary.",
    "The dashed lines show the margin band; points inside it are the visually important support vectors.",
  ],
  engine: (dataset, params) => {
    const points = dataset.points;
    const margin = numberParam(params, "margin", 0.9);
    const c = numberParam(params, "c", 2.4);
    const w1 = 0.82 + c * 0.025;
    const w2 = 0.78;
    const bias = -0.05;
    const supportCount = points.filter((point) => Math.abs(w1 * point.x + w2 * point.y + bias) < margin).length;
    const boundary = boundarySeries(points, w1, w2, bias);
    const upper = boundarySeries(points, w1, w2, bias - margin, "+ margin");
    const lower = boundarySeries(points, w1, w2, bias + margin, "- margin");

    return conceptResult(
      {
        type: "concept-demo",
        iteration: Math.round(c * 10),
        points,
        series: [
          ...(boundary ? [boundary] : []),
          ...(upper ? [{ ...upper, dashed: true, color: "#b7791f" }] : []),
          ...(lower ? [{ ...lower, dashed: true, color: "#b7791f" }] : []),
        ],
        summary: `${supportCount} support candidates · margin ${margin.toFixed(1)}`,
      },
      [
        { label: "Support", value: String(supportCount) },
        { label: "Penalty C", value: c.toFixed(1) },
        { label: "Margin", value: margin.toFixed(1) },
        { label: "Kernel", value: "linear" },
      ],
    );
  },
  python: (params) => `from sklearn.svm import SVC

model = SVC(kernel="linear", C=${numberParam(params, "c", 2.4).toFixed(1)})
model.fit(X, y)
support_vectors = model.support_vectors_`,
  javascript: (params) => `const c = ${numberParam(params, "c", 2.4).toFixed(1)};
const margin = ${numberParam(params, "margin", 0.9).toFixed(1)};
const support = points.filter((p) => Math.abs(w1 * p.x + w2 * p.y + b) < margin);`,
});

const knnClassifier = makeConceptAlgorithm({
  id: "knn-classifier",
  name: "KNN Classifier",
  category: "KNN & Distance Metrics",
  summary: "Classifies a query point by nearest-neighbor voting.",
  parameters: [
    { kind: "range", id: "k", label: "Neighbors k", min: 1, max: 15, step: 2, defaultValue: 5, format: "integer" },
    { kind: "range", id: "queryX", label: "Query x", min: -4, max: 4, step: 0.1, defaultValue: 0.8, format: "decimal" },
    { kind: "range", id: "queryY", label: "Query y", min: -4, max: 4, step: 0.1, defaultValue: 0.4, format: "decimal" },
    {
      kind: "select",
      id: "metric",
      label: "Distance metric",
      defaultValue: "euclidean",
      options: [
        { label: "Euclidean", value: "euclidean" },
        { label: "Manhattan", value: "manhattan" },
      ],
    },
  ],
  sample: () => makeClassificationSample("Generated KNN sample"),
  formulas: [
    { title: "Euclidean", expression: "d(x,z)=\\sqrt{\\sum_j(x_j-z_j)^2}" },
    { title: "Vote", expression: "\\hat{y}=mode(y_i\\mid x_i\\in N_k(x))" },
  ],
  explanation: [
    "KNN stores the training data and waits until prediction time to compare distances.",
    "Move the query point and change k to see how local neighborhoods alter the vote.",
  ],
  engine: (dataset, params) => {
    const k = Math.round(numberParam(params, "k", 5));
    const metric = stringParam(params, "metric", "euclidean");
    const query = {
      x: numberParam(params, "queryX", 0.8),
      y: numberParam(params, "queryY", 0.4),
      label: "query",
    };
    const neighbors = [...dataset.points]
      .sort((a, b) => distance(a, query, metric) - distance(b, query, metric))
      .slice(0, k);
    const positiveVotes = neighbors.filter((point) => classValue(point) === 1).length;
    const prediction = positiveVotes >= k / 2 ? "positive" : "negative";

    return conceptResult(
      {
        type: "concept-demo",
        iteration: k,
        points: [...dataset.points, query],
        series: neighbors.map((neighbor) => ({
          label: "neighbor link",
          color: "#b7791f",
          dashed: true,
          points: [query, neighbor],
        })),
        summary: `${prediction} · ${positiveVotes}/${k} positive votes`,
      },
      [
        { label: "Prediction", value: prediction },
        { label: "Positive votes", value: `${positiveVotes}/${k}` },
        { label: "Metric", value: metric },
        { label: "Neighbors", value: String(k) },
      ],
    );
  },
  python: (params) => `from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=${Math.round(numberParam(params, "k", 5))},
    metric="${stringParam(params, "metric", "euclidean")}",
)
model.fit(X, y)
model.predict([[query_x, query_y]])`,
  javascript: (params) => `const k = ${Math.round(numberParam(params, "k", 5))};
const metric = "${stringParam(params, "metric", "euclidean")}";
const neighbors = points
  .sort((a, b) => distance(a, query, metric) - distance(b, query, metric))
  .slice(0, k);`,
});

const featureSelection = barAlgorithm({
  id: "feature-selection",
  name: "Feature Selection Scores",
  category: "Feature Engineering & Selection",
  summary: "Ranks engineered features by signal strength after preprocessing.",
  parameters: [
    { kind: "range", id: "topK", label: "Top features", min: 2, max: 8, step: 1, defaultValue: 5, format: "integer" },
    {
      kind: "select",
      id: "scaling",
      label: "Scaling",
      defaultValue: "standard",
      options: [
        { label: "Standard", value: "standard" },
        { label: "Min-max", value: "minmax" },
        { label: "Robust", value: "robust" },
      ],
    },
  ],
  bars: (params) => {
    const topK = Math.round(numberParam(params, "topK", 5));
    const scaling = stringParam(params, "scaling", "standard");
    const multiplier = scaling === "robust" ? 0.94 : scaling === "minmax" ? 0.88 : 1;
    return [
      "age",
      "income",
      "income_x_age",
      "region_code",
      "tenure",
      "usage_rate",
      "missing_flag",
      "season",
    ]
      .map((label, index) => ({
        label,
        value: round((0.93 - index * 0.085 + Math.sin(index) * 0.035) * multiplier),
      }))
      .slice(0, topK);
  },
  formulas: [
    { title: "Standard scaling", expression: "z=\\frac{x-\\mu}{\\sigma}" },
    { title: "Mutual information", expression: "I(X;Y)=\\sum p(x,y)\\log\\frac{p(x,y)}{p(x)p(y)}" },
  ],
  explanation: [
    "Feature engineering creates candidate signals; feature selection ranks which ones deserve model capacity.",
    "The bars show relative signal after the selected scaling strategy is applied.",
  ],
  python: (params) => `from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
scores = mutual_info_classif(X_scaled, y)
top_k = scores.argsort()[-${Math.round(numberParam(params, "topK", 5))}:]`,
  javascript: (params) => `const scaling = "${stringParam(params, "scaling", "standard")}";
const topK = ${Math.round(numberParam(params, "topK", 5))};
const scores = features.map((feature) => mutualInformation(feature, labels));
const selected = scores.sort(descending).slice(0, topK);`,
});

const modelEvaluation = barAlgorithm({
  id: "model-evaluation",
  name: "Metrics & Cross-Validation",
  category: "Model Evaluation: Metrics, Cross-Validation",
  summary: "Compares accuracy, precision, recall, and F1 across validation folds.",
  parameters: [
    { kind: "range", id: "threshold", label: "Decision threshold", min: 0.1, max: 0.9, step: 0.05, defaultValue: 0.5, format: "decimal" },
    { kind: "range", id: "folds", label: "CV folds", min: 2, max: 10, step: 1, defaultValue: 5, format: "integer" },
  ],
  bars: (params) => {
    const threshold = numberParam(params, "threshold", 0.5);
    const precision = clamp(0.92 - Math.abs(threshold - 0.66) * 0.7);
    const recall = clamp(0.94 - Math.abs(threshold - 0.34) * 0.85);
    const f1 = (2 * precision * recall) / (precision + recall);
    const accuracy = clamp(0.86 - Math.abs(threshold - 0.52) * 0.3);
    return [
      { label: "accuracy", value: accuracy },
      { label: "precision", value: precision },
      { label: "recall", value: recall },
      { label: "f1", value: f1 },
    ];
  },
  formulas: [
    { title: "Precision", expression: "\\frac{TP}{TP+FP}" },
    { title: "Recall", expression: "\\frac{TP}{TP+FN}" },
    { title: "K-fold CV", expression: "\\frac{1}{K}\\sum_{k=1}^{K}score_k" },
  ],
  explanation: [
    "Evaluation depends on the operating threshold and the validation design.",
    "Cross-validation estimates model stability by measuring performance across multiple train/test splits.",
  ],
  python: (params) => `from sklearn.model_selection import cross_validate

scores = cross_validate(
    model,
    X,
    y,
    cv=${Math.round(numberParam(params, "folds", 5))},
    scoring=["accuracy", "precision", "recall", "f1"],
)`,
  javascript: (params) => `const threshold = ${numberParam(params, "threshold", 0.5).toFixed(2)};
const folds = ${Math.round(numberParam(params, "folds", 5))};
const metrics = crossValidate(model, dataset, folds, threshold);`,
});

const biasVariance = makeConceptAlgorithm({
  id: "bias-variance",
  name: "Bias-Variance Learning Curve",
  category: "Bias, Variance & the Learning Curve",
  summary: "Shows how model complexity trades bias for variance.",
  parameters: [
    { kind: "range", id: "complexity", label: "Model complexity", min: 1, max: 12, step: 1, defaultValue: 5, format: "integer" },
    { kind: "range", id: "noise", label: "Data noise", min: 0, max: 0.5, step: 0.05, defaultValue: 0.18, format: "decimal" },
  ],
  sample: () => makeLearningCurveDataset(),
  formulas: [
    { title: "Expected error", expression: "E[(y-\\hat{f})^2]=Bias^2+Variance+Noise" },
    { title: "Generalization gap", expression: "J_{val}-J_{train}" },
  ],
  explanation: [
    "Low-complexity models tend to underfit; high-complexity models can memorize noise.",
    "The validation curve usually bottoms out between those extremes.",
  ],
  engine: (_, params) => {
    const complexity = Math.round(numberParam(params, "complexity", 5));
    const noise = numberParam(params, "noise", 0.18);
    const train = Array.from({ length: 12 }, (_, index) => {
      const x = index + 1;
      return { x, y: clamp(0.74 / (x + 0.5) + noise * 0.35) };
    });
    const validation = Array.from({ length: 12 }, (_, index) => {
      const x = index + 1;
      return { x, y: clamp(0.22 + 0.42 / (x + 0.4) + (x / 12) ** 2 * 0.38 + noise * 0.5) };
    });
    const selected = validation[complexity - 1];

    return conceptResult(
      {
        type: "concept-demo",
        iteration: complexity,
        points: [...train, ...validation],
        series: [
          { label: "training error", points: train, color: "#0f766e" },
          { label: "validation error", points: validation, color: "#d34a43" },
        ],
        markers: [{ axis: "x", value: complexity, label: "chosen complexity" }],
        summary: `Validation error ${selected.y.toFixed(3)} at complexity ${complexity}`,
      },
      [
        { label: "Complexity", value: String(complexity) },
        { label: "Val error", value: selected.y.toFixed(3) },
        { label: "Noise", value: noise.toFixed(2) },
        { label: "Gap", value: (selected.y - train[complexity - 1].y).toFixed(3) },
      ],
    );
  },
  python: (params) => `from sklearn.model_selection import validation_curve

train_scores, val_scores = validation_curve(
    estimator,
    X,
    y,
    param_name="max_depth",
    param_range=range(1, 13),
)
chosen_complexity = ${Math.round(numberParam(params, "complexity", 5))}`,
  javascript: (params) => `const complexity = ${Math.round(numberParam(params, "complexity", 5))};
const validation = complexities.map((c) => estimateValidationError(c, noise));
const best = argmin(validation);`,
});

const ensembleMethods = barAlgorithm({
  id: "ensemble-methods",
  name: "Ensemble Comparison",
  category: "Ensemble Methods: Boosting, Bagging, Stacking",
  summary: "Compares base learners with bagging, boosting, and stacking.",
  parameters: [
    { kind: "range", id: "estimators", label: "Estimators", min: 5, max: 100, step: 5, defaultValue: 40, format: "integer" },
    { kind: "range", id: "diversity", label: "Learner diversity", min: 0, max: 1, step: 0.05, defaultValue: 0.55, format: "decimal" },
  ],
  bars: (params) => {
    const estimators = numberParam(params, "estimators", 40);
    const diversity = numberParam(params, "diversity", 0.55);
    const lift = Math.log1p(estimators) / 18;
    return [
      { label: "base", value: 0.68 },
      { label: "bagging", value: clamp(0.7 + lift + diversity * 0.1) },
      { label: "boosting", value: clamp(0.72 + lift * 1.2 + (1 - diversity) * 0.05) },
      { label: "stacking", value: clamp(0.74 + lift + diversity * 0.12) },
    ];
  },
  formulas: [
    { title: "Bagging", expression: "\\hat{f}(x)=\\frac{1}{B}\\sum_{b=1}^{B}f_b(x)" },
    { title: "Boosting", expression: "F_m(x)=F_{m-1}(x)+\\eta h_m(x)" },
  ],
  explanation: [
    "Ensembles combine multiple learners to reduce variance, bias, or both.",
    "Diversity helps bagging and stacking; boosting focuses each learner on previous errors.",
  ],
  python: (params) => `from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

bagging = RandomForestClassifier(n_estimators=${Math.round(numberParam(params, "estimators", 40))})
boosting = GradientBoostingClassifier(n_estimators=${Math.round(numberParam(params, "estimators", 40))})
bagging.fit(X, y)
boosting.fit(X, y)`,
  javascript: (params) => `const estimators = ${Math.round(numberParam(params, "estimators", 40))};
const bagged = bootstrapModels(baseLearner, estimators);
const boosted = fitBoostingRounds(baseLearner, estimators);`,
});

const hyperparameterTuning = barAlgorithm({
  id: "hyperparameter-tuning",
  name: "Hyperparameter Search",
  category: "Hyperparameter Tuning",
  summary: "Searches candidate settings and ranks validation scores.",
  parameters: [
    { kind: "range", id: "trials", label: "Trials", min: 4, max: 18, step: 1, defaultValue: 9, format: "integer" },
    {
      kind: "select",
      id: "strategy",
      label: "Search strategy",
      defaultValue: "grid",
      options: [
        { label: "Grid", value: "grid" },
        { label: "Random", value: "random" },
        { label: "Bayesian", value: "bayesian" },
      ],
    },
  ],
  bars: (params) => {
    const trials = Math.round(numberParam(params, "trials", 9));
    const strategy = stringParam(params, "strategy", "grid");
    const bonus = strategy === "bayesian" ? 0.07 : strategy === "random" ? 0.03 : 0;
    return Array.from({ length: trials }, (_, index) => ({
      label: `trial ${index + 1}`,
      value: clamp(0.55 + Math.sin(index * 1.8) * 0.08 + Math.log1p(index + 1) * 0.08 + bonus),
    }));
  },
  formulas: [
    { title: "Search objective", expression: "\\theta^*=\\arg\\max_\\theta CV(\\theta)" },
    { title: "Validation score", expression: "CV(\\theta)=\\frac{1}{K}\\sum_k score_k(\\theta)" },
  ],
  explanation: [
    "Hyperparameter tuning treats model settings as a search space.",
    "The best setting is chosen by validation score, not training score.",
  ],
  python: (params) => `from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

search = GridSearchCV(model, param_grid, cv=5)
search.fit(X, y)
print(search.best_params_)  # ${stringParam(params, "strategy", "grid")} search idea`,
  javascript: (params) => `const strategy = "${stringParam(params, "strategy", "grid")}";
const trials = ${Math.round(numberParam(params, "trials", 9))};
const candidates = proposeCandidates(searchSpace, strategy, trials);
const best = candidates.sort(byValidationScore)[0];`,
});

const naiveBayes = barAlgorithm({
  id: "naive-bayes",
  name: "Naive Bayes Posterior",
  category: "Naive Bayes",
  summary: "Combines feature likelihoods into class posterior probabilities.",
  parameters: [
    { kind: "range", id: "smoothing", label: "Laplace smoothing", min: 0, max: 2, step: 0.1, defaultValue: 1, format: "decimal" },
    { kind: "range", id: "positivePrior", label: "Positive prior", min: 0.1, max: 0.9, step: 0.05, defaultValue: 0.5, format: "decimal" },
  ],
  bars: (params) => {
    const smoothing = numberParam(params, "smoothing", 1);
    const prior = numberParam(params, "positivePrior", 0.5);
    const positive = clamp(prior * 0.8 + 0.25 - smoothing * 0.035);
    return [
      { label: "negative", value: 1 - positive },
      { label: "positive", value: positive },
    ];
  },
  formulas: [
    { title: "Bayes rule", expression: "P(y|x)=\\frac{P(y)\\prod_jP(x_j|y)}{P(x)}" },
    { title: "Laplace smoothing", expression: "\\frac{count(x,y)+\\alpha}{count(y)+\\alpha V}" },
  ],
  explanation: [
    "Naive Bayes assumes features are conditionally independent given the class.",
    "Smoothing prevents unseen feature values from zeroing out the whole posterior.",
  ],
  python: (params) => `from sklearn.naive_bayes import GaussianNB, MultinomialNB

model = MultinomialNB(alpha=${numberParam(params, "smoothing", 1).toFixed(1)})
model.fit(X_counts, y)
model.predict_proba(query)`,
  javascript: (params) => `const alpha = ${numberParam(params, "smoothing", 1).toFixed(1)};
const prior = ${numberParam(params, "positivePrior", 0.5).toFixed(2)};
const posterior = normalize(prior * likelihoods.map((p) => smooth(p, alpha)));`,
});

const anomalyDetection = makeConceptAlgorithm({
  id: "anomaly-detection",
  name: "Anomaly Score",
  category: "Anomaly Detection",
  summary: "Scores unusual points by distance from normal behavior.",
  parameters: [
    { kind: "range", id: "sensitivity", label: "Sensitivity", min: 0.5, max: 2.5, step: 0.1, defaultValue: 1.3, format: "decimal" },
    { kind: "range", id: "contamination", label: "Expected anomalies", min: 0.01, max: 0.25, step: 0.01, defaultValue: 0.08, format: "percent" },
  ],
  sample: () => makeAnomalyDataset(),
  formulas: [
    { title: "Z score", expression: "z=\\frac{x-\\mu}{\\sigma}" },
    { title: "Distance score", expression: "s_i=\\lVert x_i-\\mu\\rVert_2" },
  ],
  explanation: [
    "Anomaly detection flags observations that are unlike the normal cluster.",
    "Higher sensitivity lowers the score threshold and marks more points as unusual.",
  ],
  engine: (dataset, params) => {
    const sensitivity = numberParam(params, "sensitivity", 1.3);
    const center = mean(dataset.points);
    const scores = dataset.points.map((point) => distance(point, center, "euclidean"));
    const threshold = average(scores) + (2.3 - sensitivity * 0.45) * standardDeviation(scores);
    const points = dataset.points.map((point, index) => ({
      ...point,
      label: scores[index] > threshold ? "anomaly" : "normal",
    }));
    const anomalyCount = points.filter((point) => point.label === "anomaly").length;

    return conceptResult(
      {
        type: "concept-demo",
        iteration: Math.round(sensitivity * 10),
        points,
        summary: `${anomalyCount} anomalies above score threshold ${threshold.toFixed(2)}`,
      },
      [
        { label: "Anomalies", value: String(anomalyCount) },
        { label: "Threshold", value: threshold.toFixed(2) },
        { label: "Sensitivity", value: sensitivity.toFixed(1) },
        { label: "Method", value: "distance" },
      ],
    );
  },
  python: (params) => `from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=${numberParam(params, "contamination", 0.08).toFixed(2)},
    random_state=7,
)
labels = model.fit_predict(X)`,
  javascript: (params) => `const sensitivity = ${numberParam(params, "sensitivity", 1.3).toFixed(1)};
const scores = points.map((point) => euclideanDistance(point, center));
const threshold = mean(scores) + sensitivity * std(scores);
const labels = scores.map((score) => score > threshold ? "anomaly" : "normal");`,
});

const imbalancedData = barAlgorithm({
  id: "imbalanced-data",
  name: "Imbalanced Data Thresholds",
  category: "Handling Imbalanced Data",
  summary: "Tunes threshold and class weighting for rare positive classes.",
  parameters: [
    { kind: "range", id: "threshold", label: "Threshold", min: 0.05, max: 0.95, step: 0.05, defaultValue: 0.35, format: "decimal" },
    { kind: "range", id: "positiveWeight", label: "Positive class weight", min: 1, max: 12, step: 1, defaultValue: 5, format: "integer" },
  ],
  bars: (params) => {
    const threshold = numberParam(params, "threshold", 0.35);
    const weight = numberParam(params, "positiveWeight", 5);
    const recall = clamp(0.98 - threshold * 0.78 + weight * 0.01);
    const precision = clamp(0.24 + threshold * 0.7 - weight * 0.006);
    const f1 = (2 * precision * recall) / (precision + recall);
    const balancedAccuracy = clamp((recall + (0.66 + threshold * 0.25)) / 2);
    return [
      { label: "precision", value: precision },
      { label: "recall", value: recall },
      { label: "f1", value: f1 },
      { label: "balanced acc", value: balancedAccuracy },
    ];
  },
  formulas: [
    { title: "Class weight", expression: "w_c=\\frac{n}{K\\cdot n_c}" },
    { title: "Balanced accuracy", expression: "\\frac{TPR+TNR}{2}" },
  ],
  explanation: [
    "In imbalanced data, plain accuracy can hide poor minority-class performance.",
    "Class weights and thresholds let you trade precision for recall explicitly.",
  ],
  python: (params) => `from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight={1: ${Math.round(numberParam(params, "positiveWeight", 5))}})
model.fit(X, y)
pred = model.predict_proba(X)[:, 1] >= ${numberParam(params, "threshold", 0.35).toFixed(2)}`,
  javascript: (params) => `const threshold = ${numberParam(params, "threshold", 0.35).toFixed(2)};
const positiveWeight = ${Math.round(numberParam(params, "positiveWeight", 5))};
const weightedLoss = examples.map((row) => classWeight(row.y, positiveWeight) * logLoss(row));`,
});

const timeSeries = makeConceptAlgorithm({
  id: "time-series",
  name: "Moving Average Forecast",
  category: "Time Series Fundamentals",
  summary: "Separates trend and seasonality with a simple rolling forecast.",
  parameters: [
    { kind: "range", id: "window", label: "Window size", min: 2, max: 12, step: 1, defaultValue: 5, format: "integer" },
    { kind: "range", id: "horizon", label: "Forecast horizon", min: 2, max: 14, step: 1, defaultValue: 7, format: "integer" },
  ],
  sample: () => makeTimeSeriesDataset(),
  formulas: [
    { title: "Lag feature", expression: "x_t=[y_{t-1}, y_{t-2}, ...]" },
    { title: "Moving average", expression: "\\hat{y}_{t+1}=\\frac{1}{w}\\sum_{i=0}^{w-1}y_{t-i}" },
  ],
  explanation: [
    "Time series models must respect temporal order.",
    "A moving average smooths noise and gives a baseline forecast before richer ARIMA or neural methods.",
  ],
  engine: (dataset, params) => {
    const window = Math.round(numberParam(params, "window", 5));
    const horizon = Math.round(numberParam(params, "horizon", 7));
    const observed = dataset.points;
    const smoothed = observed.map((point, index) => ({
      x: point.x,
      y: average(observed.slice(Math.max(0, index - window + 1), index + 1).map((item) => item.y)),
    }));
    const future = Array.from({ length: horizon }, (_, index) => {
      const x = observed.length + index;
      const trend = smoothed[smoothed.length - 1].y + index * 0.08;
      const seasonal = Math.sin((x / 4) * Math.PI) * 0.35;
      return { x, y: trend + seasonal };
    });

    return conceptResult(
      {
        type: "concept-demo",
        iteration: horizon,
        points: [...observed, ...future],
        series: [
          { label: "observed", points: observed, color: "#2f6fbe" },
          { label: "moving average", points: smoothed, color: "#0f766e" },
          { label: "forecast", points: future, color: "#d34a43", dashed: true },
        ],
        markers: [{ axis: "x", value: observed[observed.length - 1].x, label: "forecast starts" }],
        summary: `${horizon}-step forecast · window ${window}`,
      },
      [
        { label: "Window", value: String(window) },
        { label: "Horizon", value: String(horizon) },
        { label: "Last value", value: observed[observed.length - 1].y.toFixed(2) },
        { label: "Next forecast", value: future[0].y.toFixed(2) },
      ],
    );
  },
  python: (params) => `series = dataset["y"]
window = ${Math.round(numberParam(params, "window", 5))}
moving_average = series.rolling(window=window).mean()
forecast = [moving_average.iloc[-1]] * ${Math.round(numberParam(params, "horizon", 7))}`,
  javascript: (params) => `const windowSize = ${Math.round(numberParam(params, "window", 5))};
const horizon = ${Math.round(numberParam(params, "horizon", 7))};
const movingAverage = rollingMean(series, windowSize);
const forecast = extrapolate(movingAverage, horizon);`,
});

const stochasticStates = [
  { id: "word-a", label: "Word A", color: "#2f6fbe" },
  { id: "word-b", label: "Word B", color: "#0f766e" },
  { id: "word-c", label: "Word C", color: "#d34a43" },
];

const defaultTransitionMatrix = [
  [0.12, 0.68, 0.2],
  [0.34, 0.16, 0.5],
  [0.52, 0.28, 0.2],
];

const stochasticProcesses = makeConceptAlgorithm({
  id: "stochastic-processes",
  name: "Stochastic Processes",
  category: "Stochastic Processes",
  summary: "Animates random-walk trajectories and Markov transition sampling.",
  parameters: [
    {
      kind: "range",
      id: "drift",
      label: "Drift mu",
      min: -0.08,
      max: 0.08,
      step: 0.005,
      defaultValue: 0.015,
      format: "decimal",
    },
    {
      kind: "range",
      id: "volatility",
      label: "Volatility sigma",
      min: 0.05,
      max: 0.65,
      step: 0.01,
      defaultValue: 0.22,
      format: "decimal",
    },
    {
      kind: "stepper",
      id: "pathCount",
      label: "Particle paths",
      min: 12,
      max: 48,
      step: 4,
      defaultValue: 28,
      format: "integer",
    },
    {
      kind: "range",
      id: "timeSteps",
      label: "Timeline steps",
      min: 24,
      max: 120,
      step: 4,
      defaultValue: 72,
      format: "integer",
    },
    {
      kind: "matrix",
      id: "transitionMatrix",
      label: "Transition matrix P",
      rowLabels: ["A", "B", "C"],
      columnLabels: ["A", "B", "C"],
      defaultValue: defaultTransitionMatrix,
      min: 0,
      max: 1,
      step: 0.05,
      format: "decimal",
    },
    {
      kind: "action",
      id: "markovStep",
      label: "Markov sample",
      buttonLabel: "Step",
      min: 0,
      max: 999,
      step: 1,
      defaultValue: 0,
      format: "integer",
    },
  ],
  sample: makeStochasticDataset,
  formulas: [
    { title: "Random walk", expression: "X_{t+1}=X_t+\\mu+\\sigma\\epsilon_t,\\quad \\epsilon_t\\sim\\mathcal{N}(0,1)" },
    { title: "Markov transition", expression: "\\Pr(S_{t+1}=j\\mid S_t=i)=P_{ij}" },
    { title: "State distribution", expression: "\\pi_{t+1}=\\pi_tP" },
  ],
  explanation: [
    "A stochastic process is a time-indexed system whose future includes randomness instead of a single deterministic path.",
    "Drift shifts the expected direction of a random walk, while volatility controls how rapidly many sample paths fan out.",
    "A Markov transition matrix stores the probability of hopping from one state to the next; the next token depends on the current state and one row of probabilities.",
  ],
  engine: (_, params) => {
    const drift = numberParam(params, "drift", 0.015);
    const volatility = numberParam(params, "volatility", 0.22);
    const pathCount = Math.round(numberParam(params, "pathCount", 28));
    const timeSteps = Math.round(numberParam(params, "timeSteps", 72));
    const markovStep = Math.round(numberParam(params, "markovStep", 0));
    const transitionMatrix = matrixParam(params, "transitionMatrix", defaultTransitionMatrix);
    const normalizedTransitionMatrix = normalizeTransitionRows(transitionMatrix);
    const fullPaths = makeStochasticPaths(pathCount, timeSteps, drift, volatility);
    const markovSequence = makeMarkovSequence(
      normalizedTransitionMatrix,
      markovStep + Math.ceil(timeSteps / 4) + 16,
      stochasticSeed(drift, volatility, pathCount, timeSteps) + markovStep * 17,
    );
    const frameCount = Math.min(80, timeSteps + 1);
    const terminalValues = fullPaths.map((path) => path.points[path.points.length - 1].y);
    const terminalMean = average(terminalValues);
    const terminalStd = standardDeviation(terminalValues);

    const frames = Array.from({ length: frameCount }, (_, frameIndex) => {
      const progress = frameCount === 1 ? 1 : frameIndex / (frameCount - 1);
      const visibleStep = Math.min(timeSteps, Math.round(progress * timeSteps));
      const markovIndex = markovStep + Math.floor(frameIndex / 8);
      const currentState = markovSequence[markovIndex] ?? 0;
      const nextState = markovSequence[markovIndex + 1] ?? currentState;
      const pulseProgress = (frameIndex % 8) / 7;
      const paths = fullPaths.map((path) => {
        const points = path.points.slice(0, visibleStep + 1);
        return {
          ...path,
          points,
          current: points[points.length - 1] ?? path.points[0],
        };
      });

      return {
        type: "concept-demo" as const,
        iteration: visibleStep,
        points: paths.map((path) => ({
          x: path.current.t,
          y: path.current.y,
          label: path.id,
        })),
        stochastic: {
          drift,
          volatility,
          visibleStep,
          timeSteps,
          pathCount,
          paths,
          terminalMean,
          terminalStd,
          states: stochasticStates,
          transitionMatrix,
          normalizedTransitionMatrix,
          currentState,
          nextState,
          pulseProgress,
          markovStep: markovIndex,
          sequence: markovSequence.slice(0, markovIndex + 2),
          entropy: transitionEntropy(normalizedTransitionMatrix),
        },
        summary: `t=${visibleStep}/${timeSteps} · ${stochasticStates[currentState]?.label ?? "Word A"} -> ${stochasticStates[nextState]?.label ?? "Word A"} · spread ${terminalStd.toFixed(2)}`,
      };
    });

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Paths", value: pathCount.toLocaleString() },
        { label: "Terminal mean", value: terminalMean.toFixed(2) },
        { label: "Terminal spread", value: terminalStd.toFixed(2) },
        { label: "Matrix entropy", value: transitionEntropy(normalizedTransitionMatrix).toFixed(2) },
      ],
    };
  },
  python: (params) => {
    const drift = numberParam(params, "drift", 0.015);
    const volatility = numberParam(params, "volatility", 0.22);
    const pathCount = Math.round(numberParam(params, "pathCount", 28));
    const timeSteps = Math.round(numberParam(params, "timeSteps", 72));
    const matrix = matrixParam(params, "transitionMatrix", defaultTransitionMatrix);

    return `import numpy as np

mu = ${drift.toFixed(3)}
sigma = ${volatility.toFixed(3)}
n_paths = ${pathCount}
steps = ${timeSteps}
P = np.array(${JSON.stringify(matrix)}, dtype=float)
P = P / P.sum(axis=1, keepdims=True)

paths = []
for path_id in range(n_paths):
    x = [0.0]
    for t in range(steps):
        noise = np.random.normal(0, 1)
        x.append(x[-1] + mu + sigma * noise)
    paths.append(x)

tokens = np.array(["Word A", "Word B", "Word C"])
state = 0
sequence = [tokens[state]]
for _ in range(16):
    state = np.random.choice(len(tokens), p=P[state])
    sequence.append(tokens[state])`;
  },
  javascript: (params) => {
    const drift = numberParam(params, "drift", 0.015);
    const volatility = numberParam(params, "volatility", 0.22);
    const pathCount = Math.round(numberParam(params, "pathCount", 28));
    const timeSteps = Math.round(numberParam(params, "timeSteps", 72));
    const matrix = matrixParam(params, "transitionMatrix", defaultTransitionMatrix);

    return `const mu = ${drift.toFixed(3)};
const sigma = ${volatility.toFixed(3)};
const pathCount = ${pathCount};
const steps = ${timeSteps};
const P = normalizeRows(${JSON.stringify(matrix)});

function weightedIndex(weights, random = Math.random()) {
  let cumulative = 0;
  for (let index = 0; index < weights.length; index += 1) {
    cumulative += weights[index];
    if (random <= cumulative) return index;
  }
  return weights.length - 1;
}

const paths = Array.from({ length: pathCount }, () => {
  const x = [0];
  for (let t = 0; t < steps; t += 1) {
    x.push(x.at(-1) + mu + sigma * gaussianNoise());
  }
  return x;
});

const tokens = ["Word A", "Word B", "Word C"];
let state = 0;
const sequence = [tokens[state]];
for (let step = 0; step < 16; step += 1) {
  state = weightedIndex(P[state]);
  sequence.push(tokens[state]);
}`;
  },
});

function makeStochasticDataset() {
  return makeDataset(
    "Generated stochastic process sample",
    Array.from({ length: 32 }, (_, index) => ({
      x: index,
      y: Math.sin(index * 0.34) * 0.35,
      label: index % 3 === 0 ? "Word A" : index % 3 === 1 ? "Word B" : "Word C",
    })),
  );
}

function makeStochasticPaths(
  pathCount: number,
  timeSteps: number,
  drift: number,
  volatility: number,
) {
  const rng = makeSeededRandom(stochasticSeed(drift, volatility, pathCount, timeSteps));

  return Array.from({ length: pathCount }, (_, pathIndex) => {
    const points = [{ t: 0, y: 0 }];
    let current = 0;

    for (let step = 1; step <= timeSteps; step += 1) {
      current += drift + volatility * gaussianNoise(rng);
      points.push({ t: step, y: round(current, 4) });
    }

    return {
      id: `path-${pathIndex + 1}`,
      color: colors[pathIndex % colors.length],
      points,
      current: points[0],
      terminal: points[points.length - 1].y,
    };
  });
}

function stochasticSeed(
  drift: number,
  volatility: number,
  pathCount: number,
  timeSteps: number,
) {
  return Math.abs(
    Math.round((drift + 1) * 100_000) +
      Math.round(volatility * 1_000_000) +
      pathCount * 997 +
      timeSteps * 389,
  );
}

function makeSeededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussianNoise(rng: () => number) {
  const u1 = Math.max(1e-9, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(Math.PI * 2 * u2);
}

function normalizeTransitionRows(matrix: number[][]) {
  return matrix.map((row) => {
    const total = row.reduce((sum, value) => sum + Math.max(0, value), 0);
    if (total <= 0) {
      return row.map(() => 1 / row.length);
    }

    return row.map((value) => Math.max(0, value) / total);
  });
}

function makeMarkovSequence(matrix: number[][], length: number, seed: number) {
  const rng = makeSeededRandom(seed);
  const sequence = [0];

  for (let index = 1; index < length; index += 1) {
    const current = sequence[sequence.length - 1];
    sequence.push(weightedRandomIndex(matrix[current] ?? matrix[0], rng()));
  }

  return sequence;
}

function weightedRandomIndex(weights: number[], sample: number) {
  let cumulative = 0;
  for (let index = 0; index < weights.length; index += 1) {
    cumulative += weights[index];
    if (sample <= cumulative) {
      return index;
    }
  }

  return Math.max(0, weights.length - 1);
}

function transitionEntropy(matrix: number[][]) {
  const rowEntropies = matrix.map((row) =>
    row.reduce((total, value) => (value > 0 ? total - value * Math.log2(value) : total), 0),
  );
  return average(rowEntropies);
}

const defaultGridWorld = makeDefaultGridWorld();

const dynamicProgramming = makeConceptAlgorithm({
  id: "dynamic-programming",
  name: "Dynamic Programming",
  category: "Dynamic Programming",
  summary: "Propagates Gridworld rewards with Bellman value updates and extracts an optimal policy.",
  parameters: [
    {
      kind: "gridworld",
      id: "gridWorld",
      label: "Gridworld layout",
      defaultValue: defaultGridWorld,
    },
    {
      kind: "select",
      id: "paintTool",
      label: "Paint tool",
      defaultValue: "wall",
      options: [
        { label: "Wall", value: "wall" },
        { label: "Fire pit", value: "fire" },
        { label: "Gold chest", value: "gold" },
        { label: "Start", value: "start" },
        { label: "Open cell", value: "empty" },
      ],
    },
    {
      kind: "range",
      id: "discount",
      label: "Discount factor gamma",
      min: 0.5,
      max: 0.98,
      step: 0.01,
      defaultValue: 0.9,
      format: "decimal",
    },
    {
      kind: "select",
      id: "dpMethod",
      label: "DP method",
      defaultValue: "value-iteration",
      options: [
        { label: "Value Iteration", value: "value-iteration" },
        { label: "Policy Iteration", value: "policy-iteration" },
      ],
    },
    {
      kind: "action",
      id: "dpStep",
      label: "Bellman sweep",
      buttonLabel: "Step",
      min: 0,
      max: 80,
      step: 1,
      defaultValue: 0,
      format: "integer",
    },
  ],
  sample: makeDynamicProgrammingDataset,
  formulas: [
    { title: "Bellman optimality", expression: "V_{k+1}(s)=\\max_a\\sum_{s'}P(s'\\mid s,a)[R(s,a,s')+\\gamma V_k(s')]" },
    { title: "Policy evaluation", expression: "V^{\\pi}_{k+1}(s)=\\sum_{s'}P(s'\\mid s,\\pi(s))[R(s,\\pi(s),s')+\\gamma V^{\\pi}_k(s')]" },
    { title: "Policy extraction", expression: "\\pi^*(s)=\\arg\\max_a\\sum_{s'}P(s'\\mid s,a)[R(s,a,s')+\\gamma V(s')]" },
  ],
  explanation: [
    "Dynamic programming solves a known Markov decision process by repeatedly applying Bellman updates to a value matrix.",
    "In Gridworld, each coordinate is a state. Walls block movement, fire pits are negative terminal rewards, and gold chests are positive terminal rewards.",
    "As sweeps accumulate, reward information propagates outward through the grid; once the values stabilize, the best neighboring action becomes the optimal policy arrow.",
  ],
  engine: (_, params) => {
    const grid = gridWorldParam(params, "gridWorld", defaultGridWorld);
    const gamma = Math.max(0.5, Math.min(0.98, numberParam(params, "discount", 0.9)));
    const method = dynamicProgrammingMethodParam(params, "dpMethod", "value-iteration");
    const baseSweep = Math.max(0, Math.min(80, Math.round(numberParam(params, "dpStep", 0))));
    const playSignal = Math.round(numberParam(params, "playSignal", 0));
    const shouldPlay = playSignal > baseSweep;
    const frameCount = shouldPlay ? Math.max(2, Math.min(44, 81 - baseSweep)) : 1;
    const maxSweep = Math.min(80, baseSweep + frameCount - 1);
    const snapshots = runDynamicProgrammingGrid(grid, gamma, method, maxSweep);
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const snapshot = snapshots[Math.min(maxSweep, baseSweep + index)] ?? snapshots[snapshots.length - 1];
      const openCells = grid.cells.flat().filter((cell) => cell !== "wall").length;

      return {
        type: "concept-demo" as const,
        iteration: snapshot.sweep + 1,
        points: snapshot.values.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => ({
            x: columnIndex,
            y: rowIndex,
            label: grid.cells[rowIndex]?.[columnIndex] ?? "empty",
            value,
          })),
        ),
        dynamicProgramming: {
          grid,
          method,
          gamma,
          sweep: snapshot.sweep,
          values: snapshot.values,
          previousValues: snapshot.previousValues,
          delta: snapshot.delta,
          stable: snapshot.stable,
          activeCell: snapshot.activeCell,
          actionValues: snapshot.actionValues,
          policy: snapshot.policy,
        },
        summary: `${dynamicProgrammingMethodLabel(method)} · sweep ${snapshot.sweep} · delta ${snapshot.delta.toFixed(3)} · ${snapshot.stable ? "policy stable" : `${openCells} states`}`,
      };
    });
    const final = frames[frames.length - 1].dynamicProgramming;
    const counts = countGridCells(grid);

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Method", value: dynamicProgrammingMethodLabel(method) },
        { label: "Discount", value: gamma.toFixed(2) },
        { label: "Delta", value: final.delta.toFixed(3) },
        { label: "Terminals", value: `${counts.gold + counts.fire}` },
      ],
    };
  },
  python: (params) => {
    const gamma = numberParam(params, "discount", 0.9).toFixed(2);
    const method = dynamicProgrammingMethodParam(params, "dpMethod", "value-iteration");

    return `import numpy as np

gamma = ${gamma}
method = "${method}"
actions = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
grid = np.array(${formatGridWorldLiteral(gridWorldParam(params, "gridWorld", defaultGridWorld))})
V = np.zeros(grid.shape, dtype=float)
V[grid == "gold"] = 10
V[grid == "fire"] = -10
policy = np.full(grid.shape, "right", dtype=object)

def move(r, c, action):
    dr, dc = actions[action]
    nr, nc = r + dr, c + dc
    if nr < 0 or nc < 0 or nr >= grid.shape[0] or nc >= grid.shape[1]:
        return r, c
    if grid[nr, nc] == "wall":
        return r, c
    return nr, nc

def reward(cell):
    if cell == "gold":
        return 10
    if cell == "fire":
        return -10
    return -0.04

def transition_matrix(action):
    side = {
        "up": ("left", "right"),
        "right": ("up", "down"),
        "down": ("left", "right"),
        "left": ("up", "down"),
    }
    return [(action, 0.8), (side[action][0], 0.1), (side[action][1], 0.1)]

def q_value(r, c, action, values):
    total = 0.0
    for candidate, prob in transition_matrix(action):
        nr, nc = move(r, c, candidate)
        terminal = grid[nr, nc] in ("gold", "fire")
        total += prob * (reward(grid[nr, nc]) + gamma * (0 if terminal else values[nr, nc]))
    return total

for sweep in range(40):
    next_V = V.copy()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] in ("wall", "gold", "fire"):
                continue
            if method == "value-iteration":
                next_V[r, c] = max(q_value(r, c, a, V) for a in actions)
            else:
                next_V[r, c] = q_value(r, c, policy[r, c], V)
    V = next_V
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] not in ("wall", "gold", "fire"):
                policy[r, c] = max(actions, key=lambda a: q_value(r, c, a, V))`;
  },
  javascript: (params) => {
    const gamma = numberParam(params, "discount", 0.9).toFixed(2);
    const method = dynamicProgrammingMethodParam(params, "dpMethod", "value-iteration");

    return `const gamma = ${gamma};
const method = "${method}";
const actions = {
  up: [-1, 0],
  right: [0, 1],
  down: [1, 0],
  left: [0, -1],
};
const grid = ${formatGridWorldLiteral(gridWorldParam(params, "gridWorld", defaultGridWorld))};
let V = grid.map((row) =>
  row.map((cell) => cell === "gold" ? 10 : cell === "fire" ? -10 : 0),
);
let policy = grid.map((row) => row.map(() => "right"));

for (let sweep = 0; sweep < 40; sweep += 1) {
  const nextV = V.map((row) => [...row]);
  for (let r = 0; r < grid.length; r += 1) {
    for (let c = 0; c < grid[0].length; c += 1) {
      if (["wall", "gold", "fire"].includes(grid[r][c])) continue;
      if (method === "value-iteration") {
        nextV[r][c] = Math.max(...Object.keys(actions).map((a) => qValue(r, c, a, V)));
      } else {
        nextV[r][c] = qValue(r, c, policy[r][c], V);
      }
    }
  }
  V = nextV;
  policy = improvePolicy(V);
}

// For larger mazes, run the same loop in a Web Worker and post V back to the canvas.
function qValue(r, c, action, values) {
  return transitionMatrix(action).reduce((total, [candidate, prob]) => {
    const [nr, nc] = move(r, c, candidate);
    const terminal = grid[nr][nc] === "gold" || grid[nr][nc] === "fire";
    return total + prob * (reward(grid[nr][nc]) + gamma * (terminal ? 0 : values[nr][nc]));
  }, 0);
}

function transitionMatrix(action) {
  const side = {
    up: ["left", "right"],
    right: ["up", "down"],
    down: ["left", "right"],
    left: ["up", "down"],
  };
  return [[action, 0.8], [side[action][0], 0.1], [side[action][1], 0.1]];
}

function move(r, c, action) {
  const [dr, dc] = actions[action];
  const nr = r + dr;
  const nc = c + dc;
  if (nr < 0 || nc < 0 || nr >= grid.length || nc >= grid[0].length) return [r, c];
  return grid[nr][nc] === "wall" ? [r, c] : [nr, nc];
}

function reward(cell) {
  if (cell === "gold") return 10;
  if (cell === "fire") return -10;
  return -0.04;
}

function improvePolicy(values) {
  return grid.map((row, r) =>
    row.map((cell, c) => {
      if (["wall", "gold", "fire"].includes(cell)) return "right";
      return Object.keys(actions).sort((a, b) => qValue(r, c, b, values) - qValue(r, c, a, values))[0];
    }),
  );
}`;
  },
});

type DynamicProgrammingSnapshot = {
  sweep: number;
  values: number[][];
  previousValues: number[][];
  delta: number;
  stable: boolean;
  activeCell: {
    row: number;
    column: number;
  };
  actionValues: Array<{ action: DynamicProgrammingAction; label: string; value: number }>;
  policy: Array<{ row: number; column: number; action: DynamicProgrammingAction; value: number }>;
};

const dynamicProgrammingActions: Array<{
  id: DynamicProgrammingAction;
  label: string;
  delta: [number, number];
}> = [
  { id: "up", label: "Up", delta: [-1, 0] },
  { id: "right", label: "Right", delta: [0, 1] },
  { id: "down", label: "Down", delta: [1, 0] },
  { id: "left", label: "Left", delta: [0, -1] },
];

function makeDefaultGridWorld(): GridWorldValue {
  const cells: GridWorldCell[][] = [
    ["start", "empty", "empty", "wall", "empty", "empty", "empty", "gold"],
    ["empty", "wall", "empty", "wall", "empty", "wall", "empty", "empty"],
    ["empty", "wall", "empty", "empty", "empty", "wall", "empty", "wall"],
    ["empty", "empty", "empty", "wall", "empty", "empty", "empty", "empty"],
    ["wall", "wall", "empty", "wall", "empty", "wall", "wall", "empty"],
    ["fire", "empty", "empty", "empty", "empty", "empty", "wall", "empty"],
    ["empty", "empty", "wall", "wall", "empty", "empty", "empty", "empty"],
    ["empty", "empty", "empty", "fire", "empty", "wall", "empty", "gold"],
  ];

  return {
    kind: "gridworld",
    rows: cells.length,
    columns: cells[0].length,
    cells,
  };
}

function makeDynamicProgrammingDataset() {
  return makeDataset(
    "Generated Gridworld states",
    defaultGridWorld.cells.flatMap((row, rowIndex) =>
      row.map((cell, columnIndex) => ({
        x: columnIndex,
        y: rowIndex,
        label: cell,
      })),
    ),
  );
}

function dynamicProgrammingMethodParam(
  params: ParameterState,
  key: string,
  fallback: DynamicProgrammingMethod,
): DynamicProgrammingMethod {
  const value = stringParam(params, key, fallback);
  return value === "policy-iteration" || value === "value-iteration" ? value : fallback;
}

function runDynamicProgrammingGrid(
  grid: GridWorldValue,
  gamma: number,
  method: DynamicProgrammingMethod,
  maxSweep: number,
) {
  let values: number[][] = initialGridWorldValues(grid);
  let policy = initialGridWorldPolicy(grid);
  const snapshots: DynamicProgrammingSnapshot[] = [
    makeDynamicProgrammingSnapshot(grid, gamma, values, values, policy, 0, 0),
  ];

  for (let sweep = 1; sweep <= maxSweep; sweep += 1) {
    const previousValues = values.map((row) => [...row]);
    if (method === "value-iteration") {
      values = bellmanOptimalitySweep(grid, gamma, previousValues);
    } else {
      values = bellmanPolicyEvaluationSweep(grid, gamma, previousValues, policy);
      policy = greedyGridWorldPolicy(grid, gamma, values);
    }
    if (method === "value-iteration") {
      policy = greedyGridWorldPolicy(grid, gamma, values);
    }
    snapshots.push(makeDynamicProgrammingSnapshot(grid, gamma, values, previousValues, policy, sweep, gridWorldDelta(values, previousValues, grid)));
  }

  return snapshots;
}

function initialGridWorldValues(grid: GridWorldValue) {
  return grid.cells.map((row) =>
    row.map((cell) => {
      if (cell === "gold") return 10;
      if (cell === "fire") return -10;
      return 0;
    }),
  );
}

function initialGridWorldPolicy(grid: GridWorldValue) {
  const goals = grid.cells.flatMap((row, rowIndex) =>
    row.flatMap((cell, columnIndex) => (cell === "gold" ? [{ row: rowIndex, column: columnIndex }] : [])),
  );

  return grid.cells.map((row, rowIndex) =>
    row.map((cell, columnIndex) => {
      if (cell === "wall" || cell === "gold" || cell === "fire") {
        return "right" as DynamicProgrammingAction;
      }
      const nearestGoal = [...goals].sort(
        (a, b) =>
          Math.abs(a.row - rowIndex) + Math.abs(a.column - columnIndex) -
          (Math.abs(b.row - rowIndex) + Math.abs(b.column - columnIndex)),
      )[0];
      if (!nearestGoal) {
        return "right" as DynamicProgrammingAction;
      }
      const rowGap = nearestGoal.row - rowIndex;
      const columnGap = nearestGoal.column - columnIndex;
      if (Math.abs(columnGap) >= Math.abs(rowGap)) {
        return columnGap >= 0 ? "right" : "left";
      }
      return rowGap >= 0 ? "down" : "up";
    }),
  );
}

function bellmanOptimalitySweep(grid: GridWorldValue, gamma: number, values: number[][]) {
  return grid.cells.map((row, rowIndex) =>
    row.map((cell, columnIndex) => {
      if (cell === "wall") return 0;
      if (cell === "gold") return 10;
      if (cell === "fire") return -10;
      return Math.max(
        ...dynamicProgrammingActions.map((action) =>
          dynamicProgrammingActionValue(grid, gamma, values, rowIndex, columnIndex, action.id),
        ),
      );
    }),
  );
}

function bellmanPolicyEvaluationSweep(
  grid: GridWorldValue,
  gamma: number,
  values: number[][],
  policy: DynamicProgrammingAction[][],
) {
  return grid.cells.map((row, rowIndex) =>
    row.map((cell, columnIndex) => {
      if (cell === "wall") return 0;
      if (cell === "gold") return 10;
      if (cell === "fire") return -10;
      return dynamicProgrammingActionValue(grid, gamma, values, rowIndex, columnIndex, policy[rowIndex][columnIndex]);
    }),
  );
}

function greedyGridWorldPolicy(grid: GridWorldValue, gamma: number, values: number[][]) {
  return grid.cells.map((row, rowIndex) =>
    row.map((cell, columnIndex) => {
      if (cell === "wall" || cell === "gold" || cell === "fire") {
        return "right" as DynamicProgrammingAction;
      }
      return bestDynamicProgrammingAction(grid, gamma, values, rowIndex, columnIndex).action;
    }),
  );
}

function makeDynamicProgrammingSnapshot(
  grid: GridWorldValue,
  gamma: number,
  values: number[][],
  previousValues: number[][],
  policy: DynamicProgrammingAction[][],
  sweep: number,
  delta: number,
): DynamicProgrammingSnapshot {
  const states = grid.cells.flatMap((row, rowIndex) =>
    row.flatMap((cell, columnIndex) =>
      cell === "wall" || cell === "gold" || cell === "fire" ? [] : [{ row: rowIndex, column: columnIndex }],
    ),
  );
  const activeCell = states[sweep % Math.max(1, states.length)] ?? { row: 0, column: 0 };
  const actionValues = dynamicProgrammingActions.map((action) => ({
    action: action.id,
    label: action.label,
    value: dynamicProgrammingActionValue(grid, gamma, values, activeCell.row, activeCell.column, action.id),
  }));
  const policyCells = grid.cells.flatMap((row, rowIndex) =>
    row.flatMap((cell, columnIndex) => {
      if (cell === "wall" || cell === "gold" || cell === "fire") {
        return [];
      }
      const action = policy[rowIndex][columnIndex];
      return [{ row: rowIndex, column: columnIndex, action, value: values[rowIndex][columnIndex] }];
    }),
  );

  return {
    sweep,
    values: values.map((row) => row.map((value) => round(value, 3))),
    previousValues: previousValues.map((row) => row.map((value) => round(value, 3))),
    delta: round(delta, 4),
    stable: sweep > 0 && (delta < 0.03 || sweep >= 36),
    activeCell,
    actionValues,
    policy: policyCells,
  };
}

function dynamicProgrammingActionValue(
  grid: GridWorldValue,
  gamma: number,
  values: number[][],
  row: number,
  column: number,
  action: DynamicProgrammingAction,
) {
  return transitionOutcomes(action).reduce((total, outcome) => {
    const next = moveGridWorld(grid, row, column, outcome.action);
    const nextCell = grid.cells[next.row][next.column];
    const terminal = nextCell === "gold" || nextCell === "fire";
    return total + outcome.probability * (gridWorldReward(nextCell) + gamma * (terminal ? 0 : values[next.row][next.column]));
  }, 0);
}

function bestDynamicProgrammingAction(
  grid: GridWorldValue,
  gamma: number,
  values: number[][],
  row: number,
  column: number,
) {
  return dynamicProgrammingActions
    .map((action) => ({
      action: action.id,
      value: dynamicProgrammingActionValue(grid, gamma, values, row, column, action.id),
    }))
    .sort((a, b) => b.value - a.value)[0];
}

function transitionOutcomes(action: DynamicProgrammingAction) {
  const perpendiculars: Record<DynamicProgrammingAction, DynamicProgrammingAction[]> = {
    up: ["left", "right"],
    right: ["up", "down"],
    down: ["left", "right"],
    left: ["up", "down"],
  };
  return [
    { action, probability: 0.8 },
    { action: perpendiculars[action][0], probability: 0.1 },
    { action: perpendiculars[action][1], probability: 0.1 },
  ];
}

function moveGridWorld(
  grid: GridWorldValue,
  row: number,
  column: number,
  action: DynamicProgrammingAction,
) {
  const delta = dynamicProgrammingActions.find((item) => item.id === action)?.delta ?? [0, 0];
  const nextRow = row + delta[0];
  const nextColumn = column + delta[1];
  if (
    nextRow < 0 ||
    nextColumn < 0 ||
    nextRow >= grid.rows ||
    nextColumn >= grid.columns ||
    grid.cells[nextRow][nextColumn] === "wall"
  ) {
    return { row, column };
  }
  return { row: nextRow, column: nextColumn };
}

function gridWorldReward(cell: GridWorldCell) {
  if (cell === "gold") return 10;
  if (cell === "fire") return -10;
  return -0.04;
}

function gridWorldDelta(values: number[][], previousValues: number[][], grid: GridWorldValue) {
  let delta = 0;
  for (let row = 0; row < values.length; row += 1) {
    for (let column = 0; column < values[row].length; column += 1) {
      if (grid.cells[row][column] !== "wall") {
        delta = Math.max(delta, Math.abs(values[row][column] - previousValues[row][column]));
      }
    }
  }
  return delta;
}

function countGridCells(grid: GridWorldValue) {
  return grid.cells.flat().reduce<Record<GridWorldCell, number>>(
    (counts, cell) => ({
      ...counts,
      [cell]: counts[cell] + 1,
    }),
    { empty: 0, wall: 0, fire: 0, gold: 0, start: 0 },
  );
}

function dynamicProgrammingMethodLabel(method: DynamicProgrammingMethod) {
  return method === "policy-iteration" ? "Policy Iteration" : "Value Iteration";
}

function formatGridWorldLiteral(grid: GridWorldValue) {
  return `[
  ${grid.cells.map((row) => `[${row.map((cell) => `"${cell}"`).join(", ")}]`).join(",\n  ")}
]`;
}

const defaultSvdImage = makeDefaultSvdImage(36);

const singularValueDecomposition = makeConceptAlgorithm({
  id: "singular-value-decomposition",
  name: "Singular Value Decomposition",
  category: "Singular Value Decomposition",
  summary: "Compresses grayscale images with low-rank SVD and animates U Sigma V^T geometry.",
  parameters: [
    {
      kind: "image",
      id: "imageMatrix",
      label: "Grayscale image",
      buttonLabel: "Upload image",
      maxSize: 40,
      defaultValue: defaultSvdImage,
    },
    {
      kind: "range",
      id: "rank",
      label: "Rank k",
      min: 1,
      max: 40,
      step: 1,
      defaultValue: 8,
      format: "integer",
    },
  ],
  sample: makeSvdDataset,
  formulas: [
    { title: "SVD factorization", expression: "A=U\\Sigma V^T" },
    { title: "Low-rank approximation", expression: "A_k=U_k\\Sigma_kV_k^T=\\sum_{i=1}^{k}\\sigma_i u_iv_i^T" },
    { title: "Energy retained", expression: "\\frac{\\sum_{i=1}^{k}\\sigma_i^2}{\\sum_i\\sigma_i^2}" },
  ],
  explanation: [
    "SVD decomposes a matrix into two rotations/reflections and one diagonal scaling stage.",
    "For images, the largest singular values capture broad structure first; keeping more rank-one components restores edges and texture.",
    "The geometric view shows the same factorization as transformations: V^T rotates into principal directions, Sigma stretches by variance, and U rotates into the output basis.",
  ],
  engine: (_, params) => {
    const source = imageMatrixParam(params, "imageMatrix", defaultSvdImage);
    const maxRank = Math.max(1, Math.min(source.height, source.width));
    const rank = Math.max(1, Math.min(maxRank, Math.round(numberParam(params, "rank", 8))));
    const decomposition = lowRankSvdApproximation(source.values, rank);
    const frameCount = 36;
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const progress = index / Math.max(1, frameCount - 1);

      return {
        type: "concept-demo" as const,
        iteration: index + 1,
        points: decomposition.singularValues.map((value, singularIndex) => ({
          x: singularIndex + 1,
          y: value,
          label: `sigma ${singularIndex + 1}`,
        })),
        svd: {
          source,
          original: source.values,
          approximation: decomposition.approximation,
          rank,
          maxRank,
          singularValues: decomposition.singularValues,
          retainedEnergy: decomposition.retainedEnergy,
          reconstructionError: decomposition.reconstructionError,
          geometry: makeSvdGeometry(progress),
        },
        summary: `rank ${rank}/${maxRank} · energy ${(decomposition.retainedEnergy * 100).toFixed(1)}% · error ${decomposition.reconstructionError.toFixed(3)}`,
      };
    });

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Rank", value: `${rank}/${maxRank}` },
        { label: "Energy retained", value: `${Math.round(decomposition.retainedEnergy * 100)}%` },
        { label: "Reconstruction error", value: decomposition.reconstructionError.toFixed(3) },
        { label: "Image matrix", value: `${source.height}x${source.width}` },
      ],
    };
  },
  python: (params) => {
    const rank = Math.round(numberParam(params, "rank", 8));

    return `import numpy as np

# A is a grayscale image matrix with values in [0, 1].
k = ${rank}
gram = A.T @ A
eigenvalues, V = np.linalg.eig(gram)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = np.maximum(eigenvalues[order], 0)
V = V[:, order]

singular_values = np.sqrt(eigenvalues)
Sigma = np.diag(singular_values[:k])
U = A @ V[:, :k] / (singular_values[:k] + 1e-12)

A_k = U @ Sigma @ V[:, :k].T
energy = np.sum(singular_values[:k] ** 2) / np.sum(singular_values ** 2)`;
  },
  javascript: (params) => {
    const rank = Math.round(numberParam(params, "rank", 8));

    return `const k = ${rank};
const gram = matMul(transpose(A), A);
const { values, vectors: V } = eigenSymmetric(gram);
const order = values
  .map((value, index) => ({ value: Math.max(0, value), index }))
  .sort((a, b) => b.value - a.value);

const singularValues = order.map((item) => Math.sqrt(item.value));
const Vk = order.slice(0, k).map((item) => column(V, item.index));

const Ak = zeros(A.length, A[0].length);
for (let component = 0; component < Vk.length; component += 1) {
  const v = Vk[component];
  const Av = matVec(A, v);
  for (let row = 0; row < A.length; row += 1) {
    for (let col = 0; col < A[0].length; col += 1) {
      Ak[row][col] += Av[row] * v[col];
    }
  }
}

// TensorFlow.js equivalent pieces:
// const gram = tf.matMul(A, A, true, false);
// const Ak = tf.matMul(tf.matMul(Uk, SigmaK), Vk.transpose());`;
  },
});

function makeDefaultSvdImage(size: number): ImageMatrixParameterValue {
  const center = (size - 1) / 2;
  const values = Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => {
      const nx = (column - center) / center;
      const ny = (row - center) / center;
      const radius = Math.hypot(nx, ny);
      const face = Math.max(0, 1 - radius * 0.88);
      const diagonal = ((row + column) % 9) / 9;
      const bars = column > size * 0.62 && row > size * 0.18 && row < size * 0.76 ? 0.34 : 0;
      const eyeLeft = Math.exp(-((nx + 0.33) ** 2 + (ny + 0.22) ** 2) * 46) * 0.55;
      const eyeRight = Math.exp(-((nx - 0.33) ** 2 + (ny + 0.22) ** 2) * 46) * 0.55;
      const smile = Math.exp(-((ny - 0.28 - 0.18 * nx ** 2) ** 2) * 90) * (Math.abs(nx) < 0.55 ? 0.28 : 0);
      return clamp(0.18 + face * 0.62 + diagonal * 0.1 + bars - eyeLeft - eyeRight - smile);
    }),
  );

  return {
    kind: "image-matrix",
    name: "Generated grayscale image",
    width: size,
    height: size,
    values,
  };
}

function makeSvdDataset() {
  return makeDataset(
    "Generated SVD basis sample",
    Array.from({ length: 24 }, (_, index) => ({
      x: Math.cos((index / 24) * Math.PI * 2),
      y: Math.sin((index / 24) * Math.PI * 2),
      label: index % 2 === 0 ? "basis" : "sample",
    })),
  );
}

function lowRankSvdApproximation(matrix: number[][], rank: number) {
  const height = matrix.length;
  const width = matrix[0]?.length ?? 0;
  const gram = multiplyTransposeByMatrix(matrix);
  const eigen = jacobiEigenSymmetric(gram);
  const components = eigen.values
    .map((value, index) => ({
      value: Math.max(0, value),
      vector: eigen.vectors[index],
    }))
    .sort((a, b) => b.value - a.value);
  const singularValues = components.map((component) => Math.sqrt(component.value));
  const approximation = zeroMatrix(height, width);
  const usableRank = Math.min(rank, components.length);

  for (let componentIndex = 0; componentIndex < usableRank; componentIndex += 1) {
    const component = components[componentIndex];
    const sigma = Math.sqrt(component.value);
    if (sigma < 1e-10) {
      continue;
    }

    const av = multiplyMatrixVector(matrix, component.vector);
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        approximation[row][column] += av[row] * component.vector[column];
      }
    }
  }

  const totalEnergy = Math.max(1e-12, singularValues.reduce((sum, value) => sum + value ** 2, 0));
  const retainedEnergy =
    singularValues.slice(0, usableRank).reduce((sum, value) => sum + value ** 2, 0) / totalEnergy;
  const originalEnergy = Math.max(1e-12, matrix.flat().reduce((sum, value) => sum + value ** 2, 0));
  let squaredError = 0;
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      squaredError += (matrix[row][column] - approximation[row][column]) ** 2;
      approximation[row][column] = clamp(approximation[row][column]);
    }
  }

  return {
    approximation,
    singularValues,
    retainedEnergy,
    reconstructionError: Math.sqrt(squaredError / originalEnergy),
  };
}

function multiplyTransposeByMatrix(matrix: number[][]) {
  const height = matrix.length;
  const width = matrix[0]?.length ?? 0;
  const result = zeroMatrix(width, width);

  for (let row = 0; row < height; row += 1) {
    for (let left = 0; left < width; left += 1) {
      for (let right = left; right < width; right += 1) {
        result[left][right] += matrix[row][left] * matrix[row][right];
      }
    }
  }

  for (let left = 0; left < width; left += 1) {
    for (let right = left + 1; right < width; right += 1) {
      result[right][left] = result[left][right];
    }
  }

  return result;
}

function multiplyMatrixVector(matrix: number[][], vector: number[]) {
  return matrix.map((row) =>
    row.reduce((sum, value, index) => sum + value * (vector[index] ?? 0), 0),
  );
}

function zeroMatrix(height: number, width: number) {
  return Array.from({ length: height }, () => Array.from({ length: width }, () => 0));
}

function identityMatrix(size: number): number[][] {
  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => (row === column ? 1 : 0)),
  );
}

function jacobiEigenSymmetric(matrix: number[][]) {
  const size = matrix.length;
  const values = matrix.map((row) => [...row]);
  const vectors = identityMatrix(size);
  const maxSweeps = Math.max(40, size * 3);

  for (let sweep = 0; sweep < maxSweeps; sweep += 1) {
    let p = 0;
    let q = 1;
    let largest = 0;

    for (let row = 0; row < size; row += 1) {
      for (let column = row + 1; column < size; column += 1) {
        const magnitude = Math.abs(values[row][column]);
        if (magnitude > largest) {
          largest = magnitude;
          p = row;
          q = column;
        }
      }
    }

    if (largest < 1e-10 || size < 2) {
      break;
    }

    const app = values[p][p];
    const aqq = values[q][q];
    const apq = values[p][q];
    const angle = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(angle);
    const s = Math.sin(angle);

    for (let index = 0; index < size; index += 1) {
      if (index !== p && index !== q) {
        const aip = values[index][p];
        const aiq = values[index][q];
        values[index][p] = c * aip - s * aiq;
        values[p][index] = values[index][p];
        values[index][q] = s * aip + c * aiq;
        values[q][index] = values[index][q];
      }
    }

    values[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    values[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    values[p][q] = 0;
    values[q][p] = 0;

    for (let row = 0; row < size; row += 1) {
      const vip = vectors[row][p];
      const viq = vectors[row][q];
      vectors[row][p] = c * vip - s * viq;
      vectors[row][q] = s * vip + c * viq;
    }
  }

  return {
    values: values.map((row, index) => row[index]),
    vectors: values.map((_, column) => vectors.map((row) => row[column])),
  };
}

function makeSvdGeometry(progress: number) {
  const inputVectors = [
    { x: 1, y: 0, color: "#2f6fbe", label: "e1" },
    { x: 0, y: 1, color: "#0f766e", label: "e2" },
    { x: 0.72, y: 0.54, color: "#d34a43", label: "x" },
  ];
  const angleV = -0.62;
  const angleU = 0.74;
  const scaleX = 1.46;
  const scaleY = 0.54;
  const phaseIndex = Math.min(2, Math.floor(progress * 3));
  const phaseProgress = phaseIndex === 2 ? progress * 3 - 2 : progress * 3 - phaseIndex;
  const eased = 0.5 - Math.cos(Math.max(0, Math.min(1, phaseProgress)) * Math.PI) / 2;

  let phase: "V^T rotation" | "Sigma scaling" | "U rotation" = "V^T rotation";
  let currentVectors = inputVectors.map((vector) => ({
    ...rotateSvdVector(vector, angleV * eased),
    color: vector.color,
    label: vector.label,
  }));

  if (phaseIndex === 1) {
    phase = "Sigma scaling";
    const rotated = inputVectors.map((vector) => rotateSvdVector(vector, angleV));
    currentVectors = rotated.map((vector, index) => ({
      x: vector.x * (1 + (scaleX - 1) * eased),
      y: vector.y * (1 + (scaleY - 1) * eased),
      color: inputVectors[index].color,
      label: inputVectors[index].label,
    }));
  } else if (phaseIndex === 2) {
    phase = "U rotation";
    const scaled = inputVectors.map((vector) => {
      const rotated = rotateSvdVector(vector, angleV);
      return { x: rotated.x * scaleX, y: rotated.y * scaleY };
    });
    currentVectors = scaled.map((vector, index) => ({
      ...rotateSvdVector(vector, angleU * eased),
      color: inputVectors[index].color,
      label: inputVectors[index].label,
    }));
  }

  return {
    phase,
    progress: eased,
    angleV,
    angleU,
    scaleX,
    scaleY,
    inputVectors,
    currentVectors,
  };
}

function rotateSvdVector(vector: { x: number; y: number }, angle: number) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return {
    x: vector.x * c - vector.y * s,
    y: vector.x * s + vector.y * c,
  };
}

const convexOptimization = makeConceptAlgorithm({
  id: "convex-optimization",
  name: "Convex Optimization",
  category: "Convex Optimization",
  summary: "Shows constrained descent on a convex bowl and detects non-convex function morphs.",
  parameters: [
    {
      kind: "select",
      id: "optimizer",
      label: "Optimizer",
      defaultValue: "projected-gradient",
      options: [
        { label: "Projected Gradient", value: "projected-gradient" },
        { label: "Newton's Method", value: "newton" },
      ],
    },
    {
      kind: "range",
      id: "curvatureX",
      label: "Hessian xx",
      min: 0.2,
      max: 2.4,
      step: 0.05,
      defaultValue: 1.25,
      format: "decimal",
    },
    {
      kind: "range",
      id: "curvatureY",
      label: "Hessian yy",
      min: -0.8,
      max: 2.4,
      step: 0.05,
      defaultValue: 0.9,
      format: "decimal",
    },
    {
      kind: "range",
      id: "crossTerm",
      label: "Hessian xy",
      min: -1.1,
      max: 1.1,
      step: 0.05,
      defaultValue: 0.15,
      format: "decimal",
    },
    {
      kind: "range",
      id: "sCurve",
      label: "S-curve morph",
      min: 0,
      max: 1.4,
      step: 0.05,
      defaultValue: 0.15,
      format: "decimal",
    },
    {
      kind: "range",
      id: "constraintSize",
      label: "Constraint box",
      min: 0.55,
      max: 1.25,
      step: 0.05,
      defaultValue: 0.85,
      format: "decimal",
    },
  ],
  sample: makeConvexDataset,
  formulas: [
    { title: "Quadratic objective", expression: "f(x)=\\frac{1}{2}x^THx+c^Tx" },
    { title: "Convexity check", expression: "H\\succeq0\\quad\\Rightarrow\\quad f\\;\\text{is convex}" },
    { title: "Projected gradient", expression: "x_{t+1}=\\Pi_C(x_t-\\eta\\nabla f(x_t))" },
    { title: "Newton step", expression: "x_{t+1}=\\Pi_C(x_t-H^{-1}\\nabla f(x_t))" },
  ],
  explanation: [
    "Convex optimization is stable because any local minimum is also global when the objective curvature stays positive semidefinite.",
    "The box walls represent inequality constraints. Projection clips each proposed update back into the feasible set.",
    "The morph sliders intentionally break convexity: negative curvature or a strong S-curve creates local traps, which is why objective structure matters in real training systems.",
  ],
  engine: (_, params) => {
    const optimizer = convexOptimizerParam(params, "optimizer", "projected-gradient");
    const curvatureX = numberParam(params, "curvatureX", 1.25);
    const curvatureY = numberParam(params, "curvatureY", 0.9);
    const crossTerm = numberParam(params, "crossTerm", 0.15);
    const sCurve = numberParam(params, "sCurve", 0.15);
    const constraintSize = numberParam(params, "constraintSize", 0.85);
    const objective = {
      curvatureX,
      curvatureY,
      crossTerm,
      sCurve,
      tiltX: -1.42,
      tiltY: 1.18,
    };
    const constraint = {
      minX: -constraintSize,
      maxX: constraintSize,
      minY: -constraintSize,
      maxY: constraintSize,
    };
    const determinant = curvatureX * curvatureY - crossTerm ** 2;
    const isConvex = curvatureX > 0 && curvatureY > 0 && determinant > 0.02 && sCurve < 0.7;
    const statusReason = isConvex
      ? "Positive definite Hessian and mild morph"
      : sCurve >= 0.7
        ? "S-curve morph creates local minima traps"
        : "Hessian is not positive definite";
    const trajectory = makeConvexTrajectory(optimizer, objective, constraint, isConvex);
    const unconstrained = convexUnconstrainedPoint(objective);
    const constrainedOptimum = approximateConstrainedOptimum(objective, constraint);
    const frameCount = Math.min(48, trajectory.length);
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const sourceIndex = Math.min(trajectory.length - 1, Math.round((index / Math.max(1, frameCount - 1)) * (trajectory.length - 1)));
      const path = trajectory.slice(0, sourceIndex + 1);
      const current = path[path.length - 1];
      const gradient = convexGradient(objective, current.x, current.y);
      const gradientNorm = Math.hypot(gradient.x, gradient.y);

      return {
        type: "concept-demo" as const,
        iteration: index + 1,
        points: path.map((point, pointIndex) => ({ x: point.x, y: point.y, label: pointIndex })),
        convex: {
          optimizer,
          status: isConvex ? "convex" as const : "non-convex" as const,
          statusReason,
          step: sourceIndex,
          objective,
          constraint,
          path,
          current,
          unconstrained,
          constrainedOptimum,
          gradientNorm,
        },
        summary: `Status: ${isConvex ? "Convex" : "Non-Convex"} · ${optimizerLabel(optimizer)} · ${current.activeConstraint ?? "interior"}`,
      };
    });
    const final = frames[frames.length - 1].convex;

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Status", value: final.status === "convex" ? "Convex" : "Non-convex" },
        { label: "Optimizer", value: optimizerLabel(optimizer) },
        { label: "Gradient norm", value: final.gradientNorm.toFixed(3) },
        { label: "Constraint", value: final.current.activeConstraint ?? "interior" },
      ],
    };
  },
  python: (params) => {
    const optimizer = convexOptimizerParam(params, "optimizer", "projected-gradient");
    return `import numpy as np

H = np.array([
    [${numberParam(params, "curvatureX", 1.25).toFixed(2)}, ${numberParam(params, "crossTerm", 0.15).toFixed(2)}],
    [${numberParam(params, "crossTerm", 0.15).toFixed(2)}, ${numberParam(params, "curvatureY", 0.9).toFixed(2)}],
])
c = np.array([-1.42, 1.18])
lower = np.array([-${numberParam(params, "constraintSize", 0.85).toFixed(2)}, -${numberParam(params, "constraintSize", 0.85).toFixed(2)}])
upper = np.array([${numberParam(params, "constraintSize", 0.85).toFixed(2)}, ${numberParam(params, "constraintSize", 0.85).toFixed(2)}])

def project(x):
    return np.minimum(upper, np.maximum(lower, x))

def grad(x):
    return H @ x + c

eigvals = np.linalg.eigvals(H)
status = "convex" if np.all(eigvals > 0) else "non-convex"
x = np.array([-0.72, 0.72])
track = [x.copy()]
for step in range(32):
    g = grad(x)
    if "${optimizer}" == "newton":
        proposal = x - np.linalg.inv(H + 1e-6 * np.eye(2)) @ g
    else:
        proposal = x - 0.28 * g
    x = project(proposal)
    track.append(x.copy())`;
  },
  javascript: (params) => {
    const optimizer = convexOptimizerParam(params, "optimizer", "projected-gradient");
    return `const H = [
  [${numberParam(params, "curvatureX", 1.25).toFixed(2)}, ${numberParam(params, "crossTerm", 0.15).toFixed(2)}],
  [${numberParam(params, "crossTerm", 0.15).toFixed(2)}, ${numberParam(params, "curvatureY", 0.9).toFixed(2)}],
];
const c = [-1.42, 1.18];
const box = ${numberParam(params, "constraintSize", 0.85).toFixed(2)};

const project = ([x, y]) => [
  Math.min(box, Math.max(-box, x)),
  Math.min(box, Math.max(-box, y)),
];
const grad = ([x, y]) => [
  H[0][0] * x + H[0][1] * y + c[0],
  H[1][0] * x + H[1][1] * y + c[1],
];

let x = [-0.72, 0.72];
const track = [x];
for (let step = 0; step < 32; step += 1) {
  const g = grad(x);
  const proposal = "${optimizer}" === "newton"
    ? subtract(x, matVec(inverse2x2(H), g))
    : subtract(x, scale(g, 0.28));
  x = project(proposal);
  track.push(x);
}

// TensorFlow.js sketch:
// const hessian = tf.hessian(objectiveFn)(tf.tensor(x));
// const step = tf.linalg.solve(hessian, gradient.reshape([2, 1]));`;
  },
});

function makeConvexDataset() {
  return makeDataset(
    "Generated convex optimization sample",
    Array.from({ length: 36 }, (_, index) => {
      const angle = (index / 36) * Math.PI * 2;
      return { x: Math.cos(angle), y: Math.sin(angle), label: "level set" };
    }),
  );
}

function convexOptimizerParam(
  params: ParameterState,
  key: string,
  fallback: ConvexOptimizerKey,
): ConvexOptimizerKey {
  const value = stringParam(params, key, fallback);
  return value === "newton" || value === "projected-gradient" ? value : fallback;
}

function makeConvexTrajectory(
  optimizer: ConvexOptimizerKey,
  objective: {
    curvatureX: number;
    curvatureY: number;
    crossTerm: number;
    sCurve: number;
    tiltX: number;
    tiltY: number;
  },
  constraint: { minX: number; maxX: number; minY: number; maxY: number },
  isConvex: boolean,
) {
  const path = [
    convexPoint(objective, {
      x: constraint.minX * 0.85,
      y: constraint.maxY * 0.85,
      projected: false,
    }),
  ];
  let point = { x: path[0].x, y: path[0].y };
  const learningRate = optimizer === "newton" ? 0.74 : 0.28;

  for (let step = 0; step < 44; step += 1) {
    const gradient = convexGradient(objective, point.x, point.y);
    let delta = { x: gradient.x, y: gradient.y };
    if (optimizer === "newton" && isConvex) {
      delta = solveConvexNewtonDelta(objective, gradient);
    }
    const proposal = {
      x: point.x - learningRate * delta.x,
      y: point.y - learningRate * delta.y,
    };
    const projected = projectToConvexBox(proposal, constraint);
    point = { x: projected.x, y: projected.y };
    path.push(convexPoint(objective, projected));
  }

  return path;
}

function convexPoint(
  objective: {
    curvatureX: number;
    curvatureY: number;
    crossTerm: number;
    sCurve: number;
    tiltX: number;
    tiltY: number;
  },
  point: { x: number; y: number; projected: boolean; activeConstraint?: string },
) {
  return {
    ...point,
    z: convexObjectiveValue(objective, point.x, point.y),
  };
}

function convexObjectiveValue(
  objective: {
    curvatureX: number;
    curvatureY: number;
    crossTerm: number;
    sCurve: number;
    tiltX: number;
    tiltY: number;
  },
  x: number,
  y: number,
) {
  const quadratic =
    0.5 * objective.curvatureX * x ** 2 +
    objective.crossTerm * x * y +
    0.5 * objective.curvatureY * y ** 2 +
    objective.tiltX * x +
    objective.tiltY * y;
  const morph = objective.sCurve * (0.18 * x ** 4 - 0.7 * x ** 2 + 0.14 * Math.sin(3 * y));
  return quadratic + morph + 2.6;
}

function convexGradient(
  objective: {
    curvatureX: number;
    curvatureY: number;
    crossTerm: number;
    sCurve: number;
    tiltX: number;
    tiltY: number;
  },
  x: number,
  y: number,
) {
  return {
    x:
      objective.curvatureX * x +
      objective.crossTerm * y +
      objective.tiltX +
      objective.sCurve * (0.72 * x ** 3 - 1.4 * x),
    y:
      objective.crossTerm * x +
      objective.curvatureY * y +
      objective.tiltY +
      objective.sCurve * 0.42 * Math.cos(3 * y),
  };
}

function solveConvexNewtonDelta(
  objective: { curvatureX: number; curvatureY: number; crossTerm: number },
  gradient: { x: number; y: number },
) {
  const det = objective.curvatureX * objective.curvatureY - objective.crossTerm ** 2;
  if (Math.abs(det) < 1e-6) {
    return gradient;
  }

  return {
    x: (objective.curvatureY * gradient.x - objective.crossTerm * gradient.y) / det,
    y: (-objective.crossTerm * gradient.x + objective.curvatureX * gradient.y) / det,
  };
}

function projectToConvexBox(
  point: { x: number; y: number },
  constraint: { minX: number; maxX: number; minY: number; maxY: number },
) {
  const x = Math.min(constraint.maxX, Math.max(constraint.minX, point.x));
  const y = Math.min(constraint.maxY, Math.max(constraint.minY, point.y));
  const projected = Math.abs(x - point.x) > 1e-6 || Math.abs(y - point.y) > 1e-6;
  const active: string[] = [];
  if (Math.abs(x - constraint.minX) < 1e-5) active.push("x min");
  if (Math.abs(x - constraint.maxX) < 1e-5) active.push("x max");
  if (Math.abs(y - constraint.minY) < 1e-5) active.push("y min");
  if (Math.abs(y - constraint.maxY) < 1e-5) active.push("y max");

  return {
    x,
    y,
    projected,
    activeConstraint: active.length ? active.join(" + ") : undefined,
  };
}

function convexUnconstrainedPoint(objective: {
  curvatureX: number;
  curvatureY: number;
  crossTerm: number;
  sCurve: number;
  tiltX: number;
  tiltY: number;
}) {
  const det = objective.curvatureX * objective.curvatureY - objective.crossTerm ** 2;
  if (Math.abs(det) < 1e-6) {
    return convexPoint(objective, { x: 0, y: 0, projected: false });
  }

  const x = (-objective.curvatureY * objective.tiltX + objective.crossTerm * objective.tiltY) / det;
  const y = (objective.crossTerm * objective.tiltX - objective.curvatureX * objective.tiltY) / det;
  return convexPoint(objective, { x, y, projected: false });
}

function approximateConstrainedOptimum(
  objective: {
    curvatureX: number;
    curvatureY: number;
    crossTerm: number;
    sCurve: number;
    tiltX: number;
    tiltY: number;
  },
  constraint: { minX: number; maxX: number; minY: number; maxY: number },
) {
  let best = convexPoint(objective, { x: constraint.minX, y: constraint.minY, projected: false });
  const samples = 34;
  for (let row = 0; row <= samples; row += 1) {
    for (let column = 0; column <= samples; column += 1) {
      const x = constraint.minX + (column / samples) * (constraint.maxX - constraint.minX);
      const y = constraint.minY + (row / samples) * (constraint.maxY - constraint.minY);
      const candidate = convexPoint(objective, { x, y, projected: false });
      if (candidate.z < best.z) {
        best = candidate;
      }
    }
  }

  return best;
}

function optimizerLabel(optimizer: ConvexOptimizerKey) {
  return optimizer === "newton" ? "Newton" : "Projected GD";
}

const convolutionKernels: Record<ConvolutionKernelPreset, number[][]> = {
  "sobel-x": [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ],
  "gaussian-blur": [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  custom: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
};

const defaultConvolutionImage = makeDefaultConvolutionImage(28);

const convolutionsFromScratch = makeConceptAlgorithm({
  id: "convolutions-from-scratch",
  name: "Convolutions from Scratch",
  category: "Convolutions from Scratch",
  summary: "Slides editable 3x3 kernels across grayscale images with padding, stride, and live arithmetic.",
  parameters: [
    {
      kind: "image",
      id: "convImage",
      label: "Source image",
      buttonLabel: "Upload image",
      maxSize: 32,
      defaultValue: defaultConvolutionImage,
    },
    {
      kind: "select",
      id: "kernelPreset",
      label: "Kernel preset",
      defaultValue: "sobel-x",
      options: [
        { label: "Sobel Filter", value: "sobel-x" },
        { label: "Gaussian Blur", value: "gaussian-blur" },
        { label: "Sharpen", value: "sharpen" },
        { label: "Custom", value: "custom" },
      ],
    },
    {
      kind: "matrix",
      id: "customKernel",
      label: "Custom 3x3 kernel",
      rowLabels: ["r1", "r2", "r3"],
      columnLabels: ["c1", "c2", "c3"],
      defaultValue: convolutionKernels.custom,
      min: -5,
      max: 5,
      step: 0.0625,
      format: "decimal",
    },
    {
      kind: "range",
      id: "stride",
      label: "Stride",
      min: 1,
      max: 3,
      step: 1,
      defaultValue: 1,
      format: "integer",
    },
    {
      kind: "toggle",
      id: "padding",
      label: "Zero padding",
      defaultValue: true,
    },
  ],
  sample: makeConvolutionDataset,
  formulas: [
    { title: "Convolution", expression: "Y_{i,j}=\\sum_{u=0}^{2}\\sum_{v=0}^{2}X_{iS+u,jS+v}K_{u,v}" },
    { title: "Output shape", expression: "H_{out}=\\left\\lfloor\\frac{H+2P-K}{S}\\right\\rfloor+1" },
    { title: "Sobel X filter", expression: "K=\\begin{bmatrix}-1&0&1\\\\-2&0&2\\\\-1&0&1\\end{bmatrix}" },
  ],
  explanation: [
    "A convolution kernel is a tiny matrix that scans an image and writes one output pixel from the weighted sum of each local patch.",
    "Stride controls how far the window jumps. Larger stride skips positions and visibly shrinks the feature map.",
    "Padding adds a border of zeros so edge pixels can still become centers of a 3x3 calculation instead of being discarded.",
  ],
  engine: (_, params) => {
    const source = imageMatrixParam(params, "convImage", defaultConvolutionImage);
    const preset = convolutionPresetParam(params, "kernelPreset", "sobel-x");
    const kernel =
      preset === "custom"
        ? signedMatrixParam(params, "customKernel", convolutionKernels.custom)
        : convolutionKernels[preset];
    const padding = Boolean(params.padding ?? true);
    const stride = Math.max(1, Math.min(3, Math.round(numberParam(params, "stride", 1))));
    const padded = padMatrix(source.values, padding ? 1 : 0);
    const output = convolveMatrix(padded, kernel, stride);
    const normalizedOutput = normalizeConvolutionOutput(output);
    const positions = makeConvolutionPositions(output, stride);
    const frameCount = Math.min(72, Math.max(1, positions.length));
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const sourceIndex = Math.min(
        positions.length - 1,
        Math.round((index / Math.max(1, frameCount - 1)) * (positions.length - 1)),
      );
      const position = positions[sourceIndex] ?? {
        inputRow: 0,
        inputColumn: 0,
        outputRow: 0,
        outputColumn: 0,
      };
      const patch = convolutionPatch(padded, position.inputRow, position.inputColumn);
      const terms = convolutionTerms(patch, kernel);
      const currentValue = terms.reduce((sum, term) => sum + term.product, 0);

      return {
        type: "concept-demo" as const,
        iteration: index + 1,
        points: output.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => ({
            x: columnIndex,
            y: rowIndex,
            label: round(value, 3),
          })),
        ),
        convolution: {
          source,
          kernelPreset: preset,
          kernel,
          padded,
          output,
          normalizedOutput,
          padding,
          stride,
          cursor: position,
          patch,
          terms,
          currentValue: round(currentValue, 4),
          outputShape: {
            height: output.length,
            width: output[0]?.length ?? 0,
          },
        },
        summary: `${convolutionKernelLabel(preset)} · stride ${stride} · padding ${padding ? "on" : "off"} · output ${output.length}x${output[0]?.length ?? 0}`,
      };
    });
    const final = frames[frames.length - 1].convolution;

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Kernel", value: convolutionKernelLabel(preset) },
        { label: "Stride", value: String(stride) },
        { label: "Padding", value: padding ? "on" : "off" },
        { label: "Output", value: `${final.outputShape.height}x${final.outputShape.width}` },
      ],
    };
  },
  python: (params) => {
    const preset = convolutionPresetParam(params, "kernelPreset", "sobel-x");
    const kernel =
      preset === "custom"
        ? signedMatrixParam(params, "customKernel", convolutionKernels.custom)
        : convolutionKernels[preset];
    const stride = Math.max(1, Math.min(3, Math.round(numberParam(params, "stride", 1))));
    const padding = Boolean(params.padding ?? true) ? 1 : 0;

    return `import numpy as np

kernel = np.array(${formatMatrixLiteral(kernel)}, dtype=float)
stride = ${stride}
pad = ${padding}

# image is a grayscale matrix with values in [0, 1].
if pad:
    image = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")

kh, kw = kernel.shape
out_h = (image.shape[0] - kh) // stride + 1
out_w = (image.shape[1] - kw) // stride + 1
output = np.zeros((out_h, out_w))

for i in range(out_h):
    for j in range(out_w):
        h = i * stride
        w = j * stride
        patch = image[h:h + kh, w:w + kw]
        output[i, j] = np.sum(patch * kernel)`;
  },
  javascript: (params) => {
    const preset = convolutionPresetParam(params, "kernelPreset", "sobel-x");
    const kernel =
      preset === "custom"
        ? signedMatrixParam(params, "customKernel", convolutionKernels.custom)
        : convolutionKernels[preset];
    const stride = Math.max(1, Math.min(3, Math.round(numberParam(params, "stride", 1))));
    const padding = Boolean(params.padding ?? true) ? 1 : 0;

    return `const kernel = ${formatMatrixLiteral(kernel)};
const stride = ${stride};
const pad = ${padding};

const padded = pad
  ? padWithZeros(image, pad)
  : image.map((row) => [...row]);
const outH = Math.floor((padded.length - 3) / stride) + 1;
const outW = Math.floor((padded[0].length - 3) / stride) + 1;
const output = Array.from({ length: outH }, () => Array(outW).fill(0));

for (let i = 0; i < outH; i += 1) {
  for (let j = 0; j < outW; j += 1) {
    const h = i * stride;
    const w = j * stride;
    let sum = 0;
    for (let kh = 0; kh < 3; kh += 1) {
      for (let kw = 0; kw < 3; kw += 1) {
        sum += padded[h + kh][w + kw] * kernel[kh][kw];
      }
    }
    output[i][j] = sum;
  }
}

// TensorFlow.js path for hardware acceleration:
// const x = tf.tensor4d(image, [1, height, width, 1]);
// const k = tf.tensor4d(kernel.flat(), [3, 3, 1, 1]);
// const y = tf.conv2d(x, k, stride, pad ? "same" : "valid");`;
  },
});

function makeDefaultConvolutionImage(size: number): ImageMatrixParameterValue {
  const values = Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => {
      const base = column > size * 0.52 ? 0.72 : 0.24;
      const square =
        column > size * 0.14 &&
        column < size * 0.42 &&
        row > size * 0.55 &&
        row < size * 0.82
          ? 0.48
          : 0;
      const bar = row > size * 0.2 && row < size * 0.32 ? 0.18 : 0;
      const diagonal = Math.abs(row - column * 0.68 - size * 0.08) < 1.4 ? 0.22 : 0;
      const ripple = Math.sin(row * 0.55) * Math.cos(column * 0.38) * 0.035;
      return Math.max(0, Math.min(1, base + square + bar + diagonal + ripple));
    }),
  );

  return {
    kind: "image-matrix",
    name: "Generated edge sample",
    width: size,
    height: size,
    values,
  };
}

function makeConvolutionDataset() {
  return makeDataset(
    "Generated convolution grid sample",
    Array.from({ length: 32 }, (_, index) => ({
      x: index % 8,
      y: Math.floor(index / 8),
      label: "pixel",
    })),
  );
}

function convolutionPresetParam(
  params: ParameterState,
  key: string,
  fallback: ConvolutionKernelPreset,
): ConvolutionKernelPreset {
  const value = stringParam(params, key, fallback);
  return value === "sobel-x" || value === "gaussian-blur" || value === "sharpen" || value === "custom"
    ? value
    : fallback;
}

function padMatrix(matrix: number[][], amount: number) {
  if (amount <= 0) {
    return matrix.map((row) => [...row]);
  }

  const width = matrix[0]?.length ?? 0;
  const paddedWidth = width + amount * 2;
  const zeros = () => Array.from({ length: paddedWidth }, () => 0);
  const padded: number[][] = Array.from({ length: amount }, zeros);
  matrix.forEach((row) => {
    padded.push([
      ...Array.from({ length: amount }, () => 0),
      ...row,
      ...Array.from({ length: amount }, () => 0),
    ]);
  });
  padded.push(...Array.from({ length: amount }, zeros));
  return padded;
}

function convolveMatrix(matrix: number[][], kernel: number[][], stride: number) {
  const kernelHeight = kernel.length;
  const kernelWidth = kernel[0]?.length ?? 0;
  const outputHeight = Math.max(1, Math.floor((matrix.length - kernelHeight) / stride) + 1);
  const outputWidth = Math.max(1, Math.floor(((matrix[0]?.length ?? 0) - kernelWidth) / stride) + 1);

  return Array.from({ length: outputHeight }, (_, outputRow) =>
    Array.from({ length: outputWidth }, (_, outputColumn) => {
      const inputRow = outputRow * stride;
      const inputColumn = outputColumn * stride;
      let total = 0;
      for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) {
        for (let kernelColumn = 0; kernelColumn < kernelWidth; kernelColumn += 1) {
          total += (matrix[inputRow + kernelRow]?.[inputColumn + kernelColumn] ?? 0) * kernel[kernelRow][kernelColumn];
        }
      }
      return round(total, 5);
    }),
  );
}

function makeConvolutionPositions(output: number[][], stride: number) {
  return output.flatMap((row, outputRow) =>
    row.map((_, outputColumn) => ({
      inputRow: outputRow * stride,
      inputColumn: outputColumn * stride,
      outputRow,
      outputColumn,
    })),
  );
}

function convolutionPatch(matrix: number[][], row: number, column: number) {
  return Array.from({ length: 3 }, (_, patchRow) =>
    Array.from({ length: 3 }, (_, patchColumn) => matrix[row + patchRow]?.[column + patchColumn] ?? 0),
  );
}

function convolutionTerms(patch: number[][], kernel: number[][]) {
  return patch.flatMap((row, rowIndex) =>
    row.map((imageValue, columnIndex) => {
      const kernelValue = kernel[rowIndex]?.[columnIndex] ?? 0;
      return {
        imageValue,
        kernelValue,
        product: round(imageValue * kernelValue, 5),
        row: rowIndex,
        column: columnIndex,
      };
    }),
  );
}

function normalizeConvolutionOutput(matrix: number[][]) {
  const values = matrix.flat();
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-8) {
    return matrix.map((row) => row.map(() => 0.5));
  }

  return matrix.map((row) => row.map((value) => (value - min) / (max - min)));
}

function convolutionKernelLabel(preset: ConvolutionKernelPreset) {
  if (preset === "sobel-x") return "Sobel X";
  if (preset === "gaussian-blur") return "Gaussian Blur";
  if (preset === "sharpen") return "Sharpen";
  return "Custom";
}

function formatMatrixLiteral(matrix: number[][]) {
  return `[
  ${matrix
    .map((row) =>
      `[${row
        .map((value) =>
          Math.abs(value) < 1e-8
            ? "0"
            : Number.isInteger(value)
              ? String(value)
              : Number(value.toFixed(4)).toString(),
        )
        .join(", ")}]`,
    )
    .join(",\n  ")}
]`;
}

const neuralNetwork = makeConceptAlgorithm({
  id: "neural-network",
  name: "Neural Network Training Curve",
  category: "Neural Networks",
  summary: "Shows loss and accuracy changing through training epochs.",
  parameters: [
    { kind: "range", id: "hiddenUnits", label: "Hidden units", min: 2, max: 64, step: 2, defaultValue: 16, format: "integer" },
    { kind: "range", id: "learningRate", label: "Learning rate", min: 0.001, max: 0.1, step: 0.001, defaultValue: 0.025, format: "decimal" },
    { kind: "range", id: "epochs", label: "Epochs", min: 20, max: 200, step: 10, defaultValue: 100, format: "integer" },
  ],
  sample: () => makeLearningCurveDataset("Generated neural training sample"),
  formulas: [
    { title: "Layer", expression: "h=\\phi(Wx+b)" },
    { title: "Backprop", expression: "W\\leftarrow W-\\alpha\\frac{\\partial L}{\\partial W}" },
  ],
  explanation: [
    "Neural networks stack differentiable layers and learn weights with gradient descent.",
    "The training curve highlights how learning rate and capacity affect convergence.",
  ],
  engine: (_, params) => {
    const hiddenUnits = Math.round(numberParam(params, "hiddenUnits", 16));
    const learningRate = numberParam(params, "learningRate", 0.025);
    const epochs = Math.round(numberParam(params, "epochs", 100));
    const capacity = Math.log2(hiddenUnits + 2) / 7;
    const loss = Array.from({ length: 24 }, (_, index) => {
      const epoch = Math.round((index / 23) * epochs);
      return {
        x: epoch,
        y: clamp(0.85 * Math.exp(-index * learningRate * 5.4) + 0.08 + (1 - capacity) * 0.08),
      };
    });
    const accuracy = loss.map((point) => ({
      x: point.x,
      y: clamp(1 - point.y + capacity * 0.08),
    }));

    return conceptResult(
      {
        type: "concept-demo",
        iteration: epochs,
        points: [...loss, ...accuracy],
        series: [
          { label: "loss", points: loss, color: "#d34a43" },
          { label: "accuracy", points: accuracy, color: "#0f766e" },
        ],
        summary: `Loss ${loss[loss.length - 1].y.toFixed(3)} · accuracy ${metricValue(accuracy[accuracy.length - 1].y)}`,
      },
      [
        { label: "Loss", value: loss[loss.length - 1].y.toFixed(3) },
        { label: "Accuracy", value: metricValue(accuracy[accuracy.length - 1].y) },
        { label: "Hidden units", value: String(hiddenUnits) },
        { label: "Epochs", value: String(epochs) },
      ],
    );
  },
  python: (params) => `import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(${Math.round(numberParam(params, "hiddenUnits", 16))}, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(${numberParam(params, "learningRate", 0.025)}), loss="binary_crossentropy")
history = model.fit(X, y, epochs=${Math.round(numberParam(params, "epochs", 100))})`,
  javascript: (params) => `const model = tf.sequential({
  layers: [
    tf.layers.dense({ units: ${Math.round(numberParam(params, "hiddenUnits", 16))}, activation: "relu" }),
    tf.layers.dense({ units: 1, activation: "sigmoid" }),
  ],
});
await model.fit(xs, ys, { epochs: ${Math.round(numberParam(params, "epochs", 100))} });`,
});

const multiLayerNetwork = makeConceptAlgorithm({
  id: "multi-layer-network",
  name: "MLP Topology",
  category: "Multi-Layer Networks",
  summary: "Visualizes hidden-layer neurons, weight matrices, and nonlinear decision boundaries.",
  parameters: [
    { kind: "range", id: "hiddenLayers", label: "Hidden layers", min: 1, max: 4, step: 1, defaultValue: 2, format: "integer" },
    { kind: "stepper", id: "hiddenNeurons", label: "Hidden neurons", min: 2, max: 16, step: 1, defaultValue: 6, format: "integer" },
    { kind: "range", id: "trainingSteps", label: "Training steps", min: 20, max: 240, step: 10, defaultValue: 120, format: "integer" },
    { kind: "range", id: "weightScale", label: "Weight scale", min: 0.4, max: 2.4, step: 0.1, defaultValue: 1.2, format: "decimal" },
    {
      kind: "select",
      id: "datasetKind",
      label: "Dataset",
      defaultValue: "xor",
      options: [
        { label: "XOR grid", value: "xor" },
        { label: "Concentric circles", value: "circles" },
        { label: "Linear blobs", value: "linear" },
      ],
    },
    {
      kind: "select",
      id: "activation",
      label: "Activation",
      defaultValue: "tanh",
      options: [
        { label: "Tanh", value: "tanh" },
        { label: "ReLU", value: "relu" },
        { label: "Sigmoid", value: "sigmoid" },
      ],
    },
  ],
  sample: () => makeNetworkLessonDataset("xor"),
  formulas: [
    { title: "Layer transform", expression: "h^{(l)}=\\phi(W^{(l)}h^{(l-1)}+b^{(l)})" },
    { title: "Matrix shape", expression: "W^{(l)}\\in\\mathbb{R}^{d_{l-1}\\times d_l}" },
    { title: "Decision boundary", expression: "\\{x:p(y=1\\mid x)=0.5\\}" },
  ],
  explanation: [
    "This lesson links the algebra to the picture: circles are neurons, lines are weights, and each dense layer applies a matrix multiply followed by a nonlinearity.",
    "Use the hidden-neuron buttons to append or remove neurons. The architecture graph updates immediately, and the code tabs show the corresponding matrix shapes.",
    "On XOR or concentric-circle data, the heat map starts close to a straight boundary and becomes curved as hidden layers add nonlinear feature composition.",
  ],
  engine: (_, params) => {
    const hiddenLayers = Math.round(numberParam(params, "hiddenLayers", 3));
    const hiddenNeurons = Math.round(numberParam(params, "hiddenNeurons", 6));
    const trainingSteps = Math.round(numberParam(params, "trainingSteps", 120));
    const weightScale = numberParam(params, "weightScale", 1.2);
    const datasetKind = stringParam(params, "datasetKind", "xor");
    const activation = stringParam(params, "activation", "relu");
    const dataset = makeNetworkLessonDataset(datasetKind);
    const layerUnits = [
      { label: "input", units: 2 },
      ...Array.from({ length: hiddenLayers }, (_, index) => ({
        label: `hidden ${index + 1}`,
        units: hiddenNeurons,
      })),
      { label: "output", units: 1 },
    ];
    const totalParameters = countNetworkParameters(layerUnits.map((layer) => layer.units));
    const frameCount = 34;
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const progress = index / (frameCount - 1);
      return {
        type: "concept-demo" as const,
        iteration: Math.round(progress * trainingSteps),
        points: dataset.points,
        heatmap: makeDecisionHeatmap(datasetKind, hiddenLayers, hiddenNeurons, progress),
        network: makeNetworkGraph(layerUnits, progress, weightScale, activation, datasetKind),
        summary: `${hiddenLayers} hidden layer${hiddenLayers === 1 ? "" : "s"} · ${hiddenNeurons} neurons/layer · ${matrixShapeSummary(layerUnits.map((layer) => layer.units))}`,
      };
    });
    const finalAccuracy = estimateBoundaryAccuracy(dataset.points, datasetKind, hiddenLayers, hiddenNeurons, 1);

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Accuracy", value: metricValue(finalAccuracy) },
        { label: "Parameters", value: totalParameters.toLocaleString() },
        { label: "Matrix shapes", value: matrixShapeSummary(layerUnits.map((layer) => layer.units)) },
        { label: "Dataset", value: datasetKind },
      ],
    };
  },
  python: (params) => {
    const hiddenLayers = Math.round(numberParam(params, "hiddenLayers", 3));
    const hiddenNeurons = Math.round(numberParam(params, "hiddenNeurons", 6));
    const activation = stringParam(params, "activation", "tanh");
    const layerSizes = [2, ...Array.from({ length: hiddenLayers }, () => hiddenNeurons), 1];

    return `import numpy as np

layer_sizes = ${JSON.stringify(layerSizes)}
weights = [
    np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.4
    for i in range(len(layer_sizes) - 1)
]
biases = [np.zeros((1, width)) for width in layer_sizes[1:]]

def activate(z):
    return ${activation === "relu" ? "np.maximum(0, z)" : activation === "sigmoid" ? "1 / (1 + np.exp(-z))" : "np.tanh(z)"}

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(X):
    a = X
    activations = [a]
    for W, b in zip(weights[:-1], biases[:-1]):
        z = a @ W + b
        a = activate(z)
        activations.append(a)

    logits = a @ weights[-1] + biases[-1]
    return sigmoid(logits), activations

probabilities, hidden_states = forward(X)`;
  },
  javascript: (params) => {
    const hiddenLayers = Math.round(numberParam(params, "hiddenLayers", 3));
    const hiddenNeurons = Math.round(numberParam(params, "hiddenNeurons", 6));
    const activation = stringParam(params, "activation", "tanh");
    const layerSizes = [2, ...Array.from({ length: hiddenLayers }, () => hiddenNeurons), 1];

    return `const layerSizes = ${JSON.stringify(layerSizes)};
const weights = layerSizes.slice(0, -1).map((width, index) =>
  randomMatrix(width, layerSizes[index + 1], 0.4),
);
const biases = layerSizes.slice(1).map((width) => [new Array(width).fill(0)]);

function matMul(a, b) {
  return a.map((row) =>
    b[0].map((_, column) =>
      row.reduce((sum, value, index) => sum + value * b[index][column], 0),
    ),
  );
}

function activate(value) {
  if ("${activation}" === "relu") return Math.max(0, value);
  if ("${activation}" === "sigmoid") return 1 / (1 + Math.exp(-value));
  return Math.tanh(value);
}

function forward(batch) {
  let a = batch;
  for (let layer = 0; layer < weights.length - 1; layer += 1) {
    a = addBias(matMul(a, weights[layer]), biases[layer])
      .map((row) => row.map(activate));
  }
  return addBias(matMul(a, weights.at(-1)), biases.at(-1))
    .map((row) => row.map((logit) => 1 / (1 + Math.exp(-logit))));
}`;
  },
});

const activationFunctions = makeConceptAlgorithm({
  id: "activation-functions",
  name: "Activation Functions",
  category: "Activation Functions — ReLU, Sigmoid, GELU",
  summary: "Compares activation curves, derivative flow, and dying-ReLU neuron states.",
  parameters: [
    {
      kind: "range",
      id: "inputX",
      label: "Input x",
      min: -6,
      max: 6,
      step: 0.1,
      defaultValue: -1.5,
      format: "decimal",
    },
    {
      kind: "select",
      id: "selectedActivation",
      label: "Activation",
      defaultValue: "relu",
      options: [
        { label: "ReLU", value: "relu" },
        { label: "Sigmoid", value: "sigmoid" },
        { label: "LeakyReLU", value: "leaky-relu" },
        { label: "GELU", value: "gelu" },
      ],
    },
    {
      kind: "range",
      id: "leakySlope",
      label: "Leaky slope",
      min: 0.01,
      max: 0.3,
      step: 0.01,
      defaultValue: 0.06,
      format: "decimal",
    },
    {
      kind: "range",
      id: "negativeShift",
      label: "Data shift",
      min: -4,
      max: 2,
      step: 0.1,
      defaultValue: -1.6,
      format: "decimal",
    },
    {
      kind: "range",
      id: "learningRate",
      label: "Learning rate",
      min: 0.01,
      max: 1.2,
      step: 0.01,
      defaultValue: 0.35,
      format: "decimal",
    },
  ],
  sample: makeActivationDataset,
  formulas: [
    { title: "Sigmoid", expression: "\\sigma(x)=\\frac{1}{1+e^{-x}},\\quad \\sigma'(x)=\\sigma(x)(1-\\sigma(x))" },
    { title: "ReLU", expression: "\\operatorname{ReLU}(x)=\\max(0,x),\\quad \\operatorname{ReLU}'(x)=\\mathbb{1}_{x>0}" },
    { title: "LeakyReLU", expression: "f(x)=\\max(\\alpha x,x),\\quad f'(x)=\\alpha\\;\\text{if}\\;x<0\\;\\text{else}\\;1" },
    { title: "GELU", expression: "\\operatorname{GELU}(x)\\approx \\frac{x}{2}\\left(1+\\tanh\\left(\\sqrt{2/\\pi}(x+0.044715x^3)\\right)\\right)" },
  ],
  explanation: [
    "Activation functions decide how much signal a neuron passes forward and how much gradient can return during backpropagation.",
    "Sigmoid squashes values into 0 to 1, but its derivative gets tiny at both extremes, which is the classic vanishing-gradient shape.",
    "ReLU is fast and sparse, but a neuron pushed negative can receive exactly zero gradient. LeakyReLU and GELU preserve some negative-side gradient, so the same units can recover.",
  ],
  engine: (_, params) => {
    const selected = activationKeyParam(params, "selectedActivation", "relu");
    const targetX = numberParam(params, "inputX", -1.5);
    const leakySlope = numberParam(params, "leakySlope", 0.06);
    const negativeShift = numberParam(params, "negativeShift", -1.6);
    const learningRate = numberParam(params, "learningRate", 0.35);
    const frameCount = 22;
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const progress = index / (frameCount - 1);
      const animatedX = targetX * (0.18 + progress * 0.82);
      const curves = makeActivationCurves(animatedX, leakySlope);
      const neurons = makeActivationNeurons(selected, negativeShift, learningRate, leakySlope);
      const selectedCurve = curves.find((curve) => curve.id === selected) ?? curves[0];
      const deadCount = neurons.filter((neuron) => neuron.status === "dead").length;
      const saturatedCount = neurons.filter((neuron) => neuron.status === "saturated").length;
      const recoveryCount = neurons.filter((neuron) => neuron.status === "recovering").length;

      return {
        type: "concept-demo" as const,
        iteration: index + 1,
        points: selectedCurve.points,
        activation: {
          selected,
          inputX: animatedX,
          targetX,
          learningRate,
          negativeShift,
          leakySlope,
          curves,
          neurons,
          deadCount,
          saturatedCount,
          recoveryCount,
        },
        summary: `${selectedCurve.label} f(${animatedX.toFixed(2)})=${selectedCurve.value.toFixed(3)} · f'=${selectedCurve.derivative.toFixed(3)} · ${deadCount} dead · ${recoveryCount} recovering`,
      };
    });
    const final = frames[frames.length - 1].activation;
    const selectedCurve = final.curves.find((curve) => curve.id === selected) ?? final.curves[0];

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "f(x)", value: selectedCurve.value.toFixed(4) },
        { label: "f'(x)", value: selectedCurve.derivative.toFixed(4) },
        { label: "Dead neurons", value: String(final.deadCount) },
        {
          label: "Recovering",
          value: selected === "leaky-relu" || selected === "gelu" ? String(final.recoveryCount) : "0",
        },
      ],
    };
  },
  python: (params) => {
    const slope = numberParam(params, "leakySlope", 0.06);
    return `import numpy as np

alpha = ${slope}
x = np.linspace(-6, 6, 241)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    y = sigmoid(x)
    return y * (1 - y)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    mask = x > 0
    return mask.astype(float)

def leaky_relu(x, alpha=alpha):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_grad(x, alpha=alpha):
    return np.where(x > 0, 1.0, alpha)

def gelu(x):
    u = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1 + np.tanh(u))

dead_relu_mask = (x < 0) & (relu_grad(x) == 0)
print(dead_relu_mask.sum())`;
  },
  javascript: (params) => {
    const slope = numberParam(params, "leakySlope", 0.06);
    return `const alpha = ${slope};

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const sigmoidGrad = (x) => {
  const y = sigmoid(x);
  return y * (1 - y);
};

const relu = (x) => Math.max(0, x);
const reluGrad = (x) => (x > 0 ? 1 : 0);

const leakyRelu = (x) => (x > 0 ? x : alpha * x);
const leakyReluGrad = (x) => (x > 0 ? 1 : alpha);

function gelu(x) {
  const u = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3);
  return 0.5 * x * (1 + Math.tanh(u));
}

// TensorFlow.js equivalent:
const yRelu = tf.relu(tf.tensor([-2, -1, 0, 1, 2]));
const deadReluMask = [-2, -1, 0, 1, 2].map((value) => reluGrad(value) === 0);`;
  },
});

function makeActivationDataset() {
  return makeDataset(
    "Activation input sweep",
    Array.from({ length: 121 }, (_, index) => {
      const x = -6 + index * 0.1;
      return { x: round(x), y: round(sigmoid(x)), label: "sigmoid" };
    }),
  );
}

function activationKeyParam(
  params: ParameterState,
  key: string,
  fallback: ActivationFunctionKey,
): ActivationFunctionKey {
  const value = stringParam(params, key, fallback);
  return value === "sigmoid" || value === "relu" || value === "leaky-relu" || value === "gelu"
    ? value
    : fallback;
}

function makeActivationCurves(inputX: number, leakySlope: number): ActivationCurve[] {
  const definitions: Array<{ id: ActivationFunctionKey; label: string; color: string }> = [
    { id: "sigmoid", label: "Sigmoid", color: "#2f6fbe" },
    { id: "relu", label: "ReLU", color: "#d34a43" },
    { id: "leaky-relu", label: "LeakyReLU", color: "#0f766e" },
    { id: "gelu", label: "GELU", color: "#6f58c9" },
  ];
  const xs = Array.from({ length: 241 }, (_, index) => -6 + index * 0.05);

  return definitions.map((definition) => ({
    ...definition,
    points: xs.map((x) => ({
      x: round(x, 3),
      y: round(activationValue(definition.id, x, leakySlope), 4),
      label: definition.id,
    })),
    derivativePoints: xs.map((x) => ({
      x: round(x, 3),
      y: round(activationDerivative(definition.id, x, leakySlope), 4),
      label: definition.id,
    })),
    value: activationValue(definition.id, inputX, leakySlope),
    derivative: activationDerivative(definition.id, inputX, leakySlope),
  }));
}

function activationValue(kind: ActivationFunctionKey, x: number, leakySlope: number) {
  if (kind === "sigmoid") {
    return sigmoid(x);
  }

  if (kind === "relu") {
    return Math.max(0, x);
  }

  if (kind === "leaky-relu") {
    return x > 0 ? x : leakySlope * x;
  }

  return gelu(x);
}

function activationDerivative(kind: ActivationFunctionKey, x: number, leakySlope: number) {
  if (kind === "sigmoid") {
    const y = sigmoid(x);
    return y * (1 - y);
  }

  if (kind === "relu") {
    return x > 0 ? 1 : 0;
  }

  if (kind === "leaky-relu") {
    return x > 0 ? 1 : leakySlope;
  }

  return geluDerivative(x);
}

function gelu(x: number) {
  const u = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3);
  return 0.5 * x * (1 + Math.tanh(u));
}

function geluDerivative(x: number) {
  const u = Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3);
  const tanh = Math.tanh(u);
  const du = Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * x ** 2);
  return 0.5 * (1 + tanh) + 0.5 * x * (1 - tanh ** 2) * du;
}

function makeActivationNeurons(
  selected: ActivationFunctionKey,
  negativeShift: number,
  learningRate: number,
  leakySlope: number,
): ActivationNeuronState[] {
  const highStepPenalty = Math.max(0, learningRate - 0.72) * 2.8;
  const deathZone = selected === "relu" && (negativeShift < -1.25 || learningRate > 0.78);

  return Array.from({ length: 8 }, (_, index) => {
    const preActivation = negativeShift + (index - 3.5) * 0.52 - highStepPenalty;
    const activation = activationValue(selected, preActivation, leakySlope);
    const derivative = activationDerivative(selected, preActivation, leakySlope);
    let status: ActivationNeuronState["status"] = derivative > 0.08 ? "alive" : "inactive";

    if (selected === "relu" && preActivation <= 0) {
      status = deathZone ? "dead" : "inactive";
    } else if ((selected === "leaky-relu" || selected === "gelu") && preActivation < 0) {
      status = "recovering";
    } else if (selected === "sigmoid" && derivative < 0.035) {
      status = "saturated";
    }

    return {
      id: `activation-h${index}`,
      index,
      label: `h${index + 1}`,
      preActivation: round(preActivation, 4),
      activation: round(activation, 4),
      derivative: round(derivative, 4),
      status,
    };
  });
}

const lossFunctions = makeConceptAlgorithm({
  id: "loss-functions",
  name: "Loss Functions",
  category: "Loss Functions — MSE, Cross-Entropy, Contrastive",
  summary: "Tracks training loss over epochs and shows contrastive pairs pulling or pushing in vector space.",
  parameters: [
    {
      kind: "select",
      id: "lossFunction",
      label: "Loss function",
      defaultValue: "cross-entropy",
      options: [
        { label: "Cross-Entropy", value: "cross-entropy" },
        { label: "MSE", value: "mse" },
        { label: "Contrastive", value: "contrastive" },
      ],
    },
    {
      kind: "range",
      id: "learningRate",
      label: "Learning rate",
      min: 0.01,
      max: 1.4,
      step: 0.01,
      defaultValue: 0.18,
      format: "decimal",
    },
    {
      kind: "range",
      id: "epochs",
      label: "Epochs",
      min: 12,
      max: 120,
      step: 4,
      defaultValue: 64,
      format: "integer",
    },
    {
      kind: "range",
      id: "margin",
      label: "Contrastive margin",
      min: 0.5,
      max: 3,
      step: 0.05,
      defaultValue: 1.35,
      format: "decimal",
    },
  ],
  sample: makeLossDataset,
  formulas: [
    { title: "Mean squared error", expression: "\\operatorname{MSE}=\\frac{1}{N}\\sum_i(\\hat{y}_i-y_i)^2" },
    { title: "Cross-entropy", expression: "\\operatorname{CE}=-\\frac{1}{N}\\sum_i\\sum_c y_{ic}\\log(\\hat{y}_{ic}+\\epsilon)" },
    { title: "Contrastive loss", expression: "L=yD^2+(1-y)\\max(0,m-D)^2" },
  ],
  explanation: [
    "A loss function turns model behavior into one scalar objective that training can minimize.",
    "The loss curve shows whether the current learning rate is helping the model descend or making the objective oscillate upward.",
    "Contrastive loss works in vector space: similar pairs are pulled together, while dissimilar pairs inside the margin are pushed apart.",
  ],
  engine: (_, params) => {
    const selected = lossKeyParam(params, "lossFunction", "cross-entropy");
    const learningRate = numberParam(params, "learningRate", 0.18);
    const epochs = Math.round(numberParam(params, "epochs", 64));
    const margin = numberParam(params, "margin", 1.35);
    const mlpTrace = selected === "mse" ? trainMseMlp(epochs, learningRate) : trainCrossEntropyMlp(epochs, learningRate);
    const contrastiveTrace = trainContrastiveSpace(epochs, learningRate, margin);
    const selectedHistory =
      selected === "contrastive" ? contrastiveTrace.lossHistory : mlpTrace.lossHistory;
    const frames = selectedHistory.map((point, index) => {
      const contrastiveFrame = contrastiveTrace.frames[Math.min(index, contrastiveTrace.frames.length - 1)];
      const mlpFrame = mlpTrace.frames[Math.min(index, mlpTrace.frames.length - 1)];
      const history = selectedHistory.slice(0, index + 1);
      const status = lossTrendStatus(history);

      return {
        type: "concept-demo" as const,
        iteration: point.x,
        points: history,
        loss: {
          selected,
          epoch: point.x,
          learningRate,
          margin,
          lossHistory: history,
          currentLoss: point.y,
          status,
          predictions: selected === "contrastive" ? contrastivePredictions(contrastiveFrame.pairs) : mlpFrame.predictions,
          contrastivePairs: contrastiveFrame.pairs,
        },
        summary: `${lossLabel(selected)} epoch ${point.x} · loss ${point.y.toFixed(4)} · ${status}`,
      };
    });
    const finalFrame = frames[frames.length - 1].loss;
    const firstLoss = selectedHistory[0]?.y ?? 0;
    const finalLoss = finalFrame.currentLoss;
    const percentDrop = firstLoss > 0 ? Math.max(-9.99, Math.min(0.999, (firstLoss - finalLoss) / firstLoss)) : 0;

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Final loss", value: finalLoss.toFixed(4) },
        { label: "Loss drop", value: `${Math.round(percentDrop * 100)}%` },
        { label: "Trend", value: finalFrame.status },
        { label: "Objective", value: lossLabel(selected) },
      ],
    };
  },
  python: () => `import numpy as np

eps = 1e-15
N = y.shape[0]

def mse(y, y_hat):
    return np.mean((y_hat - y) ** 2)

def cross_entropy(y, y_hat):
    return -np.sum(y * np.log(y_hat + 1e-15)) / N

def contrastive_loss(anchor, pair, similar, margin=1.35):
    diff = anchor - pair
    distance = np.sqrt(np.sum(diff * diff, axis=1) + eps)
    positive = similar * distance**2
    negative = (1 - similar) * np.maximum(0, margin - distance)**2
    return np.mean(positive + negative)

loss = cross_entropy(y_one_hot, y_hat)`,
  javascript: () => `const EPS = 1e-15;

function mse(y, yHat) {
  let total = 0;
  for (let i = 0; i < y.length; i += 1) {
    total += (yHat[i] - y[i]) ** 2;
  }
  return total / y.length;
}

function crossEntropy(y, yHat) {
  let total = 0;
  for (let row = 0; row < y.length; row += 1) {
    for (let column = 0; column < y[row].length; column += 1) {
      total -= y[row][column] * Math.log(yHat[row][column] + EPS);
    }
  }
  return total / y.length;
}

function contrastiveLoss(anchor, pair, similar, margin = 1.35) {
  const dx = anchor.x - pair.x;
  const dy = anchor.y - pair.y;
  const distance = Math.hypot(dx, dy);
  return similar
    ? distance ** 2
    : Math.max(0, margin - distance) ** 2;
}

const tfLoss = tf.losses.softmaxCrossEntropy(
  tf.tensor(yOneHot),
  tf.tensor(yHat),
);`,
});

function makeLossDataset() {
  return makeDataset(
    "Generated loss training sample",
    xorTrainingSet().map((item) => ({
      x: item.x[0],
      y: item.x[1],
      label: item.y === 1 ? "positive" : "negative",
    })),
  );
}

function lossKeyParam(
  params: ParameterState,
  key: string,
  fallback: LossFunctionKey,
): LossFunctionKey {
  const value = stringParam(params, key, fallback);
  return value === "mse" || value === "cross-entropy" || value === "contrastive"
    ? value
    : fallback;
}

function xorTrainingSet() {
  return [
    { x: [-1, -1] as [number, number], y: 0 },
    { x: [-1, 1] as [number, number], y: 1 },
    { x: [1, -1] as [number, number], y: 1 },
    { x: [1, 1] as [number, number], y: 0 },
  ];
}

function trainCrossEntropyMlp(epochs: number, learningRate: number) {
  const data = xorTrainingSet();
  const hiddenWidth = 4;
  const w1 = [
    [0.45, -0.6, 0.28, 0.7],
    [-0.35, 0.52, 0.62, -0.24],
  ];
  const b1 = [0.02, -0.04, 0.01, 0.03];
  const w2 = [
    [0.36, -0.31],
    [-0.22, 0.41],
    [0.29, -0.18],
    [-0.44, 0.35],
  ];
  const b2 = [0.02, -0.02];
  const frames: Array<{ loss: number; predictions: LossPrediction[] }> = [];

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const gradW1 = zeros(2, hiddenWidth);
    const gradB1 = new Array(hiddenWidth).fill(0);
    const gradW2 = zeros(hiddenWidth, 2);
    const gradB2 = [0, 0];
    let loss = 0;
    const predictions: LossPrediction[] = [];

    data.forEach((sample, sampleIndex) => {
      const hiddenRaw = b1.map((bias, hidden) => sample.x[0] * w1[0][hidden] + sample.x[1] * w1[1][hidden] + bias);
      const hidden = hiddenRaw.map(Math.tanh);
      const logits = b2.map((bias, output) =>
        bias + hidden.reduce((total, value, hiddenIndex) => total + value * w2[hiddenIndex][output], 0),
      );
      const probabilities = softmax(logits);
      const target = [sample.y === 0 ? 1 : 0, sample.y === 1 ? 1 : 0];
      loss += -target.reduce((total, value, classIndex) => total + value * Math.log(probabilities[classIndex] + 1e-15), 0);
      predictions.push({
        label: `x${sampleIndex + 1}`,
        target: sample.y,
        prediction: probabilities[1],
      });
      const deltaOut = probabilities.map((probability, classIndex) => probability - target[classIndex]);

      hidden.forEach((hiddenValue, hiddenIndex) => {
        deltaOut.forEach((delta, classIndex) => {
          gradW2[hiddenIndex][classIndex] += hiddenValue * delta;
        });
      });
      deltaOut.forEach((delta, classIndex) => {
        gradB2[classIndex] += delta;
      });

      const deltaHidden = hidden.map((hiddenValue, hiddenIndex) => {
        const flow = deltaOut.reduce((total, delta, classIndex) => total + delta * w2[hiddenIndex][classIndex], 0);
        return flow * (1 - hiddenValue ** 2);
      });
      sample.x.forEach((inputValue, inputIndex) => {
        deltaHidden.forEach((delta, hiddenIndex) => {
          gradW1[inputIndex][hiddenIndex] += inputValue * delta;
        });
      });
      deltaHidden.forEach((delta, hiddenIndex) => {
        gradB1[hiddenIndex] += delta;
      });
    });

    const scale = learningRate / data.length;
    applyMatrixStep(w1, gradW1, scale);
    applyVectorStep(b1, gradB1, scale);
    applyMatrixStep(w2, gradW2, scale);
    applyVectorStep(b2, gradB2, scale);
    frames.push({ loss: loss / data.length, predictions });
  }

  return {
    frames,
    lossHistory: frames.map((frame, index) => ({ x: index + 1, y: frame.loss, label: "cross-entropy" })),
  };
}

function trainMseMlp(epochs: number, learningRate: number) {
  const data = xorTrainingSet();
  const hiddenWidth = 4;
  const w1 = [
    [0.5, -0.42, 0.36, 0.66],
    [-0.31, 0.47, 0.58, -0.38],
  ];
  const b1 = [0.04, -0.02, 0.02, -0.03];
  const w2 = [0.42, -0.35, 0.28, -0.46];
  let b2 = 0.02;
  const frames: Array<{ loss: number; predictions: LossPrediction[] }> = [];

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    const gradW1 = zeros(2, hiddenWidth);
    const gradB1 = new Array(hiddenWidth).fill(0);
    const gradW2 = new Array(hiddenWidth).fill(0);
    let gradB2 = 0;
    let loss = 0;
    const predictions: LossPrediction[] = [];

    data.forEach((sample, sampleIndex) => {
      const hiddenRaw = b1.map((bias, hidden) => sample.x[0] * w1[0][hidden] + sample.x[1] * w1[1][hidden] + bias);
      const hidden = hiddenRaw.map(Math.tanh);
      const logit = b2 + hidden.reduce((total, value, hiddenIndex) => total + value * w2[hiddenIndex], 0);
      const prediction = sigmoid(logit);
      const error = prediction - sample.y;
      loss += error ** 2;
      predictions.push({ label: `x${sampleIndex + 1}`, target: sample.y, prediction });

      const deltaOut = 2 * error * prediction * (1 - prediction);
      hidden.forEach((hiddenValue, hiddenIndex) => {
        gradW2[hiddenIndex] += hiddenValue * deltaOut;
      });
      gradB2 += deltaOut;
      const deltaHidden = hidden.map((hiddenValue, hiddenIndex) => deltaOut * w2[hiddenIndex] * (1 - hiddenValue ** 2));
      sample.x.forEach((inputValue, inputIndex) => {
        deltaHidden.forEach((delta, hiddenIndex) => {
          gradW1[inputIndex][hiddenIndex] += inputValue * delta;
        });
      });
      deltaHidden.forEach((delta, hiddenIndex) => {
        gradB1[hiddenIndex] += delta;
      });
    });

    const scale = learningRate / data.length;
    applyMatrixStep(w1, gradW1, scale);
    applyVectorStep(b1, gradB1, scale);
    applyVectorStep(w2, gradW2, scale);
    b2 -= scale * gradB2;
    frames.push({ loss: loss / data.length, predictions });
  }

  return {
    frames,
    lossHistory: frames.map((frame, index) => ({ x: index + 1, y: frame.loss, label: "mse" })),
  };
}

type ContrastiveTraceFrame = {
  loss: number;
  pairs: ContrastivePair[];
};

function trainContrastiveSpace(epochs: number, learningRate: number, margin: number) {
  const points = new Map<string, ContrastivePoint>(
    [
      { id: "a1", x: -2.1, y: 0.72, label: "A1" },
      { id: "a2", x: -1.16, y: -0.26, label: "A2" },
      { id: "a3", x: -2.42, y: -0.84, label: "A3" },
      { id: "b1", x: 1.44, y: 0.8, label: "B1" },
      { id: "b2", x: 2.2, y: -0.5, label: "B2" },
      { id: "b3", x: 1.05, y: -1.12, label: "B3" },
    ].map((point) => [point.id, { ...point }]),
  );
  const pairDefs = [
    { id: "p1", from: "a1", to: "a2", relation: "similar" as const },
    { id: "p2", from: "a2", to: "a3", relation: "similar" as const },
    { id: "p3", from: "b1", to: "b2", relation: "similar" as const },
    { id: "p4", from: "b2", to: "b3", relation: "similar" as const },
    { id: "p5", from: "a1", to: "b2", relation: "different" as const },
    { id: "p6", from: "a3", to: "b3", relation: "different" as const },
  ];
  const frames: ContrastiveTraceFrame[] = [];
  const stepScale = Math.min(0.09, Math.max(0.004, learningRate * 0.04));

  for (let epoch = 1; epoch <= epochs; epoch += 1) {
    let totalLoss = 0;
    const pairsForFrame: ContrastivePair[] = [];

    pairDefs.forEach((pair) => {
      const from = points.get(pair.from);
      const to = points.get(pair.to);
      if (!from || !to) {
        return;
      }

      const dx = from.x - to.x;
      const dy = from.y - to.y;
      const distance = Math.max(0.0001, Math.hypot(dx, dy));
      let loss = 0;
      let gradX = 0;
      let gradY = 0;

      if (pair.relation === "similar") {
        loss = distance ** 2;
        gradX = 2 * dx;
        gradY = 2 * dy;
      } else if (distance < margin) {
        const marginError = margin - distance;
        loss = marginError ** 2;
        gradX = (-2 * marginError * dx) / distance;
        gradY = (-2 * marginError * dy) / distance;
      }

      from.x -= stepScale * gradX;
      from.y -= stepScale * gradY;
      to.x += stepScale * gradX;
      to.y += stepScale * gradY;
      totalLoss += loss;
      pairsForFrame.push({
        id: pair.id,
        from: { ...from },
        to: { ...to },
        relation: pair.relation,
        distance,
        loss,
        gradientMagnitude: Math.hypot(gradX, gradY),
      });
    });

    frames.push({ loss: totalLoss / pairDefs.length, pairs: pairsForFrame });
  }

  return {
    frames,
    lossHistory: frames.map((frame, index) => ({ x: index + 1, y: frame.loss, label: "contrastive" })),
  };
}

function contrastivePredictions(pairs: ContrastivePair[]): LossPrediction[] {
  return pairs.slice(0, 4).map((pair) => ({
    label: pair.id,
    target: pair.relation === "similar" ? 1 : 0,
    prediction: pair.distance,
  }));
}

function lossTrendStatus(history: DataPoint[]): "converging" | "diverging" | "oscillating" {
  if (history.length < 5) {
    return "converging";
  }

  const first = history[0].y;
  const last = history[history.length - 1].y;
  const previous = history[history.length - 2].y;
  const recent = history.slice(-8).map((point) => point.y);
  const directionChanges = recent.slice(2).filter((value, index) => {
    const a = recent[index + 1] - recent[index];
    const b = value - recent[index + 1];
    return Math.sign(a) !== Math.sign(b) && Math.abs(a) > 0.002 && Math.abs(b) > 0.002;
  }).length;

  if (last > first * 1.18 || last > previous * 1.16) {
    return "diverging";
  }

  if (directionChanges >= 3) {
    return "oscillating";
  }

  return "converging";
}

function lossLabel(loss: LossFunctionKey) {
  if (loss === "mse") {
    return "MSE";
  }

  if (loss === "contrastive") {
    return "Contrastive";
  }

  return "Cross-Entropy";
}

function zeros(rows: number, columns: number) {
  return Array.from({ length: rows }, () => new Array(columns).fill(0));
}

function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function applyMatrixStep(matrix: number[][], gradients: number[][], scale: number) {
  matrix.forEach((row, rowIndex) => {
    row.forEach((_, columnIndex) => {
      row[columnIndex] -= scale * gradients[rowIndex][columnIndex];
    });
  });
}

function applyVectorStep(values: number[], gradients: number[], scale: number) {
  values.forEach((_, index) => {
    values[index] -= scale * gradients[index];
  });
}

const optimizerRace = makeConceptAlgorithm({
  id: "optimizer-race",
  name: "Optimizer Race",
  category: "SGD, Momentum, Adam, AdamW",
  summary: "Races SGD, Momentum, Adam, and AdamW across a ravine-shaped loss surface.",
  parameters: [
    {
      kind: "select",
      id: "surface",
      label: "Loss surface",
      defaultValue: "rosenbrock",
      options: [
        { label: "Rosenbrock banana", value: "rosenbrock" },
        { label: "Beale function", value: "beale" },
      ],
    },
    {
      kind: "range",
      id: "learningRate",
      label: "Learning rate eta",
      min: 0.001,
      max: 0.08,
      step: 0.001,
      defaultValue: 0.018,
      format: "decimal",
    },
    {
      kind: "range",
      id: "beta1",
      label: "beta1",
      min: 0.2,
      max: 0.99,
      step: 0.01,
      defaultValue: 0.9,
      format: "decimal",
    },
    {
      kind: "range",
      id: "beta2",
      label: "beta2",
      min: 0.7,
      max: 0.999,
      step: 0.001,
      defaultValue: 0.98,
      format: "decimal",
    },
    {
      kind: "range",
      id: "weightDecay",
      label: "Weight decay lambda",
      min: 0,
      max: 0.2,
      step: 0.005,
      defaultValue: 0.035,
      format: "decimal",
    },
    {
      kind: "range",
      id: "startX",
      label: "Start x",
      min: -2.2,
      max: 2.2,
      step: 0.1,
      defaultValue: -1.4,
      format: "decimal",
    },
    {
      kind: "range",
      id: "startY",
      label: "Start y",
      min: -1.2,
      max: 2.8,
      step: 0.1,
      defaultValue: 2.1,
      format: "decimal",
    },
    {
      kind: "range",
      id: "steps",
      label: "Race steps",
      min: 30,
      max: 150,
      step: 5,
      defaultValue: 90,
      format: "integer",
    },
  ],
  sample: makeOptimizerDataset,
  formulas: [
    { title: "SGD", expression: "\\theta_t=\\theta_{t-1}-\\eta\\nabla L(\\theta_{t-1})" },
    { title: "Momentum", expression: "v_t=\\beta v_{t-1}+\\eta\\nabla L(\\theta),\\quad \\theta_t=\\theta_{t-1}-v_t" },
    { title: "Adam", expression: "m_t=\\beta_1m_{t-1}+(1-\\beta_1)g_t,\\;v_t=\\beta_2v_{t-1}+(1-\\beta_2)g_t^2" },
    { title: "AdamW", expression: "\\theta_t=\\theta_{t-1}-\\eta\\left(\\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}+\\lambda\\theta_{t-1}\\right)" },
  ],
  explanation: [
    "Optimizers are update rules for moving parameters downhill on a loss surface.",
    "SGD follows the raw gradient, Momentum accumulates velocity, Adam scales each coordinate by a running second moment, and AdamW decouples weight decay from the gradient estimate.",
    "The adaptive-rate graph shows why Adam calms large-gradient weights: a bigger denominator sqrt(v-hat) compresses that parameter's effective step size.",
  ],
  engine: (_, params) => {
    const surface = surfaceParam(params, "surface", "rosenbrock");
    const learningRate = numberParam(params, "learningRate", 0.018);
    const beta1 = numberParam(params, "beta1", 0.9);
    const beta2 = numberParam(params, "beta2", 0.98);
    const weightDecay = numberParam(params, "weightDecay", 0.035);
    const steps = Math.round(numberParam(params, "steps", 90));
    const rawStart = {
      x: numberParam(params, "startX", -1.4),
      y: numberParam(params, "startY", 2.1),
    };
    const start = clampSurfacePoint(surface, rawStart);
    const histories = runOptimizerRace(surface, start, steps, learningRate, beta1, beta2, weightDecay);
    const frames = Array.from({ length: steps + 1 }, (_, step) => {
      const runners = optimizerDefinitions.map((optimizer) => {
        const history = histories[optimizer.id].slice(0, step + 1);
        return {
          id: optimizer.id,
          label: optimizer.label,
          color: optimizer.color,
          history,
          position: history[history.length - 1],
        };
      });
      const leader = [...runners].sort((a, b) => a.position.z - b.position.z)[0];

      return {
        type: "concept-demo" as const,
        iteration: step,
        points: runners.map((runner) => ({ x: runner.position.x, y: runner.position.z, label: runner.label })),
        optimizer: {
          surface,
          step,
          learningRate,
          beta1,
          beta2,
          weightDecay,
          start,
          runners,
          adaptiveWeights: makeAdaptiveOptimizerWeights(step, learningRate, beta1, beta2, weightDecay),
        },
        summary: `Step ${step} · leader ${leader.label} · loss ${leader.position.z.toFixed(4)} · eta ${learningRate.toFixed(3)}`,
      };
    });
    const final = frames[frames.length - 1].optimizer;
    const finalRunners = final.runners;
    const leader = [...finalRunners].sort((a, b) => a.position.z - b.position.z)[0];
    const adam = finalRunners.find((runner) => runner.id === "adam");
    const sgd = finalRunners.find((runner) => runner.id === "sgd");

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Leader", value: leader.label },
        { label: "Leader loss", value: leader.position.z.toFixed(4) },
        { label: "Adam loss", value: adam ? adam.position.z.toFixed(4) : "-" },
        { label: "SGD loss", value: sgd ? sgd.position.z.toFixed(4) : "-" },
      ],
    };
  },
  python: (params) => `class AdamWFromScratch:
    def __init__(self, lr=${numberParam(params, "learningRate", 0.018)}, beta1=${numberParam(params, "beta1", 0.9)}, beta2=${numberParam(params, "beta2", 0.98)}, weight_decay=${numberParam(params, "weightDecay", 0.035)}):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, weights, grads):
        self.t += 1
        for w_id, dw in grads.items():
            self.m[w_id] = self.beta1 * self.m.get(w_id, 0.0) + (1 - self.beta1) * dw
            self.v[w_id] = self.beta2 * self.v.get(w_id, 0.0) + (1 - self.beta2) * (dw ** 2)
            m_hat = self.m[w_id] / (1 - self.beta1 ** self.t)
            v_hat = self.v[w_id] / (1 - self.beta2 ** self.t)
            adam_step = m_hat / ((v_hat ** 0.5) + 1e-8)
            weights[w_id] -= self.lr * (adam_step + self.weight_decay * weights[w_id])

        return weights`,
  javascript: (params) => `const state = { m: {}, v: {}, t: 0 };
const lr = ${numberParam(params, "learningRate", 0.018)};
const beta1 = ${numberParam(params, "beta1", 0.9)};
const beta2 = ${numberParam(params, "beta2", 0.98)};
const weightDecay = ${numberParam(params, "weightDecay", 0.035)};

function adamWStep(weights, gradients) {
  state.t += 1;
  for (const [wId, dw] of Object.entries(gradients)) {
    state.m[wId] = beta1 * (state.m[wId] ?? 0) + (1 - beta1) * dw;
    state.v[wId] = beta2 * (state.v[wId] ?? 0) + (1 - beta2) * dw ** 2;
    const mHat = state.m[wId] / (1 - beta1 ** state.t);
    const vHat = state.v[wId] / (1 - beta2 ** state.t);
    const adaptiveStep = mHat / (Math.sqrt(vHat) + 1e-8);
    weights[wId] -= lr * (adaptiveStep + weightDecay * weights[wId]);
  }
}

const tfAdamW = tf.train.adam(lr, beta1, beta2);
// Apply decoupled weight decay manually around tfAdamW if needed.`,
});

const optimizerDefinitions: Array<{ id: OptimizerKey; label: string; color: string }> = [
  { id: "sgd", label: "SGD", color: "#d34a43" },
  { id: "momentum", label: "Momentum", color: "#b7791f" },
  { id: "adam", label: "Adam", color: "#2f6fbe" },
  { id: "adamw", label: "AdamW", color: "#0f766e" },
];

function makeOptimizerDataset() {
  return makeDataset(
    "Optimizer surface starts",
    optimizerDefinitions.map((optimizer, index) => ({
      x: -1.4 + index * 0.06,
      y: 2.1 - index * 0.06,
      label: optimizer.label,
    })),
  );
}

function surfaceParam(
  params: ParameterState,
  key: string,
  fallback: LossSurfaceKey,
): LossSurfaceKey {
  const value = stringParam(params, key, fallback);
  return value === "beale" || value === "rosenbrock" ? value : fallback;
}

function runOptimizerRace(
  surface: LossSurfaceKey,
  start: { x: number; y: number },
  steps: number,
  learningRate: number,
  beta1: number,
  beta2: number,
  weightDecay: number,
) {
  const histories = Object.fromEntries(
    optimizerDefinitions.map((optimizer) => [
      optimizer.id,
      [{ step: 0, x: start.x, y: start.y, z: surfaceValue(surface, start.x, start.y) }],
    ]),
  ) as Record<OptimizerKey, Array<{ step: number; x: number; y: number; z: number }>>;
  const states = Object.fromEntries(
    optimizerDefinitions.map((optimizer) => [
      optimizer.id,
      {
        x: start.x,
        y: start.y,
        velocity: [0, 0],
        m: [0, 0],
        v: [0, 0],
      },
    ]),
  ) as Record<OptimizerKey, { x: number; y: number; velocity: number[]; m: number[]; v: number[] }>;

  for (let step = 1; step <= steps; step += 1) {
    optimizerDefinitions.forEach((optimizer) => {
      const state = states[optimizer.id];
      const gradient = clippedGradient(surfaceGradient(surface, state.x, state.y), 90);
      const next = optimizerStep(optimizer.id, state, gradient, step, learningRate, beta1, beta2, weightDecay);
      const clamped = clampSurfacePoint(surface, next);
      state.x = clamped.x;
      state.y = clamped.y;
      histories[optimizer.id].push({
        step,
        x: state.x,
        y: state.y,
        z: surfaceValue(surface, state.x, state.y),
      });
    });
  }

  return histories;
}

function optimizerStep(
  optimizer: OptimizerKey,
  state: { x: number; y: number; velocity: number[]; m: number[]; v: number[] },
  gradient: [number, number],
  step: number,
  learningRate: number,
  beta1: number,
  beta2: number,
  weightDecay: number,
) {
  if (optimizer === "sgd") {
    const wobble = Math.sin(step * 0.8) * learningRate * 1.4;
    return {
      x: state.x - learningRate * gradient[0] + wobble,
      y: state.y - learningRate * gradient[1] - wobble * 0.5,
    };
  }

  if (optimizer === "momentum") {
    state.velocity[0] = beta1 * state.velocity[0] + learningRate * gradient[0];
    state.velocity[1] = beta1 * state.velocity[1] + learningRate * gradient[1];
    return {
      x: state.x - state.velocity[0],
      y: state.y - state.velocity[1],
    };
  }

  state.m[0] = beta1 * state.m[0] + (1 - beta1) * gradient[0];
  state.m[1] = beta1 * state.m[1] + (1 - beta1) * gradient[1];
  state.v[0] = beta2 * state.v[0] + (1 - beta2) * gradient[0] ** 2;
  state.v[1] = beta2 * state.v[1] + (1 - beta2) * gradient[1] ** 2;
  const mHat = [
    state.m[0] / (1 - beta1 ** step),
    state.m[1] / (1 - beta1 ** step),
  ];
  const vHat = [
    state.v[0] / (1 - beta2 ** step),
    state.v[1] / (1 - beta2 ** step),
  ];
  const adamMove = [
    learningRate * mHat[0] / (Math.sqrt(vHat[0]) + 1e-8),
    learningRate * mHat[1] / (Math.sqrt(vHat[1]) + 1e-8),
  ];

  if (optimizer === "adamw") {
    return {
      x: state.x - adamMove[0] - learningRate * weightDecay * state.x,
      y: state.y - adamMove[1] - learningRate * weightDecay * state.y,
    };
  }

  return {
    x: state.x - adamMove[0],
    y: state.y - adamMove[1],
  };
}

function surfaceValue(surface: LossSurfaceKey, x: number, y: number) {
  if (surface === "beale") {
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2;
  }

  return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2;
}

function surfaceGradient(surface: LossSurfaceKey, x: number, y: number): [number, number] {
  if (surface === "beale") {
    const a = 1.5 - x + x * y;
    const b = 2.25 - x + x * y ** 2;
    const c = 2.625 - x + x * y ** 3;
    return [
      2 * a * (-1 + y) + 2 * b * (-1 + y ** 2) + 2 * c * (-1 + y ** 3),
      2 * a * x + 4 * b * x * y + 6 * c * x * y ** 2,
    ];
  }

  return [
    -2 * (1 - x) - 400 * x * (y - x ** 2),
    200 * (y - x ** 2),
  ];
}

function clippedGradient(gradient: [number, number], maxNorm: number): [number, number] {
  const norm = Math.hypot(gradient[0], gradient[1]);
  if (norm <= maxNorm) {
    return gradient;
  }

  const scale = maxNorm / norm;
  return [gradient[0] * scale, gradient[1] * scale];
}

function clampSurfacePoint(surface: LossSurfaceKey, point: { x: number; y: number }) {
  const bounds = surfaceBounds(surface);
  return {
    x: Math.max(bounds.x[0], Math.min(bounds.x[1], point.x)),
    y: Math.max(bounds.y[0], Math.min(bounds.y[1], point.y)),
  };
}

function surfaceBounds(surface: LossSurfaceKey) {
  if (surface === "beale") {
    return { x: [-4.5, 4.5] as [number, number], y: [-4.5, 4.5] as [number, number] };
  }

  return { x: [-2.2, 2.2] as [number, number], y: [-1.2, 2.8] as [number, number] };
}

function makeAdaptiveOptimizerWeights(
  step: number,
  learningRate: number,
  beta1: number,
  beta2: number,
  weightDecay: number,
) {
  const edges = [
    ["x1", "h1"],
    ["x1", "h2"],
    ["x1", "h3"],
    ["x2", "h1"],
    ["x2", "h2"],
    ["x2", "h3"],
    ["h1", "y"],
    ["h2", "y"],
    ["h3", "y"],
  ];

  return edges.map(([from, to], index) => {
    const raw = Math.abs(Math.sin(step * 0.19 + index * 0.77) * (1.1 + (index % 3) * 0.48));
    const gradient = raw + weightDecay * 0.35 + (index % 2 === 0 ? 0.12 : 0.03);
    const vHat = (gradient ** 2) * (1 - beta2 ** Math.max(1, step + 1)) / Math.max(0.0001, 1 - beta2);
    const denominator = Math.sqrt(vHat) + 1e-8;
    const mCompression = 1 - beta1 ** Math.max(1, step + 1);
    return {
      from,
      to,
      gradient,
      denominator,
      adaptiveRate: learningRate * mCompression / denominator,
    };
  });
}

const miniFramework = makeConceptAlgorithm({
  id: "mini-framework",
  name: "Mini-Framework Builder",
  category: "Build a Mini-Framework",
  summary: "Builds a Sequential network from Layer objects and animates forward/backward hooks.",
  parameters: [
    {
      kind: "stepper",
      id: "hiddenLayers",
      label: "Hidden layers",
      min: 1,
      max: 5,
      step: 1,
      defaultValue: 2,
      format: "integer",
    },
    {
      kind: "stepper",
      id: "hiddenUnits",
      label: "Hidden units",
      min: 4,
      max: 32,
      step: 4,
      defaultValue: 8,
      format: "integer",
    },
    {
      kind: "select",
      id: "activation",
      label: "Activation",
      defaultValue: "GELU",
      options: [
        { label: "GELU", value: "GELU" },
        { label: "ReLU", value: "ReLU" },
        { label: "Tanh", value: "Tanh" },
      ],
    },
    {
      kind: "toggle",
      id: "dropout",
      label: "Add dropout masks",
      defaultValue: false,
    },
    {
      kind: "range",
      id: "learningRate",
      label: "Trainer learning rate",
      min: 0.001,
      max: 0.2,
      step: 0.001,
      defaultValue: 0.035,
      format: "decimal",
    },
    {
      kind: "range",
      id: "batchSize",
      label: "Batch size",
      min: 4,
      max: 64,
      step: 4,
      defaultValue: 16,
      format: "integer",
    },
    {
      kind: "action",
      id: "flowStep",
      label: "Flow hook",
      buttonLabel: "Step",
      min: 0,
      max: 999,
      step: 1,
      defaultValue: 0,
      format: "integer",
    },
  ],
  sample: makeFrameworkDataset,
  formulas: [
    { title: "Layer interface", expression: "\\operatorname{forward}(X)\\rightarrow Y,\\quad \\operatorname{backward}(\\partial L/\\partial Y)\\rightarrow \\partial L/\\partial X" },
    { title: "Linear forward", expression: "Y=XW+b" },
    { title: "Linear backward", expression: "\\frac{\\partial L}{\\partial X}=\\frac{\\partial L}{\\partial Y}W^T,\\quad \\frac{\\partial L}{\\partial W}=X^T\\frac{\\partial L}{\\partial Y}" },
    { title: "Sequential backward", expression: "\\text{for layer in reversed(layers): } g=layer.backward(g)" },
  ],
  explanation: [
    "A tiny deep-learning framework is mostly a contract: every layer exposes forward and backward methods.",
    "Linear layers own parameters and gradients; activation and dropout layers cache enough state to transform gradients as they flow backward.",
    "The Sequential container turns an array of objects into a trainable network, while the Trainer coordinates batches, loss, optimizer steps, and visual hooks.",
  ],
  engine: (_, params) => {
    const hiddenLayers = Math.round(numberParam(params, "hiddenLayers", 2));
    const hiddenUnits = Math.round(numberParam(params, "hiddenUnits", 8));
    const activation = stringParam(params, "activation", "GELU");
    const dropout = Boolean(params.dropout ?? false);
    const learningRate = numberParam(params, "learningRate", 0.035);
    const batchSize = Math.round(numberParam(params, "batchSize", 16));
    const flowStep = Math.round(numberParam(params, "flowStep", 0));
    const layers = makeFrameworkLayers(hiddenLayers, hiddenUnits, activation, dropout);
    const flowEvents = makeFrameworkFlowEvents(layers);
    const baseEventIndex = ((flowStep % flowEvents.length) + flowEvents.length) % flowEvents.length;
    const frames = Array.from({ length: 24 }, (_, index) => {
      const progress = index / 23;
      const eventIndex = Math.min(flowEvents.length - 1, baseEventIndex + Math.floor(progress * 1.6));
      const rawEvent = flowEvents[eventIndex % flowEvents.length];
      const event = {
        ...rawEvent,
        progress,
      };
      const totalParameters = layers.reduce((total, layer) => total + layer.parameters, 0);
      const loss = Math.max(0.018, 0.72 * Math.exp(-(flowStep + progress) * learningRate * 1.85));

      return {
        type: "concept-demo" as const,
        iteration: flowStep + index + 1,
        points: layers.map((layer, layerIndex) => ({ x: layerIndex, y: layer.units, label: layer.label })),
        framework: {
          layers,
          event,
          totalParameters,
          batchSize,
          learningRate,
          loss,
        },
        summary: `${event.phase} · ${layers[event.layerIndex]?.label ?? "Trainer"} · ${totalParameters.toLocaleString()} params · loss ${loss.toFixed(3)}`,
      };
    });
    const final = frames[frames.length - 1].framework;

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Layers", value: String(final.layers.length) },
        { label: "Parameters", value: final.totalParameters.toLocaleString() },
        { label: "Active hook", value: final.event.phase },
        { label: "Loss", value: final.loss.toFixed(4) },
      ],
    };
  },
  python: () => `import numpy as np

class Layer:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, in_features, out_features, optimizer):
        scale = np.sqrt(2 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros((1, out_features))
        self.optimizer = optimizer

    def forward(self, input_data):
        self.X = input_data
        return input_data @ self.W + self.b

    def backward(self, output_gradient):
        dX = output_gradient @ self.W.T
        dW = self.X.T @ output_gradient
        db = np.sum(output_gradient, axis=0, keepdims=True)
        self.optimizer.update(self.W, dW)
        self.optimizer.update(self.b, db)
        return dX

class ReLU(Layer):
    def forward(self, input_data):
        self.mask = input_data > 0
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        return output_gradient * self.mask

class Sequential(Layer):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

class Trainer:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, X, y, epochs=20):
        for epoch in range(epochs):
            y_hat = self.model.forward(X)
            value, grad = self.loss(y_hat, y)
            self.model.backward(grad)
            self.optimizer.step()`,
  javascript: (params) => `class Layer {
  forward(inputData) {
    throw new Error("forward() must be implemented");
  }

  backward(outputGradient) {
    throw new Error("backward() must be implemented");
  }
}

class Linear extends Layer {
  constructor(inFeatures, outFeatures, optimizer) {
    super();
    this.W = randomMatrix(inFeatures, outFeatures, Math.sqrt(2 / inFeatures));
    this.b = [new Array(outFeatures).fill(0)];
    this.optimizer = optimizer;
  }

  forward(inputData) {
    this.X = inputData;
    return addBias(matMul(inputData, this.W), this.b);
  }

  backward(outputGradient) {
    const dX = matMul(outputGradient, transpose(this.W));
    const dW = matMul(transpose(this.X), outputGradient);
    const db = sumRows(outputGradient);
    this.optimizer.update(this.W, dW);
    this.optimizer.update(this.b, db);
    return dX;
  }
}

class GELU extends Layer {
  forward(inputData) {
    this.X = inputData;
    return inputData.map((row) => row.map(gelu));
  }

  backward(outputGradient) {
    return outputGradient.map((row, r) =>
      row.map((value, c) => value * geluGrad(this.X[r][c])),
    );
  }
}

class Sequential extends Layer {
  constructor(layers = []) {
    super();
    this.layers = layers;
  }

  push(layer) {
    this.layers.push(layer);
  }

  forward(X) {
    return this.layers.reduce((value, layer) => layer.forward(value), X);
  }

  backward(gradient) {
    return [...this.layers].reverse()
      .reduce((value, layer) => layer.backward(value), gradient);
  }
}

const network = new Sequential();
network.push(new Linear(2, ${Math.round(numberParam(params, "hiddenUnits", 8))}, optimizer));
network.push(new ${stringParam(params, "activation", "GELU")}());
network.push(new Linear(${Math.round(numberParam(params, "hiddenUnits", 8))}, 1, optimizer));`,
});

function makeFrameworkDataset() {
  return makeDataset(
    "Mini-framework builder sample",
    Array.from({ length: 10 }, (_, index) => ({
      x: index,
      y: index % 2 === 0 ? 1 : 0,
      label: index % 2 === 0 ? "forward" : "backward",
    })),
  );
}

function makeFrameworkLayers(
  hiddenLayers: number,
  hiddenUnits: number,
  activation: string,
  dropout: boolean,
): FrameworkLayerState[] {
  const layers: FrameworkLayerState[] = [
    {
      id: "input",
      kind: "input",
      label: "Input X",
      units: 2,
      parameters: 0,
      cacheShape: "batch x 2",
      gradientShape: "batch x 2",
    },
  ];
  let currentUnits = 2;

  for (let hiddenIndex = 0; hiddenIndex < hiddenLayers; hiddenIndex += 1) {
    layers.push({
      id: `linear-${hiddenIndex}`,
      kind: "linear",
      label: `Linear ${currentUnits}->${hiddenUnits}`,
      units: hiddenUnits,
      parameters: currentUnits * hiddenUnits + hiddenUnits,
      cacheShape: `batch x ${currentUnits}`,
      gradientShape: `dW ${currentUnits}x${hiddenUnits}`,
    });
    layers.push({
      id: `activation-${hiddenIndex}`,
      kind: "activation",
      label: activation,
      units: hiddenUnits,
      activation,
      parameters: 0,
      cacheShape: `z${hiddenIndex + 1}`,
      gradientShape: `${activation}'(z)`,
    });

    if (dropout) {
      layers.push({
        id: `dropout-${hiddenIndex}`,
        kind: "dropout",
        label: "Dropout",
        units: hiddenUnits,
        parameters: 0,
        cacheShape: "mask",
        gradientShape: "mask * grad",
      });
    }

    currentUnits = hiddenUnits;
  }

  layers.push({
    id: "linear-output",
    kind: "linear",
    label: `Linear ${currentUnits}->1`,
    units: 1,
    parameters: currentUnits + 1,
    cacheShape: `batch x ${currentUnits}`,
    gradientShape: `dW ${currentUnits}x1`,
  });
  layers.push({
    id: "loss",
    kind: "loss",
    label: "Loss",
    units: 1,
    parameters: 0,
    cacheShape: "y_hat, y",
    gradientShape: "dL/dy_hat",
  });

  return layers;
}

function makeFrameworkFlowEvents(layers: FrameworkLayerState[]) {
  const forward = layers.map((layer, layerIndex) => ({
    phase: "forward" as const,
    layerIndex,
    progress: 0,
    description: `${layer.label}.forward(input_data)`,
  }));
  const backward = layers
    .map((layer, layerIndex) => ({
      phase: "backward" as const,
      layerIndex,
      progress: 0,
      description: `${layer.label}.backward(output_gradient)`,
    }))
    .reverse();

  return [...forward, ...backward];
}

const backpropagationFromScratch = makeConceptAlgorithm({
  id: "backpropagation-from-scratch",
  name: "Backpropagation from Scratch",
  category: "Backpropagation from scratch",
  summary: "Steps through a manual forward pass, chain-rule deltas, and weight gradients.",
  parameters: [
    {
      kind: "action",
      id: "stepIndex",
      label: "Sample step",
      buttonLabel: "Step",
      min: 0,
      max: 999,
      step: 1,
      defaultValue: 0,
      format: "integer",
    },
    {
      kind: "range",
      id: "learningRate",
      label: "Learning rate",
      min: 0.05,
      max: 1,
      step: 0.05,
      defaultValue: 0.35,
      format: "decimal",
    },
  ],
  sample: makeBackpropDataset,
  formulas: [
    { title: "Forward pass", expression: "a^{[l]}=\\sigma(a^{[l-1]}W^{[l]}+b^{[l]})" },
    { title: "Output delta", expression: "\\delta^{[2]}=(a^{[2]}-y)\\odot a^{[2]}(1-a^{[2]})" },
    { title: "Hidden delta", expression: "\\delta^{[1]}=(\\delta^{[2]}W^{[2]T})\\odot a^{[1]}(1-a^{[1]})" },
    { title: "Weight gradient", expression: "\\frac{\\partial L}{\\partial W^{[l]}}=a^{[l-1]T}\\delta^{[l]}" },
  ],
  explanation: [
    "Backpropagation applies the chain rule from the output layer back toward the inputs.",
    "The blue pulse is the forward pass that creates activations. The red pulse is the backward pass that sends error signals through the same weights.",
    "Line brightness represents the current gradient magnitude, so a brighter edge would receive a larger update for this sample.",
  ],
  engine: (_, params) => {
    const stepIndex = Math.round(numberParam(params, "stepIndex", 0));
    const learningRate = numberParam(params, "learningRate", 0.35);
    const samples = backpropSamples();
    const sample = samples[((stepIndex % samples.length) + samples.length) % samples.length];
    const state = computeBackpropStep(sample.x, sample.y);
    const frameCount = 30;
    const frames = Array.from({ length: frameCount }, (_, index) => {
      const switchFrame = 13;
      const phase: "forward" | "backward" = index <= switchFrame ? "forward" : "backward";
      const progress =
        phase === "forward"
          ? index / switchFrame
          : (index - switchFrame - 1) / Math.max(1, frameCount - switchFrame - 2);

      return {
        type: "concept-demo" as const,
        iteration: index + 1,
        points: makeBackpropDataset().points,
        network: makeBackpropGraph(state, phase, learningRate),
        backprop: {
          pulse: {
            phase,
            progress: Math.max(0, Math.min(1, progress)),
          },
          sample,
          prediction: state.a2,
          loss: state.loss,
          deltas: {
            output: state.delta2,
            hidden: [state.delta1[0], state.delta1[1]] as [number, number],
          },
          formulas: makeBackpropFormulaValues(state),
        },
        summary: `${phase === "forward" ? "Forward pass" : "Backward pass"} · x=[${sample.x.join(", ")}] · y=${sample.y} · yhat=${state.a2.toFixed(3)} · loss=${state.loss.toFixed(4)}`,
      };
    });
    const gradients = [
      ...state.dW1.flat(),
      ...state.dW2,
      state.db2,
      ...state.db1,
    ].map(Math.abs);
    const largestGradient = Math.max(...gradients);
    const w2Preview = backpropWeights.w2[0] - learningRate * state.dW2[0];

    return {
      frames,
      runtime: "JavaScript",
      metrics: [
        { label: "Prediction", value: state.a2.toFixed(4) },
        { label: "Loss", value: state.loss.toFixed(5) },
        { label: "Largest gradient", value: largestGradient.toFixed(5) },
        { label: "W2[0] after lr", value: w2Preview.toFixed(4) },
      ],
    };
  },
  python: (params) => `import numpy as np

learning_rate = ${numberParam(params, "learningRate", 0.35)}
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

W1 = np.array([[0.60, -0.40], [0.20, 0.80]])
b1 = np.array([[0.05, -0.10]])
W2 = np.array([[0.70], [-0.50]])
b2 = np.array([[0.03]])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def backward_pass(X, y):
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    delta2 = (a2 - y) * a2 * (1 - a2)
    dW2 = a1.T.dot(delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    delta1 = delta2.dot(W2.T) * a1 * (1 - a1)
    dW1 = X.T.dot(delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

dW1, db1, dW2, db2 = backward_pass(X, y)
W1 -= learning_rate * dW1
W2 -= learning_rate * dW2`,
  javascript: (params) => `const learningRate = ${numberParam(params, "learningRate", 0.35)};
const X = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y = [[0], [1], [1], [0]];

let W1 = [[0.60, -0.40], [0.20, 0.80]];
let b1 = [[0.05, -0.10]];
let W2 = [[0.70], [-0.50]];
let b2 = [[0.03]];

const sigmoid = (value) => 1 / (1 + Math.exp(-value));
const transpose = (matrix) => matrix[0].map((_, column) => matrix.map((row) => row[column]));
const matMul = (a, b) => a.map((row) =>
  b[0].map((_, column) =>
    row.reduce((sum, value, index) => sum + value * b[index][column], 0),
  ),
);
const addBias = (matrix, bias) => matrix.map((row) => row.map((value, i) => value + bias[0][i]));
const mapMatrix = (matrix, fn) => matrix.map((row) => row.map(fn));
const hadamard = (a, b) => a.map((row, r) => row.map((value, c) => value * b[r][c]));

function backwardPass(batch, labels) {
  const z1 = addBias(matMul(batch, W1), b1);
  const a1 = mapMatrix(z1, sigmoid);
  const z2 = addBias(matMul(a1, W2), b2);
  const a2 = mapMatrix(z2, sigmoid);

  const error = a2.map((row, r) => row.map((value, c) => value - labels[r][c]));
  const sigmoidPrime2 = a2.map((row) => row.map((value) => value * (1 - value)));
  const delta2 = hadamard(error, sigmoidPrime2);
  const dW2 = matMul(transpose(a1), delta2);

  const sigmoidPrime1 = a1.map((row) => row.map((value) => value * (1 - value)));
  const delta1 = hadamard(matMul(delta2, transpose(W2)), sigmoidPrime1);
  const dW1 = matMul(transpose(batch), delta1);

  return { dW1, dW2, delta1, delta2 };
}

const gradients = backwardPass(X, y);
W1 = W1.map((row, r) => row.map((w, c) => w - learningRate * gradients.dW1[r][c]));
W2 = W2.map((row, r) => row.map((w, c) => w - learningRate * gradients.dW2[r][c]));`,
});

type BackpropSample = {
  x: [number, number];
  y: number;
};

type BackpropStep = {
  z1: [number, number];
  a1: [number, number];
  z2: number;
  a2: number;
  loss: number;
  delta2: number;
  delta1: [number, number];
  dW1: [[number, number], [number, number]];
  dW2: [number, number];
  db1: [number, number];
  db2: number;
};

const backpropWeights = {
  w1: [
    [0.6, -0.4],
    [0.2, 0.8],
  ],
  b1: [0.05, -0.1],
  w2: [0.7, -0.5],
  b2: 0.03,
};

function backpropSamples(): BackpropSample[] {
  return [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ];
}

function makeBackpropDataset() {
  return makeDataset(
    "Generated XOR backprop samples",
    backpropSamples().map((sample) => ({
      x: sample.x[0],
      y: sample.x[1],
      label: sample.y === 1 ? "positive" : "negative",
    })),
  );
}

function computeBackpropStep(x: [number, number], y: number): BackpropStep {
  const z1: [number, number] = [
    x[0] * backpropWeights.w1[0][0] + x[1] * backpropWeights.w1[1][0] + backpropWeights.b1[0],
    x[0] * backpropWeights.w1[0][1] + x[1] * backpropWeights.w1[1][1] + backpropWeights.b1[1],
  ];
  const a1: [number, number] = [sigmoid(z1[0]), sigmoid(z1[1])];
  const z2 = a1[0] * backpropWeights.w2[0] + a1[1] * backpropWeights.w2[1] + backpropWeights.b2;
  const a2 = sigmoid(z2);
  const loss = 0.5 * (a2 - y) ** 2;
  const delta2 = (a2 - y) * a2 * (1 - a2);
  const dW2: [number, number] = [a1[0] * delta2, a1[1] * delta2];
  const delta1: [number, number] = [
    delta2 * backpropWeights.w2[0] * a1[0] * (1 - a1[0]),
    delta2 * backpropWeights.w2[1] * a1[1] * (1 - a1[1]),
  ];
  const dW1: [[number, number], [number, number]] = [
    [x[0] * delta1[0], x[0] * delta1[1]],
    [x[1] * delta1[0], x[1] * delta1[1]],
  ];

  return {
    z1,
    a1,
    z2,
    a2,
    loss,
    delta2,
    delta1,
    dW1,
    dW2,
    db1: delta1,
    db2: delta2,
  };
}

function makeBackpropGraph(
  state: BackpropStep,
  phase: "forward" | "backward",
  learningRate: number,
) {
  const gradientScale = phase === "backward" ? 90 * Math.max(0.1, learningRate) : 1;
  const edgeValue = (weight: number, gradient: number) =>
    phase === "forward" ? weight : gradient * gradientScale;
  const layers = [
    { label: "input", units: 2 },
    { label: "hidden", units: 2 },
    { label: "output", units: 1 },
  ];
  const nodes = [
    { id: "bp-i0", layer: 0, index: 0, label: "x1" },
    { id: "bp-i1", layer: 0, index: 1, label: "x2" },
    { id: "bp-h0", layer: 1, index: 0, label: "h1" },
    { id: "bp-h1", layer: 1, index: 1, label: "h2" },
    { id: "bp-o0", layer: 2, index: 0, label: "yhat" },
  ];
  const weights = [
    { from: "bp-i0", to: "bp-h0", value: edgeValue(backpropWeights.w1[0][0], state.dW1[0][0]) },
    { from: "bp-i0", to: "bp-h1", value: edgeValue(backpropWeights.w1[0][1], state.dW1[0][1]) },
    { from: "bp-i1", to: "bp-h0", value: edgeValue(backpropWeights.w1[1][0], state.dW1[1][0]) },
    { from: "bp-i1", to: "bp-h1", value: edgeValue(backpropWeights.w1[1][1], state.dW1[1][1]) },
    { from: "bp-h0", to: "bp-o0", value: edgeValue(backpropWeights.w2[0], state.dW2[0]) },
    { from: "bp-h1", to: "bp-o0", value: edgeValue(backpropWeights.w2[1], state.dW2[1]) },
  ];

  return {
    layers,
    nodes,
    weights,
  };
}

function makeBackpropFormulaValues(state: BackpropStep): BackpropFormulaValue[] {
  return [
    {
      title: "Forward activation",
      expression: "\\displaystyle a^{[2]}=\\sigma(a^{[1]}W^{[2]}+b^{[2]})",
      substitution: `a^{[2]} = sigma(${bpNumber(state.a1[0])}*0.70 + ${bpNumber(state.a1[1])}*(-0.50) + 0.03)`,
      value: `a^{[2]} = ${bpNumber(state.a2)}`,
    },
    {
      title: "Output delta",
      expression: "\\displaystyle \\delta^{[2]}=(a^{[2]}-y)a^{[2]}(1-a^{[2]})",
      substitution: `delta^{[2]} = (${bpNumber(state.a2)} - y) * ${bpNumber(state.a2)} * (1 - ${bpNumber(state.a2)})`,
      value: `delta^{[2]} = ${bpNumber(state.delta2)}`,
    },
    {
      title: "Hidden delta",
      expression: "\\displaystyle \\delta^{[1]}=(\\delta^{[2]}W^{[2]T})\\odot a^{[1]}(1-a^{[1]})",
      substitution: `delta^{[1]} = ${bpNumber(state.delta2)} * [0.70, -0.50] .* [${bpNumber(state.a1[0])}, ${bpNumber(state.a1[1])}] .* (1-a^{[1]})`,
      value: `delta^{[1]} = [${bpNumber(state.delta1[0])}, ${bpNumber(state.delta1[1])}]`,
    },
    {
      title: "Weight gradients",
      expression: "\\displaystyle \\frac{\\partial L}{\\partial W^{[l]}}=a^{[l-1]T}\\delta^{[l]}",
      substitution: `dW2 = [${bpNumber(state.a1[0])}, ${bpNumber(state.a1[1])}]^T * ${bpNumber(state.delta2)}`,
      value: `dW2 = [${bpNumber(state.dW2[0])}, ${bpNumber(state.dW2[1])}]`,
    },
  ];
}

function bpNumber(value: number) {
  return value.toFixed(4);
}

function makeNetworkLessonDataset(kind: string): NormalizedDataset {
  if (kind === "circles") {
    const points = Array.from({ length: 84 }, (_, index) => {
      const inner = index % 3 === 0;
      const angle = index * 1.41;
      const radius = inner
        ? 0.75 + ((index * 11) % 9) * 0.035
        : 2.05 + ((index * 17) % 13) * 0.045;
      return {
        x: round(Math.cos(angle) * radius),
        y: round(Math.sin(angle) * radius),
        label: inner ? "positive" : "negative",
      };
    });

    return makeDataset("Generated concentric circles sample", points);
  }

  if (kind === "linear") {
    const points = Array.from({ length: 76 }, (_, index) => {
      const positive = index % 2 === 0;
      const angle = index * 1.27;
      const center = positive ? 1.25 : -1.25;
      return {
        x: round(center + Math.cos(angle) * 0.75),
        y: round(center + Math.sin(angle * 1.1) * 0.75),
        label: positive ? "positive" : "negative",
      };
    });

    return makeDataset("Generated linear blobs sample", points);
  }

  const points = Array.from({ length: 88 }, (_, index) => {
    const x = -2.8 + (((index * 37) % 100) / 99) * 5.6;
    const y = -2.8 + (((index * 61 + 19) % 100) / 99) * 5.6;
    const adjustedX = Math.abs(x) < 0.22 ? x + 0.32 * Math.sign(x || 1) : x;
    const adjustedY = Math.abs(y) < 0.22 ? y + 0.32 * Math.sign(y || 1) : y;

    return {
      x: round(adjustedX),
      y: round(adjustedY),
      label: adjustedX * adjustedY < 0 ? "positive" : "negative",
    };
  });

  return makeDataset("Generated XOR grid sample", points);
}

function makeDecisionHeatmap(
  kind: string,
  hiddenLayers: number,
  hiddenNeurons: number,
  progress: number,
) {
  const gridSize = 34;
  const cells = [];

  for (let row = 0; row < gridSize; row += 1) {
    for (let column = 0; column < gridSize; column += 1) {
      const x = -3 + (column / (gridSize - 1)) * 6;
      const y = -3 + (row / (gridSize - 1)) * 6;
      cells.push({
        x,
        y,
        value: networkLessonScore(x, y, kind, hiddenLayers, hiddenNeurons, progress),
      });
    }
  }

  return cells;
}

function networkLessonScore(
  x: number,
  y: number,
  kind: string,
  hiddenLayers: number,
  hiddenNeurons: number,
  progress: number,
) {
  const capacity = Math.min(1, hiddenNeurons / 10);
  const depthBlend = Math.min(1, Math.max(0, hiddenLayers - 1) / 2);
  const learned = progress * (0.35 + capacity * 0.65);
  const nonlinearBlend = Math.min(1, depthBlend * learned);
  const linearScore = sigmoid((x + y) * 1.15);
  let nonlinearScore = linearScore;

  if (kind === "circles") {
    nonlinearScore = sigmoid((1.85 - (x ** 2 + y ** 2)) * 2.2);
  } else if (kind === "xor") {
    nonlinearScore = sigmoid((-x * y) * 2.2);
  }

  return linearScore * (1 - nonlinearBlend) + nonlinearScore * nonlinearBlend;
}

function estimateBoundaryAccuracy(
  points: DataPoint[],
  kind: string,
  hiddenLayers: number,
  hiddenNeurons: number,
  progress: number,
) {
  const correct = points.filter((point) => {
    const prediction =
      networkLessonScore(point.x, point.y, kind, hiddenLayers, hiddenNeurons, progress) >= 0.5
        ? 1
        : 0;
    return prediction === classValue(point);
  }).length;

  return correct / points.length;
}

function makeNetworkGraph(
  layers: Array<{ label: string; units: number }>,
  progress: number,
  weightScale: number,
  activation: string,
  datasetKind: string,
) {
  const nodes = layers.flatMap((layer, layerIndex) =>
    Array.from({ length: layer.units }, (_, nodeIndex) => ({
      id: `l${layerIndex}n${nodeIndex}`,
      layer: layerIndex,
      index: nodeIndex,
      label: `${layer.label} ${nodeIndex + 1}`,
    })),
  );
  const activationFactor = activation === "relu" ? 1.15 : activation === "sigmoid" ? 0.75 : 1;
  const datasetOffset = datasetKind === "circles" ? 1.7 : datasetKind === "linear" ? 0.6 : 2.8;
  const weights: Array<{ from: string; to: string; value: number }> = [];

  for (let layerIndex = 0; layerIndex < layers.length - 1; layerIndex += 1) {
    const fromNodes = nodes.filter((node) => node.layer === layerIndex);
    const toNodes = nodes.filter((node) => node.layer === layerIndex + 1);

    fromNodes.forEach((from) => {
      toNodes.forEach((to) => {
        const wave =
          Math.sin(
            datasetOffset +
              (layerIndex + 1) * 1.73 +
              from.index * 0.61 -
              to.index * 0.47 +
              progress * 3.2,
          ) * Math.cos((from.index + 1) * (to.index + 1) * 0.09 + progress);
        weights.push({
          from: from.id,
          to: to.id,
          value: wave * weightScale * activationFactor * (0.2 + progress * 0.8),
        });
      });
    });
  }

  return {
    layers,
    nodes,
    weights,
  };
}

function matrixShapeSummary(layerSizes: number[]) {
  return layerSizes
    .slice(0, -1)
    .map((size, index) => `${size}x${layerSizes[index + 1]}`)
    .join(" -> ");
}

function countNetworkParameters(layerSizes: number[]) {
  return layerSizes
    .slice(1)
    .reduce((total, size, index) => total + (layerSizes[index] + 1) * size, 0);
}

type ConceptConfig = {
  id: string;
  name: string;
  category: AlgorithmCategory;
  summary: string;
  parameters: AlgorithmDefinition["parameters"];
  sample: () => NormalizedDataset;
  formulas: AlgorithmDefinition["formulas"];
  explanation: string[];
  engine: (dataset: NormalizedDataset, params: ParameterState) => EngineResult;
  python: (params: ParameterState) => string;
  javascript: (params: ParameterState) => string;
};

function makeConceptAlgorithm(config: ConceptConfig): AlgorithmDefinition {
  return {
    id: config.id,
    name: config.name,
    category: config.category,
    summary: config.summary,
    parameters: config.parameters,
    makeSampleDataset: config.sample,
    engine: config.engine,
    formulas: config.formulas,
    explanation: config.explanation,
    code: {
      python: config.python,
      javascript: config.javascript,
    },
  };
}

type BarConfig = Omit<ConceptConfig, "sample" | "engine"> & {
  bars: (params: ParameterState) => ConceptBar[];
};

function barAlgorithm(config: BarConfig): AlgorithmDefinition {
  return makeConceptAlgorithm({
    ...config,
    sample: () => makeBarsDataset(config.name, config.bars({})),
    engine: (_, params) => {
      const bars = config.bars(params).map((bar, index) => ({
        ...bar,
        color: colors[index % colors.length],
      }));
      const best = [...bars].sort((a, b) => b.value - a.value)[0];

      return conceptResult(
        {
          type: "concept-demo",
          iteration: bars.length,
          points: bars.map((bar, index) => ({ x: index, y: bar.value, label: bar.label })),
          bars,
          summary: `${best.label} leads at ${metricValue(best.value)}`,
        },
        [
          { label: "Best", value: best.label },
          { label: "Score", value: metricValue(best.value) },
          { label: "Items", value: String(bars.length) },
          { label: "Range", value: `${metricValue(Math.min(...bars.map((bar) => bar.value)))}-${metricValue(Math.max(...bars.map((bar) => bar.value)))}` },
        ],
      );
    },
  });
}

function gini(points: DataPoint[]) {
  if (points.length === 0) {
    return 0;
  }

  const positives = points.filter((point) => classValue(point) === 1).length / points.length;
  return 1 - positives ** 2 - (1 - positives) ** 2;
}

function distance(a: DataPoint, b: DataPoint, metric: string) {
  if (metric === "manhattan") {
    return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
  }

  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(value: number) {
  return Math.max(0.03, Math.min(0.99, value));
}

function mean(points: DataPoint[]): DataPoint {
  const total = points.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), {
    x: 0,
    y: 0,
  });

  return {
    x: total.x / points.length,
    y: total.y / points.length,
  };
}

function average(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function standardDeviation(values: number[]) {
  const center = average(values);
  return Math.sqrt(average(values.map((value) => (value - center) ** 2)));
}

function makeLearningCurveDataset(name = "Generated learning curve sample") {
  return makeDataset(
    name,
    Array.from({ length: 12 }, (_, index) => ({ x: index + 1, y: 1 / (index + 2) })),
  );
}

function makeAnomalyDataset() {
  const normal = Array.from({ length: 62 }, (_, index) => {
    const angle = index * 1.41;
    const radius = 0.4 + ((index * 13) % 19) * 0.06;
    return {
      x: round(Math.cos(angle) * radius),
      y: round(Math.sin(angle * 1.13) * radius),
      label: "normal",
    };
  });
  const anomalies = [
    { x: -3.1, y: 2.4, label: "anomaly" },
    { x: 3.4, y: -2.2, label: "anomaly" },
    { x: 2.7, y: 2.8, label: "anomaly" },
    { x: -2.9, y: -2.5, label: "anomaly" },
  ];

  return makeDataset("Generated anomaly sample", [...normal, ...anomalies]);
}

function makeTimeSeriesDataset() {
  return makeDataset(
    "Generated time series sample",
    Array.from({ length: 34 }, (_, index) => ({
      x: index,
      y: round(4 + index * 0.08 + Math.sin((index / 4) * Math.PI) * 0.65 + Math.cos(index * 0.7) * 0.16),
    })),
  );
}

export const categoryDemos: AlgorithmDefinition[] = [
  logisticRegression,
  decisionTreeSplit,
  supportVectorMachine,
  knnClassifier,
  featureSelection,
  modelEvaluation,
  biasVariance,
  ensembleMethods,
  hyperparameterTuning,
  naiveBayes,
  anomalyDetection,
  imbalancedData,
  timeSeries,
  stochasticProcesses,
  dynamicProgramming,
  singularValueDecomposition,
  convexOptimization,
  convolutionsFromScratch,
  neuralNetwork,
  multiLayerNetwork,
  activationFunctions,
  lossFunctions,
  optimizerRace,
  miniFramework,
  backpropagationFromScratch,
];
