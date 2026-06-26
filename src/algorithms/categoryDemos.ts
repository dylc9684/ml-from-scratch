import type {
  AlgorithmCategory,
  AlgorithmDefinition,
  ConceptBar,
  ConceptFrame,
  ConceptSeries,
  DataPoint,
  EngineResult,
  NormalizedDataset,
  ParameterState,
} from "../types/algorithm";

const colors = ["#0f766e", "#2f6fbe", "#b7791f", "#d34a43", "#6f58c9", "#258f66"];

const numberParam = (params: ParameterState, key: string, fallback: number) => {
  const value = Number(params[key]);
  return Number.isFinite(value) ? value : fallback;
};

const stringParam = (params: ParameterState, key: string, fallback: string) =>
  typeof params[key] === "string" ? String(params[key]) : fallback;

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
  neuralNetwork,
];
