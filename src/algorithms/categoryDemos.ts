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
  neuralNetwork,
  multiLayerNetwork,
  activationFunctions,
  backpropagationFromScratch,
];
