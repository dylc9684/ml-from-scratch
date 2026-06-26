import * as tf from "@tensorflow/tfjs";
import { makeRegressionDataset } from "../data/datasets";
import type {
  AlgorithmDefinition,
  DataPoint,
  EngineResult,
  LinearRegressionFrame,
  NormalizedDataset,
  ParameterState,
} from "../types/algorithm";

const numberParam = (
  params: ParameterState,
  key: string,
  fallback: number,
) => {
  const value = Number(params[key]);
  return Number.isFinite(value) ? value : fallback;
};

const boolParam = (
  params: ParameterState,
  key: string,
  fallback: boolean,
) => (typeof params[key] === "boolean" ? Boolean(params[key]) : fallback);

export const linearRegression: AlgorithmDefinition = {
  id: "linear-regression",
  name: "Linear Regression",
  category: "Regression",
  summary: "Fits a line by minimizing mean squared error with gradient descent.",
  parameters: [
    {
      kind: "range",
      id: "learningRate",
      label: "Learning rate",
      min: 0.001,
      max: 0.08,
      step: 0.001,
      defaultValue: 0.018,
      format: "decimal",
    },
    {
      kind: "range",
      id: "epochs",
      label: "Epochs",
      min: 20,
      max: 500,
      step: 10,
      defaultValue: 180,
      format: "integer",
    },
    {
      kind: "toggle",
      id: "fitIntercept",
      label: "Fit intercept",
      defaultValue: true,
    },
  ],
  makeSampleDataset: makeRegressionDataset,
  engine: runLinearRegression,
  formulas: [
    {
      title: "Prediction",
      expression: "\\hat{y}=mx+b",
    },
    {
      title: "Objective",
      expression: "J(m,b)=\\frac{1}{n}\\sum_{i=1}^{n}(\\hat{y}_i-y_i)^2",
    },
    {
      title: "Gradient step",
      expression: "\\theta \\leftarrow \\theta - \\alpha \\nabla J(\\theta)",
    },
  ],
  explanation: [
    "Linear regression models a continuous target as a weighted combination of input features. This playground focuses on the one-feature case so the fitted line can be animated directly.",
    "Each run starts with a flat line. The JavaScript execution engine computes gradients, updates slope and intercept, and emits intermediate frames for the canvas visualization.",
    "The Python tab mirrors the same parameters using NumPy-style code for notebook learning; the browser still executes the JavaScript/TensorFlow.js path.",
  ],
  code: {
    python: (params) => {
      const learningRate = numberParam(params, "learningRate", 0.018);
      const epochs = Math.round(numberParam(params, "epochs", 180));
      const fitIntercept = boolParam(params, "fitIntercept", true);

      return `import numpy as np

X = dataset["x"].to_numpy(dtype=float)
y = dataset["y"].to_numpy(dtype=float)

m = 0.0
b = 0.0
learning_rate = ${learningRate}
epochs = ${epochs}
fit_intercept = ${fitIntercept ? "True" : "False"}

for epoch in range(epochs):
    y_hat = m * X + b
    error = y_hat - y

    dm = (2 / len(X)) * np.sum(error * X)
    db = (2 / len(X)) * np.sum(error) if fit_intercept else 0.0

    m -= learning_rate * dm
    b -= learning_rate * db

mse = np.mean((m * X + b - y) ** 2)
print({"slope": m, "intercept": b, "mse": mse})`;
    },
    javascript: (params) => {
      const learningRate = numberParam(params, "learningRate", 0.018);
      const epochs = Math.round(numberParam(params, "epochs", 180));
      const fitIntercept = boolParam(params, "fitIntercept", true);

      return `import * as tf from "@tensorflow/tfjs";

let slope = 0;
let intercept = 0;
const learningRate = ${learningRate};
const epochs = ${epochs};
const fitIntercept = ${fitIntercept};

for (let epoch = 0; epoch < epochs; epoch += 1) {
  const gradients = points.reduce(
    (acc, point) => {
      const prediction = slope * point.x + intercept;
      const error = prediction - point.y;
      acc.slope += error * point.x;
      acc.intercept += error;
      return acc;
    },
    { slope: 0, intercept: 0 },
  );

  slope -= learningRate * (2 / points.length) * gradients.slope;
  intercept -= fitIntercept
    ? learningRate * (2 / points.length) * gradients.intercept
    : 0;
}

const mse = tf.tidy(() => {
  const xs = tf.tensor1d(points.map((point) => point.x));
  const ys = tf.tensor1d(points.map((point) => point.y));
  return tf.mean(tf.square(tf.sub(tf.add(tf.mul(xs, slope), intercept), ys)));
});`;
    },
  },
};

function runLinearRegression(
  dataset: NormalizedDataset,
  params: ParameterState,
): EngineResult {
  const points = dataset.points;

  if (points.length === 0) {
    return {
      frames: [],
      metrics: [{ label: "Rows", value: "0 usable" }],
      runtime: "TensorFlow.js",
    };
  }

  const learningRate = numberParam(params, "learningRate", 0.018);
  const epochs = Math.round(numberParam(params, "epochs", 180));
  const fitIntercept = boolParam(params, "fitIntercept", true);
  const frameEvery = Math.max(1, Math.floor(epochs / 90));
  const frames: LinearRegressionFrame[] = [];

  let slope = 0;
  let intercept = 0;

  for (let epoch = 0; epoch <= epochs; epoch += 1) {
    if (epoch % frameEvery === 0 || epoch === epochs) {
      frames.push({
        type: "linear-regression",
        iteration: epoch,
        points,
        slope,
        intercept,
        loss: mseWithTensorFlow(points, slope, intercept),
      });
    }

    const gradient = points.reduce(
      (acc, point) => {
        const prediction = slope * point.x + intercept;
        const error = prediction - point.y;
        acc.slope += error * point.x;
        acc.intercept += error;
        return acc;
      },
      { slope: 0, intercept: 0 },
    );

    slope -= learningRate * (2 / points.length) * gradient.slope;
    intercept -= fitIntercept
      ? learningRate * (2 / points.length) * gradient.intercept
      : 0;
  }

  const finalLoss = mseWithTensorFlow(points, slope, intercept);

  return {
    frames,
    metrics: [
      { label: "MSE", value: finalLoss.toFixed(4) },
      { label: "Slope", value: slope.toFixed(3) },
      { label: "Intercept", value: intercept.toFixed(3) },
      { label: "Frames", value: String(frames.length) },
    ],
    runtime: "TensorFlow.js",
  };
}

function mseWithTensorFlow(
  points: DataPoint[],
  slope: number,
  intercept: number,
) {
  return tf.tidy(() => {
    const xs = tf.tensor1d(points.map((point) => point.x));
    const ys = tf.tensor1d(points.map((point) => point.y));
    const predictions = tf.add(tf.mul(xs, slope), intercept);
    const loss = tf.mean(tf.square(tf.sub(predictions, ys)));
    return loss.dataSync()[0];
  });
}
