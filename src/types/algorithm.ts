export const algorithmCategories = [
  "Regression",
  "Logistic Regression & Classification",
  "Decision Trees & Random Forests",
  "Support Vector Machines",
  "KNN & Distance Metrics",
  "Clustering",
  "Feature Engineering & Selection",
  "Model Evaluation: Metrics, Cross-Validation",
  "Bias, Variance & the Learning Curve",
  "Ensemble Methods: Boosting, Bagging, Stacking",
  "Hyperparameter Tuning",
  "Naive Bayes",
  "Anomaly Detection",
  "Handling Imbalanced Data",
  "Time Series Fundamentals",
  "Neural Networks",
] as const;

export type AlgorithmCategory = (typeof algorithmCategories)[number];

export type ParameterValue = number | string | boolean;

export type ParameterState = Record<string, ParameterValue>;

export type NumericParameter = {
  kind: "range";
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  format?: "integer" | "decimal" | "percent";
};

export type SelectParameter = {
  kind: "select";
  id: string;
  label: string;
  defaultValue: string;
  options: Array<{ label: string; value: string }>;
};

export type ToggleParameter = {
  kind: "toggle";
  id: string;
  label: string;
  defaultValue: boolean;
};

export type ParameterDefinition =
  | NumericParameter
  | SelectParameter
  | ToggleParameter;

export type DataPoint = {
  x: number;
  y: number;
  label?: string | number;
};

export type DatasetColumn = {
  name: string;
  type: "number" | "string";
};

export type RawDataset = {
  name: string;
  rows: Record<string, unknown>[];
  columns: DatasetColumn[];
};

export type DatasetMapping = {
  x: string;
  y: string;
  label?: string;
};

export type NormalizedDataset = {
  name: string;
  points: DataPoint[];
  columns: DatasetColumn[];
  mapping: DatasetMapping;
  source: "sample" | "upload";
};

export type Metric = {
  label: string;
  value: string;
};

export type FormulaBlock = {
  title: string;
  expression: string;
};

export type CodeTemplate = (params: ParameterState) => string;

export type LinearRegressionFrame = {
  type: "linear-regression";
  iteration: number;
  points: DataPoint[];
  slope: number;
  intercept: number;
  loss: number;
};

export type KMeansFrame = {
  type: "k-means";
  iteration: number;
  points: DataPoint[];
  centroids: DataPoint[];
  assignments: number[];
  inertia: number;
};

export type ConceptSeries = {
  label: string;
  points: DataPoint[];
  color?: string;
  dashed?: boolean;
};

export type ConceptMarker = {
  axis: "x" | "y";
  value: number;
  label: string;
  color?: string;
};

export type ConceptBar = {
  label: string;
  value: number;
  color?: string;
};

export type ConceptFrame = {
  type: "concept-demo";
  iteration: number;
  points: DataPoint[];
  series?: ConceptSeries[];
  markers?: ConceptMarker[];
  bars?: ConceptBar[];
  summary: string;
};

export type AlgorithmFrame = LinearRegressionFrame | KMeansFrame | ConceptFrame;

export type EngineResult = {
  frames: AlgorithmFrame[];
  metrics: Metric[];
  runtime: "JavaScript" | "TensorFlow.js";
};

export type AlgorithmEngine = (
  dataset: NormalizedDataset,
  params: ParameterState,
) => EngineResult;

export type AlgorithmDefinition = {
  id: string;
  name: string;
  category: AlgorithmCategory;
  summary: string;
  parameters: ParameterDefinition[];
  makeSampleDataset: () => NormalizedDataset;
  engine: AlgorithmEngine;
  formulas: FormulaBlock[];
  explanation: string[];
  code: {
    python: CodeTemplate;
    javascript: CodeTemplate;
  };
};
