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
  "Multi-Layer Networks",
  "Activation Functions — ReLU, Sigmoid, GELU",
  "Loss Functions — MSE, Cross-Entropy, Contrastive",
  "SGD, Momentum, Adam, AdamW",
  "Backpropagation from scratch",
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

export type StepperParameter = {
  kind: "stepper";
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
  format?: "integer" | "decimal" | "percent";
};

export type ActionParameter = {
  kind: "action";
  id: string;
  label: string;
  buttonLabel: string;
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
  | StepperParameter
  | ActionParameter
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

export type NetworkNode = {
  id: string;
  layer: number;
  index: number;
  label?: string;
};

export type NetworkWeight = {
  from: string;
  to: string;
  value: number;
};

export type NetworkLayer = {
  label: string;
  units: number;
};

export type NetworkGraph = {
  layers: NetworkLayer[];
  nodes: NetworkNode[];
  weights: NetworkWeight[];
};

export type HeatmapCell = {
  x: number;
  y: number;
  value: number;
};

export type BackpropFormulaValue = {
  title: string;
  expression: string;
  substitution: string;
  value: string;
};

export type BackpropPulse = {
  phase: "forward" | "backward";
  progress: number;
};

export type BackpropState = {
  pulse: BackpropPulse;
  sample: {
    x: [number, number];
    y: number;
  };
  prediction: number;
  loss: number;
  deltas: {
    output: number;
    hidden: [number, number];
  };
  formulas: BackpropFormulaValue[];
};

export type ActivationFunctionKey = "sigmoid" | "relu" | "leaky-relu" | "gelu";

export type ActivationCurve = {
  id: ActivationFunctionKey;
  label: string;
  color: string;
  points: DataPoint[];
  derivativePoints: DataPoint[];
  value: number;
  derivative: number;
};

export type ActivationNeuronState = {
  id: string;
  index: number;
  label: string;
  preActivation: number;
  activation: number;
  derivative: number;
  status: "alive" | "recovering" | "inactive" | "dead" | "saturated";
};

export type ActivationState = {
  selected: ActivationFunctionKey;
  inputX: number;
  targetX: number;
  learningRate: number;
  negativeShift: number;
  leakySlope: number;
  curves: ActivationCurve[];
  neurons: ActivationNeuronState[];
  deadCount: number;
  saturatedCount: number;
  recoveryCount: number;
};

export type LossFunctionKey = "mse" | "cross-entropy" | "contrastive";

export type LossPrediction = {
  label: string;
  target: number;
  prediction: number;
};

export type ContrastivePoint = {
  id: string;
  x: number;
  y: number;
  label: string;
};

export type ContrastivePair = {
  id: string;
  from: ContrastivePoint;
  to: ContrastivePoint;
  relation: "similar" | "different";
  distance: number;
  loss: number;
  gradientMagnitude: number;
};

export type LossState = {
  selected: LossFunctionKey;
  epoch: number;
  learningRate: number;
  margin: number;
  lossHistory: DataPoint[];
  currentLoss: number;
  status: "converging" | "diverging" | "oscillating";
  predictions: LossPrediction[];
  contrastivePairs: ContrastivePair[];
};

export type OptimizerKey = "sgd" | "momentum" | "adam" | "adamw";

export type LossSurfaceKey = "rosenbrock" | "beale";

export type OptimizerTracePoint = {
  step: number;
  x: number;
  y: number;
  z: number;
};

export type OptimizerRunnerState = {
  id: OptimizerKey;
  label: string;
  color: string;
  history: OptimizerTracePoint[];
  position: OptimizerTracePoint;
};

export type AdaptiveWeightState = {
  from: string;
  to: string;
  gradient: number;
  denominator: number;
  adaptiveRate: number;
};

export type OptimizerState = {
  surface: LossSurfaceKey;
  step: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  weightDecay: number;
  start: {
    x: number;
    y: number;
  };
  runners: OptimizerRunnerState[];
  adaptiveWeights: AdaptiveWeightState[];
};

export type ConceptFrame = {
  type: "concept-demo";
  iteration: number;
  points: DataPoint[];
  series?: ConceptSeries[];
  markers?: ConceptMarker[];
  bars?: ConceptBar[];
  network?: NetworkGraph;
  heatmap?: HeatmapCell[];
  backprop?: BackpropState;
  activation?: ActivationState;
  loss?: LossState;
  optimizer?: OptimizerState;
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
