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
  "Stochastic Processes",
  "Building a Tokenizer",
  "Dynamic Programming",
  "Singular Value Decomposition",
  "Non-negative Matrix Factorization",
  "Convex Optimization",
  "Convolutions from Scratch",
  "Neural Networks",
  "Multi-Layer Networks",
  "Activation Functions — ReLU, Sigmoid, GELU",
  "Loss Functions — MSE, Cross-Entropy, Contrastive",
  "SGD, Momentum, Adam, AdamW",
  "Build a Mini-Framework",
  "Backpropagation from scratch",
] as const;

export type AlgorithmCategory = (typeof algorithmCategories)[number];

export type MatrixParameterValue = number[][];

export type ImageMatrixParameterValue = {
  kind: "image-matrix";
  name: string;
  width: number;
  height: number;
  values: MatrixParameterValue;
  dataUrl?: string;
};

export type GridWorldCell = "empty" | "wall" | "fire" | "gold" | "start";

export type GridWorldValue = {
  kind: "gridworld";
  rows: number;
  columns: number;
  cells: GridWorldCell[][];
};

export type PointSetParameterValue = {
  kind: "point-set";
  points: DataPoint[];
};

export type TextQuickInsert = {
  label: string;
  value: string;
};

export type ParameterValue =
  | number
  | string
  | boolean
  | MatrixParameterValue
  | ImageMatrixParameterValue
  | GridWorldValue
  | PointSetParameterValue;

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

export type TextParameter = {
  kind: "text";
  id: string;
  label: string;
  defaultValue: string;
  placeholder?: string;
  rows?: number;
  quickInserts?: TextQuickInsert[];
};

export type MatrixParameter = {
  kind: "matrix";
  id: string;
  label: string;
  rowLabels: string[];
  columnLabels: string[];
  defaultValue: MatrixParameterValue;
  min: number;
  max: number;
  step: number;
  format?: "integer" | "decimal" | "percent";
};

export type ImageParameter = {
  kind: "image";
  id: string;
  label: string;
  buttonLabel: string;
  maxSize: number;
  defaultValue: ImageMatrixParameterValue;
};

export type GridWorldParameter = {
  kind: "gridworld";
  id: string;
  label: string;
  defaultValue: GridWorldValue;
};

export type PointSetParameter = {
  kind: "point-set";
  id: string;
  label: string;
  defaultValue: PointSetParameterValue;
  minPoints?: number;
  maxPoints?: number;
};

export type ParameterDefinition =
  | NumericParameter
  | StepperParameter
  | ActionParameter
  | SelectParameter
  | ToggleParameter
  | TextParameter
  | MatrixParameter
  | ImageParameter
  | GridWorldParameter
  | PointSetParameter;

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

export type FrameworkLayerKind = "input" | "linear" | "activation" | "dropout" | "loss";

export type FrameworkFlowPhase = "forward" | "backward";

export type FrameworkLayerState = {
  id: string;
  kind: FrameworkLayerKind;
  label: string;
  units: number;
  activation?: string;
  parameters: number;
  cacheShape?: string;
  gradientShape?: string;
};

export type FrameworkFlowEvent = {
  phase: FrameworkFlowPhase;
  layerIndex: number;
  progress: number;
  description: string;
};

export type FrameworkState = {
  layers: FrameworkLayerState[];
  event: FrameworkFlowEvent;
  totalParameters: number;
  batchSize: number;
  learningRate: number;
  loss: number;
};

export type StochasticPathPoint = {
  t: number;
  y: number;
};

export type StochasticPathState = {
  id: string;
  color: string;
  points: StochasticPathPoint[];
  current: StochasticPathPoint;
  terminal: number;
};

export type MarkovNodeState = {
  id: string;
  label: string;
  color: string;
};

export type StochasticState = {
  drift: number;
  volatility: number;
  visibleStep: number;
  timeSteps: number;
  pathCount: number;
  paths: StochasticPathState[];
  terminalMean: number;
  terminalStd: number;
  states: MarkovNodeState[];
  transitionMatrix: MatrixParameterValue;
  normalizedTransitionMatrix: MatrixParameterValue;
  currentState: number;
  nextState: number;
  pulseProgress: number;
  markovStep: number;
  sequence: number[];
  entropy: number;
};

export type SvdGeometryPhase = "V^T rotation" | "Sigma scaling" | "U rotation";

export type SvdVector2D = {
  x: number;
  y: number;
  color: string;
  label: string;
};

export type SvdGeometryState = {
  phase: SvdGeometryPhase;
  progress: number;
  angleV: number;
  angleU: number;
  scaleX: number;
  scaleY: number;
  inputVectors: SvdVector2D[];
  currentVectors: SvdVector2D[];
};

export type MatrixDecompositionMethod = "svd" | "nmf";

export type MatrixFactorBlock = {
  label: string;
  role: string;
  matrix: MatrixParameterValue;
  color: string;
  signed?: boolean;
};

export type SvdUseCaseState = {
  frame: MatrixParameterValue;
  background: MatrixParameterValue;
  foreground: MatrixParameterValue;
  backgroundEnergy: number;
  foregroundEnergy: number;
};

export type SvdState = {
  source: ImageMatrixParameterValue;
  method: MatrixDecompositionMethod;
  original: MatrixParameterValue;
  approximation: MatrixParameterValue;
  factors: MatrixFactorBlock[];
  splitProgress: number;
  useCase: SvdUseCaseState;
  rank: number;
  maxRank: number;
  singularValues: number[];
  retainedEnergy: number;
  reconstructionError: number;
  geometry: SvdGeometryState;
};

export type NmfTopicWord = {
  term: string;
  weight: number;
};

export type NmfTopicState = {
  label: string;
  words: NmfTopicWord[];
  documentWeights: number[];
  topDocument: string;
};

export type NmfFacePartState = {
  label: string;
  weight: number;
  matrix: MatrixParameterValue;
  color: string;
};

export type NmfState = {
  documents: string[];
  vocabulary: string[];
  topicCount: number;
  iterations: number;
  topics: NmfTopicState[];
  documentTopicMatrix: MatrixParameterValue;
  topicWordMatrix: MatrixParameterValue;
  faceOriginal: MatrixParameterValue;
  faceReconstruction: MatrixParameterValue;
  faceParts: NmfFacePartState[];
  faceReconstructionError: number;
  sparsity: number;
};

export type PolynomialFeatureRow = {
  power: number;
  value: number;
};

export type PolynomialState = {
  degree: number;
  points: DataPoint[];
  coefficients: number[];
  curve: DataPoint[];
  featureRows: PolynomialFeatureRow[][];
  mse: number;
  overfitScore: number;
  xDomain: [number, number];
  yDomain: [number, number];
};

export type ConvexOptimizerKey = "projected-gradient" | "newton";

export type ConvexStatus = "convex" | "non-convex";

export type ConvexTrajectoryPoint = {
  x: number;
  y: number;
  z: number;
  projected: boolean;
  activeConstraint?: string;
};

export type ConvexState = {
  optimizer: ConvexOptimizerKey;
  status: ConvexStatus;
  statusReason: string;
  step: number;
  objective: {
    curvatureX: number;
    curvatureY: number;
    crossTerm: number;
    sCurve: number;
    tiltX: number;
    tiltY: number;
  };
  constraint: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  };
  path: ConvexTrajectoryPoint[];
  current: ConvexTrajectoryPoint;
  unconstrained: ConvexTrajectoryPoint;
  constrainedOptimum: ConvexTrajectoryPoint;
  gradientNorm: number;
};

export type ConvolutionKernelPreset = "sobel-x" | "gaussian-blur" | "sharpen" | "custom";

export type ConvolutionTerm = {
  imageValue: number;
  kernelValue: number;
  product: number;
  row: number;
  column: number;
};

export type ConvolutionState = {
  source: ImageMatrixParameterValue;
  kernelPreset: ConvolutionKernelPreset;
  kernel: MatrixParameterValue;
  padded: MatrixParameterValue;
  output: MatrixParameterValue;
  normalizedOutput: MatrixParameterValue;
  padding: boolean;
  stride: number;
  cursor: {
    inputRow: number;
    inputColumn: number;
    outputRow: number;
    outputColumn: number;
  };
  patch: MatrixParameterValue;
  terms: ConvolutionTerm[];
  currentValue: number;
  outputShape: {
    height: number;
    width: number;
  };
};

export type DynamicProgrammingMethod = "value-iteration" | "policy-iteration";

export type DynamicProgrammingAction = "up" | "right" | "down" | "left";

export type DynamicProgrammingActionValue = {
  action: DynamicProgrammingAction;
  label: string;
  value: number;
};

export type DynamicProgrammingPolicyCell = {
  row: number;
  column: number;
  action: DynamicProgrammingAction;
  value: number;
};

export type DynamicProgrammingState = {
  grid: GridWorldValue;
  method: DynamicProgrammingMethod;
  gamma: number;
  sweep: number;
  values: MatrixParameterValue;
  previousValues: MatrixParameterValue;
  delta: number;
  stable: boolean;
  activeCell: {
    row: number;
    column: number;
  };
  actionValues: DynamicProgrammingActionValue[];
  policy: DynamicProgrammingPolicyCell[];
};

export type TokenizerPiece = {
  text: string;
  id: number;
  bytes: number[];
  byteFallback: boolean;
  colorIndex: number;
};

export type TokenizerMerge = {
  left: string;
  right: string;
  merged: string;
  count: number;
};

export type TokenizerState = {
  input: string;
  pieces: TokenizerPiece[];
  tokenIds: number[];
  byteLength: number;
  characterLength: number;
  vocabSize: number;
  mergeCount: number;
  byteFallbackCount: number;
  compressionRatio: number;
  topMerges: TokenizerMerge[];
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
  framework?: FrameworkState;
  stochastic?: StochasticState;
  svd?: SvdState;
  nmf?: NmfState;
  polynomial?: PolynomialState;
  convex?: ConvexState;
  convolution?: ConvolutionState;
  dynamicProgramming?: DynamicProgrammingState;
  tokenizer?: TokenizerState;
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

export type AlgorithmRunController = {
  primaryActionLabel?: string;
  prepareRunParams?: (params: ParameterState) => ParameterState;
  shouldAutoPlay?: (result: EngineResult) => boolean;
};

export type AlgorithmDefinition = {
  id: string;
  name: string;
  category: AlgorithmCategory;
  summary: string;
  parameters: ParameterDefinition[];
  makeSampleDataset: () => NormalizedDataset;
  engine: AlgorithmEngine;
  controller?: AlgorithmRunController;
  formulas: FormulaBlock[];
  explanation: string[];
  code: {
    python: CodeTemplate;
    javascript: CodeTemplate;
  };
};
