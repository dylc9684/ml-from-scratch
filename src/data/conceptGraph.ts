import type { AlgorithmDefinition } from "../types/algorithm";

export type ConceptRelationshipKind =
  | "foundation"
  | "alternative"
  | "depends-on"
  | "evaluates"
  | "optimizes"
  | "extends";

export type ConceptRelationship = {
  source: string;
  target: string;
  label: string;
  kind: ConceptRelationshipKind;
};

export type ConceptDomain =
  | "foundations"
  | "supervised"
  | "unsupervised"
  | "neural"
  | "optimization"
  | "data"
  | "probability";

export type ConceptGraphNode = {
  id: string;
  algorithm: AlgorithmDefinition;
  domain: ConceptDomain;
  depth: 0 | 1;
};

export type ConceptGraphLink = ConceptRelationship & {
  sourceDepth: 0 | 1;
  targetDepth: 0 | 1;
};

export type ConceptGraphNeighborhood = {
  active: AlgorithmDefinition;
  nodes: ConceptGraphNode[];
  links: ConceptGraphLink[];
  directRelationships: ConceptRelationship[];
};

export const conceptDomainColors: Record<ConceptDomain, string> = {
  foundations: "#64748b",
  supervised: "#2f6fbe",
  unsupervised: "#0f766e",
  neural: "#6f58c9",
  optimization: "#b7791f",
  data: "#d34a43",
  probability: "#258f66",
};

const conceptDomains: Record<string, ConceptDomain> = {
  "linear-regression": "foundations",
  "logistic-regression": "supervised",
  "decision-tree-split": "supervised",
  "support-vector-machine": "supervised",
  "knn-classifier": "supervised",
  "feature-selection": "data",
  "polynomial-features": "data",
  "model-evaluation": "data",
  "bias-variance": "foundations",
  "ensemble-methods": "supervised",
  "hyperparameter-tuning": "optimization",
  "naive-bayes": "probability",
  "anomaly-detection": "unsupervised",
  "imbalanced-data": "data",
  "time-series": "probability",
  "building-tokenizer": "data",
  "stochastic-processes": "probability",
  "dynamic-programming": "optimization",
  "singular-value-decomposition": "foundations",
  "non-negative-matrix-factorization": "unsupervised",
  "convex-optimization": "optimization",
  "convolutions-from-scratch": "neural",
  "neural-network": "neural",
  "multi-layer-network": "neural",
  "activation-functions": "neural",
  "loss-functions": "neural",
  "optimizer-race": "optimization",
  "mini-framework": "neural",
  "backpropagation-from-scratch": "neural",
  "k-means": "unsupervised",
};

export const conceptRelationships: ConceptRelationship[] = [
  { source: "linear-regression", target: "logistic-regression", label: "linear model family", kind: "extends" },
  { source: "linear-regression", target: "loss-functions", label: "MSE objective", kind: "depends-on" },
  { source: "linear-regression", target: "optimizer-race", label: "gradient descent", kind: "optimizes" },
  { source: "linear-regression", target: "feature-selection", label: "feature weights", kind: "depends-on" },
  { source: "linear-regression", target: "polynomial-features", label: "expanded design matrix", kind: "extends" },
  { source: "linear-regression", target: "bias-variance", label: "fit complexity", kind: "evaluates" },
  { source: "logistic-regression", target: "model-evaluation", label: "classification metrics", kind: "evaluates" },
  { source: "logistic-regression", target: "loss-functions", label: "cross-entropy", kind: "depends-on" },
  { source: "logistic-regression", target: "activation-functions", label: "sigmoid link", kind: "depends-on" },
  { source: "logistic-regression", target: "naive-bayes", label: "probabilistic classifier", kind: "alternative" },
  { source: "logistic-regression", target: "support-vector-machine", label: "linear boundary", kind: "alternative" },
  { source: "decision-tree-split", target: "ensemble-methods", label: "tree ensembles", kind: "extends" },
  { source: "decision-tree-split", target: "feature-selection", label: "split importance", kind: "depends-on" },
  { source: "decision-tree-split", target: "hyperparameter-tuning", label: "depth and leaves", kind: "optimizes" },
  { source: "decision-tree-split", target: "model-evaluation", label: "validation score", kind: "evaluates" },
  { source: "support-vector-machine", target: "convex-optimization", label: "margin objective", kind: "depends-on" },
  { source: "support-vector-machine", target: "knn-classifier", label: "geometry and distance", kind: "alternative" },
  { source: "support-vector-machine", target: "feature-selection", label: "scaled features", kind: "depends-on" },
  { source: "knn-classifier", target: "k-means", label: "distance geometry", kind: "alternative" },
  { source: "knn-classifier", target: "anomaly-detection", label: "neighbor distance", kind: "extends" },
  { source: "knn-classifier", target: "model-evaluation", label: "accuracy and F1", kind: "evaluates" },
  { source: "k-means", target: "anomaly-detection", label: "cluster residuals", kind: "extends" },
  { source: "k-means", target: "singular-value-decomposition", label: "latent geometry", kind: "depends-on" },
  { source: "k-means", target: "non-negative-matrix-factorization", label: "unsupervised grouping", kind: "alternative" },
  { source: "feature-selection", target: "singular-value-decomposition", label: "low-rank signals", kind: "depends-on" },
  { source: "feature-selection", target: "non-negative-matrix-factorization", label: "interpretable factors", kind: "depends-on" },
  { source: "feature-selection", target: "polynomial-features", label: "engineered columns", kind: "foundation" },
  { source: "feature-selection", target: "model-evaluation", label: "validation lift", kind: "evaluates" },
  { source: "model-evaluation", target: "bias-variance", label: "generalization gap", kind: "evaluates" },
  { source: "model-evaluation", target: "hyperparameter-tuning", label: "selection loop", kind: "optimizes" },
  { source: "model-evaluation", target: "imbalanced-data", label: "precision and recall", kind: "evaluates" },
  { source: "polynomial-features", target: "bias-variance", label: "degree versus overfit", kind: "evaluates" },
  { source: "polynomial-features", target: "hyperparameter-tuning", label: "degree search", kind: "optimizes" },
  { source: "bias-variance", target: "ensemble-methods", label: "variance reduction", kind: "extends" },
  { source: "ensemble-methods", target: "hyperparameter-tuning", label: "learner search", kind: "optimizes" },
  { source: "naive-bayes", target: "stochastic-processes", label: "probability rules", kind: "foundation" },
  { source: "time-series", target: "stochastic-processes", label: "temporal noise", kind: "foundation" },
  { source: "time-series", target: "model-evaluation", label: "rolling validation", kind: "evaluates" },
  { source: "building-tokenizer", target: "neural-network", label: "model input ids", kind: "foundation" },
  { source: "building-tokenizer", target: "stochastic-processes", label: "next-token chains", kind: "foundation" },
  { source: "building-tokenizer", target: "feature-selection", label: "text features", kind: "depends-on" },
  { source: "building-tokenizer", target: "non-negative-matrix-factorization", label: "document terms", kind: "foundation" },
  { source: "building-tokenizer", target: "loss-functions", label: "language-model targets", kind: "depends-on" },
  { source: "building-tokenizer", target: "mini-framework", label: "preprocessing layer", kind: "extends" },
  { source: "stochastic-processes", target: "dynamic-programming", label: "Markov state transitions", kind: "foundation" },
  { source: "dynamic-programming", target: "convex-optimization", label: "iterative updates", kind: "alternative" },
  { source: "singular-value-decomposition", target: "non-negative-matrix-factorization", label: "matrix factors", kind: "alternative" },
  { source: "singular-value-decomposition", target: "convolutions-from-scratch", label: "matrix transforms", kind: "foundation" },
  { source: "singular-value-decomposition", target: "neural-network", label: "linear algebra", kind: "foundation" },
  { source: "convex-optimization", target: "optimizer-race", label: "loss surfaces", kind: "foundation" },
  { source: "convolutions-from-scratch", target: "multi-layer-network", label: "learned filters", kind: "extends" },
  { source: "neural-network", target: "multi-layer-network", label: "deeper topology", kind: "extends" },
  { source: "neural-network", target: "loss-functions", label: "training objective", kind: "depends-on" },
  { source: "neural-network", target: "optimizer-race", label: "parameter updates", kind: "optimizes" },
  { source: "multi-layer-network", target: "activation-functions", label: "hidden nonlinearities", kind: "depends-on" },
  { source: "multi-layer-network", target: "backpropagation-from-scratch", label: "gradient flow", kind: "depends-on" },
  { source: "activation-functions", target: "backpropagation-from-scratch", label: "local derivatives", kind: "depends-on" },
  { source: "loss-functions", target: "backpropagation-from-scratch", label: "loss gradients", kind: "depends-on" },
  { source: "loss-functions", target: "optimizer-race", label: "surface shape", kind: "optimizes" },
  { source: "optimizer-race", target: "backpropagation-from-scratch", label: "weight updates", kind: "optimizes" },
  { source: "backpropagation-from-scratch", target: "mini-framework", label: "layer API", kind: "extends" },
  { source: "mini-framework", target: "multi-layer-network", label: "network builder", kind: "extends" },
];

export function buildConceptNeighborhood(
  algorithms: AlgorithmDefinition[],
  activeId: string,
): ConceptGraphNeighborhood {
  const byId = new Map(algorithms.map((algorithm) => [algorithm.id, algorithm]));
  const active = byId.get(activeId) ?? algorithms[0];
  const directRelationships = conceptRelationships.filter((relationship) =>
    relationship.source === active.id || relationship.target === active.id,
  );
  const visibleIds = new Set([
    active.id,
    ...directRelationships.map((relationship) =>
      relationship.source === active.id ? relationship.target : relationship.source,
    ),
  ]);
  const nodes = [...visibleIds]
    .map((id) => byId.get(id))
    .filter((algorithm): algorithm is AlgorithmDefinition => Boolean(algorithm))
    .map<ConceptGraphNode>((algorithm) => ({
      id: algorithm.id,
      algorithm,
      domain: conceptDomains[algorithm.id] ?? "foundations",
      depth: algorithm.id === active.id ? 0 : 1,
    }));
  const depthById = new Map(nodes.map((node) => [node.id, node.depth]));
  const links = conceptRelationships
    .filter((relationship) => visibleIds.has(relationship.source) && visibleIds.has(relationship.target))
    .map<ConceptGraphLink>((relationship) => ({
      ...relationship,
      sourceDepth: depthById.get(relationship.source) ?? 1,
      targetDepth: depthById.get(relationship.target) ?? 1,
    }));

  return {
    active,
    nodes,
    links,
    directRelationships,
  };
}
