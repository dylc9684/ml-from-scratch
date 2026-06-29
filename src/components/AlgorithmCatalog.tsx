import { ChevronDown, Compass, History } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  algorithmCategories,
  type AlgorithmCategory,
  type AlgorithmDefinition,
} from "../types/algorithm";

type CatalogFilter = "all" | "beginner" | "math" | "neural" | "linear-algebra" | "optimization";

type Props = {
  algorithms: AlgorithmDefinition[];
  activeId: string;
  query: string;
  recentIds: string[];
  onSelect: (id: string) => void;
};

export function AlgorithmCatalog({
  algorithms,
  activeId,
  query,
  recentIds,
  onSelect,
}: Props) {
  const [activeFilter, setActiveFilter] = useState<CatalogFilter>("all");
  const activeAlgorithm = algorithms.find((algorithm) => algorithm.id === activeId) ?? algorithms[0];
  const [collapsedCategories, setCollapsedCategories] = useState<Set<AlgorithmCategory>>(
    () => new Set(algorithmCategories.filter((category) => category !== activeAlgorithm.category)),
  );
  const normalizedQuery = query.trim().toLowerCase();

  useEffect(() => {
    setCollapsedCategories((current) => {
      if (!current.has(activeAlgorithm.category)) {
        return current;
      }
      const next = new Set(current);
      next.delete(activeAlgorithm.category);
      return next;
    });
  }, [activeAlgorithm.category]);

  const filteredAlgorithms = algorithms.filter((algorithm) => {
    const haystack = [
      algorithm.name,
      algorithm.category,
      algorithm.summary,
    ].join(" ").toLowerCase();

    return haystack.includes(normalizedQuery) && matchesCatalogFilter(algorithm, activeFilter);
  });

  const groups = filteredAlgorithms.reduce<Partial<Record<AlgorithmCategory, AlgorithmDefinition[]>>>(
    (grouped, algorithm) => {
      const categoryAlgorithms = grouped[algorithm.category] ?? [];
      categoryAlgorithms.push(algorithm);
      grouped[algorithm.category] = categoryAlgorithms;
      return grouped;
    },
    {},
  );
  const visibleCategories = algorithmCategories.filter((category) => {
    const items = groups[category] ?? [];
    const categoryMatches = category.toLowerCase().includes(normalizedQuery);

    if (!normalizedQuery) {
      return true;
    }

    return categoryMatches || items.length > 0;
  });
  const byId = useMemo(() => new Map(algorithms.map((algorithm) => [algorithm.id, algorithm])), [algorithms]);
  const recentAlgorithms = recentIds
    .map((id) => byId.get(id))
    .filter((algorithm): algorithm is AlgorithmDefinition => Boolean(algorithm));
  const recommendedAlgorithms = recommendedPathIds
    .map((id) => byId.get(id))
    .filter((algorithm): algorithm is AlgorithmDefinition => Boolean(algorithm));

  const toggleCategory = (category: AlgorithmCategory) => {
    setCollapsedCategories((current) => {
      const next = new Set(current);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  return (
    <aside className="catalog-panel" aria-label="Algorithm catalog">
      <div className="panel-heading">
        <p>Learning Path</p>
        <span>
          {algorithms.length} algorithms · {algorithmCategories.length} categories
        </span>
      </div>

      <div className="catalog-filter-row" aria-label="Catalog filters">
        {catalogFilters.map((filter) => (
          <button
            key={filter.id}
            className={`catalog-filter ${activeFilter === filter.id ? "active" : ""}`}
            type="button"
            onClick={() => setActiveFilter(filter.id)}
          >
            {filter.label}
          </button>
        ))}
      </div>

      <section className="catalog-quick-section" aria-label="Recommended learning path">
        <div className="quick-section-heading">
          <Compass size={14} aria-hidden="true" />
          <span>Recommended Path</span>
        </div>
        <div className="recommended-path">
          {recommendedAlgorithms.map((algorithm, index) => (
            <button
              key={algorithm.id}
              className={`path-step ${algorithm.id === activeId ? "active" : ""}`}
              type="button"
              onClick={() => onSelect(algorithm.id)}
            >
              <span>{index + 1}</span>
              <strong>{algorithm.name}</strong>
            </button>
          ))}
        </div>
      </section>

      {recentAlgorithms.length > 0 && (
        <section className="catalog-quick-section" aria-label="Recently viewed lessons">
          <div className="quick-section-heading">
            <History size={14} aria-hidden="true" />
            <span>Recently Viewed</span>
          </div>
          <div className="recent-row">
            {recentAlgorithms.map((algorithm) => (
              <button
                key={algorithm.id}
                className={`recent-chip ${algorithm.id === activeId ? "active" : ""}`}
                type="button"
                onClick={() => onSelect(algorithm.id)}
              >
                {algorithm.name}
              </button>
            ))}
          </div>
        </section>
      )}

      <nav className="catalog-list">
        {visibleCategories.map((category) => {
          const items = groups[category] ?? [];
          const collapsed = collapsedCategories.has(category) && !normalizedQuery;

          return (
          <section className="category-group" key={category}>
            <button className="category-title-button" type="button" onClick={() => toggleCategory(category)}>
              <span>{category}</span>
              <small>{items.length}</small>
              <ChevronDown size={15} className={collapsed ? "" : "open"} aria-hidden="true" />
            </button>
            {!collapsed && items.map((algorithm) => (
              <button
                key={algorithm.id}
                className={`algorithm-row ${
                  algorithm.id === activeId ? "active" : ""
                }`}
                type="button"
                onClick={() => onSelect(algorithm.id)}
              >
                <span className="algorithm-dot" aria-hidden="true" />
                <span className="algorithm-main">
                  <strong>{algorithm.name}</strong>
                  <small>{algorithm.summary}</small>
                </span>
              </button>
            ))}
            {!collapsed && items.length === 0 && (
              <div className="category-placeholder">
                <span aria-hidden="true" />
                <p>{categoryNotes[category]}</p>
              </div>
            )}
          </section>
          );
        })}

        {visibleCategories.length === 0 && (
          <p className="empty-state">No categories or algorithms match that search.</p>
        )}
      </nav>
    </aside>
  );
}

const catalogFilters: Array<{ id: CatalogFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "beginner", label: "Beginner" },
  { id: "math", label: "Math Heavy" },
  { id: "neural", label: "Neural Nets" },
  { id: "linear-algebra", label: "Linear Algebra" },
  { id: "optimization", label: "Optimization" },
];

const recommendedPathIds = [
  "linear-regression",
  "bayesian-regression",
  "polynomial-features",
  "regularization-and-noise",
  "determinant-visualizer",
  "find-magic-directions",
  "loss-functions",
  "optimizer-race",
  "backpropagation-from-scratch",
];

function matchesCatalogFilter(algorithm: AlgorithmDefinition, filter: CatalogFilter) {
  if (filter === "all") {
    return true;
  }

  if (filter === "beginner") {
    return beginnerIds.has(algorithm.id) || beginnerCategories.has(algorithm.category);
  }

  if (filter === "math") {
    return mathHeavyIds.has(algorithm.id) || mathHeavyCategories.has(algorithm.category);
  }

  if (filter === "neural") {
    return neuralCategories.has(algorithm.category) || algorithm.category.includes("Neural");
  }

  if (filter === "linear-algebra") {
    return linearAlgebraIds.has(algorithm.id) || linearAlgebraCategories.has(algorithm.category);
  }

  return optimizationIds.has(algorithm.id) || optimizationCategories.has(algorithm.category);
}

const beginnerIds = new Set([
  "linear-regression",
  "bayesian-regression",
  "logistic-regression",
  "knn-classifier",
  "k-means",
  "polynomial-features",
  "model-evaluation",
  "bias-variance",
]);

const beginnerCategories = new Set<AlgorithmCategory>([
  "Regression",
  "KNN & Distance Metrics",
  "Clustering",
  "Model Evaluation: Metrics, Cross-Validation",
  "Bias, Variance & the Learning Curve",
]);

const mathHeavyIds = new Set([
  "bayesian-regression",
  "singular-value-decomposition",
  "non-negative-matrix-factorization",
  "determinant-visualizer",
  "find-magic-directions",
  "convex-optimization",
  "dynamic-programming",
  "backpropagation-from-scratch",
]);

const mathHeavyCategories = new Set<AlgorithmCategory>([
  "Singular Value Decomposition",
  "Non-negative Matrix Factorization",
  "Determinants & Linear Transformations",
  "Convex Optimization",
  "Dynamic Programming",
  "Backpropagation from scratch",
]);

const neuralCategories = new Set<AlgorithmCategory>([
  "Neural Networks",
  "Multi-Layer Networks",
  "Activation Functions — ReLU, Sigmoid, GELU",
  "Loss Functions — MSE, Cross-Entropy, Contrastive",
  "SGD, Momentum, Adam, AdamW",
  "Build a Mini-Framework",
  "Backpropagation from scratch",
  "Convolutions from Scratch",
]);

const linearAlgebraIds = new Set([
  "singular-value-decomposition",
  "non-negative-matrix-factorization",
  "determinant-visualizer",
  "find-magic-directions",
  "convolutions-from-scratch",
]);

const linearAlgebraCategories = new Set<AlgorithmCategory>([
  "Singular Value Decomposition",
  "Non-negative Matrix Factorization",
  "Determinants & Linear Transformations",
  "Convolutions from Scratch",
]);

const optimizationIds = new Set([
  "hyperparameter-tuning",
  "regularization-and-noise",
  "convex-optimization",
  "optimizer-race",
  "dynamic-programming",
]);

const optimizationCategories = new Set<AlgorithmCategory>([
  "Hyperparameter Tuning",
  "Regularization and Noise",
  "Convex Optimization",
  "SGD, Momentum, Adam, AdamW",
  "Dynamic Programming",
]);

const categoryNotes: Record<AlgorithmCategory, string> = {
  Regression: "Linear and probabilistic models for continuous prediction.",
  "Logistic Regression & Classification":
    "Binary and multiclass decision boundaries.",
  "Decision Trees & Random Forests":
    "Tree splits, impurity, bagging, and feature importance.",
  "Support Vector Machines": "Margins, kernels, and support vectors.",
  "KNN & Distance Metrics": "Nearest-neighbor voting and distance geometry.",
  Clustering: "Unsupervised grouping and centroid movement.",
  "Feature Engineering & Selection":
    "Polynomial transforms, encodings, scaling, and feature ranking.",
  "Model Evaluation: Metrics, Cross-Validation":
    "Scores, folds, confusion matrices, and validation strategy.",
  "Bias, Variance & the Learning Curve":
    "Underfitting, overfitting, and sample-size effects.",
  "Ensemble Methods: Boosting, Bagging, Stacking":
    "Combining learners for stronger predictions.",
  "Hyperparameter Tuning": "Search strategies and validation feedback.",
  "Regularization and Noise": "Ridge shrinkage, Lasso pruning, noisy fits, and coefficient stability.",
  "Naive Bayes": "Probabilistic classification with conditional assumptions.",
  "Anomaly Detection": "Outlier scoring and unusual-pattern discovery.",
  "Handling Imbalanced Data": "Sampling, class weights, and threshold tradeoffs.",
  "Time Series Fundamentals": "Trend, seasonality, lag features, and forecasting.",
  "Building a Tokenizer": "Subword units, byte fallback, BPE merges, and token IDs.",
  "Stochastic Processes": "Random walks, noise, transition matrices, and Markov chains.",
  "Dynamic Programming": "Bellman updates, Gridworld values, and optimal policies.",
  "Singular Value Decomposition": "Matrix splitting, low-rank compression, and background/foreground separation.",
  "Non-negative Matrix Factorization": "Additive topic extraction and localized parts-based image factors.",
  "Determinants & Linear Transformations": "Area scale, orientation, grid warps, eigenvectors, and eigenvalue products.",
  "Convex Optimization": "Constrained bowls, projected steps, Hessians, and convexity checks.",
  "Convolutions from Scratch": "Kernels, padding, stride, and image feature maps.",
  "Neural Networks": "Layered differentiable models trained by backpropagation.",
  "Multi-Layer Networks": "Deep feed-forward topologies with stacked hidden layers.",
  "Activation Functions — ReLU, Sigmoid, GELU":
    "Nonlinearities, gradients, saturation, and dying-neuron behavior.",
  "Loss Functions — MSE, Cross-Entropy, Contrastive":
    "Training objectives, loss curves, and metric-learning distances.",
  "SGD, Momentum, Adam, AdamW":
    "Optimizer trajectories, adaptive rates, momentum, and decoupled decay.",
  "Build a Mini-Framework":
    "Layer interfaces, Sequential containers, Trainer loops, and visual hooks.",
  "Backpropagation from scratch":
    "Forward activations, backward deltas, and gradient updates by hand.",
};
