import {
  algorithmCategories,
  type AlgorithmCategory,
  type AlgorithmDefinition,
} from "../types/algorithm";

type Props = {
  algorithms: AlgorithmDefinition[];
  activeId: string;
  query: string;
  onSelect: (id: string) => void;
};

export function AlgorithmCatalog({
  algorithms,
  activeId,
  query,
  onSelect,
}: Props) {
  const normalizedQuery = query.trim().toLowerCase();
  const filteredAlgorithms = algorithms.filter((algorithm) => {
    const haystack = [
      algorithm.name,
      algorithm.category,
      algorithm.summary,
    ].join(" ").toLowerCase();

    return haystack.includes(normalizedQuery);
  });

  const groups = filteredAlgorithms.reduce<
    Partial<Record<AlgorithmCategory, AlgorithmDefinition[]>>
  >(
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

  return (
    <aside className="catalog-panel" aria-label="Algorithm catalog">
      <div className="panel-heading">
        <p>Catalog</p>
        <span>
          {algorithms.length} algorithms · {algorithmCategories.length} categories
        </span>
      </div>

      <nav className="catalog-list">
        {visibleCategories.map((category) => {
          const items = groups[category] ?? [];

          return (
          <section className="category-group" key={category}>
            <h3 className="category-title">{category}</h3>
            {items.map((algorithm) => (
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
            {items.length === 0 && (
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

const categoryNotes: Record<AlgorithmCategory, string> = {
  Regression: "Linear models for continuous prediction.",
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
  "Determinants & Linear Transformations": "Area scale, orientation, grid warps, and eigenvalue products.",
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
