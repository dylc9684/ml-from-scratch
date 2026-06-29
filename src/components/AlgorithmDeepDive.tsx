import type {
  AlgorithmCategory,
  AlgorithmDefinition,
  AlgorithmFrame,
  Metric,
} from "../types/algorithm";

type Props = {
  algorithm: AlgorithmDefinition;
  frame: AlgorithmFrame | null;
  metrics: Metric[];
};

export type ComplexityTerm = {
  label: string;
  weight: number;
};

export type DeepDiveContent = {
  graphTitle: string;
  graphNotes: string[];
  complexity: {
    time: string;
    prediction?: string;
    space: string;
    plainEnglish: string;
    terms: ComplexityTerm[];
  };
  realWorld: string[];
  keyDetails: string[];
  beginnerTerms?: Array<{ term: string; definition: string }>;
};

export function AlgorithmDeepDive({ algorithm, frame, metrics }: Props) {
  const content = getDeepDiveContent(algorithm);
  const currentSummary = frameSummary(frame);
  const visibleMetrics = metrics.slice(0, 4);
  const terms = content.beginnerTerms ?? defaultBeginnerTerms;

  return (
    <section className="algorithm-deep-dive" aria-label={`${algorithm.name} deep dive`}>
      <div className="deep-dive-header">
        <span>Deep dive</span>
        <h3>{algorithm.name}: what matters beyond the graph</h3>
        <p>
          A slower read of the visualization, how the algorithm scales, where it shows up in the
          real world, and the details that usually matter when you move from a toy demo to real data.
        </p>
      </div>

      <div className="deep-dive-grid">
        <article className="deep-dive-block deep-dive-block-wide">
          <span>{content.graphTitle}</span>
          <ul>
            {content.graphNotes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </article>

        <article className="deep-dive-block complexity-block">
          <span>Running time and Big O</span>
          <dl className="complexity-list">
            <div>
              <dt>Training</dt>
              <dd>{content.complexity.time}</dd>
            </div>
            {content.complexity.prediction && (
              <div>
                <dt>Prediction</dt>
                <dd>{content.complexity.prediction}</dd>
              </div>
            )}
            <div>
              <dt>Memory</dt>
              <dd>{content.complexity.space}</dd>
            </div>
          </dl>
          <p>{content.complexity.plainEnglish}</p>
          <div className="complexity-bars" aria-label="Complexity drivers">
            {content.complexity.terms.map((term) => (
              <div className="complexity-bar-row" key={term.label}>
                <span>{term.label}</span>
                <div>
                  <i style={{ width: `${Math.max(10, Math.min(100, term.weight))}%` }} />
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="deep-dive-block">
          <span>Real-world examples</span>
          <ul>
            {content.realWorld.map((example) => (
              <li key={example}>{example}</li>
            ))}
          </ul>
        </article>

        <article className="deep-dive-block">
          <span>Key details to remember</span>
          <ul>
            {content.keyDetails.map((detail) => (
              <li key={detail}>{detail}</li>
            ))}
          </ul>
        </article>
      </div>

      <div className="deep-dive-current">
        <div>
          <span>Current run</span>
          <strong>{currentSummary}</strong>
        </div>
        <div className="deep-dive-metric-strip">
          {visibleMetrics.map((metric) => (
            <span key={metric.label}>
              {metric.label}: <strong>{metric.value}</strong>
            </span>
          ))}
        </div>
      </div>

      <div className="big-o-glossary">
        {terms.map((item) => (
          <div key={item.term}>
            <strong>{item.term}</strong>
            <span>{item.definition}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function getDeepDiveContent(algorithm: AlgorithmDefinition) {
  return (
    deepDiveById[algorithm.id] ??
    deepDiveByCategory[algorithm.category] ??
    makeFallbackDive(algorithm)
  );
}

export function frameSummary(frame: AlgorithmFrame | null) {
  if (!frame) {
    return "Run the algorithm to generate a live summary.";
  }

  if (frame.type === "linear-regression") {
    return `MSE ${frame.loss.toFixed(4)} with line y = ${frame.slope.toFixed(2)}x + ${frame.intercept.toFixed(2)}.`;
  }

  if (frame.type === "k-means") {
    return `Inertia ${frame.inertia.toFixed(3)} across ${frame.centroids.length} centroids.`;
  }

  return frame.summary;
}

function makeFallbackDive(algorithm: AlgorithmDefinition): DeepDiveContent {
  return {
    graphTitle: `How to read ${algorithm.name}`,
    graphNotes: [
      "Watch which quantities move first: model parameters, assignments, losses, or values.",
      "A stable graph usually means the update rule has converged or stopped changing much.",
      "Large jumps usually point to sensitivity in learning rate, initialization, or data scale.",
    ],
    complexity: {
      time: "Depends on the number of data points, parameters, and update steps.",
      space: "Depends on the data matrix plus any cached model state.",
      plainEnglish:
        "Big O is a rough growth estimate. It tells you what gets expensive when the dataset gets larger.",
      terms: [
        { label: "data size n", weight: 68 },
        { label: "features d", weight: 54 },
        { label: "iterations t", weight: 48 },
      ],
    },
    realWorld: [
      "Use the demo to build intuition before picking production libraries or model sizes.",
      "Most real systems add validation, monitoring, and guardrails around the core math.",
      "The same update pattern often appears inside larger AI pipelines.",
    ],
    keyDetails: [
      "Scale inputs when distances, gradients, or dot products drive the result.",
      "Check whether the algorithm needs repeated passes over the data.",
      "Good visual convergence does not guarantee good generalization on new data.",
    ],
  };
}

const defaultBeginnerTerms = [
  { term: "n", definition: "number of rows, samples, states, pixels, or data points." },
  { term: "d", definition: "number of features or dimensions in each sample." },
  { term: "k", definition: "number of neighbors, clusters, rank terms, classes, or selected items." },
  { term: "t", definition: "number of iterations, epochs, sweeps, or optimization steps." },
];

const deepDiveById: Record<string, DeepDiveContent> = {
  "linear-regression": {
    graphTitle: "What the graph is showing",
    graphNotes: [
      "The line moves toward the center of the point cloud as gradient descent reduces mean squared error.",
      "Steeper slopes mean the model thinks x has a stronger relationship with y.",
      "If the line bounces around, the learning rate is probably too high for the scale of the data.",
    ],
    complexity: {
      time: "O(t n d) for gradient descent.",
      prediction: "O(d) per new example.",
      space: "O(n d) for the data plus O(d) for weights.",
      plainEnglish:
        "Each step scans the rows and features once. More epochs or more columns increase runtime directly.",
      terms: [
        { label: "samples n", weight: 72 },
        { label: "features d", weight: 44 },
        { label: "epochs t", weight: 64 },
      ],
    },
    realWorld: [
      "Forecasting price from size, age, mileage, or other numeric variables.",
      "Estimating demand, salary bands, baseline risk, or simple trend lines.",
      "Serving as a transparent baseline before trying larger models.",
    ],
    keyDetails: [
      "It assumes the target changes roughly linearly with the inputs.",
      "Outliers can dominate squared error because large mistakes are punished heavily.",
      "Feature scaling makes gradient descent smoother and easier to tune.",
    ],
  },
  "bayes-rule-visualizer": {
    graphTitle: "What the grid and tree are showing",
    graphNotes: [
      "The grid turns probabilities into 1,000 concrete people so the denominator is visible.",
      "The positive-test group contains both true positives and false positives; the posterior is the true-positive share of that group.",
      "Changing the prior often moves the posterior more than people expect, especially for rare events.",
    ],
    complexity: {
      time: "O(n) to render a frequency grid with n people; the probability update itself is O(1).",
      prediction: "O(1) for one posterior calculation.",
      space: "O(n) for the displayed people, or O(1) if you keep only the counts.",
      plainEnglish:
        "Bayes' Rule is just a few multiplications and a division. The visual grid costs more than the math because it draws every person.",
      terms: [
        { label: "people n", weight: 64 },
        { label: "posterior O(1)", weight: 86 },
        { label: "false positives", weight: 72 },
      ],
    },
    realWorld: [
      "Medical screening, where a positive test can still be uncertain if the condition is rare.",
      "Fraud and security alerts, where most flagged events may be false positives unless the base rate is high enough.",
      "Spam, moderation, and monitoring systems that need calibrated confidence after evidence arrives.",
    ],
    keyDetails: [
      "P(A) is the prior: what you believed before seeing the evidence.",
      "P(B|A) is sensitivity: how often evidence appears when A is truly present.",
      "P(B|not A) is the false positive rate: how often evidence appears even when A is absent.",
      "Posterior means useful evidence divided by all evidence, not just test accuracy.",
    ],
  },
  "hidden-markov-models": {
    graphTitle: "What the trellis is showing",
    graphNotes: [
      "Each column is one visible observation; each row is one hidden state candidate.",
      "Viterbi stores the best previous state for every cell, then backtraces the highest-scoring final cell.",
      "The glowing path is the single most likely hidden-state explanation for the entire sequence.",
    ],
    complexity: {
      time: "O(T K^2), where T is sequence length and K is the number of hidden states.",
      prediction: "O(T K^2) for one decoded sequence.",
      space: "O(T K) for the trellis and backpointers.",
      plainEnglish:
        "Every time step compares every current state against every previous state. More hidden states are the main cost multiplier.",
      terms: [
        { label: "steps T", weight: 72 },
        { label: "states K", weight: 84 },
        { label: "trellis T*K", weight: 70 },
      ],
    },
    realWorld: [
      "Early part-of-speech tagging and speech recognition before neural sequence models became dominant.",
      "Market-regime and weather-state decoding from visible observations.",
      "Fault detection in sensors where the true machine state is hidden but emissions are observable.",
    ],
    keyDetails: [
      "Use log-probabilities so long sequences do not underflow toward zero.",
      "Transition probabilities describe hidden-state movement; emission probabilities describe visible evidence from a state.",
      "Viterbi finds the best single path, while forward-backward estimates marginal state probabilities.",
      "HMMs struggle when the next state depends on long-range context beyond the previous hidden state.",
    ],
  },
  "gaussian-discriminant-analysis": {
    graphTitle: "What the contours are showing",
    graphNotes: [
      "Each ring is a constant-density contour from a learned 2D Gaussian profile for one class.",
      "Dragging the training-size constraint low inflates and fades the contours, representing uncertainty in the estimated mean and covariance.",
      "LDA shares one covariance shape across classes; QDA lets each class stretch and rotate independently, which can bend the boundary.",
    ],
    complexity: {
      time: "Fitting is O(n d^2 + k d^3); drawing the probability grid adds O(g k d^2).",
      prediction: "O(k d^2) per point because each class evaluates a quadratic Gaussian score.",
      space: "O(k d^2) for class covariance matrices plus O(g) for the rendered grid.",
      plainEnglish:
        "With two features this is tiny. The expensive part in the playground is drawing many background cells, not inverting 2 by 2 matrices.",
      terms: [
        { label: "samples n", weight: 68 },
        { label: "features d", weight: 72 },
        { label: "classes k", weight: 58 },
        { label: "grid cells g", weight: 64 },
      ],
    },
    realWorld: [
      "Sensor classification where each machine state has a characteristic cloud of readings.",
      "Medical or biological screening when class-conditional measurements are roughly bell-shaped.",
      "Fast client-side routing when a small probabilistic classifier must score streaming points without a backend.",
    ],
    keyDetails: [
      "GDA is generative: it models p(x|class) and then uses Bayes' Rule to classify.",
      "QDA is more flexible but needs more data because every class gets its own covariance matrix.",
      "LDA is more stable with small data because it shares one covariance matrix across classes.",
      "Poorly estimated covariance can overfit or become singular, so production systems often add shrinkage or a ridge term.",
    ],
  },
  "bayesian-regression": {
    graphTitle: "What the band and sampled lines are showing",
    graphNotes: [
      "The green line is the posterior mean: the center of the model's current belief about the linear trend.",
      "The blue shaded region is predictive uncertainty. It tightens near evidence and widens where the model is extrapolating.",
      "Sampled lines are plausible parameter draws from the posterior; they fan out with little data and cluster as evidence grows.",
    ],
    complexity: {
      time: "O(n d^2 + d^3) for the closed-form posterior; here d=2 for intercept and slope.",
      prediction: "O(d^2) per x value when computing uncertainty, or O(d) for the mean line only.",
      space: "O(d^2) for the covariance matrix plus O(n d) for the design matrix if stored.",
      plainEnglish:
        "The tiny browser demo is cheap because it only learns intercept and slope. Bigger Bayesian linear models get more expensive as feature count grows because covariance is a d by d matrix.",
      terms: [
        { label: "samples n", weight: 68 },
        { label: "features d", weight: 62 },
        { label: "covariance d^2", weight: 78 },
      ],
    },
    realWorld: [
      "Forecasting demand, revenue, or risk while returning an interval instead of a single overconfident number.",
      "Active learning and experiment planning, where the next sample should be collected where uncertainty is largest.",
      "Small-data scientific modeling, where a prior keeps the model reasonable until enough measurements arrive.",
    ],
    keyDetails: [
      "The prior controls how strongly the model trusts small weights before seeing data.",
      "The noise setting controls how much scatter the model expects around the true line.",
      "Uncertainty is lowest near observed x values and grows when the model extrapolates far away from evidence.",
      "The demo assumes a linear relationship and Gaussian noise; curved patterns still need richer features or another model.",
    ],
  },
  "k-means": {
    graphTitle: "What the graph is showing",
    graphNotes: [
      "Points are repeatedly assigned to the nearest centroid, then each centroid moves to its assigned mean.",
      "Inertia is the total within-cluster distance. Lower is usually tighter, but not always more meaningful.",
      "Different initial centroids can produce different cluster layouts.",
    ],
    complexity: {
      time: "O(t n k d).",
      prediction: "O(k d) to assign one new point.",
      space: "O(n + k d) after the data is loaded.",
      plainEnglish:
        "Every iteration compares every point with every centroid, so k and n are the main cost drivers.",
      terms: [
        { label: "samples n", weight: 78 },
        { label: "clusters k", weight: 70 },
        { label: "features d", weight: 48 },
        { label: "iterations t", weight: 58 },
      ],
    },
    realWorld: [
      "Customer segmentation, image color quantization, and grouping geographic locations.",
      "Finding rough groups before labeling data for supervised learning.",
      "Compressing many points into a smaller set of representative centers.",
    ],
    keyDetails: [
      "K-Means likes round-ish clusters and struggles with curved or uneven-density groups.",
      "Always scale features when distances mix units like dollars, age, and clicks.",
      "Pick k with validation, domain knowledge, or an elbow/silhouette analysis.",
    ],
  },
  "polynomial-features": {
    graphTitle: "What the curve and matrix are showing",
    graphNotes: [
      "The curve is still linear regression, but it is fitting columns like x, x^2, and x^3 instead of only raw x.",
      "Hovering a point reveals the exact feature row created from that one x coordinate.",
      "Higher degree lowers training error at first, then can create sharp bends that memorize noise.",
    ],
    complexity: {
      time: "Feature expansion is O(n d) with d=degree+1; solving least squares is roughly O(d^3 + n d^2).",
      prediction: "O(d) per x value after the coefficients are learned.",
      space: "O(n d) for the expanded feature matrix Phi.",
      plainEnglish:
        "The expensive part is not making powers of x; it is solving for more coefficients as degree grows.",
      terms: [
        { label: "points n", weight: 68 },
        { label: "degree d", weight: 78 },
        { label: "matrix solve d^3", weight: 62 },
      ],
    },
    realWorld: [
      "Adding curvature to simple regression baselines before trying a neural network.",
      "Modeling calibration curves, dose-response curves, sensor drift, and smooth physical trends.",
      "Teaching overfitting because high-degree polynomials can look excellent on training points and awful between them.",
    ],
    keyDetails: [
      "Polynomial features make linear models nonlinear in x while staying linear in the learned coefficients.",
      "Scaling x matters because high powers can explode numerically.",
      "Validation error, not training error, should decide the useful degree.",
    ],
  },
  "regularization-and-noise": {
    graphTitle: "What the regularization graph is showing",
    graphNotes: [
      "The red alpha-0 curve shows what a flexible polynomial does when it is allowed to chase noisy training points.",
      "The regularized curve adds a penalty for large coefficients, so it trades a little training fit for a smoother shape.",
      "The ledger shows the difference between Ridge and Lasso: Ridge shrinks weights together, while Lasso can drive specific columns to exactly zero.",
    ],
    complexity: {
      time: "Ridge closed form is roughly O(d^3 + n d^2); coordinate-descent Lasso is O(t n d).",
      prediction: "O(d) per x value after polynomial features are expanded.",
      space: "O(n d) for the expanded design matrix plus O(d) for weights.",
      plainEnglish:
        "Regularization does not make prediction much slower; the main cost is fitting the expanded feature matrix while adding the penalty.",
      terms: [
        { label: "samples n", weight: 68 },
        { label: "features d", weight: 78 },
        { label: "lasso sweeps t", weight: 54 },
        { label: "matrix solve d^3", weight: 62 },
      ],
    },
    realWorld: [
      "Stabilizing noisy forecasting curves, pricing models, and scientific calibration fits.",
      "Keeping high-dimensional tabular models from leaning too hard on weak or redundant columns.",
      "Using Lasso as a practical feature-pruning baseline before more complex feature-selection workflows.",
    ],
    keyDetails: [
      "Ridge uses an L2 penalty, which prefers many small weights over a few huge ones.",
      "Lasso uses an L1 penalty, which creates sparsity and can remove features entirely.",
      "Alpha is a bias-variance control: too low overfits noise, too high underfits the real signal.",
      "Scale features before regularization so one unit-heavy column does not get penalized unfairly.",
    ],
  },
  "determinant-visualizer": {
    graphTitle: "What the transformation graph is showing",
    graphNotes: [
      "The blue/green arrows are the matrix columns: where the original basis vectors land.",
      "The transformed parallelogram's signed area is the determinant.",
      "Dashed eigenvector lines are directions that stay on the same line after transformation.",
    ],
    complexity: {
      time: "O(1) for a 2x2 determinant and closed-form eigenvalues.",
      prediction: "O(1) to transform one 2D point.",
      space: "O(1) for the matrix, determinant, and eigenvalue state.",
      plainEnglish:
        "For a 2x2 matrix this is tiny math: a few multiplications for det, and a quadratic formula for eigenvalues.",
      terms: [
        { label: "matrix entries", weight: 46 },
        { label: "area scale", weight: 78 },
        { label: "eigenvalues", weight: 68 },
      ],
    },
    realWorld: [
      "Detecting whether a 2D transform preserves, flips, scales, or collapses geometry.",
      "Understanding Jacobian determinants in change-of-variables, graphics, robotics, and simulations.",
      "Building intuition for why zero determinants mean information is lost.",
    ],
    keyDetails: [
      "The determinant is signed: negative means the orientation flipped, not that area became negative.",
      "A determinant near zero means the plane is nearly collapsed, which makes inverse computations unstable.",
      "For any square matrix, the determinant equals the product of eigenvalues, counting complex eigenvalues too.",
    ],
  },
  "find-magic-directions": {
    graphTitle: "What the magic-direction interface is showing",
    graphNotes: [
      "The blue vector is a test direction v. The after-vector Av shows what the matrix does to it.",
      "Most vectors change direction, but eigenvectors stay on the same line and only scale or reverse.",
      "Power iteration repeatedly applies A and often drifts toward the strongest preserved direction.",
    ],
    complexity: {
      time: "O(1) for a 2x2 closed-form check; O(t d^2) for power iteration on a d by d matrix.",
      prediction: "O(d^2) to multiply one vector by a dense matrix.",
      space: "O(d) for the current vector, or O(t d) if every power-iteration step is stored.",
      plainEnglish:
        "For this 2D demo, everything is tiny. In larger systems the cost is dominated by repeated matrix-vector multiplication.",
      terms: [
        { label: "dimension d", weight: 78 },
        { label: "iterations t", weight: 58 },
        { label: "matrix multiply d^2", weight: 72 },
      ],
    },
    realWorld: [
      "Principal component analysis, PageRank-style ranking, vibration modes, stability analysis, and spectral clustering.",
      "Understanding why some transformations have natural axes or dominant directions.",
      "Diagnosing neural-network weight matrices, covariance matrices, and linear dynamical systems.",
    ],
    keyDetails: [
      "An eigenvector is about direction, not position. Any nonzero scalar multiple of an eigenvector is still an eigenvector.",
      "A negative eigenvalue means the vector reverses direction while staying on the same line.",
      "Complex eigenvalues mean no real direction is preserved in the simple arrow-line sense.",
      "The determinant equals the product of eigenvalues, tying this lesson back to area scaling.",
    ],
  },
  "building-tokenizer": {
    graphTitle: "What the tokenizer graph is showing",
    graphNotes: [
      "Each colored chip is one token that will become one integer ID before text reaches a model.",
      "A larger vocabulary allows more frequent character pairs to merge into longer subwords.",
      "Red byte fallback chips show characters that were not merged into familiar text pieces.",
    ],
    complexity: {
      time: "Training toy BPE is roughly O(m L) here, where m is merges and L is corpus length.",
      prediction: "O(m L) for a simple educational merge loop over one input.",
      space: "O(V + L) for merge rules, token IDs, and the current symbol sequence.",
      plainEnglish:
        "Production tokenizers use optimized tables, but the core idea is still repeated pair lookup and replacement.",
      terms: [
        { label: "input length L", weight: 76 },
        { label: "merge rules m", weight: 68 },
        { label: "vocab V", weight: 54 },
        { label: "bytes", weight: 62 },
      ],
    },
    realWorld: [
      "Counting context-window cost before sending prompts to a language model.",
      "Understanding why emoji, rare scripts, or mixed-language text can consume more tokens.",
      "Building client-side prompt previews, truncators, and text preprocessing tools.",
    ],
    keyDetails: [
      "Token IDs are not words; they are learned chunks that may be words, word pieces, spaces, or bytes.",
      "Small vocabularies reduce the lookup table but increase token overhead.",
      "Byte fallback makes every string encodable, but poorly represented text becomes more expensive.",
    ],
  },
  "non-negative-matrix-factorization": {
    graphTitle: "What the NMF panels are showing",
    graphNotes: [
      "The topic panel factors a document-term matrix into document-topic weights and topic-word weights.",
      "The sorted bars show which words contribute positive evidence for each topic.",
      "The face panel uses the same additive rule: changing one W weight changes one localized part.",
    ],
    complexity: {
      time: "O(t n d k) for multiplicative updates on an n by d matrix with k topics.",
      prediction: "O(d k) to infer topic weights for a new document with fixed topic-word factors.",
      space: "O(n k + k d) for W and H, plus O(n d) if the full input matrix is kept.",
      plainEnglish:
        "Every update multiplies the document-term matrix by the current factors, so documents, vocabulary size, topics, and iterations all matter.",
      terms: [
        { label: "documents n", weight: 68 },
        { label: "terms d", weight: 76 },
        { label: "topics k", weight: 58 },
        { label: "iterations t", weight: 62 },
      ],
    },
    realWorld: [
      "Topic modeling for article collections, support tickets, research abstracts, and product reviews.",
      "Parts-based decomposition for faces, medical imagery, audio spectrograms, and nonnegative sensor readings.",
      "Interpretable feature discovery when negative cancellation would make factors harder to explain.",
    ],
    keyDetails: [
      "NMF is not unique; different initial factors or k values can produce different but valid decompositions.",
      "Nonnegativity encourages parts to add together, which often makes topics and face components easier to inspect than SVD factors.",
      "Text preprocessing matters: stop words, vocabulary size, and TF-IDF scaling strongly shape the topics.",
    ],
  },
};

const deepDiveByCategory: Partial<Record<AlgorithmCategory, DeepDiveContent>> = {
  "Logistic Regression & Classification": {
    graphTitle: "How to read the classifier graph",
    graphNotes: [
      "The boundary separates regions where the predicted class probability changes from low to high.",
      "Points near the boundary are uncertain; points far away are high-confidence predictions.",
      "Regularization keeps weights smaller, which often makes boundaries less brittle.",
    ],
    complexity: {
      time: "O(t n d) for batch gradient descent.",
      prediction: "O(d) per example.",
      space: "O(n d) for the data plus O(d) for weights.",
      plainEnglish:
        "It is usually fast because each training step is a matrix/vector pass over the dataset.",
      terms: [
        { label: "samples n", weight: 70 },
        { label: "features d", weight: 50 },
        { label: "epochs t", weight: 62 },
      ],
    },
    realWorld: [
      "Spam detection, churn prediction, lead scoring, and medical risk screening.",
      "Ranking candidates by probability when you need a simple, explainable model.",
      "A strong baseline for tabular binary classification.",
    ],
    keyDetails: [
      "It draws a linear boundary unless you add engineered nonlinear features.",
      "Predicted probabilities need calibration checks before high-stakes use.",
      "Class imbalance can make accuracy misleading; inspect precision, recall, and thresholds.",
    ],
  },
  "Decision Trees & Random Forests": {
    graphTitle: "How to read the split graph",
    graphNotes: [
      "Each split asks one feature question and tries to make child nodes more pure.",
      "Deeper trees memorize more details; shallow trees generalize more smoothly.",
      "Random forests average many trees to reduce variance.",
    ],
    complexity: {
      time: "About O(n d log n) for a balanced tree; forests multiply by number of trees.",
      prediction: "O(depth) per tree.",
      space: "O(number of nodes).",
      plainEnglish:
        "Trees spend most of their time searching for useful thresholds across features.",
      terms: [
        { label: "samples n", weight: 76 },
        { label: "features d", weight: 60 },
        { label: "depth", weight: 46 },
        { label: "trees", weight: 72 },
      ],
    },
    realWorld: [
      "Credit risk rules, fraud triage, medical decision support, and feature importance reports.",
      "Random forests are common for messy tabular data with mixed feature types.",
      "Tree splits are useful when stakeholders need rule-like explanations.",
    ],
    keyDetails: [
      "Single trees overfit easily; pruning, depth limits, or ensembles help.",
      "Trees do not require feature scaling.",
      "Feature importance can be biased toward high-cardinality features.",
    ],
  },
  "Support Vector Machines": {
    graphTitle: "How to read the margin graph",
    graphNotes: [
      "The separating line tries to maximize margin between the classes.",
      "Support vectors are the boundary-defining points; moving them changes the model most.",
      "The penalty C controls how much the model tolerates mistakes versus margin width.",
    ],
    complexity: {
      time: "Often between O(n^2 d) and O(n^3) for kernel SVM training.",
      prediction: "O(s d), where s is the number of support vectors.",
      space: "O(n^2) can appear with kernel matrices.",
      plainEnglish:
        "SVMs can be excellent on medium-sized data, but kernel versions become expensive as rows grow.",
      terms: [
        { label: "samples n", weight: 92 },
        { label: "features d", weight: 42 },
        { label: "support vectors s", weight: 68 },
      ],
    },
    realWorld: [
      "Text classification, bioinformatics, small/medium tabular classification, and image descriptors.",
      "Good when the margin idea matches the problem and data size is manageable.",
      "Kernel SVMs can model nonlinear boundaries without building deep networks.",
    ],
    keyDetails: [
      "Feature scaling matters a lot.",
      "Kernels improve flexibility but can make training much slower.",
      "C is a tradeoff between cleaner margins and tolerating misclassified points.",
    ],
  },
  "KNN & Distance Metrics": {
    graphTitle: "How to read the neighbor graph",
    graphNotes: [
      "The query point is classified by the labels of its nearest neighbors.",
      "Changing k smooths the decision: small k is sensitive, large k is more stable but blurrier.",
      "Euclidean and Manhattan distance create different neighborhoods, especially in high dimensions.",
    ],
    complexity: {
      time: "Training is O(1) if you just store the data.",
      prediction: "O(n d) for a simple scan, or faster with an index on low-dimensional data.",
      space: "O(n d), because the dataset is the model.",
      plainEnglish:
        "KNN is lazy: it is cheap to train but can be slow at prediction time because it compares against stored examples.",
      terms: [
        { label: "stored rows n", weight: 90 },
        { label: "features d", weight: 60 },
        { label: "neighbors k", weight: 35 },
      ],
    },
    realWorld: [
      "Recommendation prototypes, similarity search, duplicate detection, and local anomaly checks.",
      "Useful when examples that are physically close in feature space should behave similarly.",
      "Often replaced by approximate nearest-neighbor systems at large scale.",
    ],
    keyDetails: [
      "Scale features before using distances.",
      "High-dimensional distance can become less meaningful; this is the curse of dimensionality.",
      "Prediction cost grows with stored data unless you use indexing or approximation.",
    ],
  },
  Clustering: deepDiveById["k-means"],
  "Feature Engineering & Selection": {
    graphTitle: "How to read the feature score graph",
    graphNotes: [
      "Taller bars mean a feature carries stronger signal under the chosen scoring method.",
      "Correlated features can split credit between themselves.",
      "A useful feature should improve validation performance, not just look strong in isolation.",
    ],
    complexity: {
      time: "Common filters are O(n d); wrapper methods can be O(trials x model_cost).",
      prediction: "No prediction cost by itself, but fewer features make later models faster.",
      space: "O(n d) for the data plus feature metadata.",
      plainEnglish:
        "Simple feature scoring is cheap. Searching combinations of features can become expensive quickly.",
      terms: [
        { label: "samples n", weight: 58 },
        { label: "features d", weight: 82 },
        { label: "search trials", weight: 70 },
      ],
    },
    realWorld: [
      "Reducing noisy columns in marketing, finance, healthcare, and sensor datasets.",
      "Making models faster and easier to explain.",
      "Finding which raw signals deserve better collection or cleaning.",
    ],
    keyDetails: [
      "Feature leakage can make a feature look powerful during training and fail in production.",
      "Scaling, encoding, and missing-value handling often matter as much as the model.",
      "Removing useless features can improve speed, stability, and generalization.",
    ],
  },
  "Model Evaluation: Metrics, Cross-Validation": {
    graphTitle: "How to read metric tradeoffs",
    graphNotes: [
      "Accuracy can hide bad behavior when one class is rare.",
      "Precision asks, 'when the model predicts positive, how often is it right?'",
      "Recall asks, 'of all real positives, how many did the model catch?'",
    ],
    complexity: {
      time: "O(f x model_cost), where f is the number of folds.",
      prediction: "Evaluation adds O(n) metric counting after predictions.",
      space: "Usually O(n) for predictions and labels.",
      plainEnglish:
        "Cross-validation repeats training several times, so it is more reliable but more expensive.",
      terms: [
        { label: "folds f", weight: 72 },
        { label: "samples n", weight: 54 },
        { label: "model cost", weight: 88 },
      ],
    },
    realWorld: [
      "Choosing fraud thresholds, medical screening cutoffs, and recommendation ranking metrics.",
      "Comparing models before deployment.",
      "Monitoring whether live performance drifts after launch.",
    ],
    keyDetails: [
      "Split data by time or group when random splits would leak information.",
      "Pick metrics based on the cost of false positives and false negatives.",
      "Always keep a final untouched test set for a last honest check.",
    ],
  },
  "Bias, Variance & the Learning Curve": {
    graphTitle: "How to read the learning curve",
    graphNotes: [
      "High training and validation error means high bias: the model is too simple or underfit.",
      "Low training error but high validation error means high variance: the model memorized too much.",
      "More data helps variance more than bias.",
    ],
    complexity: {
      time: "Depends on the model, repeated across dataset sizes or complexity settings.",
      prediction: "Depends on the chosen final model.",
      space: "Depends on cached training curves and model state.",
      plainEnglish:
        "Learning curves are diagnostic experiments; the cost is training many models to see the pattern.",
      terms: [
        { label: "model fits", weight: 78 },
        { label: "samples n", weight: 62 },
        { label: "complexity", weight: 70 },
      ],
    },
    realWorld: [
      "Diagnosing whether to collect more data, simplify a model, or engineer better features.",
      "Explaining why a model looks good in training but fails in validation.",
      "Budgeting experiments before scaling training runs.",
    ],
    keyDetails: [
      "Bias means systematic underfitting; variance means sensitivity to training data.",
      "Regularization usually reduces variance but can increase bias.",
      "Better features can improve both sides of the tradeoff.",
    ],
  },
  "Ensemble Methods: Boosting, Bagging, Stacking": {
    graphTitle: "How to read ensemble comparisons",
    graphNotes: [
      "Bagging reduces variance by averaging many unstable learners.",
      "Boosting reduces bias by training learners to focus on previous mistakes.",
      "Stacking trains a meta-model that learns how to combine other models.",
    ],
    complexity: {
      time: "O(m x model_cost), where m is the number of estimators.",
      prediction: "O(m x prediction_cost).",
      space: "O(m x model_size).",
      plainEnglish:
        "Ensembles are powerful because they repeat models, but that repetition costs training time, memory, and serving latency.",
      terms: [
        { label: "estimators m", weight: 86 },
        { label: "base model cost", weight: 78 },
        { label: "data n", weight: 56 },
      ],
    },
    realWorld: [
      "Gradient boosted trees for tabular business data.",
      "Random forests for robust baselines.",
      "Model blending in forecasting, ranking, and competition-style pipelines.",
    ],
    keyDetails: [
      "Diversity between learners is what makes averaging useful.",
      "Boosting is sensitive to noisy labels if pushed too far.",
      "Serving many models can be slower than serving one model.",
    ],
  },
  "Hyperparameter Tuning": {
    graphTitle: "How to read tuning traces",
    graphNotes: [
      "Each point is a trial: a parameter choice and its validation score.",
      "Random search often beats grid search when only a few parameters matter.",
      "A good search narrows toward better regions without trusting one lucky split too much.",
    ],
    complexity: {
      time: "O(trials x validation_cost).",
      prediction: "No extra prediction cost after selecting the final model.",
      space: "O(trials) for logs plus model artifacts if saved.",
      plainEnglish:
        "Tuning is expensive because every candidate usually trains and validates another model.",
      terms: [
        { label: "trials", weight: 88 },
        { label: "folds", weight: 68 },
        { label: "model cost", weight: 80 },
      ],
    },
    realWorld: [
      "Choosing learning rates, tree depth, regularization, and batch sizes.",
      "Automating model selection in ML platforms.",
      "Finding a good speed/accuracy tradeoff before deployment.",
    ],
    keyDetails: [
      "Tune on validation data, not the test set.",
      "Search spaces should be informed by model behavior, not random guesses alone.",
      "More tuning can overfit validation data if you keep trying indefinitely.",
    ],
  },
  "Naive Bayes": {
    graphTitle: "How to read probability bars",
    graphNotes: [
      "The model multiplies conditional evidence from features and combines it with class priors.",
      "Smoothing prevents zero-count features from wiping out a class probability.",
      "Large prior changes move the baseline before features are considered.",
    ],
    complexity: {
      time: "O(n d) to count feature statistics.",
      prediction: "O(d c), where c is number of classes.",
      space: "O(d c) for feature/class statistics.",
      plainEnglish:
        "Naive Bayes is fast because training is mostly counting and prediction is mostly adding log probabilities.",
      terms: [
        { label: "samples n", weight: 48 },
        { label: "features d", weight: 64 },
        { label: "classes c", weight: 46 },
      ],
    },
    realWorld: [
      "Spam filtering, lightweight text classification, and quick probabilistic baselines.",
      "Situations where speed and interpretability matter more than maximum accuracy.",
      "Streaming systems where counts can be updated incrementally.",
    ],
    keyDetails: [
      "The 'naive' assumption treats features as conditionally independent.",
      "It can work surprisingly well even when that assumption is not perfectly true.",
      "Use log probabilities to avoid numeric underflow.",
    ],
  },
  "Anomaly Detection": {
    graphTitle: "How to read anomaly scores",
    graphNotes: [
      "Points far from dense normal regions receive higher anomaly scores.",
      "The contamination setting controls how many points you expect to flag.",
      "Outlier labels should be reviewed because rare does not always mean bad.",
    ],
    complexity: {
      time: "Depends on method; distance scans are often O(n^2 d), isolation trees are closer to O(t n log n).",
      prediction: "Usually O(model_size) or O(n d) for direct distance methods.",
      space: "O(n d) or O(model_size).",
      plainEnglish:
        "Anomaly detection ranges from cheap tree methods to expensive pairwise-distance methods.",
      terms: [
        { label: "samples n", weight: 80 },
        { label: "features d", weight: 54 },
        { label: "trees/models", weight: 58 },
      ],
    },
    realWorld: [
      "Fraud spikes, broken sensors, network intrusion, and unusual manufacturing defects.",
      "Monitoring systems where labels are rare or delayed.",
      "Flagging candidates for human review rather than automatic punishment.",
    ],
    keyDetails: [
      "Anomalies are context-dependent.",
      "Feature scaling and seasonality can completely change anomaly scores.",
      "Evaluate with domain review, not only generic metrics.",
    ],
  },
  "Handling Imbalanced Data": {
    graphTitle: "How to read threshold movement",
    graphNotes: [
      "Lowering the threshold catches more positives but usually creates more false alarms.",
      "Class weights make mistakes on rare classes count more during training.",
      "Precision-recall curves are often more useful than accuracy for rare events.",
    ],
    complexity: {
      time: "Usually the base model cost plus resampling or weighting overhead.",
      prediction: "Same as the base classifier.",
      space: "Can increase if oversampling creates extra training rows.",
      plainEnglish:
        "Imbalance handling changes the training data or loss, not the basic prediction mechanics.",
      terms: [
        { label: "minority rows", weight: 70 },
        { label: "resampling", weight: 52 },
        { label: "model cost", weight: 78 },
      ],
    },
    realWorld: [
      "Fraud, medical diagnosis, machine failure, security alerts, and rare content moderation events.",
      "Cases where missing a positive is much more expensive than a false alarm.",
      "Operational systems with review queues and alert budgets.",
    ],
    keyDetails: [
      "Accuracy can be almost meaningless when one class dominates.",
      "Choose thresholds with real costs in mind.",
      "Resampling must happen inside cross-validation folds to avoid leakage.",
    ],
  },
  "Time Series Fundamentals": {
    graphTitle: "How to read time-series graphs",
    graphNotes: [
      "Trend is long-term direction; seasonality is repeating pattern; noise is random movement.",
      "Lag features let past values become inputs for future predictions.",
      "Forecast uncertainty usually grows as the horizon gets farther away.",
    ],
    complexity: {
      time: "Simple rolling features are O(n w), where w is window size.",
      prediction: "Often O(w) for simple window methods.",
      space: "O(n) for the series plus O(w) rolling state.",
      plainEnglish:
        "Window methods get slower as you look farther back, but they are still usually lightweight.",
      terms: [
        { label: "time points n", weight: 62 },
        { label: "window w", weight: 58 },
        { label: "horizon h", weight: 44 },
      ],
    },
    realWorld: [
      "Demand forecasting, stock indicators, server metrics, weather, and energy load.",
      "Detecting seasonal patterns before picking a forecasting model.",
      "Building lagged features for larger ML models.",
    ],
    keyDetails: [
      "Never train on future data when validating a time-series model.",
      "Use time-based splits rather than random splits.",
      "External events can break patterns that looked stable historically.",
    ],
  },
  "Stochastic Processes": {
    graphTitle: "How to read stochastic paths",
    graphNotes: [
      "Drift moves the average path; volatility controls how widely paths fan out.",
      "A Markov transition matrix stores one-step probabilities between states.",
      "Random samples differ run to run, but distributions should show stable patterns.",
    ],
    complexity: {
      time: "O(paths x steps) for random walks, O(steps x states) for Markov simulation.",
      prediction: "O(states^2) for full distribution updates with dense matrices.",
      space: "O(paths x steps) if every trajectory is stored.",
      plainEnglish:
        "Simulation is usually simple loops, but storing many paths can become the memory cost.",
      terms: [
        { label: "paths", weight: 72 },
        { label: "steps t", weight: 78 },
        { label: "states s", weight: 56 },
      ],
    },
    realWorld: [
      "Financial random walks, queueing systems, weather scenarios, and language token transitions.",
      "Monte Carlo risk estimates.",
      "Modeling uncertainty when one deterministic line would be misleading.",
    ],
    keyDetails: [
      "The random seed affects individual paths, not the underlying probability rule.",
      "Transition rows should sum to 1.",
      "More simulations reduce sampling noise but cost more time.",
    ],
  },
  "Dynamic Programming": {
    graphTitle: "How to read the value grid",
    graphNotes: [
      "Numbers in cells are estimates of long-term reward from that state.",
      "Reward bleeds outward from terminal states as Bellman sweeps repeat.",
      "Once values stabilize, policy arrows point toward the best expected next move.",
    ],
    complexity: {
      time: "O(t |S| |A|) for value iteration on a small deterministic grid.",
      prediction: "O(1) to follow a stored policy per step.",
      space: "O(|S|) for the value table plus O(|S|) for the policy.",
      plainEnglish:
        "Every sweep visits each state and tries each action, so grid size and action count dominate.",
      terms: [
        { label: "states |S|", weight: 82 },
        { label: "actions |A|", weight: 48 },
        { label: "sweeps t", weight: 70 },
      ],
    },
    realWorld: [
      "Robot grid navigation, inventory control, small games, and planning under known transition rules.",
      "The foundation behind many reinforcement-learning ideas.",
      "Teaching how rewards become policies before using neural RL.",
    ],
    keyDetails: [
      "Dynamic programming assumes you know the transition and reward model.",
      "Discount gamma controls how much future rewards matter.",
      "Large state spaces need approximation because exact tables become too big.",
    ],
  },
  "Singular Value Decomposition": {
    graphTitle: "How to read the decomposition panels",
    graphNotes: [
      "The large matrix A splits into smaller factors whose product rebuilds the original signal.",
      "Low rank keeps the broad background first; the residual panel shows details, edges, or motion left behind.",
      "SVD factors can contain positive and negative rotations, while NMF factors stay nonnegative and read more like additive parts.",
    ],
    complexity: {
      time: "Roughly O(min(m n^2, m^2 n)) for a full SVD.",
      prediction: "O(k(m+n)) storage and roughly O(m n k) arithmetic to reconstruct a rank-k matrix.",
      space: "O(m n) for the matrix plus factors.",
      plainEnglish:
        "Full SVD is expensive on large matrices, but truncated factors let you keep the useful structure without storing every pixel or table entry.",
      terms: [
        { label: "rows m", weight: 72 },
        { label: "columns n", weight: 72 },
        { label: "rank k", weight: 46 },
      ],
    },
    realWorld: [
      "Image compression, background subtraction in video, recommender systems, search indexing, and PCA-like dimensionality reduction.",
      "Noise reduction by keeping stable low-rank structure and discarding high-frequency residuals.",
      "Topic or parts discovery when nonnegative matrices are decomposed with NMF.",
    ],
    keyDetails: [
      "SVD is deterministic for a fixed matrix, unlike many iterative ML methods.",
      "Top singular values often explain most of the useful signal.",
      "NMF is iterative and approximate, but its nonnegative factors are often easier to interpret.",
      "Randomized/truncated algorithms are common for huge data because full decompositions are too expensive.",
    ],
  },
  "Convex Optimization": {
    graphTitle: "How to read the constraint surface",
    graphNotes: [
      "A convex bowl has one global minimum, so downhill movement is reliable.",
      "Constraints clip or project proposed updates back into the feasible region.",
      "When the function becomes non-convex, local traps can appear.",
    ],
    complexity: {
      time: "Gradient descent is O(t d) per simple objective; Newton often needs O(t d^3) matrix solves.",
      prediction: "Not a prediction model by itself.",
      space: "O(d) for gradient methods, O(d^2) for Hessian-based methods.",
      plainEnglish:
        "First-order methods are cheaper per step; second-order methods use curvature but solve heavier linear algebra.",
      terms: [
        { label: "variables d", weight: 82 },
        { label: "iterations t", weight: 60 },
        { label: "constraints", weight: 52 },
      ],
    },
    realWorld: [
      "Portfolio allocation, logistics, resource planning, regularized regression, and constrained ML training.",
      "Any setting where the answer must obey limits.",
      "The mathematical core of many estimators and optimizers.",
    ],
    keyDetails: [
      "Convexity gives strong guarantees: local optimum equals global optimum.",
      "Constraints change where the optimum lives.",
      "Non-convex objectives can still work, but guarantees become weaker.",
    ],
  },
  "Convolutions from Scratch": {
    graphTitle: "How to read the filter sandbox",
    graphNotes: [
      "The moving 3x3 box shows which local pixels are multiplied by the kernel.",
      "Stride skips positions, shrinking the feature map.",
      "Padding adds artificial border values so edges are not discarded immediately.",
    ],
    complexity: {
      time: "O(H W K^2 C F) for a basic convolution layer.",
      prediction: "Same order as the forward convolution.",
      space: "O(H W) for feature maps, multiplied by channels and filters.",
      plainEnglish:
        "Every output pixel does a small dot product, repeated across positions, channels, and filters.",
      terms: [
        { label: "image pixels HxW", weight: 86 },
        { label: "kernel K", weight: 42 },
        { label: "filters F", weight: 64 },
        { label: "channels C", weight: 52 },
      ],
    },
    realWorld: [
      "Edge detection, blur/sharpen filters, CNN image classifiers, OCR, and medical imaging.",
      "Feature extraction for visual inspection systems.",
      "GPU-accelerated computer vision in browsers and mobile devices.",
    ],
    keyDetails: [
      "Kernel values define what visual pattern is amplified.",
      "Convolution is translation-friendly: the same filter scans every location.",
      "Padding and stride change output size and boundary behavior.",
    ],
  },
  "Neural Networks": {
    graphTitle: "How to read training curves",
    graphNotes: [
      "Loss should generally decrease as the model learns.",
      "Accuracy can rise while loss still reveals confidence problems.",
      "Flat curves can mean low learning rate, insufficient capacity, or poor features.",
    ],
    complexity: {
      time: "O(t n P), roughly proportional to epochs, samples, and parameters.",
      prediction: "O(P) for a dense forward pass.",
      space: "O(P) for weights plus activation caches during training.",
      plainEnglish:
        "Neural networks scale with parameter count and the number of times data passes through them.",
      terms: [
        { label: "parameters P", weight: 88 },
        { label: "samples n", weight: 68 },
        { label: "epochs t", weight: 72 },
      ],
    },
    realWorld: [
      "Computer vision, language models, speech recognition, recommendation, and nonlinear tabular models.",
      "Learning representations where hand-written features are not enough.",
      "Approximating complex functions with many trainable weights.",
    ],
    keyDetails: [
      "Learning rate, initialization, and normalization strongly affect training.",
      "More parameters increase capacity but also overfitting and compute risk.",
      "Validation curves matter more than training curves alone.",
    ],
  },
  "Multi-Layer Networks": {
    graphTitle: "How to read the decision boundary",
    graphNotes: [
      "More hidden layers let the boundary bend around nonlinear patterns.",
      "Weight colors/thickness show which connections are carrying strong signal.",
      "If the boundary is too wiggly, the network may be overfitting.",
    ],
    complexity: {
      time: "O(t n P) for dense training.",
      prediction: "O(P) for one forward pass.",
      space: "O(P + activations) during training.",
      plainEnglish:
        "Dense neural networks spend most of their time multiplying activations by weight matrices.",
      terms: [
        { label: "parameters P", weight: 90 },
        { label: "layers L", weight: 54 },
        { label: "epochs t", weight: 70 },
      ],
    },
    realWorld: [
      "XOR-like nonlinear classification, embeddings, tabular deep learning, and small browser demos.",
      "Learning hidden transformations before a final prediction.",
      "Demonstrating why one straight line is not enough for nonlinear data.",
    ],
    keyDetails: [
      "Nonlinear activations are what make layers more powerful than one linear model.",
      "Too much depth for small data can overfit.",
      "Matrix shape compatibility is the first thing to debug.",
    ],
  },
  "Activation Functions — ReLU, Sigmoid, GELU": {
    graphTitle: "How to read activation curves",
    graphNotes: [
      "The value curve shows the neuron output for each input x.",
      "The derivative curve shows how much gradient can flow backward.",
      "Flat derivatives mean learning slows or stops in that region.",
    ],
    complexity: {
      time: "O(n) elementwise over activations.",
      prediction: "O(n) for n activation values.",
      space: "O(n) when training caches activations or masks.",
      plainEnglish:
        "Activation functions are cheap individually, but huge networks apply them millions of times.",
      terms: [
        { label: "activations n", weight: 74 },
        { label: "cached masks", weight: 42 },
        { label: "layers", weight: 54 },
      ],
    },
    realWorld: [
      "ReLU for efficient deep networks, sigmoid for probabilities/gates, GELU in transformer-style models.",
      "Diagnosing vanishing gradients and dying ReLU behavior.",
      "Choosing nonlinearities for stability and accuracy.",
    ],
    keyDetails: [
      "Sigmoid saturates at extremes, which can shrink gradients.",
      "ReLU is simple but can die when outputs stay negative.",
      "GELU is smoother and often works well in modern deep learning.",
    ],
  },
  "Loss Functions — MSE, Cross-Entropy, Contrastive": {
    graphTitle: "How to read loss curves",
    graphNotes: [
      "A falling curve usually means updates are improving the objective.",
      "A rising or exploding curve often means the learning rate is too high.",
      "Different losses encode different definitions of what a mistake means.",
    ],
    complexity: {
      time: "O(n c) for many batch losses, where c may be outputs/classes.",
      prediction: "Loss is training/evaluation only; prediction uses the model.",
      space: "O(n c) for predictions and targets in a batch.",
      plainEnglish:
        "Loss computation is usually cheaper than the model itself, but it drives every gradient update.",
      terms: [
        { label: "batch n", weight: 60 },
        { label: "outputs c", weight: 48 },
        { label: "pairs", weight: 66 },
      ],
    },
    realWorld: [
      "MSE for regression, cross-entropy for classification, contrastive loss for embeddings.",
      "Training classifiers, recommenders, image search, and face/product similarity systems.",
      "Monitoring whether training is healthy.",
    ],
    keyDetails: [
      "Pick the loss that matches the desired behavior.",
      "Loss can improve while a business metric stays flat.",
      "Numerical stability tricks like epsilon and log-sum-exp matter.",
    ],
  },
  "SGD, Momentum, Adam, AdamW": {
    graphTitle: "How to read optimizer trajectories",
    graphNotes: [
      "SGD follows the gradient directly and may wobble in ravines.",
      "Momentum carries velocity from past gradients.",
      "Adam adapts per-parameter step sizes using gradient history.",
    ],
    complexity: {
      time: "O(P) per update for most first-order optimizers.",
      prediction: "Optimizers do not affect prediction cost after training.",
      space: "SGD O(P); Momentum O(P); Adam/AdamW O(2P) extra moment state.",
      plainEnglish:
        "Adaptive optimizers are not much slower per step, but they store extra state for every parameter.",
      terms: [
        { label: "parameters P", weight: 86 },
        { label: "steps t", weight: 70 },
        { label: "moment state", weight: 54 },
      ],
    },
    realWorld: [
      "Training neural networks, embeddings, language models, and differentiable simulations.",
      "AdamW is common in modern transformer training.",
      "Momentum helps in noisy or curved optimization landscapes.",
    ],
    keyDetails: [
      "Learning rate is still the most important knob.",
      "AdamW decouples weight decay from gradient updates.",
      "Fast loss drops can still overfit if validation metrics do not improve.",
    ],
  },
  "Build a Mini-Framework": {
    graphTitle: "How to read the framework flow",
    graphNotes: [
      "Forward hooks show data moving layer by layer to predictions.",
      "Backward hooks show gradients flowing in reverse.",
      "Layer interfaces make the same loop work for Linear, activation, dropout, and loss layers.",
    ],
    complexity: {
      time: "O(sum(layer forward/backward costs)) per batch.",
      prediction: "O(sum(layer forward costs)).",
      space: "O(parameters + cached activations).",
      plainEnglish:
        "Frameworks are orchestration: the cost mostly comes from the layers they run.",
      terms: [
        { label: "layers L", weight: 56 },
        { label: "parameters P", weight: 84 },
        { label: "batch size b", weight: 58 },
      ],
    },
    realWorld: [
      "Understanding PyTorch, TensorFlow, Keras, and custom training loops.",
      "Building educational neural-net libraries.",
      "Debugging model code by separating forward, backward, loss, and optimizer responsibilities.",
    ],
    keyDetails: [
      "Caching inputs during forward is what makes backward possible.",
      "Shape mismatches are the most common implementation bug.",
      "Clean abstractions make visual hooks and debugging easier.",
    ],
  },
  "Backpropagation from scratch": {
    graphTitle: "How to read the pulse animation",
    graphNotes: [
      "Forward pulses compute activations and predictions.",
      "Backward pulses carry error gradients through the chain rule.",
      "Brighter weights have larger gradients and will update more strongly.",
    ],
    complexity: {
      time: "O(P) for one forward/backward pass in a dense network.",
      prediction: "O(P) for forward only.",
      space: "O(P + activations) during training.",
      plainEnglish:
        "Backprop is efficient because it reuses cached intermediate values instead of recomputing every derivative separately.",
      terms: [
        { label: "parameters P", weight: 88 },
        { label: "layers L", weight: 60 },
        { label: "batch size b", weight: 58 },
      ],
    },
    realWorld: [
      "The core algorithm behind neural network training.",
      "Manual gradient checks for custom layers.",
      "Understanding why autodiff libraries need computation graphs.",
    ],
    keyDetails: [
      "The chain rule connects each local derivative to the final loss.",
      "Gradients tell direction and magnitude of parameter updates.",
      "Exploding or vanishing gradients make deep training unstable.",
    ],
  },
};
