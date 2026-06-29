import { AlertTriangle, ArrowRight, BriefcaseBusiness, Database, Target } from "lucide-react";
import { getDeepDiveContent } from "./AlgorithmDeepDive";
import type { AlgorithmDefinition } from "../types/algorithm";

type Props = {
  algorithm: AlgorithmDefinition;
};

type ApplicationCard = {
  title: string;
  scenario: string;
  data: string;
  action: string;
  caveat: string;
};

export function RealApplicationsPanel({ algorithm }: Props) {
  const content = getDeepDiveContent(algorithm);
  const applications = makeApplicationCards(algorithm, content.realWorld, content.keyDetails);

  return (
    <section className="applications-panel" aria-label={`${algorithm.name} real applications`}>
      <div className="applications-header">
        <span className="eyebrow">Real Applications</span>
        <h3>Where this shows up outside the playground</h3>
        <p>
          Each card connects the toy visualization to a real workflow: the data people collect,
          the decision the model supports, and the caveat that keeps the demo honest.
        </p>
      </div>

      <div className="application-card-grid">
        {applications.map((application) => (
          <article className="application-card" key={application.title}>
            <div className="application-card-title">
              <span aria-hidden="true">
                <BriefcaseBusiness size={16} />
              </span>
              <h4>{application.title}</h4>
            </div>

            <p>{application.scenario}</p>

            <div className="application-flow" aria-label={`${application.title} workflow`}>
              <FlowPill icon={Database} label="Data" value={application.data} />
              <ArrowRight size={16} aria-hidden="true" />
              <FlowPill icon={Target} label="Decision" value={application.action} />
            </div>

            <div className="application-caveat">
              <AlertTriangle size={15} aria-hidden="true" />
              <span>{application.caveat}</span>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function FlowPill({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Database;
  label: string;
  value: string;
}) {
  return (
    <div className="application-flow-pill">
      <Icon size={15} aria-hidden="true" />
      <span>
        <strong>{label}</strong>
        <small>{value}</small>
      </span>
    </div>
  );
}

function makeApplicationCards(
  algorithm: AlgorithmDefinition,
  realWorld: string[],
  keyDetails: string[],
) {
  const examples = realWorld.length > 0 ? realWorld : [algorithm.summary];
  const caveats = keyDetails.length > 0 ? keyDetails : ["Validate on new data before trusting the visual pattern."];
  const templates = applicationTemplates(algorithm);

  return templates.map<ApplicationCard>((template, index) => ({
    title: template.title,
    scenario: examples[index % examples.length],
    data: template.data,
    action: template.action,
    caveat: caveats[index % caveats.length],
  }));
}

function applicationTemplates(algorithm: AlgorithmDefinition) {
  if (algorithm.id === "hidden-markov-models") {
    return [
      {
        title: "Part-of-speech tagging",
        data: "Tokenized sentences, transition counts, and word/tag emission counts",
        action: "Decode the most likely grammatical role for every word",
      },
      {
        title: "Regime detection",
        data: "Visible market, weather, or operations events over time",
        action: "Infer hidden state sequences such as bull, crash, recovery, or fault states",
      },
      {
        title: "Streaming diagnostics",
        data: "Sensor readings, alarms, and state transition assumptions",
        action: "Track likely hidden machine states as observations arrive",
      },
    ];
  }

  if (algorithm.id === "gaussian-discriminant-analysis") {
    return [
      {
        title: "Sensor-state routing",
        data: "Two or more continuous readings from devices, vehicles, or industrial equipment",
        action: "Route each new point to the class whose Gaussian profile most likely produced it",
      },
      {
        title: "Clinical measurement screening",
        data: "Lab measurements, biomarkers, or imaging-derived numeric features",
        action: "Estimate class likelihoods while tracking uncertainty from small samples",
      },
      {
        title: "Client-side probabilistic triage",
        data: "Small feature vectors arriving in a browser, edge device, or dashboard",
        action: "Score likely classes without waiting for backend inference",
      },
    ];
  }

  if (algorithm.id === "bayes-rule-visualizer") {
    return [
      {
        title: "Medical screening",
        data: "Condition prevalence, test sensitivity, and false-positive rate",
        action: "Estimate the chance a positive result means the condition is truly present",
      },
      {
        title: "Fraud alerts",
        data: "Base fraud rate, detection rate, and normal-transaction false alarms",
        action: "Rank alerts by posterior risk instead of raw detector confidence",
      },
      {
        title: "Content moderation",
        data: "Policy violation prevalence, classifier recall, and false-positive rate",
        action: "Decide which positive flags need automated action or human review",
      },
    ];
  }

  if (algorithm.id === "bayesian-regression") {
    return [
      {
        title: "Forecasting with confidence intervals",
        data: "Historical demand, prices, weather, capacity, or risk features",
        action: "Return a prediction range so planners can see best, likely, and risky outcomes",
      },
      {
        title: "Experiment planning",
        data: "Small batches of lab, product, or sensor measurements",
        action: "Collect the next sample where the model is still most uncertain",
      },
      {
        title: "Risk-aware decisions",
        data: "Sparse customer, medical, manufacturing, or financial observations",
        action: "Avoid overconfident actions when the posterior band is still wide",
      },
    ];
  }

  if (algorithm.category.includes("Linear Transformations") || algorithm.id.includes("svd") || algorithm.id.includes("matrix")) {
    return [
      {
        title: "Compression and structure discovery",
        data: "Images, matrices, embeddings, or sensor tables",
        action: "Keep the dominant directions or factors that preserve most signal",
      },
      {
        title: "Geometry diagnostics",
        data: "2D/3D transforms, covariance matrices, or model weights",
        action: "Detect stretching, collapse, preserved axes, or unstable directions",
      },
      {
        title: "Simulation and graphics",
        data: "Coordinate grids, transforms, camera geometry",
        action: "Predict how shapes scale, rotate, flip, or lose information",
      },
    ];
  }

  if (algorithm.category.includes("Neural") || algorithm.category.includes("Backpropagation") || algorithm.category.includes("Activation") || algorithm.category.includes("Loss") || algorithm.category.includes("SGD")) {
    return [
      {
        title: "Model training workflow",
        data: "Labeled batches, predictions, gradients, and losses",
        action: "Tune the model until validation loss and task metrics improve",
      },
      {
        title: "Debugging learning behavior",
        data: "Activation values, gradients, loss curves, and parameter updates",
        action: "Spot dead neurons, exploding loss, saturation, or optimizer instability",
      },
      {
        title: "Interactive prototyping",
        data: "Small browser datasets or generated nonlinear samples",
        action: "Test architecture choices before moving to larger training code",
      },
    ];
  }

  if (algorithm.category.includes("Optimization") || algorithm.category.includes("Regularization") || algorithm.category.includes("Hyperparameter")) {
    return [
      {
        title: "Search and tuning",
        data: "Validation scores, parameter settings, and training curves",
        action: "Choose settings that improve generalization without wasting compute",
      },
      {
        title: "Constrained decision-making",
        data: "Objective values, gradients, penalties, and limits",
        action: "Find the best feasible point while respecting real-world constraints",
      },
      {
        title: "Stability control",
        data: "Weights, losses, regularization strength, and noise",
        action: "Reduce overfitting, wild updates, or brittle parameter choices",
      },
    ];
  }

  if (algorithm.category.includes("Tokenizer") || algorithm.category.includes("Feature") || algorithm.category.includes("Data")) {
    return [
      {
        title: "Input preparation",
        data: "Raw text, columns, categories, missing values, or engineered features",
        action: "Convert messy inputs into model-ready signals",
      },
      {
        title: "Cost and quality control",
        data: "Token counts, feature scores, validation lift, or encoding choices",
        action: "Keep only useful representations and reduce avoidable overhead",
      },
      {
        title: "Explainability",
        data: "Feature rows, selected columns, token pieces, or score tables",
        action: "Show which input pieces actually influence downstream behavior",
      },
    ];
  }

  if (algorithm.category.includes("Clustering") || algorithm.category.includes("Anomaly") || algorithm.category.includes("NMF")) {
    return [
      {
        title: "Discovery without labels",
        data: "Customer behavior, text counts, embeddings, or sensor readings",
        action: "Find groups, topics, outliers, or latent structure",
      },
      {
        title: "Human review queues",
        data: "Distance scores, cluster residuals, or anomaly ranks",
        action: "Prioritize unusual cases for inspection",
      },
      {
        title: "Compression of behavior",
        data: "Many rows of unlabeled examples",
        action: "Summarize patterns into centroids, topics, or representative factors",
      },
    ];
  }

  return [
    {
      title: "Prediction or scoring",
      data: "Historical rows with features and outcomes",
      action: "Generate a prediction, class score, ranking, or diagnostic signal",
    },
    {
      title: "Model comparison",
      data: "Training splits, validation results, and baseline metrics",
      action: "Decide whether the simple model is enough or a richer method is needed",
    },
    {
      title: "Operational monitoring",
      data: "New inputs, prediction logs, errors, and drift indicators",
      action: "Watch whether the learned pattern still matches production data",
    },
  ];
}
