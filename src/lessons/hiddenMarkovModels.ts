import type {
  ConceptFrame,
  DataPoint,
  EngineResult,
  HmmMode,
  NormalizedDataset,
  ParameterState,
} from "../types/algorithm";
import { defineLessonModule } from "./lessonModule";

export const hiddenMarkovModelsModule = defineLessonModule({
  metadata: {
    id: "hidden-markov-models",
    name: "Hidden Markov Models",
    category: "Stochastic Processes",
    summary: "Decodes hidden state paths from visible event sequences with Viterbi dynamic programming.",
    catalog: {
      difficulty: "Advanced",
      tags: ["math-heavy", "probability", "sequence-model"],
      recommendedOrder: 3,
      prerequisites: ["bayes-rule-visualizer", "stochastic-processes", "dynamic-programming"],
    },
    parameters: [
      {
        kind: "select",
        id: "hmmMode",
        label: "HMM playground",
        defaultValue: "market",
        options: [
          { label: "Market regime decoder", value: "market" },
          { label: "Part-of-speech tagger", value: "pos" },
        ],
      },
      {
        kind: "text",
        id: "hmmInput",
        label: "Visible sequence",
        defaultValue: "Up Down Flat Up",
        placeholder: "Market: Up Down Flat Up. POS: The quick model learns markets.",
        rows: 3,
        quickInserts: [
          { label: "Market", value: "Up Down Flat Up" },
          { label: "Stress", value: "Down Down Flat Up Flat" },
          { label: "POS", value: "The quick model learns markets" },
        ],
      },
    ],
  },
  runtime: {
    makeSampleDataset: makeHmmDataset,
    engine: hiddenMarkovEngine,
    controller: {
      primaryActionLabel: "Decode",
      shouldAutoPlay: (result) => result.frames.length > 1,
    },
  },
  content: {
    formulas: [
      {
        title: "Viterbi recurrence",
        expression: "\\delta_t(j)=\\max_i[\\delta_{t-1}(i)+\\log A_{ij}]+\\log B_j(o_t)",
      },
      {
        title: "Backpointer",
        expression: "\\psi_t(j)=\\arg\\max_i[\\delta_{t-1}(i)+\\log A_{ij}]",
      },
      {
        title: "Hidden path",
        expression: "\\hat{z}_{1:T}=\\arg\\max_z P(z_{1:T},x_{1:T})",
      },
    ],
    explanation: [
      "A Hidden Markov Model assumes the visible sequence is produced by an unobserved chain of hidden states.",
      "Viterbi fills a trellis of log-probabilities, stores backpointers, then traces the single most likely hidden path.",
      "The same engine can decode market regimes from Up/Down/Flat observations or grammatical roles from words.",
    ],
    deepDive: {
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
    applications: [
      {
        title: "Part-of-speech tagging",
        scenario: "Early part-of-speech tagging and speech recognition before neural sequence models became dominant.",
        data: "Tokenized sentences, transition counts, and word/tag emission counts",
        action: "Decode the most likely grammatical role for every word",
        caveat: "HMMs struggle when the next state depends on long-range context beyond the previous hidden state.",
      },
      {
        title: "Regime detection",
        scenario: "Market-regime and weather-state decoding from visible observations.",
        data: "Visible market, weather, or operations events over time",
        action: "Infer hidden state sequences such as bull, crash, recovery, or fault states",
        caveat: "Transition probabilities describe hidden-state movement; emission probabilities describe visible evidence from a state.",
      },
      {
        title: "Streaming diagnostics",
        scenario: "Fault detection in sensors where the true machine state is hidden but emissions are observable.",
        data: "Sensor readings, alarms, and state transition assumptions",
        action: "Track likely hidden machine states as observations arrive",
        caveat: "Use log-probabilities so long sequences do not underflow toward zero.",
      },
    ],
    notebook: {
      enabled: true,
      intro:
        "Run the Viterbi recurrence in Python and compare the trellis values with the browser animation.",
      packages: ["numpy"],
    },
  },
  code: {
    python: (params) => {
      const mode = modeParam(params);
      const text = JSON.stringify(
        stringParam(params, "hmmInput", mode === "pos" ? "The quick model learns markets" : "Up Down Flat Up"),
      );

      return `import numpy as np

mode = "${mode}"
observations = ${text}.lower().replace(",", " ").split()
states = ["Bull", "Crash", "Recovery"] if mode == "market" else ["Noun", "Verb", "Adjective", "Preposition"]

log_start = np.log(start_probs)
log_A = np.log(transition_matrix)
log_B = emission_log_matrix(observations, states)

T, K = len(observations), len(states)
delta = np.full((T, K), -np.inf)
backptr = np.zeros((T, K), dtype=int)
delta[0] = log_start + log_B[0]

for t in range(1, T):
    candidates = delta[t - 1][:, None] + log_A
    backptr[t] = np.argmax(candidates, axis=0)
    delta[t] = np.max(candidates, axis=0) + log_B[t]

path = [int(np.argmax(delta[-1]))]
for t in range(T - 1, 0, -1):
    path.append(int(backptr[t, path[-1]]))
path = path[::-1]
decoded = [states[i] for i in path]`;
    },
    javascript: (params) => {
      const mode = modeParam(params);
      const text = JSON.stringify(
        stringParam(params, "hmmInput", mode === "pos" ? "The quick model learns markets" : "Up Down Flat Up"),
      );

      return `const mode = "${mode}";
const observations = tokenize(${text}, mode);
const hmm = buildHmmConfig(mode);
const trellis = [];

for (let t = 0; t < observations.length; t += 1) {
  trellis[t] = [];
  for (const state of hmm.states) {
    const emissionLog = Math.log(hmm.emission(state.key, observations[t]));
    if (t === 0) {
      trellis[t].push({
        state: state.key,
        logProbability: Math.log(hmm.start[state.key]) + emissionLog,
        backPointer: null,
      });
      continue;
    }

    const candidates = trellis[t - 1].map((previous) => ({
      previous: previous.state,
      score: previous.logProbability + Math.log(hmm.transition[previous.state][state.key]),
    }));
    const best = candidates.sort((a, b) => b.score - a.score)[0];
    trellis[t].push({
      state: state.key,
      logProbability: best.score + emissionLog,
      backPointer: best.previous,
    });
  }
}

const path = backtraceBestPath(trellis);
console.log(path.map((key) => hmm.states.find((state) => state.key === key).label));`;
    },
  },
  visualization: {
    renderer: "canvas",
  },
});

function hiddenMarkovEngine(_: NormalizedDataset, params: ParameterState): EngineResult {
  const mode = modeParam(params);
  const fallback = mode === "pos" ? "The quick model learns markets" : "Up Down Flat Up";
  const input = stringParam(params, "hmmInput", fallback).trim() || fallback;
  const observations = parseHmmObservations(mode, input);
  const decoded = runHmmViterbi(mode, observations);
  const stepsPerObservation = 6;
  const frameCount = Math.max(1, observations.length * stepsPerObservation + 1);
  const frames = Array.from({ length: frameCount }, (_, frameIndex) => {
    const terminal = frameIndex >= frameCount - 1;
    const phaseIndex = terminal
      ? observations.length - 1
      : Math.min(observations.length - 1, Math.floor(frameIndex / stepsPerObservation));
    const phaseProgress = terminal ? 1 : (frameIndex % stepsPerObservation) / Math.max(1, stepsPerObservation - 1);
    const state = makeHmmFrameState(decoded, input, phaseIndex, phaseProgress);

    return {
      type: "concept-demo" as const,
      iteration: frameIndex + 1,
      points: observations.map((observation, index) => ({
        x: index,
        y: decoded.bestPath[index] ? decoded.states.findIndex((node) => node.key === decoded.bestPath[index]) : 0,
        label: observation,
      })),
      hiddenMarkov: state,
      summary: `${mode === "pos" ? "POS tags" : "Market path"} · ${state.bestPathLabels.join(" -> ")} · logP ${state.logProbability.toFixed(2)}`,
    };
  });
  const finalState = makeHmmFrameState(decoded, input, observations.length - 1, 1);

  return {
    frames,
    runtime: "JavaScript",
    metrics: [
      { label: "Observations", value: String(observations.length) },
      { label: "Best logP", value: finalState.logProbability.toFixed(2) },
      { label: "Final state", value: finalState.bestPathLabels.at(-1) ?? "n/a" },
      { label: "Mode", value: mode === "pos" ? "POS" : "Market" },
    ],
  };
}

function makeHmmDataset() {
  return makeDataset(
    "Generated HMM sequence sample",
    ["Up", "Down", "Flat", "Up"].map((label, index) => ({
      x: index,
      y: index % 2,
      label,
    })),
  );
}

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

function parseHmmObservations(mode: HmmMode, input: string) {
  if (mode === "market") {
    const tokens = input
      .split(/[\s,>-]+/)
      .map((token) => token.trim().toLowerCase())
      .filter(Boolean)
      .map((token) => {
        if (token.startsWith("u") || token === "+") return "Up";
        if (token.startsWith("d") || token === "-") return "Down";
        return "Flat";
      });
    return tokens.length > 0 ? tokens.slice(0, 12) : ["Up", "Down", "Flat", "Up"];
  }

  const words = input
    .toLowerCase()
    .match(/[a-z']+/g)
    ?.slice(0, 12) ?? [];
  return words.length > 0 ? words : ["the", "quick", "model", "learns", "markets"];
}

function buildHmmConfig(mode: HmmMode) {
  if (mode === "pos") {
    const states = [
      { key: "noun", label: "Noun", shortLabel: "N", color: "#2f6fbe" },
      { key: "verb", label: "Verb", shortLabel: "V", color: "#d34a43" },
      { key: "adjective", label: "Adjective", shortLabel: "Adj", color: "#b7791f" },
      { key: "preposition", label: "Preposition", shortLabel: "Prep", color: "#0f766e" },
    ];
    return {
      mode,
      states,
      start: {
        noun: 0.28,
        verb: 0.12,
        adjective: 0.34,
        preposition: 0.26,
      },
      transition: {
        noun: { noun: 0.16, verb: 0.48, adjective: 0.08, preposition: 0.28 },
        verb: { noun: 0.42, verb: 0.12, adjective: 0.12, preposition: 0.34 },
        adjective: { noun: 0.66, verb: 0.08, adjective: 0.2, preposition: 0.06 },
        preposition: { noun: 0.38, verb: 0.05, adjective: 0.48, preposition: 0.09 },
      },
      emission: (stateKey: string, observation: string) => posEmissionProbability(stateKey, observation),
    };
  }

  const states = [
    { key: "bull", label: "Bull Market", shortLabel: "Bull", color: "#0f766e" },
    { key: "crash", label: "High Volatility Crash", shortLabel: "Crash", color: "#d34a43" },
    { key: "recovery", label: "Recovery", shortLabel: "Rec", color: "#b7791f" },
  ];
  return {
    mode,
    states,
    start: {
      bull: 0.55,
      crash: 0.15,
      recovery: 0.3,
    },
    transition: {
      bull: { bull: 0.72, crash: 0.08, recovery: 0.2 },
      crash: { bull: 0.06, crash: 0.62, recovery: 0.32 },
      recovery: { bull: 0.42, crash: 0.12, recovery: 0.46 },
    },
    emission: (stateKey: string, observation: string) => marketEmissionProbability(stateKey, observation),
  };
}

function marketEmissionProbability(stateKey: string, observation: string) {
  const emissions: Record<string, Record<string, number>> = {
    bull: { Up: 0.62, Flat: 0.28, Down: 0.1 },
    crash: { Up: 0.12, Flat: 0.18, Down: 0.7 },
    recovery: { Up: 0.38, Flat: 0.42, Down: 0.2 },
  };
  return emissions[stateKey]?.[observation] ?? 0.08;
}

const posLexicon = {
  noun: new Set([
    "algorithm",
    "algorithms",
    "cat",
    "data",
    "dog",
    "market",
    "markets",
    "model",
    "models",
    "pipeline",
    "pipelines",
    "robot",
    "stock",
    "stocks",
    "system",
    "trader",
    "traders",
  ]),
  verb: new Set(["are", "decode", "decodes", "fall", "falls", "is", "jump", "jumps", "learn", "learns", "rise", "rises", "route", "routes", "trade", "trades", "watch", "watches"]),
  adjective: new Set(["blue", "bright", "classic", "early", "hidden", "modern", "quick", "red", "robust", "volatile"]),
  preposition: new Set(["by", "for", "from", "in", "into", "of", "on", "over", "through", "to", "under", "with"]),
};

function posEmissionProbability(stateKey: string, observation: string) {
  const lexicon = posLexicon[stateKey as keyof typeof posLexicon];
  if (lexicon?.has(observation)) {
    return stateKey === "preposition" ? 0.76 : 0.66;
  }
  if (stateKey === "verb" && /(ed|ing|s)$/.test(observation)) {
    return 0.34;
  }
  if (stateKey === "noun" && /(tion|ment|er|or|s)$/.test(observation)) {
    return 0.32;
  }
  if (stateKey === "adjective" && /(ive|al|ous|ful|ic)$/.test(observation)) {
    return 0.3;
  }
  if (stateKey === "noun" && observation.length <= 2) {
    return 0.06;
  }
  return stateKey === "noun" ? 0.18 : stateKey === "adjective" ? 0.14 : stateKey === "verb" ? 0.12 : 0.08;
}

function runHmmViterbi(mode: HmmMode, observations: string[]) {
  const config = buildHmmConfig(mode);
  const start = config.start as unknown as Record<string, number>;
  const transition = config.transition as unknown as Record<string, Record<string, number>>;
  const trellis: NonNullable<ConceptFrame["hiddenMarkov"]>["trellis"] = [];

  observations.forEach((observation, time) => {
    trellis[time] = config.states.map((state) => {
      const emissionLog = safeLog(config.emission(state.key, observation));
      if (time === 0) {
        return {
          time,
          stateKey: state.key,
          stateLabel: state.label,
          observation,
          logProbability: safeLog(start[state.key]) + emissionLog,
          emissionLog,
          transitionLog: safeLog(start[state.key]),
          backPointer: undefined,
          active: false,
          winning: false,
        };
      }

      const candidates = trellis[time - 1].map((previousCell) => {
        const transitionProbability = transition[previousCell.stateKey]?.[state.key] ?? 1e-6;
        const transitionLog = safeLog(transitionProbability);
        return {
          previous: previousCell.stateKey,
          transitionLog,
          score: previousCell.logProbability + transitionLog,
        };
      });
      const best = candidates.reduce((winner, candidate) => (candidate.score > winner.score ? candidate : winner));
      return {
        time,
        stateKey: state.key,
        stateLabel: state.label,
        observation,
        logProbability: best.score + emissionLog,
        emissionLog,
        transitionLog: best.transitionLog,
        backPointer: best.previous,
        active: false,
        winning: false,
      };
    });
  });

  const finalCells = trellis[trellis.length - 1] ?? [];
  const finalBest = finalCells.reduce((winner, cell) => (cell.logProbability > winner.logProbability ? cell : winner), finalCells[0]);
  const bestPath = Array(observations.length).fill(finalBest?.stateKey ?? config.states[0].key);
  for (let time = observations.length - 1; time >= 0; time -= 1) {
    const stateKey = bestPath[time];
    const cell = trellis[time]?.find((candidate) => candidate.stateKey === stateKey);
    if (time > 0) {
      bestPath[time - 1] = cell?.backPointer ?? config.states[0].key;
    }
  }
  const scoreSteps = bestPath.map((stateKey, time) => {
    const cell = trellis[time]?.find((candidate) => candidate.stateKey === stateKey) ?? trellis[time]?.[0];
    const previousState = time === 0 ? "START" : bestPath[time - 1];
    const state = config.states.find((candidate) => candidate.key === stateKey) ?? config.states[0];
    return {
      time,
      observation: observations[time],
      stateKey,
      stateLabel: state.label,
      transitionLabel: time === 0 ? `START -> ${state.shortLabel}` : `${stateLabel(config, previousState)} -> ${state.shortLabel}`,
      transitionLog: cell?.transitionLog ?? 0,
      emissionLog: cell?.emissionLog ?? 0,
      cumulativeLog: cell?.logProbability ?? 0,
      color: state.color,
    };
  });

  return {
    mode,
    input: observations.join(" "),
    observations,
    states: config.states,
    trellis,
    bestPath,
    bestPathLabels: bestPath.map((stateKey) => stateLabel(config, stateKey)),
    scoreSteps,
    logProbability: finalBest?.logProbability ?? 0,
  };
}

function makeHmmFrameState(
  decoded: ReturnType<typeof runHmmViterbi>,
  input: string,
  phaseIndex: number,
  phaseProgress: number,
) {
  const trellis = decoded.trellis.map((column, time) =>
    column.map((cell) => ({
      ...cell,
      active: time <= phaseIndex,
      winning: time <= phaseIndex && cell.stateKey === decoded.bestPath[time],
    })),
  );
  const taggedTokens = decoded.observations.map((observation, index) => {
    const stateKey = decoded.bestPath[index];
    const state = decoded.states.find((candidate) => candidate.key === stateKey) ?? decoded.states[0];
    return {
      text: observation,
      stateKey,
      stateLabel: state.label,
      color: state.color,
      logProbability: decoded.scoreSteps[index]?.cumulativeLog ?? 0,
    };
  });

  return {
    ...decoded,
    input,
    trellis,
    scoreSteps: decoded.scoreSteps.slice(0, Math.max(1, phaseIndex + 1)),
    taggedTokens,
    phaseIndex,
    phaseProgress,
  };
}

function modeParam(params: ParameterState): HmmMode {
  return stringParam(params, "hmmMode", "market") === "pos" ? "pos" : "market";
}

function stringParam(params: ParameterState, key: string, fallback: string) {
  return typeof params[key] === "string" ? String(params[key]) : fallback;
}

function safeLog(value: number) {
  return Math.log(Math.max(1e-9, value));
}

function stateLabel(config: ReturnType<typeof buildHmmConfig>, stateKey: string) {
  return config.states.find((state) => state.key === stateKey)?.shortLabel ?? stateKey;
}
