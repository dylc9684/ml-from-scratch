import type { AlgorithmDefinition, AlgorithmRunController } from "../types/algorithm";

export const dynamicProgrammingController: AlgorithmRunController = {
  primaryActionLabel: "Play",
  prepareRunParams: (params) => ({
    ...params,
    playSignal: Number(params.dpStep ?? 0) + 1,
  }),
};

export const dynamicProgrammingContent = {
  formulas: [
    {
      title: "Bellman optimality",
      expression:
        "V_{k+1}(s)=\\max_a\\sum_{s'}P(s'\\mid s,a)[R(s,a,s')+\\gamma V_k(s')]",
    },
    {
      title: "Policy evaluation",
      expression:
        "V^{\\pi}_{k+1}(s)=\\sum_{s'}P(s'\\mid s,\\pi(s))[R(s,\\pi(s),s')+\\gamma V^{\\pi}_k(s')]",
    },
    {
      title: "Policy extraction",
      expression:
        "\\pi^*(s)=\\arg\\max_a\\sum_{s'}P(s'\\mid s,a)[R(s,a,s')+\\gamma V(s')]",
    },
  ],
  explanation: [
    "Dynamic programming solves a known Markov decision process by repeatedly applying Bellman updates to a value matrix.",
    "In Gridworld, each coordinate is a state. Walls block movement, fire pits are negative terminal rewards, and gold chests are positive terminal rewards.",
    "As sweeps accumulate, reward information propagates outward through the grid; once the values stabilize, the best neighboring action becomes the optimal policy arrow.",
  ],
} satisfies Pick<AlgorithmDefinition, "formulas" | "explanation">;
