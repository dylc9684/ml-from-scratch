import { categoryDemos } from "./categoryDemos";
import { kMeans } from "./kMeans";
import { linearRegression } from "./linearRegression";
import type { AlgorithmDefinition, ParameterState } from "../types/algorithm";

export const algorithms: AlgorithmDefinition[] = [
  linearRegression,
  ...categoryDemos,
  kMeans,
];

export function makeDefaultParams(algorithm: AlgorithmDefinition): ParameterState {
  return algorithm.parameters.reduce<ParameterState>((state, parameter) => {
    state[parameter.id] =
      parameter.kind === "matrix"
        ? parameter.defaultValue.map((row) => [...row])
        : parameter.kind === "image"
          ? {
              ...parameter.defaultValue,
              values: parameter.defaultValue.values.map((row) => [...row]),
            }
        : parameter.defaultValue;
    return state;
  }, {});
}
