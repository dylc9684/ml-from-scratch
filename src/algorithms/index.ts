import { lessonModules, lessonModuleToAlgorithm } from "../lessons";
import type { AlgorithmDefinition, GridWorldValue, ParameterState, PointSetParameterValue } from "../types/algorithm";

export const algorithms: AlgorithmDefinition[] = lessonModules.map(lessonModuleToAlgorithm);

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
          : parameter.kind === "gridworld"
            ? cloneGridWorld(parameter.defaultValue)
            : parameter.kind === "point-set"
              ? clonePointSet(parameter.defaultValue)
        : parameter.defaultValue;
    return state;
  }, {});
}

function clonePointSet(pointSet: PointSetParameterValue): PointSetParameterValue {
  return {
    ...pointSet,
    points: pointSet.points.map((point) => ({ ...point })),
  };
}

function cloneGridWorld(grid: GridWorldValue): GridWorldValue {
  return {
    ...grid,
    cells: grid.cells.map((row) => [...row]),
  };
}
