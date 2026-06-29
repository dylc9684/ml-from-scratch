import type { AlgorithmDefinition } from "../types/algorithm";

export type LessonMetadata = Pick<
  AlgorithmDefinition,
  "id" | "name" | "category" | "summary" | "catalog" | "parameters"
>;

export type LessonRuntime = Pick<
  AlgorithmDefinition,
  "makeSampleDataset" | "engine" | "controller"
>;

export type LessonContent = Pick<
  AlgorithmDefinition,
  "formulas" | "explanation" | "deepDive" | "applications" | "notebook"
>;

export type LessonCode = AlgorithmDefinition["code"];

export type LessonVisualization = {
  renderer: "canvas";
};

export type LessonModule = {
  metadata: LessonMetadata;
  runtime: LessonRuntime;
  content: LessonContent;
  code: LessonCode;
  visualization: LessonVisualization;
};

export function defineLessonModule(module: LessonModule): LessonModule {
  return module;
}

export function algorithmToLessonModule(algorithm: AlgorithmDefinition): LessonModule {
  return defineLessonModule({
    metadata: {
      id: algorithm.id,
      name: algorithm.name,
      category: algorithm.category,
      summary: algorithm.summary,
      catalog: algorithm.catalog,
      parameters: algorithm.parameters,
    },
    runtime: {
      makeSampleDataset: algorithm.makeSampleDataset,
      engine: algorithm.engine,
      controller: algorithm.controller,
    },
    content: {
      formulas: algorithm.formulas,
      explanation: algorithm.explanation,
      deepDive: algorithm.deepDive,
      applications: algorithm.applications,
      notebook: algorithm.notebook,
    },
    code: algorithm.code,
    visualization: {
      renderer: "canvas",
    },
  });
}

export function lessonModuleToAlgorithm(module: LessonModule): AlgorithmDefinition {
  return {
    ...module.metadata,
    ...module.runtime,
    ...module.content,
    code: module.code,
  };
}
