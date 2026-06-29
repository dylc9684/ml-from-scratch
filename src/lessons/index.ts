import { categoryDemos } from "../algorithms/categoryDemos";
import { kMeans } from "../algorithms/kMeans";
import { linearRegression } from "../algorithms/linearRegression";
import { hiddenMarkovModelsModule } from "./hiddenMarkovModels";
import { algorithmToLessonModule } from "./lessonModule";
import type { LessonModule } from "./lessonModule";

const categoryDemoModules = categoryDemos.map(algorithmToLessonModule);
const stochasticProcessesIndex = categoryDemoModules.findIndex(
  (module) => module.metadata.id === "stochastic-processes",
);
const hiddenMarkovInsertIndex =
  stochasticProcessesIndex >= 0 ? stochasticProcessesIndex + 1 : categoryDemoModules.length;

export const lessonModules: LessonModule[] = [
  algorithmToLessonModule(linearRegression),
  ...categoryDemoModules.slice(0, hiddenMarkovInsertIndex),
  hiddenMarkovModelsModule,
  ...categoryDemoModules.slice(hiddenMarkovInsertIndex),
  algorithmToLessonModule(kMeans),
];

export { lessonModuleToAlgorithm } from "./lessonModule";
export type { LessonModule } from "./lessonModule";
