import { categoryDemos } from "../algorithms/categoryDemos";
import { kMeans } from "../algorithms/kMeans";
import { linearRegression } from "../algorithms/linearRegression";
import { algorithmToLessonModule } from "./lessonModule";
import type { LessonModule } from "./lessonModule";

export const lessonModules: LessonModule[] = [
  linearRegression,
  ...categoryDemos,
  kMeans,
].map(algorithmToLessonModule);

export { lessonModuleToAlgorithm } from "./lessonModule";
export type { LessonModule } from "./lessonModule";
