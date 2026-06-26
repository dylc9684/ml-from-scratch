import katex from "katex";
import type { AlgorithmDefinition } from "../types/algorithm";

type Props = {
  algorithm: AlgorithmDefinition;
};

export function EducationPanel({ algorithm }: Props) {
  return (
    <article className="education-panel">
      <div className="education-copy">
        {algorithm.explanation.map((paragraph) => (
          <p key={paragraph}>{paragraph}</p>
        ))}
      </div>

      <div className="formula-stack">
        {algorithm.formulas.map((formula) => (
          <section className="formula-card" key={formula.title}>
            <span>{formula.title}</span>
            <Formula expression={formula.expression} />
          </section>
        ))}
      </div>
    </article>
  );
}

function Formula({ expression }: { expression: string }) {
  let html = "";

  try {
    html = katex.renderToString(expression, {
      displayMode: true,
      throwOnError: false,
    });
  } catch {
    html = expression;
  }

  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />;
}
