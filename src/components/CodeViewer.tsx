import { Check, Copy } from "lucide-react";
import Prism from "prismjs";
import "prismjs/components/prism-javascript";
import "prismjs/components/prism-python";
import { useMemo, useState } from "react";
import type { AlgorithmDefinition, ParameterState } from "../types/algorithm";

type Language = "python" | "javascript";

type Props = {
  algorithm: AlgorithmDefinition;
  params: ParameterState;
};

const labels: Record<Language, string> = {
  python: "Python",
  javascript: "JavaScript",
};

export function CodeViewer({ algorithm, params }: Props) {
  const [language, setLanguage] = useState<Language>("python");
  const [copied, setCopied] = useState(false);
  const code = useMemo(
    () => algorithm.code[language](params),
    [algorithm, language, params],
  );
  const highlighted = useMemo(() => {
    const grammar = Prism.languages[language] ?? Prism.languages.javascript;
    return Prism.highlight(code, grammar, language);
  }, [code, language]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <section className="code-viewer">
      <div className="code-viewer-header">
        <div className="segmented-control" role="tablist" aria-label="Code language">
          {(["python", "javascript"] as Language[]).map((item) => (
            <button
              key={item}
              type="button"
              className={language === item ? "active" : ""}
              role="tab"
              aria-selected={language === item}
              onClick={() => setLanguage(item)}
            >
              {labels[item]}
            </button>
          ))}
        </div>

        <button className="icon-button compact" type="button" onClick={handleCopy}>
          {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
          <span className="visually-hidden">Copy code</span>
        </button>
      </div>

      <p className="code-note">
        Runtime is always the browser JavaScript engine. The Python tab is a synchronized
        notebook reference.
      </p>

      <pre className={`language-${language}`}>
        <code dangerouslySetInnerHTML={{ __html: highlighted }} />
      </pre>
    </section>
  );
}
