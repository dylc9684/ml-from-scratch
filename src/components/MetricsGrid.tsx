import type { Metric } from "../types/algorithm";

type Props = {
  metrics: Metric[];
};

export function MetricsGrid({ metrics }: Props) {
  return (
    <section className="metrics-grid" aria-label="Model metrics">
      {metrics.map((metric) => (
        <div className="metric-card" key={metric.label}>
          <span>{metric.label}</span>
          <strong>{metric.value}</strong>
        </div>
      ))}
    </section>
  );
}
