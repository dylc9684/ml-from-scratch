import { Sparkles, Upload } from "lucide-react";
import type {
  DatasetMapping,
  NormalizedDataset,
  RawDataset,
} from "../types/algorithm";
import type { DatasetIssue } from "../data/datasets";

type Props = {
  rawDataset: RawDataset | null;
  dataset: NormalizedDataset;
  issues: DatasetIssue[];
  mapping: DatasetMapping;
  onMappingChange: (mapping: DatasetMapping) => void;
  onUploadClick: () => void;
  onUseSample: () => void;
};

export function DatasetPanel({
  rawDataset,
  dataset,
  issues,
  mapping,
  onMappingChange,
  onUploadClick,
  onUseSample,
}: Props) {
  const columns = rawDataset?.columns ?? dataset.columns;
  const numericColumns = columns.filter((column) => column.type === "number");

  const updateMapping = (key: keyof DatasetMapping, value: string) => {
    onMappingChange({
      ...mapping,
      [key]: value || undefined,
    });
  };

  return (
    <section className="dataset-panel">
      <div className="dataset-actions">
        <button className="button secondary" type="button" onClick={onUploadClick}>
          <Upload size={16} aria-hidden="true" />
          Upload CSV/JSON
        </button>
        <button className="button secondary" type="button" onClick={onUseSample}>
          <Sparkles size={16} aria-hidden="true" />
          Use sample
        </button>
      </div>

      <div className="data-summary">
        <div>
          <span>Dataset</span>
          <strong>{dataset.name}</strong>
        </div>
        <strong>{dataset.points.length} usable rows</strong>
      </div>

      {issues.length > 0 && (
        <div className="dataset-issues" role="status" aria-live="polite">
          {issues.slice(0, 5).map((issue, index) => (
            <div className={`dataset-issue ${issue.severity}`} key={`${issue.message}-${index}`}>
              <strong>{issue.severity === "error" ? "Error" : "Warning"}</strong>
              <span>
                {issue.message}
                {issue.row ? ` Row ${issue.row}.` : ""}
                {issue.column ? ` Column ${issue.column}.` : ""}
              </span>
            </div>
          ))}
          {issues.length > 5 && (
            <div className="dataset-issue warning">
              <strong>More</strong>
              <span>{issues.length - 5} additional dataset issue(s).</span>
            </div>
          )}
        </div>
      )}

      {rawDataset ? (
        <div className="mapping-controls">
          <MappingSelect
            label="X feature"
            value={mapping.x}
            columns={numericColumns}
            onChange={(value) => updateMapping("x", value)}
          />
          <MappingSelect
            label="Y feature"
            value={mapping.y}
            columns={numericColumns}
            onChange={(value) => updateMapping("y", value)}
          />
          <MappingSelect
            label="Label"
            value={mapping.label ?? ""}
            columns={columns}
            optional
            onChange={(value) => updateMapping("label", value)}
          />
        </div>
      ) : (
        <p className="empty-state">
          Generated samples are shaped to make the selected algorithm easy to inspect.
          Upload a CSV or JSON array to map custom features.
        </p>
      )}

      <div className="data-preview">
        <div className="preview-row header">
          <span>x</span>
          <span>y</span>
          <span>label</span>
        </div>
        {dataset.points.slice(0, 7).map((point, index) => (
          <div className="preview-row" key={`${point.x}-${point.y}-${index}`}>
            <span>{point.x.toFixed(3)}</span>
            <span>{point.y.toFixed(3)}</span>
            <span>{point.label ?? "none"}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function MappingSelect({
  label,
  value,
  columns,
  optional = false,
  onChange,
}: {
  label: string;
  value: string;
  columns: Array<{ name: string; type: string }>;
  optional?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <label className="mapping-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {optional && <option value="">None</option>}
        {columns.map((column) => (
          <option key={column.name} value={column.name}>
            {column.name}
          </option>
        ))}
      </select>
    </label>
  );
}
