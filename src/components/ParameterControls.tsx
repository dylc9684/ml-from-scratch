import { Minus, Plus } from "lucide-react";
import type {
  AlgorithmDefinition,
  ParameterDefinition,
  ParameterState,
  ParameterValue,
} from "../types/algorithm";

type Props = {
  algorithm: AlgorithmDefinition;
  params: ParameterState;
  autoRun: boolean;
  onAutoRunChange: (value: boolean) => void;
  onChange: (key: string, value: ParameterValue) => void;
};

export function ParameterControls({
  algorithm,
  params,
  autoRun,
  onAutoRunChange,
  onChange,
}: Props) {
  return (
    <div className="control-stack">
      <div className="control-card">
        <div className="toggle-row">
          <div>
            <strong>Auto-run</strong>
            <p>Refresh the JavaScript engine after parameter changes.</p>
          </div>
          <label className="switch">
            <input
              type="checkbox"
              checked={autoRun}
              onChange={(event) => onAutoRunChange(event.target.checked)}
            />
            <span />
          </label>
        </div>
      </div>

      {algorithm.parameters.map((parameter) => (
        <Control
          key={parameter.id}
          parameter={parameter}
          value={params[parameter.id]}
          onChange={(value) => onChange(parameter.id, value)}
        />
      ))}
    </div>
  );
}

function Control({
  parameter,
  value,
  onChange,
}: {
  parameter: ParameterDefinition;
  value: ParameterValue;
  onChange: (value: ParameterValue) => void;
}) {
  if (parameter.kind === "range") {
    const numericValue = Number(value ?? parameter.defaultValue);
    return (
      <label className="control-group">
        <span className="control-label">
          <strong>{parameter.label}</strong>
          <span>{formatValue(numericValue, parameter.format)}</span>
        </span>
        <input
          type="range"
          min={parameter.min}
          max={parameter.max}
          step={parameter.step}
          value={numericValue}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </label>
    );
  }

  if (parameter.kind === "select") {
    return (
      <label className="control-group">
        <span className="control-label">
          <strong>{parameter.label}</strong>
        </span>
        <select
          value={String(value ?? parameter.defaultValue)}
          onChange={(event) => onChange(event.target.value)}
        >
          {parameter.options.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    );
  }

  if (parameter.kind === "stepper") {
    const numericValue = Number(value ?? parameter.defaultValue);
    const nextDown = Math.max(parameter.min, numericValue - parameter.step);
    const nextUp = Math.min(parameter.max, numericValue + parameter.step);

    return (
      <div className="control-card stepper-control">
        <div>
          <span className="control-label">
            <strong>{parameter.label}</strong>
            <span>{formatValue(numericValue, parameter.format)}</span>
          </span>
        </div>
        <div className="stepper-actions" aria-label={`${parameter.label} controls`}>
          <button
            className="stepper-button"
            type="button"
            aria-label={`Decrease ${parameter.label}`}
            disabled={numericValue <= parameter.min}
            onClick={() => onChange(nextDown)}
          >
            <Minus size={16} aria-hidden="true" />
          </button>
          <button
            className="stepper-button"
            type="button"
            aria-label={`Increase ${parameter.label}`}
            disabled={numericValue >= parameter.max}
            onClick={() => onChange(nextUp)}
          >
            <Plus size={16} aria-hidden="true" />
          </button>
        </div>
      </div>
    );
  }

  if (parameter.kind === "action") {
    const numericValue = Number(value ?? parameter.defaultValue);
    const nextValue =
      numericValue >= parameter.max
        ? parameter.min
        : Math.min(parameter.max, numericValue + parameter.step);

    return (
      <div className="control-card action-control">
        <span className="control-label">
          <strong>{parameter.label}</strong>
          <span>{formatValue(numericValue, parameter.format)}</span>
        </span>
        <button
          className="button primary action-button"
          type="button"
          onClick={() => onChange(nextValue)}
        >
          {parameter.buttonLabel}
        </button>
      </div>
    );
  }

  return (
    <div className="control-card">
      <div className="toggle-row">
        <strong>{parameter.label}</strong>
        <label className="switch">
          <input
            type="checkbox"
            checked={Boolean(value ?? parameter.defaultValue)}
            onChange={(event) => onChange(event.target.checked)}
          />
          <span />
        </label>
      </div>
    </div>
  );
}

function formatValue(value: number, format?: "integer" | "decimal" | "percent") {
  if (format === "integer") {
    return Math.round(value).toString();
  }

  if (format === "percent") {
    return `${Math.round(value * 100)}%`;
  }

  return value.toFixed(3);
}
