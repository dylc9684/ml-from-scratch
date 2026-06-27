import { ImageUp, Minus, Plus } from "lucide-react";
import { Fragment } from "react";
import type {
  AlgorithmDefinition,
  GridWorldCell,
  GridWorldValue,
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

  if (parameter.kind === "matrix") {
    const matrixValue = toMatrixValue(value, parameter.defaultValue);
    const updateCell = (rowIndex: number, columnIndex: number, nextValue: number) => {
      const next = matrixValue.map((row) => [...row]);
      next[rowIndex][columnIndex] = Math.max(parameter.min, Math.min(parameter.max, nextValue));
      onChange(next);
    };

    return (
      <div className="control-card matrix-control">
        <span className="control-label">
          <strong>{parameter.label}</strong>
          <span>{matrixValue.length}x{matrixValue[0]?.length ?? 0}</span>
        </span>
        <div className="matrix-grid" role="table" aria-label={parameter.label}>
          <div className="matrix-corner" aria-hidden="true" />
          {parameter.columnLabels.map((label) => (
            <div key={label} className="matrix-heading" role="columnheader">
              {label}
            </div>
          ))}
          <div className="matrix-heading matrix-sum-heading" role="columnheader">
            Sum
          </div>
          {matrixValue.map((row, rowIndex) => {
            const rowLabel = parameter.rowLabels[rowIndex] ?? `S${rowIndex + 1}`;
            const rowSum = row.reduce((total, cell) => total + cell, 0);

            return (
              <Fragment key={rowLabel}>
                <div key={`${rowLabel}-label`} className="matrix-heading" role="rowheader">
                  {rowLabel}
                </div>
                {row.map((cell, columnIndex) => {
                  const columnLabel = parameter.columnLabels[columnIndex] ?? `S${columnIndex + 1}`;
                  return (
                    <input
                      key={`${rowLabel}-${columnLabel}`}
                      type="number"
                      min={parameter.min}
                      max={parameter.max}
                      step={parameter.step}
                      value={cell}
                      aria-label={`${rowLabel} to ${columnLabel}`}
                      onChange={(event) => updateCell(rowIndex, columnIndex, Number(event.target.value))}
                    />
                  );
                })}
                <div
                  key={`${rowLabel}-sum`}
                  className={`matrix-row-sum ${Math.abs(rowSum - 1) < 0.02 ? "balanced" : ""}`}
                >
                  {formatValue(rowSum, parameter.format)}
                </div>
              </Fragment>
            );
          })}
        </div>
      </div>
    );
  }

  if (parameter.kind === "image") {
    const imageValue = toImageMatrixValue(value, parameter.defaultValue);
    const handleImageChange = async (file: File | undefined) => {
      if (!file) {
        return;
      }

      const nextImage = await imageFileToMatrix(file, parameter.maxSize);
      onChange(nextImage);
    };

    return (
      <div className="control-card image-control">
        <span className="control-label">
          <strong>{parameter.label}</strong>
          <span>
            {imageValue.width}x{imageValue.height}
          </span>
        </span>
        <label className="image-upload-button">
          <ImageUp size={16} aria-hidden="true" />
          {parameter.buttonLabel}
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              void handleImageChange(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </label>
        <div className="image-control-meta">
          <span>{imageValue.name}</span>
          <span>{imageValue.values.length} rows</span>
        </div>
      </div>
    );
  }

  if (parameter.kind === "gridworld") {
    const gridValue = toGridWorldValue(value, parameter.defaultValue);
    const counts = countGridWorldCells(gridValue);
    const resetGrid = () => {
      onChange({
        ...parameter.defaultValue,
        cells: parameter.defaultValue.cells.map((row) => [...row]),
      });
    };

    return (
      <div className="control-card gridworld-control">
        <span className="control-label">
          <strong>{parameter.label}</strong>
          <span>{gridValue.rows}x{gridValue.columns}</span>
        </span>
        <div className="gridworld-counts" aria-label={`${parameter.label} cell summary`}>
          <span>Walls {counts.wall}</span>
          <span>Fire {counts.fire}</span>
          <span>Gold {counts.gold}</span>
        </div>
        <button className="button secondary action-button" type="button" onClick={resetGrid}>
          Reset layout
        </button>
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

function toGridWorldValue(value: ParameterValue, fallback: GridWorldValue) {
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    "kind" in value &&
    value.kind === "gridworld"
  ) {
    return value;
  }

  return fallback;
}

function countGridWorldCells(grid: GridWorldValue) {
  return grid.cells.flat().reduce<Record<GridWorldCell, number>>(
    (counts, cell) => ({
      ...counts,
      [cell]: counts[cell] + 1,
    }),
    { empty: 0, wall: 0, fire: 0, gold: 0, start: 0 },
  );
}

function toMatrixValue(value: ParameterValue, fallback: number[][]) {
  const source = Array.isArray(value) ? value : fallback;
  return fallback.map((fallbackRow, rowIndex) => {
    const row = Array.isArray(source[rowIndex]) ? source[rowIndex] : fallbackRow;
    return fallbackRow.map((fallbackCell, columnIndex) => {
      const cell = Number(row[columnIndex]);
      return Number.isFinite(cell) ? cell : fallbackCell;
    });
  });
}

function toImageMatrixValue(
  value: ParameterValue,
  fallback: {
    kind: "image-matrix";
    name: string;
    width: number;
    height: number;
    values: number[][];
    dataUrl?: string;
  },
) {
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    "kind" in value &&
    value.kind === "image-matrix"
  ) {
    return value;
  }

  return fallback;
}

function imageFileToMatrix(file: File, maxSize: number) {
  return new Promise<{
    kind: "image-matrix";
    name: string;
    width: number;
    height: number;
    values: number[][];
    dataUrl?: string;
  }>((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read image file."));
    reader.onload = () => {
      const dataUrl = String(reader.result ?? "");
      const image = new Image();
      image.onerror = () => reject(new Error("Could not load image file."));
      image.onload = () => {
        const scale = Math.min(1, maxSize / Math.max(image.naturalWidth, image.naturalHeight));
        const width = Math.max(8, Math.round(image.naturalWidth * scale));
        const height = Math.max(8, Math.round(image.naturalHeight * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        if (!context) {
          reject(new Error("Could not process image file."));
          return;
        }

        context.drawImage(image, 0, 0, width, height);
        const pixels = context.getImageData(0, 0, width, height).data;
        const values = Array.from({ length: height }, (_, row) =>
          Array.from({ length: width }, (_, column) => {
            const index = (row * width + column) * 4;
            const luminance =
              pixels[index] * 0.2126 +
              pixels[index + 1] * 0.7152 +
              pixels[index + 2] * 0.0722;
            return Number.parseFloat((luminance / 255).toFixed(4));
          }),
        );

        resolve({
          kind: "image-matrix",
          name: file.name,
          width,
          height,
          values,
          dataUrl,
        });
      };
      image.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
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
