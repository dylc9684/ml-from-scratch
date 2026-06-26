import type {
  DataPoint,
  DatasetColumn,
  DatasetMapping,
  NormalizedDataset,
  RawDataset,
} from "../types/algorithm";

const round = (value: number, digits = 3) =>
  Number.parseFloat(value.toFixed(digits));

export function makeRegressionDataset(): NormalizedDataset {
  const points = Array.from({ length: 42 }, (_, index) => {
    const x = -4 + (index / 41) * 8;
    const wave = Math.sin(index * 0.8) * 0.55;
    const drift = (index % 5) * 0.08;
    return {
      x: round(x),
      y: round(1.45 * x + 2.2 + wave + drift),
    };
  });

  return makeSample("Generated regression sample", points);
}

export function makeClusterDataset(): NormalizedDataset {
  const centers = [
    { x: -3.2, y: -1.8 },
    { x: 2.9, y: -1.3 },
    { x: 0.8, y: 3.1 },
    { x: -1.7, y: 2.7 },
  ];

  const points = centers.flatMap((center, clusterIndex) =>
    Array.from({ length: 22 }, (_, index) => {
      const angle = index * 1.71 + clusterIndex * 0.5;
      const radius = 0.4 + ((index * 17) % 11) * 0.075;
      return {
        x: round(center.x + Math.cos(angle) * radius),
        y: round(center.y + Math.sin(angle * 1.2) * radius),
        label: clusterIndex,
      };
    }),
  );

  return makeSample("Generated cluster sample", points);
}

function makeSample(name: string, points: DataPoint[]): NormalizedDataset {
  return {
    name,
    points,
    columns: [
      { name: "x", type: "number" },
      { name: "y", type: "number" },
      { name: "label", type: "string" },
    ],
    mapping: { x: "x", y: "y", label: "label" },
    source: "sample",
  };
}

export function parseDatasetFile(fileName: string, text: string): RawDataset {
  if (fileName.toLowerCase().endsWith(".json")) {
    return parseJsonDataset(fileName, text);
  }

  return parseCsvDataset(fileName, text);
}

function parseJsonDataset(fileName: string, text: string): RawDataset {
  const parsed = JSON.parse(text) as unknown;
  const rows = Array.isArray(parsed)
    ? parsed
    : typeof parsed === "object" && parsed !== null && Array.isArray((parsed as { rows?: unknown }).rows)
      ? (parsed as { rows: unknown[] }).rows
      : [];

  const objectRows = rows.filter(isRecord);
  return {
    name: fileName,
    rows: objectRows,
    columns: inferColumns(objectRows),
  };
}

function parseCsvDataset(fileName: string, text: string): RawDataset {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const headers = splitCsvLine(lines[0] ?? "");
  const rows = lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return headers.reduce<Record<string, unknown>>((row, header, index) => {
      row[header] = coerceValue(values[index] ?? "");
      return row;
    }, {});
  });

  return {
    name: fileName,
    rows,
    columns: inferColumns(rows),
  };
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];

    if (char === "\"" && next === "\"") {
      current += "\"";
      index += 1;
    } else if (char === "\"") {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  cells.push(current.trim());
  return cells;
}

function coerceValue(value: string): unknown {
  const numeric = Number(value);
  if (value !== "" && Number.isFinite(numeric)) {
    return numeric;
  }

  return value;
}

function inferColumns(rows: Record<string, unknown>[]): DatasetColumn[] {
  const names = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  return names.map((name) => ({
    name,
    type: rows.some((row) => typeof row[name] === "number") ? "number" : "string",
  }));
}

export function initialMapping(raw: RawDataset): DatasetMapping {
  const numeric = raw.columns.filter((column) => column.type === "number");
  const stringColumn = raw.columns.find((column) => column.type === "string");

  return {
    x: numeric[0]?.name ?? raw.columns[0]?.name ?? "",
    y: numeric[1]?.name ?? numeric[0]?.name ?? raw.columns[1]?.name ?? "",
    label: stringColumn?.name,
  };
}

export function normalizeDataset(
  raw: RawDataset,
  mapping: DatasetMapping,
): NormalizedDataset {
  const points = raw.rows
    .map((row) => ({
      x: Number(row[mapping.x]),
      y: Number(row[mapping.y]),
      label: mapping.label ? (row[mapping.label] as string | number | undefined) : undefined,
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  return {
    name: raw.name,
    points,
    columns: raw.columns,
    mapping,
    source: "upload",
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
