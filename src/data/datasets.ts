import { csvParseRows } from "d3";
import type {
  DataPoint,
  DatasetColumn,
  DatasetMapping,
  NormalizedDataset,
  RawDataset,
} from "../types/algorithm";

export type DatasetIssue = {
  severity: "error" | "warning";
  message: string;
  row?: number;
  column?: string;
};

export type DatasetParseResult =
  | {
      ok: true;
      dataset: RawDataset;
      issues: DatasetIssue[];
    }
  | {
      ok: false;
      issues: DatasetIssue[];
    };

export type DatasetNormalizationResult = {
  dataset: NormalizedDataset;
  issues: DatasetIssue[];
};

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

export function parseDatasetFile(fileName: string, text: string): DatasetParseResult {
  if (text.trim().length === 0) {
    return fail("The uploaded file is empty.");
  }

  if (fileName.toLowerCase().endsWith(".json")) {
    return parseJsonDataset(fileName, text);
  }

  return parseCsvDataset(fileName, text);
}

function parseJsonDataset(fileName: string, text: string): DatasetParseResult {
  let parsed: unknown;

  try {
    parsed = JSON.parse(stripBom(text));
  } catch (error) {
    return fail(`JSON could not be parsed: ${error instanceof Error ? error.message : "invalid syntax"}.`);
  }

  const rows = Array.isArray(parsed)
    ? parsed
    : typeof parsed === "object" && parsed !== null && Array.isArray((parsed as { rows?: unknown }).rows)
      ? (parsed as { rows: unknown[] }).rows
      : null;

  if (!rows) {
    return fail("JSON uploads must be an array of objects or an object with a rows array.");
  }

  const objectRows = rows.filter(isRecord);
  if (objectRows.length === 0) {
    return fail("JSON upload did not contain any object rows.");
  }

  const issues: DatasetIssue[] = [];
  if (objectRows.length !== rows.length) {
    issues.push({
      severity: "warning",
      message: `${rows.length - objectRows.length} JSON row(s) were skipped because they were not objects.`,
    });
  }

  return ok({
    name: fileName,
    rows: objectRows,
    columns: inferColumns(objectRows),
  }, issues);
}

function parseCsvDataset(fileName: string, text: string): DatasetParseResult {
  let records: string[][];

  try {
    records = csvParseRows(stripBom(text));
  } catch (error) {
    return fail(`CSV could not be parsed: ${error instanceof Error ? error.message : "invalid syntax"}.`);
  }

  const nonEmptyRecords = records.filter((row) =>
    row.some((cell) => cell.trim().length > 0),
  );

  if (nonEmptyRecords.length < 2) {
    return fail("CSV uploads need a header row and at least one data row.");
  }

  const headers = nonEmptyRecords[0].map((header) => header.trim());
  const headerIssue = validateHeaders(headers);
  if (headerIssue) {
    return fail(headerIssue);
  }

  const issues: DatasetIssue[] = [];
  const rows = nonEmptyRecords.slice(1).map((values, index) => {
    const rowNumber = index + 2;
    if (values.length !== headers.length) {
      issues.push({
        severity: "warning",
        row: rowNumber,
        message: `Row ${rowNumber} has ${values.length} cell(s); expected ${headers.length}. Missing cells were filled as blank and extra cells were ignored.`,
      });
    }

    return headers.reduce<Record<string, unknown>>((row, header, columnIndex) => {
      row[header] = coerceValue(values[columnIndex]?.trim() ?? "");
      return row;
    }, {});
  });

  return ok({
    name: fileName,
    rows,
    columns: inferColumns(rows),
  }, issues);
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
    type: isMostlyNumeric(rows.map((row) => row[name])) ? "number" : "string",
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
): DatasetNormalizationResult {
  const issues: DatasetIssue[] = [];

  if (!mapping.x || !mapping.y) {
    return {
      dataset: makeUploadedDataset(raw, [], mapping),
      issues: [
        {
          severity: "error",
          message: "Choose numeric X and Y feature columns before running the algorithm.",
        },
      ],
    };
  }

  const points: DataPoint[] = [];
  const rejectedRows: number[] = [];

  raw.rows.forEach((row, index) => {
    const point = {
      x: Number(row[mapping.x]),
      y: Number(row[mapping.y]),
      label: mapping.label ? (row[mapping.label] as string | number | undefined) : undefined,
    };

    if (Number.isFinite(point.x) && Number.isFinite(point.y)) {
      points.push(point);
    } else {
      rejectedRows.push(index + 1);
    }
  });

  if (rejectedRows.length > 0) {
    issues.push({
      severity: "warning",
      message: `${rejectedRows.length} row(s) were skipped because the mapped X/Y values were not numeric.`,
      row: rejectedRows[0],
    });
  }

  if (points.length === 0) {
    issues.push({
      severity: "error",
      message: "No usable rows remain after feature mapping. Select different numeric columns or upload cleaner data.",
    });
  }

  return {
    dataset: makeUploadedDataset(raw, points, mapping),
    issues,
  };
}

function makeUploadedDataset(
  raw: RawDataset,
  points: DataPoint[],
  mapping: DatasetMapping,
): NormalizedDataset {
  return {
    name: raw.name,
    points,
    columns: raw.columns,
    mapping,
    source: "upload",
  };
}

function stripBom(text: string) {
  return text.replace(/^\uFEFF/, "");
}

function ok(dataset: RawDataset, issues: DatasetIssue[] = []): DatasetParseResult {
  return {
    ok: true,
    dataset,
    issues,
  };
}

function fail(message: string): DatasetParseResult {
  return {
    ok: false,
    issues: [{ severity: "error", message }],
  };
}

function validateHeaders(headers: string[]) {
  if (headers.length === 0 || headers.every((header) => header.length === 0)) {
    return "CSV header row is empty.";
  }

  const blankIndex = headers.findIndex((header) => header.length === 0);
  if (blankIndex >= 0) {
    return `CSV header ${blankIndex + 1} is blank.`;
  }

  const seen = new Set<string>();
  const duplicate = headers.find((header) => {
    const key = header.toLowerCase();
    if (seen.has(key)) {
      return true;
    }
    seen.add(key);
    return false;
  });

  return duplicate ? `CSV header "${duplicate}" is duplicated.` : null;
}

function isMostlyNumeric(values: unknown[]) {
  const filled = values.filter((value) => value !== "");
  return filled.length > 0 && filled.every((value) => typeof value === "number");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
