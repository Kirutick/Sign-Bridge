import type { ExportedDatasetJson } from "./exportDataset";

export interface ImportValidationFailure {
  index?: number;
  reason: string;
}

/**
 * Import Validation (§9)
 * Strictly validates an imported JSON dataset for structural and internal consistency.
 * Rejects partially invalid files wholesale with itemized errors.
 */
export function validateImportDataset(data: any): { ok: boolean; errors: ImportValidationFailure[] } {
  const errors: ImportValidationFailure[] = [];

  if (!data || typeof data !== "object") {
    errors.push({ reason: "Dataset is not a valid JSON object" });
    return { ok: false, errors };
  }

  const { X, y, labels, metadata } = data as ExportedDatasetJson;

  if (!X || !Array.isArray(X)) errors.push({ reason: "Missing or invalid 'X' array" });
  if (!y || !Array.isArray(y)) errors.push({ reason: "Missing or invalid 'y' array" });
  if (!labels || typeof labels !== "object") errors.push({ reason: "Missing or invalid 'labels' mapping" });
  if (!metadata || typeof metadata !== "object") errors.push({ reason: "Missing or invalid 'metadata' object" });

  if (errors.length > 0) return { ok: false, errors };

  const sampleCount = metadata.sampleCount;
  if (typeof sampleCount !== "number") {
    errors.push({ reason: "metadata.sampleCount must be a number" });
    return { ok: false, errors };
  }

  // Check top-level length consistency
  if (X.length !== sampleCount) {
    errors.push({ reason: `X array length (${X.length}) does not match metadata.sampleCount (${sampleCount})` });
  }
  if (y.length !== sampleCount) {
    errors.push({ reason: `y array length (${y.length}) does not match metadata.sampleCount (${sampleCount})` });
  }

  // If top-level lengths mismatch, stop further expensive frame checks
  if (errors.length > 0) return { ok: false, errors };

  const expectedSeqLen = metadata.sequenceLength;
  const expectedFeatCount = metadata.featureCount;

  // Validate internal consistency of y vs labels vs classMapping
  const classMapping = metadata.classMapping;
  const labelKeys = new Set(Object.keys(labels));
  const mappingKeys = new Set(Object.keys(classMapping || {}));

  if (labelKeys.size !== mappingKeys.size) {
    errors.push({ reason: `labels dictionary size (${labelKeys.size}) does not match classMapping size (${mappingKeys.size})` });
  }

  for (const key of labelKeys) {
    if (labels[key] !== classMapping[Number(key)]) {
      errors.push({ reason: `Label key '${key}' mismatch between labels dict ('${labels[key]}') and classMapping ('${classMapping[Number(key)]}')` });
    }
  }

  // Validate structural integrity of X and y
  for (let i = 0; i < sampleCount; i++) {
    const classId = y[i];
    if (labels[classId.toString()] === undefined) {
      errors.push({ index: i, reason: `y value (${classId}) does not exist in labels dictionary` });
    }

    const seq = X[i];
    if (!Array.isArray(seq)) {
      errors.push({ index: i, reason: "Sequence is not an array" });
      continue;
    }
    
    if (seq.length !== expectedSeqLen) {
      errors.push({ index: i, reason: `Sequence length mismatch (expected ${expectedSeqLen}, got ${seq.length})` });
      continue;
    }

    for (let f = 0; f < seq.length; f++) {
      const frame = seq[f];
      if (!Array.isArray(frame)) {
        errors.push({ index: i, reason: `Frame ${f} is not an array` });
        break;
      }
      
      if (frame.length !== expectedFeatCount) {
        errors.push({ index: i, reason: `Frame ${f} feature count mismatch (expected ${expectedFeatCount}, got ${frame.length})` });
        break;
      }

      for (let j = 0; j < frame.length; j++) {
        const val = frame[j];
        if (typeof val !== "number" || Number.isNaN(val) || !Number.isFinite(val)) {
          errors.push({ index: i, reason: `Frame ${f} feature ${j} is NaN, Infinity, or non-numeric` });
          break;
        }
        if (Math.abs(val) > 10.0) {
          errors.push({ index: i, reason: `Frame ${f} feature ${j} is out of sane numeric bounds (${val} not in [-10.0, 10.0])` });
          break;
        }
      }
    }
  }

  return { ok: errors.length === 0, errors };
}
