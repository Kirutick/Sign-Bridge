import type { RecordedSample } from "../dataCollection/validateSample";
import { NORMALIZATION_VERSION } from "../services/landmarkProcessor";

export interface ValidationFailure {
  sampleId: string;
  label: string;
  timestamp: string;
  reasons: string[];
}

export interface ValidationReport {
  totalScanned: number;
  validSamples: number;
  invalidSamples: number;
  failures: {
    sequenceLengthMismatch: number;
    featureCountMismatch: number;
    nanOrInfinity: number;
    missingLabel: number;
    duplicateIds: number;
    malformedMetadata: number;
    outOfBounds: number;
  };
  invalidDetails: ValidationFailure[];
  datasetVersions: Record<string, number>; // Breakdown of counts by datasetVersion
}

export interface DatasetValidationResult {
  report: ValidationReport;
  validSamples: RecordedSample[]; // Validated, clean subset
}

/**
 * Structural checks (§1.1)
 * Analyzes the raw dataset and produces a validation report and a clean subset.
 * Does NOT mutate raw storage.
 */
export function validateDataset(
  rawSamples: RecordedSample[],
  expectedSequenceLength: number = 30,
  expectedFeatureCount: number = 63
): DatasetValidationResult {
  const report: ValidationReport = {
    totalScanned: rawSamples.length,
    validSamples: 0,
    invalidSamples: 0,
    failures: {
      sequenceLengthMismatch: 0,
      featureCountMismatch: 0,
      nanOrInfinity: 0,
      missingLabel: 0,
      duplicateIds: 0,
      malformedMetadata: 0,
      outOfBounds: 0,
    },
    invalidDetails: [],
    datasetVersions: {},
  };

  const validSubset: RecordedSample[] = [];
  const idCounts = new Map<string, number>();

  // First pass: identify ID collisions (§1.4)
  for (const sample of rawSamples) {
    idCounts.set(sample.id, (idCounts.get(sample.id) || 0) + 1);
  }

  for (const sample of rawSamples) {
    const reasons: string[] = [];

    // Track version breakdown (§1.6)
    const version = sample.datasetVersion || "unknown";
    report.datasetVersions[version] = (report.datasetVersions[version] || 0) + 1;

    // 1. Missing/Empty label
    if (!sample.label || typeof sample.label !== "string" || sample.label.trim() === "") {
      reasons.push("Missing or empty label");
      report.failures.missingLabel++;
    }

    // 2. Duplicate IDs (§1.4)
    if (idCounts.get(sample.id)! > 1) {
      reasons.push(`Duplicate sample ID: ${sample.id}`);
      report.failures.duplicateIds++; // This might overcount a bit per sample, but gives distinct flags
    }

    // 3. Malformed metadata
    if (
      !sample.sequence ||
      !Array.isArray(sample.sequence) ||
      sample.sequence.length === 0 ||
      !["Left", "Right", "Unknown"].includes(sample.handedness) ||
      sample.source !== "webcam" ||
      !sample.datasetVersion ||
      !sample.timestamp ||
      typeof sample.sequenceLength !== "number" ||
      typeof sample.featureCount !== "number"
    ) {
      reasons.push("Malformed metadata or empty sequence");
      report.failures.malformedMetadata++;
    }

    // 3b. Normalization version check
    if (sample.normalizationVersion && sample.normalizationVersion !== NORMALIZATION_VERSION) {
      reasons.push(`Normalization version mismatch (expected ${NORMALIZATION_VERSION}, got ${sample.normalizationVersion})`);
      report.failures.malformedMetadata++;
    }

    // 4. Sequence length check
    if (sample.sequence && Array.isArray(sample.sequence)) {
      if (sample.sequence.length !== expectedSequenceLength) {
        reasons.push(`Sequence length mismatch (expected ${expectedSequenceLength}, got ${sample.sequence.length})`);
        report.failures.sequenceLengthMismatch++;
      } else if (sample.sequenceLength !== sample.sequence.length) {
        // Metadata disagrees with reality
        reasons.push(`Sequence length metadata (${sample.sequenceLength}) disagrees with array length (${sample.sequence.length})`);
        report.failures.malformedMetadata++;
      }

      // 5. Feature count & numeric checks
      let hasFeatureMismatch = false;
      let hasNanInfinity = false;
      let hasOutOfBounds = false;

      for (let f = 0; f < sample.sequence.length; f++) {
        const frame = sample.sequence[f];

        if (!Array.isArray(frame)) {
          if (!hasFeatureMismatch) {
            reasons.push(`Frame ${f} is not an array`);
            report.failures.featureCountMismatch++;
            hasFeatureMismatch = true;
          }
          continue;
        }

        if (frame.length !== expectedFeatureCount) {
          if (!hasFeatureMismatch) {
            reasons.push(`Frame feature count mismatch (expected ${expectedFeatureCount}, got ${frame.length})`);
            report.failures.featureCountMismatch++;
            hasFeatureMismatch = true;
          }
        } else if (sample.featureCount !== frame.length) {
          if (!hasFeatureMismatch) {
            reasons.push(`Feature count metadata (${sample.featureCount}) disagrees with frame length (${frame.length})`);
            report.failures.malformedMetadata++;
            hasFeatureMismatch = true;
          }
        }

        for (let i = 0; i < frame.length; i++) {
          const val = frame[i];

          if (val === null || val === undefined || typeof val !== "number" || Number.isNaN(val) || !Number.isFinite(val)) {
            if (!hasNanInfinity) {
              reasons.push("Sequence contains NaN, Infinity, or non-numeric values");
              report.failures.nanOrInfinity++;
              hasNanInfinity = true;
            }
          } else if (Math.abs(val) > 10.0) {
            // §1.5 Numeric range bounds (cited from src/dataCollection/validateSample.ts)
            if (!hasOutOfBounds) {
              reasons.push(`Out of bounds numeric value: ${val} (expected [-10.0, 10.0])`);
              report.failures.outOfBounds++;
              hasOutOfBounds = true;
            }
          }
        }
      }
    }

    if (reasons.length > 0) {
      report.invalidSamples++;
      report.invalidDetails.push({
        sampleId: sample.id,
        label: sample.label || "UNKNOWN",
        timestamp: sample.timestamp || "UNKNOWN",
        reasons,
      });
    } else {
      report.validSamples++;
      validSubset.push(sample);
    }
  }

  // Deduplicate failure counters for duplicate IDs so we don't overcount total issue impact unnecessarily
  report.failures.duplicateIds = [...idCounts.values()].filter((c) => c > 1).length;

  return {
    report,
    validSamples: validSubset,
  };
}
