import type { RecordingConfig } from "./recordingConfig";
import { NORMALIZATION_VERSION } from "../services/landmarkProcessor";

export interface RecordedSample {
  id: string;
  label: string;
  sequence: number[][]; // [30, 63]
  sequenceLength: number;
  featureCount: number;
  timestamp: string;
  handedness: "Left" | "Right" | "Unknown";
  source: "webcam";
  datasetVersion: string;
  normalizationVersion: string;
}

export interface ValidationResult {
  valid: boolean;
  reason?: string;
}

/**
 * Pre-Save Quality Gate (§7)
 * Runs an explicit, all-or-nothing validation pass before persisting any sample.
 */
export function validateSample(
  sample: RecordedSample,
  config: RecordingConfig
): ValidationResult {
  if (!sample) {
    return { valid: false, reason: "Sample object is null or undefined" };
  }

  if (sample.normalizationVersion !== NORMALIZATION_VERSION) {
    return { valid: false, reason: `Normalization version mismatch: expected ${NORMALIZATION_VERSION}, got ${sample.normalizationVersion}` };
  }

  if (!sample.label || typeof sample.label !== "string" || sample.label.trim().length === 0) {
    return { valid: false, reason: "Sample label is empty or invalid" };
  }

  if (!Array.isArray(sample.sequence)) {
    return { valid: false, reason: "Sample sequence is not an array" };
  }

  // 1. Frame Count Check (§7)
  if (sample.sequence.length !== config.sequenceLength) {
    return {
      valid: false,
      reason: `Sequence length mismatch: expected ${config.sequenceLength}, got ${sample.sequence.length}`,
    };
  }

  // 2. Feature Count & Value Numeric Gate (§7)
  for (let frameIdx = 0; frameIdx < sample.sequence.length; frameIdx++) {
    const frame = sample.sequence[frameIdx];

    if (!Array.isArray(frame)) {
      return { valid: false, reason: `Frame ${frameIdx} is not an array` };
    }

    if (frame.length !== config.featuresPerFrame) {
      return {
        valid: false,
        reason: `Frame ${frameIdx} feature count mismatch: expected ${config.featuresPerFrame}, got ${frame.length}`,
      };
    }

    for (let featIdx = 0; featIdx < frame.length; featIdx++) {
      const val = frame[featIdx];

      if (val === null || val === undefined) {
        return { valid: false, reason: `Frame ${frameIdx} feature ${featIdx} is null/undefined` };
      }

      if (typeof val !== "number" || Number.isNaN(val)) {
        return { valid: false, reason: `Frame ${frameIdx} feature ${featIdx} is NaN` };
      }

      if (!Number.isFinite(val)) {
        return { valid: false, reason: `Frame ${frameIdx} feature ${featIdx} is Infinity` };
      }

      // Sane normalized coordinate bounds check (relative-to-wrist scale normalized coordinates roughly [-5.0, 5.0])
      if (Math.abs(val) > 10.0) {
        return { valid: false, reason: `Frame ${frameIdx} feature ${featIdx} is out of sane bounds (${val})` };
      }
    }
  }

  return { valid: true };
}
