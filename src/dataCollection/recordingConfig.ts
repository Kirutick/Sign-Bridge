export interface RecordingConfig {
  label: string;
  sequenceLength: number; // default 30
  featuresPerFrame: number; // default 63
  countdownDuration: number; // default 3 (seconds)
  requiredHandCount: number; // default 1
  maxConsecutiveInvalidFrames: number; // default 5
  maxInvalidRatio: number; // default 0.20 (20%)
  maxSamples?: number;
  datasetVersion: string; // default "1.0.0"
}

export function createDefaultRecordingConfig(
  overrides?: Partial<RecordingConfig>
): RecordingConfig {
  return {
    label: overrides?.label || "UNLABELED",
    sequenceLength: overrides?.sequenceLength ?? 30,
    featuresPerFrame: overrides?.featuresPerFrame ?? 63,
    countdownDuration: overrides?.countdownDuration ?? 3,
    requiredHandCount: overrides?.requiredHandCount ?? 1,
    maxConsecutiveInvalidFrames: overrides?.maxConsecutiveInvalidFrames ?? 5,
    maxInvalidRatio: overrides?.maxInvalidRatio ?? 0.20,
    maxSamples: overrides?.maxSamples,
    datasetVersion: overrides?.datasetVersion || "1.0.0",
  };
}
