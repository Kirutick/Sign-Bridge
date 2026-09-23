import type { RecordedSample } from "../dataCollection/validateSample";

export interface OutlierFlag {
  sampleId: string;
  label: string;
  heuristic: string;
  details: string;
}

/**
 * Heuristic constants (§4.1)
 */
const FROZEN_VARIANCE_THRESHOLD = 1e-5; // Near-zero variance
const TELEPORT_DISTANCE_THRESHOLD = 2.0; // 2.0 relative units (wrist-to-MCP is 1.0)
const STATISTICAL_STD_DEV_MULTIPLIER = 4.0;
const CORRUPTED_DUPLICATE_FRAME_RATIO = 0.5; // If > 50% of frames in a sample are near-identical

export function detectOutliers(samples: RecordedSample[]): OutlierFlag[] {
  const flags: OutlierFlag[] = [];
  if (samples.length === 0) return flags;

  const seqLen = samples[0].sequenceLength || 30;
  const featCount = samples[0].featureCount || 63;

  // 1. Calculate Dataset-Wide Feature Means & StdDevs for STATISTICAL_OUTLIER
  const featureSums = new Array(featCount).fill(0);
  const featureSquares = new Array(featCount).fill(0);
  let totalFrames = 0;

  for (const sample of samples) {
    for (const frame of sample.sequence) {
      for (let i = 0; i < featCount; i++) {
        const val = frame[i];
        featureSums[i] += val;
        featureSquares[i] += val * val;
      }
      totalFrames++;
    }
  }

  const featureMeans = new Array(featCount).fill(0);
  const featureStdDevs = new Array(featCount).fill(0);

  for (let i = 0; i < featCount; i++) {
    const mean = featureSums[i] / totalFrames;
    const variance = (featureSquares[i] / totalFrames) - (mean * mean);
    featureMeans[i] = mean;
    featureStdDevs[i] = variance > 0 ? Math.sqrt(variance) : 0;
  }

  // 2. Scan per sample — structural checks first, statistical last
  for (const sample of samples) {
    const seq = sample.sequence;
    let flagged = false;

    // Check TELEPORTING_LANDMARK first (structural: frame-to-frame jump)
    // Frame-to-frame displacement > 2.0 relative units for any single landmark (x,y,z tuple)
    for (let f = 1; f < seqLen && !flagged; f++) {
      const prev = seq[f - 1];
      const curr = seq[f];

      for (let lm = 0; lm < 21; lm++) {
        const idx = lm * 3;
        const dx = curr[idx] - prev[idx];
        const dy = curr[idx + 1] - prev[idx + 1];
        const dz = curr[idx + 2] - prev[idx + 2];
        const dist = Math.hypot(dx, dy, dz);

        if (dist > TELEPORT_DISTANCE_THRESHOLD) {
          flags.push({
            sampleId: sample.id,
            label: sample.label,
            heuristic: "TELEPORTING_LANDMARK",
            details: `Landmark ${lm} jumped by ${dist.toFixed(2)} units between frames ${f - 1} and ${f}`,
          });
          flagged = true;
          break;
        }
      }
    }

    if (flagged) continue;

    // Check CORRUPTED_SEQUENCE (structural: stuck/frozen frames)
    let duplicateFrameCount = 0;
    for (let f = 1; f < seqLen; f++) {
      const prev = seq[f - 1];
      const curr = seq[f];
      let isNearIdentical = true;
      for (let i = 0; i < featCount; i++) {
        if (Math.abs(curr[i] - prev[i]) > 1e-4) {
          isNearIdentical = false;
          break;
        }
      }
      if (isNearIdentical) duplicateFrameCount++;
    }

    if (duplicateFrameCount / seqLen > CORRUPTED_DUPLICATE_FRAME_RATIO) {
      flags.push({
        sampleId: sample.id,
        label: sample.label,
        heuristic: "CORRUPTED_SEQUENCE",
        details: `${duplicateFrameCount} out of ${seqLen} frames are near-identical stuck frames`,
      });
      flagged = true;
      continue;
    }

    if (flagged) continue;

    // Check FROZEN_HAND (structural: variance collapse mid-sequence)
    const half = Math.floor(seqLen / 2);

    const calcFrameVariance = (start: number, end: number) => {
      let sum = 0, sqSum = 0;
      let count = 0;
      for (let f = start; f < end; f++) {
        // take mean displacement of wrist for simplicity
        const dx = seq[f][0];
        sum += dx;
        sqSum += dx * dx;
        count++;
      }
      const mean = sum / count;
      return (sqSum / count) - (mean * mean);
    };

    const varHalf1 = calcFrameVariance(0, half);
    const varHalf2 = calcFrameVariance(half, seqLen);

    if (varHalf1 > 0.01 && varHalf2 < FROZEN_VARIANCE_THRESHOLD) {
      flags.push({
        sampleId: sample.id,
        label: sample.label,
        heuristic: "FROZEN_HAND",
        details: `Variance dropped from ${varHalf1.toFixed(5)} to ${varHalf2.toFixed(8)}, indicating lock-on to static artifact`,
      });
      flagged = true;
    }

    if (flagged) continue;

    // Check STATISTICAL_OUTLIER last (statistical: dataset-wide deviation)
    for (let f = 0; f < seqLen && !flagged; f++) {
      const frame = seq[f];
      for (let i = 0; i < featCount; i++) {
        const val = frame[i];
        const mean = featureMeans[i];
        const stddev = featureStdDevs[i];
        if (stddev > 0 && Math.abs(val - mean) > STATISTICAL_STD_DEV_MULTIPLIER * stddev) {
          flags.push({
            sampleId: sample.id,
            label: sample.label,
            heuristic: "STATISTICAL_OUTLIER",
            details: `Feature ${i} value ${val.toFixed(3)} is > ${STATISTICAL_STD_DEV_MULTIPLIER} stddevs from mean ${mean.toFixed(3)}`,
          });
          flagged = true;
          break;
        }
      }
    }
  }

  return flags;
}
