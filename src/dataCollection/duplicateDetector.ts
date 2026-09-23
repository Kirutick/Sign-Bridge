import type { RecordedSample } from "./validateSample";

export interface DuplicateCheckResult {
  isDuplicate: boolean;
  mostSimilarId?: string;
  similarityScore?: number; // Mean Euclidean distance across frames (lower = more similar)
}

/**
 * Calculates mean per-frame Euclidean distance between two 30x63 landmark sequences.
 */
export function calculateSequenceDistance(seqA: number[][], seqB: number[][]): number {
  if (seqA.length !== seqB.length || seqA.length === 0) return Infinity;

  let totalDist = 0;
  const frameCount = seqA.length;

  for (let f = 0; f < frameCount; f++) {
    const frameA = seqA[f];
    const frameB = seqB[f];

    if (frameA.length !== frameB.length) return Infinity;

    let frameDistSum = 0;
    for (let i = 0; i < frameA.length; i++) {
      const diff = frameA[i] - frameB[i];
      frameDistSum += diff * diff;
    }

    totalDist += Math.sqrt(frameDistSum);
  }

  return totalDist / frameCount;
}

/**
 * Non-blocking Advisory Duplicate Detector (§9).
 * Compares new sample against existing samples for the same label.
 * Returns a warning flag if mean frame distance is below threshold (default 0.05).
 * NEVER automatically rejects or deletes samples — advisory only.
 */
export function checkForNearDuplicate(
  newSample: RecordedSample,
  existingSamples: RecordedSample[],
  threshold: number = 0.05
): DuplicateCheckResult {
  if (!existingSamples || existingSamples.length === 0) {
    return { isDuplicate: false };
  }

  const sameLabelSamples = existingSamples.filter((s) => s.label === newSample.label && s.id !== newSample.id);

  if (sameLabelSamples.length === 0) {
    return { isDuplicate: false };
  }

  let minDistance = Infinity;
  let mostSimilarId: string | undefined;

  for (const sample of sameLabelSamples) {
    const dist = calculateSequenceDistance(newSample.sequence, sample.sequence);
    if (dist < minDistance) {
      minDistance = dist;
      mostSimilarId = sample.id;
    }
  }

  if (minDistance < threshold) {
    return {
      isDuplicate: true,
      mostSimilarId,
      similarityScore: Math.round(minDistance * 10000) / 10000,
    };
  }

  return {
    isDuplicate: false,
    mostSimilarId,
    similarityScore: Math.round(minDistance * 10000) / 10000,
  };
}
