import type { RecordedSample } from "../dataCollection/validateSample";
import { calculateSequenceDistance } from "../dataCollection/duplicateDetector";

export interface DuplicatePair {
  sampleId1: string;
  sampleId2: string;
  label1: string;
  label2: string;
  isCrossLabel: boolean;
  similarityScore: number;
}

export interface DatasetDuplicateReport {
  exactDuplicates: DuplicatePair[];
  potentialDuplicates: DuplicatePair[];
}

/**
 * Dataset Duplicate Scan (§3)
 * Scans valid samples for exact duplicates (identical sequences) and near-duplicates.
 * This is an offline, full-dataset operation (O(N^2) complexity).
 * Returns a report of flagged pairs. NEVER automatically removes them.
 */
export function scanForDuplicates(
  samples: RecordedSample[],
  potentialThreshold: number = 0.05
): DatasetDuplicateReport {
  const exactDuplicates: DuplicatePair[] = [];
  const potentialDuplicates: DuplicatePair[] = [];

  // Flatten sequences for quick exact match comparison
  const flattenCache = new Map<string, string>();
  for (const sample of samples) {
    // Stringify the flat array of all values for exact equality checking
    const flatStr = JSON.stringify(sample.sequence.flat());
    flattenCache.set(sample.id, flatStr);
  }

  for (let i = 0; i < samples.length; i++) {
    const s1 = samples[i];
    const flat1 = flattenCache.get(s1.id)!;

    for (let j = i + 1; j < samples.length; j++) {
      const s2 = samples[j];
      const isCrossLabel = s1.label !== s2.label;

      const flat2 = flattenCache.get(s2.id)!;

      // Exact Match (§3.1)
      if (flat1 === flat2) {
        exactDuplicates.push({
          sampleId1: s1.id,
          sampleId2: s2.id,
          label1: s1.label,
          label2: s2.label,
          isCrossLabel,
          similarityScore: 0,
        });
        continue;
      }

      // Potential Near Match (§3.2)
      // Recalibrated note: 0.05 is the session-scoped threshold. We use the same for consistency.
      const dist = calculateSequenceDistance(s1.sequence, s2.sequence);
      if (dist < potentialThreshold) {
        potentialDuplicates.push({
          sampleId1: s1.id,
          sampleId2: s2.id,
          label1: s1.label,
          label2: s2.label,
          isCrossLabel,
          similarityScore: dist,
        });
      }
    }
  }

  return {
    exactDuplicates,
    potentialDuplicates,
  };
}
