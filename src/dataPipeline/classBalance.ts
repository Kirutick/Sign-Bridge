import type { RecordedSample } from "../dataCollection/validateSample";

export interface ClassBalanceStats {
  label: string;
  count: number;
  isFlagged: boolean;
}

export interface ClassBalanceReport {
  classes: ClassBalanceStats[];
  minimum: number;
  maximum: number;
  mean: number;
  median: number;
  flaggingThreshold: number;
  flaggedCount: number;
}

/**
 * Class Balance computation (§2)
 * Computes sample distribution across classes on VALID samples only.
 * Threshold: Flags classes below 50% of the median class count.
 */
export function analyzeClassBalance(validSamples: RecordedSample[]): ClassBalanceReport {
  if (validSamples.length === 0) {
    return {
      classes: [],
      minimum: 0,
      maximum: 0,
      mean: 0,
      median: 0,
      flaggingThreshold: 0,
      flaggedCount: 0,
    };
  }

  const counts = new Map<string, number>();
  for (const sample of validSamples) {
    counts.set(sample.label, (counts.get(sample.label) || 0) + 1);
  }

  const classCounts = Array.from(counts.values()).sort((a, b) => a - b);
  
  const minimum = classCounts[0] || 0;
  const maximum = classCounts[classCounts.length - 1] || 0;
  const mean = classCounts.reduce((a, b) => a + b, 0) / classCounts.length;
  
  let median = 0;
  if (classCounts.length > 0) {
    const mid = Math.floor(classCounts.length / 2);
    median = classCounts.length % 2 !== 0 ? classCounts[mid] : (classCounts[mid - 1] + classCounts[mid]) / 2;
  }

  // Stated threshold (§2): 50% of the median sample count.
  const flaggingThreshold = Math.max(1, Math.round(median * 0.5));

  const classes: ClassBalanceStats[] = Array.from(counts.entries()).map(([label, count]) => ({
    label,
    count,
    isFlagged: count < flaggingThreshold,
  }));

  // Sort worst-covered-first
  classes.sort((a, b) => a.count - b.count);

  const flaggedCount = classes.filter((c) => c.isFlagged).length;

  return {
    classes,
    minimum,
    maximum,
    mean,
    median,
    flaggingThreshold,
    flaggedCount,
  };
}
