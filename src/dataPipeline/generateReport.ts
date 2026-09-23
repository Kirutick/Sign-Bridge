import type { DatasetValidationResult } from "./validateDataset";
import type { ClassBalanceReport } from "./classBalance";
import type { DatasetDuplicateReport } from "./datasetDuplicateScan";
import type { OutlierFlag } from "./outlierDetection";

/**
 * Report Generation (§10)
 * Compiles a human-readable text report merging validation, balance, duplicate, and outlier results.
 */
export function generateDatasetReport(
  validationResult: DatasetValidationResult,
  balanceReport: ClassBalanceReport,
  duplicateReport: DatasetDuplicateReport,
  outliers: OutlierFlag[]
): string {
  const { report } = validationResult;
  
  const lines: string[] = [];

  lines.push("========================================");
  lines.push("      DATASET VALIDATION REPORT         ");
  lines.push("========================================");
  lines.push("");

  lines.push("--- 1. OVERALL STATS ---");
  lines.push(`Total Scanned:      ${report.totalScanned}`);
  lines.push(`Structurally Valid: ${report.validSamples}`);
  lines.push(`Structurally Invalid: ${report.invalidSamples}`);
  lines.push("");

  lines.push("--- 2. STRUCTURAL FAILURES ---");
  if (report.invalidSamples === 0) {
    lines.push("None! All samples are structurally sound.");
  } else {
    lines.push(`Missing Label:           ${report.failures.missingLabel}`);
    lines.push(`Duplicate IDs:           ${report.failures.duplicateIds}`);
    lines.push(`Malformed Metadata:      ${report.failures.malformedMetadata}`);
    lines.push(`Sequence Length Mismatch:${report.failures.sequenceLengthMismatch}`);
    lines.push(`Feature Count Mismatch:  ${report.failures.featureCountMismatch}`);
    lines.push(`NaN / Infinity Values:   ${report.failures.nanOrInfinity}`);
    lines.push(`Out of Bounds Values:    ${report.failures.outOfBounds}`);
  }
  lines.push("");

  lines.push("--- 3. CLASS BALANCE ---");
  if (balanceReport.classes.length === 0) {
    lines.push("No valid classes to report.");
  } else {
    lines.push(`Median Class Count: ${balanceReport.median}`);
    lines.push(`Flag Threshold (< 50%): < ${balanceReport.flaggingThreshold}`);
    lines.push(`Flagged Classes:    ${balanceReport.flaggedCount}`);
    lines.push("");
    lines.push("Class Distribution:");
    balanceReport.classes.forEach((c) => {
      const flagStr = c.isFlagged ? " [IMBALANCED]" : "";
      lines.push(`  - ${c.label.padEnd(20, " ")}: ${c.count.toString().padStart(4, " ")}${flagStr}`);
    });
  }
  lines.push("");

  lines.push("--- 4. DUPLICATES ---");
  lines.push(`Exact Duplicate Pairs:     ${duplicateReport.exactDuplicates.length}`);
  lines.push(`Potential Duplicate Pairs: ${duplicateReport.potentialDuplicates.length}`);
  if (duplicateReport.exactDuplicates.length > 0) {
    lines.push("\nSample Exact Duplicates:");
    duplicateReport.exactDuplicates.slice(0, 5).forEach((d) => {
      lines.push(`  - ${d.sampleId1} (${d.label1}) == ${d.sampleId2} (${d.label2})`);
    });
  }
  lines.push("");

  lines.push("--- 5. OUTLIERS ---");
  lines.push(`Total Flagged Samples: ${outliers.length}`);
  if (outliers.length > 0) {
    const heuristicCounts = new Map<string, number>();
    outliers.forEach((o) => {
      heuristicCounts.set(o.heuristic, (heuristicCounts.get(o.heuristic) || 0) + 1);
    });
    
    heuristicCounts.forEach((count, heuristic) => {
      lines.push(`  ${heuristic}: ${count}`);
    });

    lines.push("\nSample Outliers:");
    outliers.slice(0, 5).forEach((o) => {
      lines.push(`  - ${o.sampleId} (${o.label}): [${o.heuristic}] ${o.details}`);
    });
  }
  lines.push("");

  lines.push("========================================");

  return lines.join("\n");
}
