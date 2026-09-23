import type { DatasetValidationResult } from "./validateDataset";
import type { OutlierFlag } from "./outlierDetection";
import type { DatasetDuplicateReport } from "./datasetDuplicateScan";
import { reviewStore } from "./reviewStore";
import { NORMALIZATION_VERSION } from "../services/landmarkProcessor";

export interface DatasetExportMetadata {
  exportVersion: string;
  createdAt: string;
  featureCount: number;
  sequenceLength: number;
  sampleCount: number;
  rawSampleCount: number;
  classMapping: Record<number, string>;
  sourceDatasetVersions: string[];
  normalizationVersion: string;
  validationSummary: {
    invalidExcluded: number;
    duplicatesExcluded: number;
    outliersExcluded: number;
    manuallyExcluded: number;
  };
}

export interface ExportedDatasetJson {
  X: number[][][]; // shape (N, 30, 63)
  y: number[];     // shape (N)
  labels: Record<string, string>; // "0": "HELLO"
  metadata: DatasetExportMetadata;
}

export interface ExportConfiguration {
  includeUnresolvedFlags?: boolean; // Default false (conservative)
}

/**
 * Computes deterministic alphabetical mapping for labels (§7).
 */
export function computeClassMapping(labels: Set<string>): Record<number, string> {
  const sorted = Array.from(labels).sort((a, b) => a.localeCompare(b));
  const mapping: Record<number, string> = {};
  sorted.forEach((label, index) => {
    mapping[index] = label;
  });
  return mapping;
}

/**
 * Applies exclusion policies and formats dataset for export (§6, §8).
 */
export function buildDatasetExport(
  rawSampleCount: number,
  validationResult: DatasetValidationResult,
  duplicateReport: DatasetDuplicateReport,
  outlierFlags: OutlierFlag[],
  config: ExportConfiguration = {}
): ExportedDatasetJson {
  const includeUnresolved = !!config.includeUnresolvedFlags;
  
  const validSamples = validationResult.validSamples;
  const invalidExcluded = rawSampleCount - validSamples.length;

  let duplicatesExcluded = 0;
  let outliersExcluded = 0;
  let manuallyExcluded = 0;

  // Build sets of flagged samples for O(1) lookup
  const duplicateSet = new Set<string>();
  duplicateReport.exactDuplicates.forEach((d) => {
    duplicateSet.add(d.sampleId1);
    duplicateSet.add(d.sampleId2);
  });
  duplicateReport.potentialDuplicates.forEach((d) => {
    duplicateSet.add(d.sampleId1);
    duplicateSet.add(d.sampleId2);
  });

  const outlierSet = new Set<string>();
  outlierFlags.forEach((o) => outlierSet.add(o.sampleId));

  // Determine which labels actually survive to be mapped
  const uniqueLabels = new Set<string>();
  validSamples.forEach((s) => uniqueLabels.add(s.label));

  const classMapping = computeClassMapping(uniqueLabels);
  // Reverse lookup for mapping
  const labelToId: Record<string, number> = {};
  for (const [id, label] of Object.entries(classMapping)) {
    labelToId[label] = Number(id);
  }

  const stringLabels: Record<string, string> = {};
  for (const [id, label] of Object.entries(classMapping)) {
    stringLabels[id] = label;
  }

  const X: number[][][] = [];
  const y: number[] = [];

  const manuallyExcludedIds = reviewStore.getExcludedSampleIds();
  const manuallyValidIds = reviewStore.getValidSampleIds();

  const sourceVersions = new Set<string>();

  for (const sample of validSamples) {
    if (sample.datasetVersion) {
      sourceVersions.add(sample.datasetVersion);
    }

    if (manuallyExcludedIds.has(sample.id)) {
      manuallyExcluded++;
      continue;
    }

    const isDuplicate = duplicateSet.has(sample.id);
    const isOutlier = outlierSet.has(sample.id);

    // Is it flagged and NOT manually marked valid?
    if ((isDuplicate || isOutlier) && !manuallyValidIds.has(sample.id)) {
      if (!includeUnresolved) {
        if (isDuplicate) duplicatesExcluded++;
        else if (isOutlier) outliersExcluded++;
        continue;
      }
    }

    // Export structural assertion check (§8.1)
    if (sample.sequence.length !== 30 || sample.sequence[0].length !== 63) {
      throw new Error(`Export assertion failed: Sample ${sample.id} has incorrect shape [${sample.sequence.length}, ${sample.sequence[0]?.length}]`);
    }

    X.push(sample.sequence);
    y.push(labelToId[sample.label]);
  }

  // Final structural assertion (§8.1)
  if (X.length !== y.length) {
    throw new Error(`Export assertion failed: X length (${X.length}) does not match y length (${y.length})`);
  }

  const metadata: DatasetExportMetadata = {
    exportVersion: "1.0.0", // §6 Distinct from datasetVersion
    createdAt: new Date().toISOString(),
    featureCount: 63,
    sequenceLength: 30,
    sampleCount: X.length,
    rawSampleCount,
    classMapping,
    sourceDatasetVersions: Array.from(sourceVersions),
    normalizationVersion: NORMALIZATION_VERSION,
    validationSummary: {
      invalidExcluded,
      duplicatesExcluded,
      outliersExcluded,
      manuallyExcluded,
    },
  };

  return {
    X,
    y,
    labels: stringLabels,
    metadata,
  };
}
