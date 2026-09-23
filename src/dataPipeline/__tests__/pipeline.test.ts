import { describe, it, expect, beforeEach } from "vitest";
import { validateDataset } from "../validateDataset";
import { analyzeClassBalance } from "../classBalance";
import { scanForDuplicates } from "../datasetDuplicateScan";
import { detectOutliers } from "../outlierDetection";
import { buildDatasetExport } from "../exportDataset";
import { validateImportDataset } from "../importDataset";
import { reviewStore } from "../reviewStore";
import type { RecordedSample } from "../../dataCollection/validateSample";

function createMockSample(id: string, label: string, modifier?: (seq: number[][]) => void): RecordedSample {
  const seq: number[][] = [];
  const seed = id.split("").reduce((acc, c) => acc + c.charCodeAt(0), 0);
  const baseFrame = new Array(63).fill(0).map((_, i) => Math.sin(i * 0.2 + seed) * 0.5);
  for (let f = 0; f < 30; f++) {
    const frame: number[] = baseFrame.map((v, i) => v + f * 0.005 + (i % 3) * 0.001);
    seq.push(frame);
  }
  
  if (modifier) modifier(seq);

  return {
    id,
    label,
    sequence: seq,
    sequenceLength: 30,
    featureCount: 63,
    timestamp: new Date().toISOString(),
    handedness: "Right",
    source: "webcam",
    datasetVersion: "1.0.0",
    normalizationVersion: "v1",
  };
}

describe("Data Collection 3 Pipeline", () => {
  beforeEach(() => {
    reviewStore.clear();
  });

  describe("§1 Structural Validation", () => {
    it("accepts valid samples", () => {
      const samples = [createMockSample("1", "A"), createMockSample("2", "B")];
      const result = validateDataset(samples);
      expect(result.report.validSamples).toBe(2);
      expect(result.report.invalidSamples).toBe(0);
      expect(result.validSamples.length).toBe(2);
    });

    it("rejects out of bounds numeric values", () => {
      const badSample = createMockSample("3", "C", (seq) => {
        seq[0][0] = 11.5; // > 10.0
      });
      const result = validateDataset([badSample]);
      expect(result.report.invalidSamples).toBe(1);
      expect(result.report.failures.outOfBounds).toBe(1);
    });

    it("rejects duplicate IDs correctly", () => {
      const s1 = createMockSample("4", "D");
      const s2 = createMockSample("4", "D"); // Same ID
      const result = validateDataset([s1, s2]);
      expect(result.report.invalidSamples).toBe(2);
      expect(result.report.failures.duplicateIds).toBe(1); // 1 collision detected
    });
  });

  describe("§2 Class Balance", () => {
    it("flags severely imbalanced classes", () => {
      const samples = [
        createMockSample("1", "A"), createMockSample("2", "A"), createMockSample("3", "A"),
        createMockSample("4", "B"), createMockSample("5", "B"), createMockSample("6", "B"),
        createMockSample("7", "C") // C has 1, A/B have 3. Median is 3. Threshold is 1.5. C is < 1.5.
      ];
      const report = analyzeClassBalance(samples);
      expect(report.median).toBe(3);
      expect(report.flaggingThreshold).toBe(2); // Math.max(1, Math.round(1.5))
      
      const c = report.classes.find((x) => x.label === "C");
      expect(c?.isFlagged).toBe(true);

      const a = report.classes.find((x) => x.label === "A");
      expect(a?.isFlagged).toBe(false);
    });
  });

  describe("§3 Duplicate Scan", () => {
    it("detects exact duplicates", () => {
      const s1 = createMockSample("1", "A");
      const s2 = JSON.parse(JSON.stringify(s1)); // Exact copy
      s2.id = "2";
      
      const report = scanForDuplicates([s1, s2]);
      expect(report.exactDuplicates.length).toBe(1);
      expect(report.potentialDuplicates.length).toBe(0);
    });

    it("detects potential near-duplicates", () => {
      const s1 = createMockSample("1", "A");
      const s2 = JSON.parse(JSON.stringify(s1));
      s2.id = "2";
      // Slightly perturb s2
      s2.sequence[0][0] += 0.01;
      
      const report = scanForDuplicates([s1, s2]);
      expect(report.exactDuplicates.length).toBe(0);
      expect(report.potentialDuplicates.length).toBe(1);
    });
  });

  describe("§4 Outlier Detection", () => {
    it("flags TELEPORTING_LANDMARK", () => {
      const s = createMockSample("1", "A", (seq) => {
        seq[1][0] = seq[0][0] + 3.0; // dx > 2.0 teleport
      });
      const flags = detectOutliers([s]);
      expect(flags[0].heuristic).toBe("TELEPORTING_LANDMARK");
    });

    it("flags CORRUPTED_SEQUENCE", () => {
      const s = createMockSample("1", "A", (seq) => {
        // Freeze first 20 frames
        for (let i = 1; i < 20; i++) {
          seq[i] = [...seq[0]];
        }
      });
      const flags = detectOutliers([s]);
      expect(flags[0].heuristic).toBe("CORRUPTED_SEQUENCE");
    });
  });

  describe("§6 Export & §8 Exclusion Policy", () => {
    it("determines class mapping alphabetically", () => {
      const samples = [createMockSample("1", "YES"), createMockSample("2", "HELLO"), createMockSample("3", "PLEASE")];
      const vResult = validateDataset(samples);
      const exported = buildDatasetExport(3, vResult, { exactDuplicates: [], potentialDuplicates: [] }, []);
      
      expect(exported.metadata.classMapping[0]).toBe("HELLO");
      expect(exported.metadata.classMapping[1]).toBe("PLEASE");
      expect(exported.metadata.classMapping[2]).toBe("YES");
    });

    it("excludes unresolved flags by default", () => {
      const s1 = createMockSample("1", "A");
      const s2 = createMockSample("2", "A");
      const vResult = validateDataset([s1, s2]);
      const duplicateReport = scanForDuplicates([s1, s2]); // Will flag s2 as duplicate of s1 if they match, but here they are random
      
      // Let's manually flag s1
      const outlierFlags = [{ sampleId: "1", label: "A", heuristic: "TEST", details: "test" }];
      
      const exported = buildDatasetExport(2, vResult, duplicateReport, outlierFlags);
      expect(exported.X.length).toBe(1); // s1 excluded
      expect(exported.metadata.validationSummary.outliersExcluded).toBe(1);
    });

    it("includes resolved flags if marked valid", () => {
      const s1 = createMockSample("1", "A");
      const vResult = validateDataset([s1]);
      const outlierFlags = [{ sampleId: "1", label: "A", heuristic: "TEST", details: "test" }];
      
      reviewStore.logAction("1", "MARK_VALID");
      
      const exported = buildDatasetExport(1, vResult, { exactDuplicates: [], potentialDuplicates: [] }, outlierFlags);
      expect(exported.X.length).toBe(1); // s1 kept
    });
  });

  describe("§9 Import Validation", () => {
    it("rejects malformed imports with itemized errors", () => {
      const badImport = {
        X: [[[1, 2]]], // Wrong shape
        y: [0],
        labels: { "0": "A" },
        metadata: { sampleCount: 1, sequenceLength: 30, featureCount: 63, classMapping: { 0: "A" } },
      };
      const result = validateImportDataset(badImport);
      expect(result.ok).toBe(false);
      expect(result.errors.length).toBeGreaterThan(0);
      expect(result.errors[0].reason).toMatch(/Sequence length mismatch/);
    });
  });
});
