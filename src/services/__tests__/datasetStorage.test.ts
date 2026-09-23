import "fake-indexeddb/auto";
import { describe, it, expect, beforeEach } from "vitest";
import { datasetStorage } from "../datasetStorage";
import type { DatasetSample, DatasetExport } from "../../types/dataset";

function createMockSample(
  label = "HELLO",
  performerId = "p1",
  seqLen = 30,
  featCount = 63
): DatasetSample {
  const sequence: number[][] = [];
  for (let f = 0; f < seqLen; f++) {
    const frame: number[] = new Array(featCount).fill(0).map((_, i) => (f + i) * 0.01);
    sequence.push(frame);
  }

  return {
    id: "", // Will be assigned
    label,
    sequence,
    sequenceLength: seqLen,
    featureCount: featCount,
    createdAt: new Date().toISOString(),
    handedness: ["Right"],
    metadata: {
      performerId,
      notes: "Unit test mock sample",
      fps: 30,
      appVersion: "0.3.0",
    },
  };
}

describe("datasetStorage Service Unit Tests", () => {
  beforeEach(async () => {
    await datasetStorage.clearAll();
  });

  it("adds and retrieves a DatasetSample from IndexedDB", async () => {
    const mock = createMockSample("THANK_YOU");
    const id = await datasetStorage.addSample(mock);

    expect(id).toBeTruthy();

    const retrieved = await datasetStorage.getSample(id);
    expect(retrieved).not.toBeNull();
    expect(retrieved?.label).toBe("THANK_YOU");
    expect(retrieved?.sequenceLength).toBe(30);
    expect(retrieved?.sequence).toHaveLength(30);
    expect(retrieved?.sequence[0]).toHaveLength(63);
  });

  it("lists samples filtered by label", async () => {
    await datasetStorage.addSample(createMockSample("HELLO", "p1"));
    await datasetStorage.addSample(createMockSample("HELLO", "p2"));
    await datasetStorage.addSample(createMockSample("BYE", "p1"));

    const helloSamples = await datasetStorage.listSamples("HELLO");
    expect(helloSamples).toHaveLength(2);

    const byeSamples = await datasetStorage.listSamples("BYE");
    expect(byeSamples).toHaveLength(1);

    const allSamples = await datasetStorage.listSamples();
    expect(allSamples).toHaveLength(3);
  });

  it("calculates label metrics and performer counts via listLabels", async () => {
    await datasetStorage.addSample(createMockSample("HELLO", "p1"));
    await datasetStorage.addSample(createMockSample("HELLO", "p1"));
    await datasetStorage.addSample(createMockSample("HELLO", "p2"));
    await datasetStorage.addSample(createMockSample("YES", "p1"));

    const labels = await datasetStorage.listLabels();
    expect(labels).toHaveLength(2);

    const helloLabel = labels.find((l) => l.label === "HELLO");
    expect(helloLabel?.count).toBe(3);
    expect(helloLabel?.performersCount).toBe(2); // p1 & p2

    const yesLabel = labels.find((l) => l.label === "YES");
    expect(yesLabel?.count).toBe(1);
    expect(yesLabel?.performersCount).toBe(1);
  });

  it("deletes individual sample and entire label group", async () => {
    const id1 = await datasetStorage.addSample(createMockSample("YES"));
    await datasetStorage.addSample(createMockSample("YES"));
    await datasetStorage.addSample(createMockSample("NO"));

    await datasetStorage.deleteSample(id1);
    const remainingYes = await datasetStorage.listSamples("YES");
    expect(remainingYes).toHaveLength(1);

    await datasetStorage.deleteLabel("YES");
    const clearedYes = await datasetStorage.listSamples("YES");
    expect(clearedYes).toHaveLength(0);

    const noSamples = await datasetStorage.listSamples("NO");
    expect(noSamples).toHaveLength(1);
  });

  it("clears entire dataset storage", async () => {
    await datasetStorage.addSample(createMockSample("A"));
    await datasetStorage.addSample(createMockSample("B"));

    await datasetStorage.clearAll();
    const all = await datasetStorage.listSamples();
    expect(all).toHaveLength(0);
  });

  it("exports dataset to version 1 JSON schema format", async () => {
    await datasetStorage.addSample(createMockSample("HELLO"));
    await datasetStorage.addSample(createMockSample("WORLD"));

    const exportData = await datasetStorage.exportAll();
    expect(exportData.version).toBe(1);
    expect(exportData.featureCount).toBe(63);
    expect(exportData.sequenceLength).toBe(30);
    expect(exportData.samples).toHaveLength(2);
    expect(exportData.exportedAt).toBeTruthy();
  });

  it("validates and imports dataset with multi-stage error checking", async () => {
    const validSample = createMockSample("VALID");
    validSample.id = "sample-1";

    const exportData: DatasetExport = {
      version: 1,
      featureCount: 63,
      sequenceLength: 30,
      exportedAt: new Date().toISOString(),
      samples: [validSample],
    };

    const res = await datasetStorage.importAll(exportData);
    expect(res.success).toBe(true);
    expect(res.importedCount).toBe(1);
    expect(res.skippedCount).toBe(0);

    const imported = await datasetStorage.listSamples("VALID");
    expect(imported).toHaveLength(1);
    expect(imported[0].id).not.toBe("sample-1"); // Regenerated UUID
  });

  it("rejects malformed samples during import", async () => {
    const badLengthSample = createMockSample("BAD");
    badLengthSample.sequence = badLengthSample.sequence.slice(0, 15); // Only 15 frames instead of 30

    const exportData: DatasetExport = {
      version: 1,
      featureCount: 63,
      sequenceLength: 30,
      exportedAt: new Date().toISOString(),
      samples: [badLengthSample],
    };

    const res = await datasetStorage.importAll(exportData, { skipInvalid: true });
    expect(res.importedCount).toBe(0);
    expect(res.skippedCount).toBe(1);
    expect(res.errors).toHaveLength(1);
    expect(res.errors[0].message).toContain("sequence length (15) does not match");
  });
});
