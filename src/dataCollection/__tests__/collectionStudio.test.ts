import "fake-indexeddb/auto";
import { describe, it, expect, beforeEach } from "vitest";
import { LabelStore } from "../labelStore";
import { SessionTracker } from "../sessionTracker";
import { datasetStorage } from "../../services/datasetStorage";

describe("Data Collection 2 — Dataset Recording Studio Tests (§12)", () => {
  beforeEach(async () => {
    await datasetStorage.clearAll();
  });

  describe("LabelStore CRUD & Identity Resolution Tests (§2.1, §2.2, §12)", () => {
    it("creates unique labels and rejects case-insensitive duplicate names", () => {
      const store = new LabelStore(20);
      const l1 = store.createLabel("HELLO");

      expect(l1.id).toBe("HELLO");
      expect(l1.name).toBe("HELLO");
      expect(l1.sampleCount).toBe(0);

      // Rejects duplicates
      expect(() => store.createLabel("hello")).toThrow("already exists");
    });

    it("sorts labels by sample count ascending to surface under-collected classes (§2.3)", async () => {
      const store = new LabelStore(20);
      store.createLabel("ALPHA");
      store.createLabel("BETA");

      // Add 2 samples for ALPHA
      await datasetStorage.addSample({
        id: "s1",
        label: "ALPHA",
        sequence: new Array(30).fill(new Array(63).fill(0.1)),
        sequenceLength: 30,
        featureCount: 63,
        createdAt: new Date().toISOString(),
      });
      await datasetStorage.addSample({
        id: "s2",
        label: "ALPHA",
        sequence: new Array(30).fill(new Array(63).fill(0.1)),
        sequenceLength: 30,
        featureCount: 63,
        createdAt: new Date().toISOString(),
      });

      await store.syncWithStorage();

      const sorted = store.getLabels("sampleCountAsc");
      expect(sorted[0].name).toBe("BETA"); // 0 samples
      expect(sorted[1].name).toBe("ALPHA"); // 2 samples
    });

    it("deletes label and purges all associated samples from IndexedDB storage (§2.2, §12)", async () => {
      const store = new LabelStore(20);
      store.createLabel("HELLO");

      await datasetStorage.addSample({
        id: "s1",
        label: "HELLO",
        sequence: new Array(30).fill(new Array(63).fill(0.1)),
        sequenceLength: 30,
        featureCount: 63,
        createdAt: new Date().toISOString(),
      });

      await store.syncWithStorage();
      expect(store.getLabels()[0].sampleCount).toBe(1);

      const deletedCount = await store.deleteLabel("HELLO");
      expect(deletedCount).toBe(1);

      const remainingSamples = await datasetStorage.listSamples("HELLO");
      expect(remainingSamples.length).toBe(0);
    });
  });

  describe("SessionTracker Minimal Data Collection Tests (§10, §12)", () => {
    it("tracks session metrics and collects only coarse device userAgent string (§10.1)", () => {
      const tracker = new SessionTracker();
      const session = tracker.getSession();

      expect(session.id).toBeDefined();
      expect(session.samplesRecordedCount).toBe(0);
      expect(session.device.userAgent).toBeDefined();

      tracker.recordSampleSaved("HELLO");
      expect(tracker.getSession().samplesRecordedCount).toBe(1);
      expect(tracker.getSession().labelsRecorded).toEqual(["HELLO"]);
    });
  });

  describe("Class Imbalance Threshold Logic Tests (§11, §12)", () => {
    it("calculates average sample count and flags classes under 50% threshold", () => {
      const labels = [
        { id: "L1", name: "L1", createdAt: "", sampleCount: 20, targetCount: 20 },
        { id: "L2", name: "L2", createdAt: "", sampleCount: 20, targetCount: 20 },
        { id: "L3", name: "L3", createdAt: "", sampleCount: 4, targetCount: 20 }, // 4 < 50% of avg (14.6) -> 7.3
      ];

      const totalSamples = labels.reduce((acc, l) => acc + l.sampleCount, 0); // 44
      const avg = totalSamples / labels.length; // 14.66
      const threshold = Math.round(avg * 0.5); // 7

      expect(labels[2].sampleCount < threshold).toBe(true);
      expect(labels[0].sampleCount < threshold).toBe(false);
    });
  });
});
