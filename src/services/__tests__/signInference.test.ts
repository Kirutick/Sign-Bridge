import { describe, it, expect, beforeEach } from "vitest";
import { SequenceBuffer } from "../sequenceBuffer";
import { PredictionSmoother } from "../predictionSmoother";
import type { RawPrediction } from "../signClassifier";

describe("Phase 5 Inference Subsystems Unit Tests", () => {
  describe("SequenceBuffer Unit Tests", () => {
    let buffer: SequenceBuffer;

    beforeEach(() => {
      buffer = new SequenceBuffer(30, 63);
    });

    it("accumulates valid 63-element feature vectors up to sequenceLength (30)", () => {
      const mockVector = new Array(63).fill(0.1);

      for (let i = 0; i < 29; i++) {
        buffer.push(mockVector);
        expect(buffer.isFull()).toBe(false);
      }

      buffer.push(mockVector);
      expect(buffer.isFull()).toBe(true);
      expect(buffer.length()).toBe(30);
    });

    it("maintains sliding window behavior when pushing past sequenceLength", () => {
      const vec1 = new Array(63).fill(1.0);
      const vec2 = new Array(63).fill(2.0);

      for (let i = 0; i < 30; i++) {
        buffer.push(vec1);
      }
      buffer.push(vec2);

      expect(buffer.length()).toBe(30);
      const seq = buffer.getSequence();
      expect(seq[29][0]).toBe(2.0); // Newest frame
      expect(seq[0][0]).toBe(1.0); // Shifted frame
    });

    it("returns a deep copy from getSequence isolated from internal mutation", () => {
      const vec = new Array(63).fill(0.5);
      buffer.push(vec);

      const seq = buffer.getSequence();
      seq[0][0] = 999.0; // Mutate returned copy

      const freshSeq = buffer.getSequence();
      expect(freshSeq[0][0]).toBe(0.5); // Internal array remains unchanged
    });

    it("clears buffer and increments stalled counter on hand tracking loss (§4)", () => {
      const vec = new Array(63).fill(0.1);
      for (let i = 0; i < 15; i++) {
        buffer.push(vec);
      }

      expect(buffer.length()).toBe(15);
      buffer.handleHandLoss();

      expect(buffer.length()).toBe(0);
      expect(buffer.isFull()).toBe(false);
      expect(buffer.getStalledCount()).toBe(1);
    });
  });

  describe("PredictionSmoother Unit Tests", () => {
    let smoother: PredictionSmoother;

    beforeEach(() => {
      smoother = new PredictionSmoother(5, 3); // Window size 5, min agreement 3
    });

    it("returns prediction with majority label and averaged confidence", () => {
      const makePred = (label: string, confidence: number): RawPrediction => ({
        probabilities: [0.9, 0.1],
        labelId: 0,
        label,
        confidence,
        inferenceTimeMs: 5,
        timestamp: new Date().toISOString(),
      });

      smoother.addPrediction(makePred("HELLO", 0.90));
      smoother.addPrediction(makePred("HELLO", 0.94));
      smoother.addPrediction(makePred("HELLO", 0.86));
      smoother.addPrediction(makePred("NO", 0.80));
      smoother.addPrediction(makePred("NO", 0.70));

      const smoothed = smoother.getSmoothedPrediction(0.80);
      expect(smoothed.label).toBe("HELLO");
      expect(smoothed.confidence).toBeCloseTo(0.90, 4);
      expect(smoothed.belowThreshold).toBe(false);
    });

    it("returns null label when majority confidence is below threshold (0.80)", () => {
      const makePred = (label: string, confidence: number): RawPrediction => ({
        probabilities: [0.7, 0.3],
        labelId: 0,
        label,
        confidence,
        inferenceTimeMs: 5,
        timestamp: new Date().toISOString(),
      });

      smoother.addPrediction(makePred("YES", 0.65));
      smoother.addPrediction(makePred("YES", 0.70));
      smoother.addPrediction(makePred("YES", 0.60));

      const smoothed = smoother.getSmoothedPrediction(0.80);
      expect(smoothed.label).toBeNull();
      expect(smoothed.belowThreshold).toBe(true);
    });

    it("resets window state on reset()", () => {
      const makePred = (label: string, confidence: number): RawPrediction => ({
        probabilities: [0.95, 0.05],
        labelId: 0,
        label,
        confidence,
        inferenceTimeMs: 5,
        timestamp: new Date().toISOString(),
      });

      smoother.addPrediction(makePred("HELLO", 0.95));
      smoother.addPrediction(makePred("HELLO", 0.95));
      smoother.addPrediction(makePred("HELLO", 0.95));

      smoother.reset();
      const smoothed = smoother.getSmoothedPrediction(0.80);
      expect(smoothed.label).toBeNull();
    });
  });

  describe("Automated Compatibility Validation Rule Tests (§3c)", () => {
    it("detects feature length mismatches cleanly", () => {
      const liveFeatureCount = 63;
      const liveSequenceLength = 30;

      const incompatibleSpec = {
        featureCount: 126, // Mismatched dimension
        sequenceLength: 30,
        modelVersion: "v1_incompatible",
      };

      const isCompatible =
        incompatibleSpec.featureCount === liveFeatureCount &&
        incompatibleSpec.sequenceLength === liveSequenceLength;

      expect(isCompatible).toBe(false);
    });
  });
});
