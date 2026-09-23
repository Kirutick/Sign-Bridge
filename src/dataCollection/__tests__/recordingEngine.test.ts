import { describe, it, expect } from "vitest";
import { RecordingSession } from "../recordingStateMachine";
import { validateSample, RecordedSample } from "../validateSample";
import { checkForNearDuplicate } from "../duplicateDetector";
import type { HandLandmarks } from "../../types/landmarks";

describe("Data Collection 1 — Landmark Sequence Recording Engine Tests (§10)", () => {
  const makeMockHand = (xOffset: number = 0.0): HandLandmarks => {
    // Generate valid 21 3D landmarks
    const landmarks = new Array(21).fill(null).map((_, i) => ({
      x: 0.1 + i * 0.02 + xOffset,
      y: 0.2 + i * 0.01,
      z: 0.0,
    }));
    return {
      landmarks,
      handedness: "Right",
      handednessScore: 0.98,
    };
  };

  describe("State Transition Table Tests (§2.2, §10)", () => {
    it("starts in IDLE state", () => {
      const session = new RecordingSession({ label: "HELLO" });
      expect(session.getState().state).toBe("IDLE");
      expect(session.getState().validFrameCount).toBe(0);
    });

    it("transitions IDLE -> COUNTDOWN -> RECORDING", () => {
      const session = new RecordingSession({ label: "HELLO", countdownDuration: 2 });
      
      expect(session.startCountdown()).toBe(true);
      expect(session.getState().state).toBe("COUNTDOWN");
      expect(session.getState().countdownRemaining).toBe(2);

      session.tickCountdown();
      expect(session.getState().countdownRemaining).toBe(1);

      session.tickCountdown();
      expect(session.getState().state).toBe("RECORDING");
      expect(session.getState().countdownRemaining).toBe(0);
    });

    it("ignores onFrame calls when not in RECORDING state (Invariant §2.2)", () => {
      const session = new RecordingSession({ label: "HELLO" });
      const mockHand = makeMockHand();

      // In IDLE state
      session.onFrame({ hands: [mockHand], timestampMs: 100 });
      expect(session.getState().validFrameCount).toBe(0);

      // In COUNTDOWN state
      session.startCountdown();
      session.onFrame({ hands: [mockHand], timestampMs: 200 });
      expect(session.getState().validFrameCount).toBe(0);
    });
  });

  describe("Successful 30-Frame Sequence Recording (§10)", () => {
    it("collects 30 valid synthetic frames and transitions to COMPLETED with valid sample shape [30, 63]", () => {
      const session = new RecordingSession({ label: "HELLO", countdownDuration: 0, sequenceLength: 30 });
      session.startCountdown();
      session.tickCountdown(); // State: RECORDING

      const mockHand = makeMockHand();

      for (let i = 0; i < 30; i++) {
        session.onFrame({ hands: [mockHand], timestampMs: i * 33 });
      }

      const state = session.getState();
      expect(state.state).toBe("COMPLETED");
      expect(state.validFrameCount).toBe(30);
      expect(state.sample).not.toBeNull();

      const sample = state.sample!;
      expect(sample.label).toBe("HELLO");
      expect(sample.sequence.length).toBe(30);
      expect(sample.sequence[0].length).toBe(63);
      expect(sample.handedness).toBe("Right");
      expect(sample.datasetVersion).toBe("1.0.0");

      // Verify Pre-Save Quality Gate Pass (§7)
      const gateRes = validateSample(sample, {
        label: "HELLO",
        sequenceLength: 30,
        featuresPerFrame: 63,
        countdownDuration: 3,
        requiredHandCount: 1,
        maxConsecutiveInvalidFrames: 5,
        maxInvalidRatio: 0.2,
        datasetVersion: "1.0.0",
      });

      expect(gateRes.valid).toBe(true);
    });
  });

  describe("Invalid Frame Cancellation Policy Tests (§4.2, §10)", () => {
    it("cancels recording and discards buffer when 5 consecutive invalid frames occur", () => {
      const session = new RecordingSession({
        label: "HELLO",
        sequenceLength: 30,
        countdownDuration: 0,
        maxConsecutiveInvalidFrames: 5,
        maxInvalidRatio: 0.50,
      });

      session.startCountdown();
      session.tickCountdown(); // RECORDING

      const mockHand = makeMockHand();

      // Push 10 valid frames
      for (let i = 0; i < 10; i++) {
        session.onFrame({ hands: [mockHand], timestampMs: i * 33 });
      }

      expect(session.getState().validFrameCount).toBe(10);

      // Inject 5 consecutive missing-hand frames
      for (let i = 0; i < 5; i++) {
        session.onFrame({ hands: [], timestampMs: (10 + i) * 33 });
      }

      const state = session.getState();
      expect(state.state).toBe("CANCELLED");
      expect(state.cancellationReason).toContain("hand lost for too long");
      expect(state.sample).toBeNull(); // Buffer discarded cleanly (§4.2)
    });
  });

  describe("Malformed Data & Quality Gate Assertions (§7, §10)", () => {
    it("detects NaN or Infinity values in validateSample quality gate", () => {
      const validSeq = new Array(30).fill(new Array(63).fill(0.1));
      const nanSeq = JSON.parse(JSON.stringify(validSeq));
      nanSeq[5][10] = NaN;

      const sampleNaN: RecordedSample = {
        id: "s1",
        label: "TEST",
        sequence: nanSeq,
        sequenceLength: 30,
        featureCount: 63,
        timestamp: new Date().toISOString(),
        handedness: "Right",
        source: "webcam",
        datasetVersion: "1.0.0",
        normalizationVersion: "v1",
      };

      const resNaN = validateSample(sampleNaN, {
        label: "TEST",
        sequenceLength: 30,
        featuresPerFrame: 63,
        countdownDuration: 3,
        requiredHandCount: 1,
        maxConsecutiveInvalidFrames: 5,
        maxInvalidRatio: 0.2,
        datasetVersion: "1.0.0",
      });

      expect(resNaN.valid).toBe(false);
      expect(resNaN.reason).toContain("NaN");
    });
  });

  describe("User Cancellation Tests (§10)", () => {
    it("handles user cancellation mid-COUNTDOWN cleanly", () => {
      const session = new RecordingSession({ label: "HELLO" });
      session.startCountdown();
      expect(session.getState().state).toBe("COUNTDOWN");

      session.cancel("User clicked cancel");
      expect(session.getState().state).toBe("CANCELLED");
      expect(session.getState().cancellationReason).toBe("User clicked cancel");
    });
  });

  describe("Advisory Duplicate Detector Tests (§9, §10)", () => {
    it("emits non-blocking duplicate warning for near-identical samples", () => {
      const seqA = new Array(30).fill(null).map(() => new Array(63).fill(0.5));
      const seqB = new Array(30).fill(null).map(() => new Array(63).fill(0.501)); // Near-identical

      const sampleA: RecordedSample = {
        id: "sa",
        label: "HELLO",
        sequence: seqA,
        sequenceLength: 30,
        featureCount: 63,
        timestamp: new Date().toISOString(),
        handedness: "Right",
        source: "webcam",
        datasetVersion: "1.0.0",
        normalizationVersion: "v1",
      };

      const sampleB: RecordedSample = {
        id: "sb",
        label: "HELLO",
        sequence: seqB,
        sequenceLength: 30,
        featureCount: 63,
        timestamp: new Date().toISOString(),
        handedness: "Right",
        source: "webcam",
        datasetVersion: "1.0.0",
        normalizationVersion: "v1",
      };

      const dupRes = checkForNearDuplicate(sampleB, [sampleA], 0.05);
      expect(dupRes.isDuplicate).toBe(true);
      expect(dupRes.mostSimilarId).toBe("sa");
    });
  });
});
