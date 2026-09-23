import { describe, it, expect, beforeEach } from "vitest";
import { PredictionStabilizer, FrameInput } from "../predictionStabilizer";

describe("Phase 6 PredictionStabilizer State Machine Unit Tests (§12)", () => {
  let stabilizer: PredictionStabilizer;

  beforeEach(() => {
    stabilizer = new PredictionStabilizer({
      confidenceThreshold: 0.80,
      votingWindowFrames: 10,
      votingMajorityRatio: 0.70,
      stabilityDurationMs: 600,
      pauseDurationMs: 500,
      movementThreshold: 0.01,
      lowConfidenceGraceFrames: 5,
      duplicateSuppressionRequired: true,
    });
  });

  const makeFrame = (
    label: string | null,
    confidence: number | null,
    handDetected: boolean = true,
    timestampMs: number = 0
  ): FrameInput => ({
    label,
    confidence,
    handDetected,
    timestampMs,
  });

  it("1. Single sign, many identical frames -> commits exactly ONCE (§12.1)", () => {
    let committedDeltas: string[] = [];

    // Feed 60 consecutive identical frames of "HELLO" over 2000ms
    for (let f = 0; f < 60; f++) {
      const res = stabilizer.processFrame(makeFrame("HELLO", 0.92, true, f * 33));
      if (res.committedTextDelta) {
        committedDeltas.push(res.committedTextDelta);
      }
    }

    expect(committedDeltas).toEqual(["HELLO"]);
    expect(stabilizer.getFullCommittedText()).toEqual(["HELLO"]);
  });

  it("2. Repeated sign with pause in between -> commits TWICE ('HELLO HELLO') (§12.2)", () => {
    let committedDeltas: string[] = [];

    // Part 1: First performance of "HELLO" (0 to 1000ms)
    for (let f = 0; f < 30; f++) {
      const res = stabilizer.processFrame(makeFrame("HELLO", 0.95, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
    }
    expect(committedDeltas).toEqual(["HELLO"]);

    // Part 2: Qualifying Pause (no hand for 600ms / 20 frames)
    for (let f = 30; f < 50; f++) {
      stabilizer.processFrame(makeFrame(null, null, false, f * 33));
    }

    // Part 3: Second performance of "HELLO" (1700ms to 2700ms)
    for (let f = 50; f < 80; f++) {
      const res = stabilizer.processFrame(makeFrame("HELLO", 0.95, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
    }

    expect(committedDeltas).toEqual(["HELLO", "HELLO"]);
    expect(stabilizer.getFullCommittedText()).toEqual(["HELLO", "HELLO"]);
  });

  it("3. Different signs in sequence -> each commits once in exact order (§12.3)", () => {
    let committedDeltas: string[] = [];

    // Sign 1: HELLO
    for (let f = 0; f < 30; f++) {
      const res = stabilizer.processFrame(makeFrame("HELLO", 0.90, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
    }

    // Sign 2: THANK_YOU
    for (let f = 30; f < 60; f++) {
      const res = stabilizer.processFrame(makeFrame("THANK_YOU", 0.95, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
    }

    // Sign 3: YES
    for (let f = 60; f < 90; f++) {
      const res = stabilizer.processFrame(makeFrame("YES", 0.88, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
    }

    expect(committedDeltas).toEqual(["HELLO", "THANK_YOU", "YES"]);
    expect(stabilizer.getFullCommittedText()).toEqual(["HELLO", "THANK_YOU", "YES"]);
    expect(stabilizer.getFullCommittedString()).toBe("HELLO THANK_YOU YES");
  });

  it("4. Low confidence throughout -> zero commits, status reflects 'Uncertain' (§12.4)", () => {
    let committedDeltas: string[] = [];
    let lastStatus = "";

    // Feed 30 frames with confidence 0.50 (below 0.80 threshold)
    for (let f = 0; f < 30; f++) {
      const res = stabilizer.processFrame(makeFrame("YES", 0.50, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
      lastStatus = res.statusMessage;
    }

    expect(committedDeltas).toEqual([]);
    expect(stabilizer.getFullCommittedText()).toEqual([]);
    expect(lastStatus).toBe("Uncertain");
  });

  it("5. No hand detected -> state IDLE, status 'No hand detected', resumes on return (§12.5)", () => {
    const resNoHand = stabilizer.processFrame(makeFrame(null, null, false, 0));
    expect(resNoHand.state).toBe("IDLE");
    expect(resNoHand.statusMessage).toBe("No hand detected");

    // Hand returns
    const resReturn = stabilizer.processFrame(makeFrame("HELLO", 0.90, true, 33));
    expect(resReturn.state).toBe("UNSTABLE");
    expect(resReturn.statusMessage).toBe("Recognizing...");
  });

  it("6. Rapid label flicker below voting majority -> zero commits, stays UNSTABLE (§12.6)", () => {
    let committedDeltas: string[] = [];
    const labels = ["HELLO", "NO", "YES", "THANK_YOU"];

    // Rapidly alternating noisy labels every frame
    for (let f = 0; f < 30; f++) {
      const noisyLabel = labels[f % labels.length];
      const res = stabilizer.processFrame(makeFrame(noisyLabel, 0.85, true, f * 33));
      if (res.committedTextDelta) committedDeltas.push(res.committedTextDelta);
    }

    expect(committedDeltas).toEqual([]);
    expect(stabilizer.getFullCommittedText()).toEqual([]);
  });

  it("7. Backspace on empty committed text -> safe no-op (§12.7)", () => {
    expect(() => {
      stabilizer.backspaceCommittedText();
    }).not.toThrow();
    expect(stabilizer.getFullCommittedText()).toEqual([]);
  });

  it("8. Clear text -> empties committed tokens, stabilizer state unaffected (§12.8)", () => {
    // Commit one sign
    for (let f = 0; f < 30; f++) {
      stabilizer.processFrame(makeFrame("HELLO", 0.95, true, f * 33));
    }
    expect(stabilizer.getFullCommittedText()).toEqual(["HELLO"]);

    stabilizer.clearCommittedText();
    expect(stabilizer.getFullCommittedText()).toEqual([]);
  });

  it("9. Confidence threshold changed mid-stream -> subsequent frames respect new threshold (§12.9)", () => {
    // At default 0.80 threshold, 0.75 confidence is low confidence
    const res1 = stabilizer.processFrame(makeFrame("HELLO", 0.75, true, 0));
    expect(res1.state).toBe("LOW_CONFIDENCE");
    expect(res1.statusMessage).toBe("Uncertain");

    // Lower threshold to 0.70 live
    stabilizer.updateConfig({ confidenceThreshold: 0.70 });

    // Same 0.75 confidence is now valid!
    const res2 = stabilizer.processFrame(makeFrame("HELLO", 0.75, true, 33));
    expect(res2.state).toBe("UNSTABLE");
  });
});
