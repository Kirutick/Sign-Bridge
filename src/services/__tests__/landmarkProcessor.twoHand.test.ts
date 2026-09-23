import { describe, it, expect } from "vitest";
import {
  extractTwoHandFeatureVector,
  processFrameForMode,
  FRAME_FEATURES_BY_MODE,
  extractFeatureVector,
  normalizeHandLandmarks,
} from "../landmarkProcessor";
import type { RawHandLandmarks } from "../../types/features";
import type { Landmark } from "../../types/landmarks";

function createMockLandmarks(offset: number = 0): Landmark[] {
  const lms: Landmark[] = [];
  for (let i = 0; i < 21; i++) {
    lms.push({
      x: 0.1 * i + offset,
      y: 0.05 * i + offset,
      z: 0.01 * i,
    });
  }
  // Ensure wrist (0) to middle MCP (9) distance is valid non-degenerate
  lms[0] = { x: 0.2 + offset, y: 0.2 + offset, z: 0.0 };
  lms[9] = { x: 0.5 + offset, y: 0.6 + offset, z: 0.1 };
  return lms;
}

describe("Two-Hand Feature Extraction Contract", () => {
  it("defines correct feature counts per mode", () => {
    expect(FRAME_FEATURES_BY_MODE.SINGLE_HAND).toBe(63);
    expect(FRAME_FEATURES_BY_MODE.TWO_HAND).toBe(126);
  });

  it("extracts 126 features when both left and right hands are present", () => {
    const leftHand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.0),
      handedness: "Left",
      handednessScore: 0.99,
    };
    const rightHand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.3),
      handedness: "Right",
      handednessScore: 0.98,
    };

    const vector = extractTwoHandFeatureVector([leftHand, rightHand]);
    expect(vector).toHaveLength(126);

    // Indices 0..62 should match left hand normalized features
    const leftNorm = normalizeHandLandmarks(leftHand.landmarks);
    expect(leftNorm.ok).toBe(true);
    if (leftNorm.ok) {
      const expectedLeftVec = extractFeatureVector(leftNorm.normalized.landmarks);
      expect(vector.slice(0, 63)).toEqual(expectedLeftVec);
    }

    // Indices 63..125 should match right hand normalized features
    const rightNorm = normalizeHandLandmarks(rightHand.landmarks);
    expect(rightNorm.ok).toBe(true);
    if (rightNorm.ok) {
      const expectedRightVec = extractFeatureVector(rightNorm.normalized.landmarks);
      expect(vector.slice(63, 126)).toEqual(expectedRightVec);
    }
  });

  it("enforces deterministic handedness ordering regardless of input array order", () => {
    const leftHand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.0),
      handedness: "Left",
      handednessScore: 0.95,
    };
    const rightHand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.3),
      handedness: "Right",
      handednessScore: 0.95,
    };

    const vectorLeftFirst = extractTwoHandFeatureVector([leftHand, rightHand]);
    const vectorRightFirst = extractTwoHandFeatureVector([rightHand, leftHand]);

    // Vector must be IDENTICAL regardless of array ordering
    expect(vectorLeftFirst).toEqual(vectorRightFirst);
  });

  it("zero-pads missing right hand while keeping left hand at indices 0..62", () => {
    const leftHand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.0),
      handedness: "Left",
      handednessScore: 0.95,
    };

    const vector = extractTwoHandFeatureVector([leftHand]);
    expect(vector).toHaveLength(126);

    // Left hand non-zero
    expect(vector.slice(0, 63).some((v) => v !== 0)).toBe(true);
    // Right hand zero-padded
    expect(vector.slice(63, 126)).toEqual(new Array(63).fill(0));
  });

  it("zero-pads missing left hand while keeping right hand at indices 63..125", () => {
    const rightHand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.3),
      handedness: "Right",
      handednessScore: 0.95,
    };

    const vector = extractTwoHandFeatureVector([rightHand]);
    expect(vector).toHaveLength(126);

    // Left hand zero-padded
    expect(vector.slice(0, 63)).toEqual(new Array(63).fill(0));
    // Right hand non-zero
    expect(vector.slice(63, 126).some((v) => v !== 0)).toBe(true);
  });

  it("processFrameForMode works in SINGLE_HAND mode (63) and TWO_HAND mode (126)", () => {
    const hand: RawHandLandmarks = {
      landmarks: createMockLandmarks(0.0),
      handedness: "Right",
      handednessScore: 0.99,
    };

    const singleRes = processFrameForMode([hand], 1000, "SINGLE_HAND");
    expect(singleRes.ok).toBe(true);
    if (singleRes.ok) {
      expect(singleRes.vector).toHaveLength(63);
    }

    const twoHandRes = processFrameForMode([hand], 1000, "TWO_HAND");
    expect(twoHandRes.ok).toBe(true);
    if (twoHandRes.ok) {
      expect(twoHandRes.vector).toHaveLength(126);
    }
  });
});
