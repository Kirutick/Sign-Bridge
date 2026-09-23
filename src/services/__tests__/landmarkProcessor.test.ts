import { describe, it, expect } from "vitest";
import {
  processFrame,
} from "../landmarkProcessor";
import type { Landmark, HandLandmarks } from "../../types/landmarks";
import type { RawHandLandmarks } from "../../types/features";

/** Helper to generate a realistic mock 21-landmark array */
function createMockHandLandmarks(
  wristX = 0.5,
  wristY = 0.5,
  wristZ = 0.0,
  scale = 0.1
): Landmark[] {
  const landmarks: Landmark[] = [];
  for (let i = 0; i < 21; i++) {
    // Index 9 is Middle Finger MCP, placed 'scale' units away from wrist
    if (i === 9) {
      landmarks.push({
        x: wristX + scale,
        y: wristY,
        z: wristZ,
      });
    } else {
      landmarks.push({
        x: wristX + (i * 0.01),
        y: wristY + (i * 0.015),
        z: wristZ + (i * 0.005),
      });
    }
  }
  // Wrist is index 0
  landmarks[0] = { x: wristX, y: wristY, z: wristZ };
  return landmarks;
}

function createMockRawHand(handedness: "Left" | "Right" = "Right"): RawHandLandmarks {
  return {
    landmarks: createMockHandLandmarks(),
    handedness,
    handednessScore: 0.98,
  };
}

describe("landmarkProcessor Unit Tests", () => {
  it("validates valid 21-landmark hand resulting in a 63-value vector", () => {
    const rawHand = createMockRawHand("Right");
    const result = processFrame([rawHand], 1000);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.features.hands).toHaveLength(1);
      const handFeature = result.features.hands[0];
      expect(handFeature.vector).toHaveLength(63);
      expect(handFeature.handedness).toBe("Right");
      expect(handFeature.handednessScore).toBe(0.98);
    }
  });

  it("normalizes wrist landmark (index 0) to exactly (0, 0, 0)", () => {
    const rawHand = createMockRawHand();
    const result = processFrame([rawHand], 1000);

    expect(result.ok).toBe(true);
    if (result.ok) {
      const vector = result.features.hands[0].vector;
      expect(vector[0]).toBeCloseTo(0, 5); // x0
      expect(vector[1]).toBeCloseTo(0, 5); // y0
      expect(vector[2]).toBeCloseTo(0, 5); // z0
    }
  });

  it("ensures deterministic processing across multiple executions", () => {
    const rawHand = createMockRawHand();
    const res1 = processFrame([rawHand], 2000);
    const res2 = processFrame([rawHand], 2000);

    expect(res1).toEqual(res2);
  });

  it("rejects missing landmarks array with MISSING_LANDMARKS reason", () => {
    // @ts-expect-error Testing invalid runtime input
    const res1 = processFrame(null, 1000);
    expect(res1.ok).toBe(false);
    if (!res1.ok) {
      expect(res1.reason).toBe("MISSING_LANDMARKS");
    }

    const invalidHand = { landmarks: null, handedness: "Right", handednessScore: 1 } as unknown as HandLandmarks;
    const res2 = processFrame([invalidHand], 1000);
    expect(res2.ok).toBe(false);
    if (!res2.ok) {
      expect(res2.reason).toBe("MISSING_LANDMARKS");
    }
  });

  it("rejects invalid landmark count with WRONG_LANDMARK_COUNT reason", () => {
    const shortLandmarks = createMockHandLandmarks().slice(0, 19); // 19 landmarks
    const hand: RawHandLandmarks = { landmarks: shortLandmarks, handedness: "Left", handednessScore: 0.9 };
    const res = processFrame([hand], 1000);

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("WRONG_LANDMARK_COUNT");
    }
  });

  it("rejects NaN in coordinate with NON_FINITE_VALUE reason", () => {
    const lms = createMockHandLandmarks();
    lms[5].x = NaN;
    const hand: RawHandLandmarks = { landmarks: lms, handedness: "Right", handednessScore: 0.9 };
    const res = processFrame([hand], 1000);

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("NON_FINITE_VALUE");
    }
  });

  it("rejects Infinity in coordinate with NON_FINITE_VALUE reason", () => {
    const lms = createMockHandLandmarks();
    lms[12].z = Infinity;
    const hand: RawHandLandmarks = { landmarks: lms, handedness: "Right", handednessScore: 0.9 };
    const res = processFrame([hand], 1000);

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("NON_FINITE_VALUE");
    }
  });

  it("rejects malformed entry with MALFORMED_INPUT reason", () => {
    const lms = createMockHandLandmarks() as unknown as Record<string, unknown>[];
    lms[3] = { x: "0.5", y: 0.5, z: 0.1 };
    const hand = { landmarks: lms, handedness: "Right", handednessScore: 0.9 } as unknown as RawHandLandmarks;
    const res = processFrame([hand], 1000);

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("MALFORMED_INPUT");
    }
  });

  it("rejects degenerate scale (wrist ≈ middle-MCP) without producing Infinity or NaN", () => {
    const lms = createMockHandLandmarks();
    // Set middle-MCP (index 9) identical to wrist (index 0)
    lms[9] = { ...lms[0] };
    const hand: RawHandLandmarks = { landmarks: lms, handedness: "Left", handednessScore: 0.9 };
    const res = processFrame([hand], 1000);

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.reason).toBe("DEGENERATE_SCALE");
    }
  });

  it("guarantees all 63 output values are finite across random valid inputs", () => {
    for (let testRun = 0; testRun < 50; testRun++) {
      const wx = Math.random();
      const wy = Math.random();
      const wz = Math.random();
      const scale = 0.05 + Math.random() * 0.3; // scale guaranteed > EPSILON

      const lms = createMockHandLandmarks(wx, wy, wz, scale);
      const hand: RawHandLandmarks = { landmarks: lms, handedness: "Right", handednessScore: 0.95 };
      const res = processFrame([hand], 1000);

      expect(res.ok).toBe(true);
      if (res.ok) {
        const vec = res.features.hands[0].vector;
        expect(vec).toHaveLength(63);
        for (const val of vec) {
          expect(Number.isFinite(val)).toBe(true);
          expect(Number.isNaN(val)).toBe(false);
        }
      }
    }
  });

  it("processes two-hand input into independent, correctly ordered 63-value vectors", () => {
    const leftHand = createMockRawHand("Left");
    const rightHand = createMockRawHand("Right");

    const res = processFrame([leftHand, rightHand], 1500);

    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.features.hands).toHaveLength(2);

      const h1 = res.features.hands[0];
      const h2 = res.features.hands[1];

      expect(h1.handedness).toBe("Left");
      expect(h1.vector).toHaveLength(63);

      expect(h2.handedness).toBe("Right");
      expect(h2.vector).toHaveLength(63);
    }
  });
});
