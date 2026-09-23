import type { HandLandmarks } from "../types/landmarks";

export interface HandSelectionResult {
  primaryHand: HandLandmarks | null;
  handCount: number;
  subState: "NO_HAND_DETECTED" | "HAND_DETECTED" | "MULTIPLE_HANDS_DETECTED";
}

/**
 * Calculates 2D bounding box area of a 21-landmark set.
 */
function calculateHandArea(landmarks: Array<{ x: number; y: number; z: number }>): number {
  if (!landmarks || landmarks.length === 0) return 0;

  let minX = Infinity, maxX = -Infinity;
  let minY = Infinity, maxY = -Infinity;

  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.y > maxY) maxY = lm.y;
  }

  const width = Math.max(0, maxX - minX);
  const height = Math.max(0, maxY - minY);
  return width * height;
}

/**
 * Pure function selecting primary hand when multiple hands are detected in frame (§6).
 * Selection policy: largest bounding box area (closer to camera) or highest handedness score.
 * Absolute Invariant: non-selected hand landmarks are NEVER merged or averaged into the output.
 */
export function selectPrimaryHand(hands: HandLandmarks[]): HandSelectionResult {
  if (!hands || !Array.isArray(hands) || hands.length === 0) {
    return {
      primaryHand: null,
      handCount: 0,
      subState: "NO_HAND_DETECTED",
    };
  }

  if (hands.length === 1) {
    return {
      primaryHand: hands[0],
      handCount: 1,
      subState: "HAND_DETECTED",
    };
  }

  // Multiple hands detected (>= 2)
  let bestHand: HandLandmarks = hands[0];
  let maxArea = calculateHandArea(hands[0].landmarks);

  for (let i = 1; i < hands.length; i++) {
    const area = calculateHandArea(hands[i].landmarks);
    // Prefer larger bounding box, or higher handedness score if areas are equal
    if (
      area > maxArea ||
      (Math.abs(area - maxArea) < 1e-4 && hands[i].handednessScore > bestHand.handednessScore)
    ) {
      maxArea = area;
      bestHand = hands[i];
    }
  }

  // Deep copy primary hand landmarks to guarantee zero landmark merging with secondary hands (§6)
  const isolatedPrimaryHand: HandLandmarks = {
    landmarks: bestHand.landmarks.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
    handedness: bestHand.handedness,
    handednessScore: bestHand.handednessScore,
  };

  return {
    primaryHand: isolatedPrimaryHand,
    handCount: hands.length,
    subState: "MULTIPLE_HANDS_DETECTED",
  };
}
