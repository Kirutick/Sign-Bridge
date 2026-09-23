import type { Landmark } from "../types/landmarks";
import type {
  RawHandLandmarks,
  NormalizedHandLandmarks,
  LandmarkFeatureVector,
  FrameFeatures,
  ProcessingResult,
  RejectionReason,
} from "../types/features";

/**
 * MediaPipe Canonical Hand Landmark Indices (0-20).
 * Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
 * 
 * 0:  WRIST
 * 1:  THUMB_CMC, 2: THUMB_MCP, 3: THUMB_IP, 4: THUMB_TIP
 * 5:  INDEX_FINGER_MCP, 6: INDEX_FINGER_PIP, 7: INDEX_FINGER_DIP, 8: INDEX_FINGER_TIP
 * 9:  MIDDLE_FINGER_MCP, 10: MIDDLE_FINGER_PIP, 11: MIDDLE_FINGER_DIP, 12: MIDDLE_FINGER_TIP
 * 13: RING_FINGER_MCP, 14: RING_FINGER_PIP, 15: RING_FINGER_DIP, 16: RING_FINGER_TIP
 * 17: PINKY_MCP, 18: PINKY_PIP, 19: PINKY_DIP, 20: PINKY_TIP
 */
export const WRIST_INDEX = 0;
export const MIDDLE_FINGER_MCP_INDEX = 9;

/**
 * Minimum Euclidean distance threshold between wrist and middle MCP joint.
 * Guards against zero/near-zero scale division during corrupt or degenerate hand detections.
 */
export const DEGENERATE_SCALE_EPSILON = 1e-6;

export interface ValidationSuccess {
  ok: true;
}

export interface ValidationFailure {
  ok: false;
  reason: RejectionReason;
  detail: string;
}

export type ValidationResult = ValidationSuccess | ValidationFailure;

/**
 * Validates raw hand landmarks array prior to normalization.
 * Checks for missing array, invalid length, missing numeric fields, NaN, or Infinity.
 */
export function validateRawLandmarks(hand: RawHandLandmarks): ValidationResult {
  if (!hand || !hand.landmarks || !Array.isArray(hand.landmarks)) {
    return {
      ok: false,
      reason: "MISSING_LANDMARKS",
      detail: "Hand object is missing or landmarks array is undefined/null.",
    };
  }

  if (hand.landmarks.length !== 21) {
    return {
      ok: false,
      reason: "WRONG_LANDMARK_COUNT",
      detail: `Expected exactly 21 landmarks, received ${hand.landmarks.length}.`,
    };
  }

  for (let i = 0; i < hand.landmarks.length; i++) {
    const lm = hand.landmarks[i];
    if (
      !lm ||
      typeof lm.x !== "number" ||
      typeof lm.y !== "number" ||
      typeof lm.z !== "number"
    ) {
      return {
        ok: false,
        reason: "MALFORMED_INPUT",
        detail: `Landmark at index ${i} is missing x, y, or z numeric properties.`,
      };
    }

    if (
      !Number.isFinite(lm.x) ||
      !Number.isFinite(lm.y) ||
      !Number.isFinite(lm.z)
    ) {
      return {
        ok: false,
        reason: "NON_FINITE_VALUE",
        detail: `Landmark at index ${i} contains NaN or Infinity values: x=${lm.x}, y=${lm.y}, z=${lm.z}.`,
      };
    }
  }

  return { ok: true };
}

export interface NormalizationSuccess {
  ok: true;
  normalized: NormalizedHandLandmarks;
}

export type NormalizationResult = NormalizationSuccess | ValidationFailure;

export const NORMALIZATION_VERSION = "v1";

/**
 * Translates landmarks relative to wrist (index 0) and scales by wrist-to-middle-MCP (index 9) distance.
 * 
 * Step 1 — Translate to wrist-relative coordinates:
 * tx_i = x_i - x_wrist, ty_i = y_i - y_wrist, tz_i = z_i - z_wrist
 * 
 * Step 2 — Scale by wrist-to-middle-finger-MCP distance:
 * scale = sqrt(tx9^2 + ty9^2 + tz9^2)
 * 
 * Why wrist-to-middle-MCP? That joint undergoes minimal articulation movement relative to fingertips,
 * making it a stable anatomical scale reference that avoids bounding-box diagonal jitter.
 */
export function normalizeHandLandmarks(landmarks: Landmark[]): NormalizationResult {
  const wrist = landmarks[WRIST_INDEX];
  const middleMcp = landmarks[MIDDLE_FINGER_MCP_INDEX];

  const tx9 = middleMcp.x - wrist.x;
  const ty9 = middleMcp.y - wrist.y;
  const tz9 = middleMcp.z - wrist.z;

  const scaleFactor = Math.hypot(tx9, ty9, tz9);

  if (scaleFactor < DEGENERATE_SCALE_EPSILON) {
    return {
      ok: false,
      reason: "DEGENERATE_SCALE",
      detail: `Scale factor (${scaleFactor}) is smaller than epsilon threshold (${DEGENERATE_SCALE_EPSILON}).`,
    };
  }

  const normalized: Landmark[] = new Array(21);

  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    normalized[i] = {
      x: (lm.x - wrist.x) / scaleFactor,
      y: (lm.y - wrist.y) / scaleFactor,
      z: (lm.z - wrist.z) / scaleFactor,
    };
  }

  return {
    ok: true,
    normalized: {
      landmarks: normalized,
      scaleFactor,
    },
  };
}

/**
 * Extracts a flat, 63-value feature vector from normalized 21-point hand landmarks.
 * Ordering: [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20].
 */
export function extractFeatureVector(landmarks: Landmark[]): LandmarkFeatureVector {
  const vector: number[] = new Array(63);
  let idx = 0;

  for (let i = 0; i < 21; i++) {
    const lm = landmarks[i];
    vector[idx++] = lm.x;
    vector[idx++] = lm.y;
    vector[idx++] = lm.z;
  }

  if (vector.length !== 63) {
    throw new Error(`Invalid feature vector length generated: ${vector.length}. Expected 63.`);
  }

  return vector;
}

/**
 * Multi-Hand Processing Policy Choice (§7):
 * We process each detected hand independently. If one hand fails validation or normalization,
 * we drop only that hand while keeping valid hands in FrameFeatures.
 * If ALL hands fail or 0 hands are present in the frame, we return an overall ProcessingResult rejection.
 */
export function processFrame(
  hands: RawHandLandmarks[],
  timestampMs: number
): ProcessingResult {
  if (!hands || !Array.isArray(hands) || hands.length === 0) {
    return {
      ok: false,
      reason: "MISSING_LANDMARKS",
      detail: "No hands present in input frame.",
    };
  }

  const processedHands: FrameFeatures["hands"] = [];
  let lastFailure: ValidationFailure = {
    ok: false,
    reason: "MISSING_LANDMARKS",
    detail: "No valid hands found in frame.",
  };

  for (let i = 0; i < hands.length; i++) {
    const rawHand = hands[i];

    // Step 1: Validation
    const valRes = validateRawLandmarks(rawHand);
    if (!valRes.ok) {
      lastFailure = valRes;
      continue;
    }

    // Step 2: Normalization
    const normRes = normalizeHandLandmarks(rawHand.landmarks);
    if (!normRes.ok) {
      lastFailure = normRes;
      continue;
    }

    // Step 3: Vector Extraction (63 floats)
    const vector = extractFeatureVector(normRes.normalized.landmarks);

    processedHands.push({
      vector,
      handedness: rawHand.handedness,
      handednessScore: rawHand.handednessScore,
      scaleFactor: normRes.normalized.scaleFactor,
      rawLandmarks: rawHand.landmarks,
      normalizedLandmarks: normRes.normalized.landmarks,
    });
  }

  if (processedHands.length === 0) {
    return lastFailure;
  }

  return {
    ok: true,
    features: {
      hands: processedHands,
      timestampMs,
    },
  };
}

export const FRAME_FEATURES_BY_MODE = {
  SINGLE_HAND: 63,
  TWO_HAND: 126,
} as const;

/**
 * Deterministic Two-Hand Feature Vector Extraction (126 values).
 * Layout:
 * Indices 0..62: Normalized Left Hand (or 63 zeros if missing/invalid)
 * Indices 63..125: Normalized Right Hand (or 63 zeros if missing/invalid)
 * 
 * Invariants:
 * - Deterministic handedness ordering (Left always first, Right always second).
 * - Never silently swaps Left and Right hands.
 * - Zero-padded deterministically when a hand is absent.
 */
export function extractTwoHandFeatureVector(hands: RawHandLandmarks[]): number[] {
  const vector: number[] = new Array(126).fill(0.0);

  if (!hands || !Array.isArray(hands)) {
    return vector;
  }

  let leftHand: RawHandLandmarks | null = null;
  let rightHand: RawHandLandmarks | null = null;

  for (const h of hands) {
    if (h.handedness === "Left" && !leftHand) {
      leftHand = h;
    } else if (h.handedness === "Right" && !rightHand) {
      rightHand = h;
    }
  }

  // If handedness is missing or ambiguous, order deterministically by x-position of wrist (wrist.x < 0.5 is left)
  if (!leftHand && !rightHand && hands.length > 0) {
    if (hands.length === 1) {
      rightHand = hands[0]; // fallback default
    } else {
      const sorted = [...hands].sort((a, b) => (a.landmarks[0]?.x ?? 0) - (b.landmarks[0]?.x ?? 0));
      leftHand = sorted[0];
      rightHand = sorted[1];
    }
  }

  // Normalize Left Hand -> indices 0..62
  if (leftHand && validateRawLandmarks(leftHand).ok) {
    const norm = normalizeHandLandmarks(leftHand.landmarks);
    if (norm.ok) {
      const leftVec = extractFeatureVector(norm.normalized.landmarks);
      for (let i = 0; i < 63; i++) {
        vector[i] = leftVec[i];
      }
    }
  }

  // Normalize Right Hand -> indices 63..125
  if (rightHand && validateRawLandmarks(rightHand).ok) {
    const norm = normalizeHandLandmarks(rightHand.landmarks);
    if (norm.ok) {
      const rightVec = extractFeatureVector(norm.normalized.landmarks);
      for (let i = 0; i < 63; i++) {
        vector[63 + i] = rightVec[i];
      }
    }
  }

  return vector;
}

/**
 * Generalized frame processor supporting both SINGLE_HAND (63) and TWO_HAND (126) modes.
 */
export function processFrameForMode(
  hands: RawHandLandmarks[],
  timestampMs: number,
  mode: "SINGLE_HAND" | "TWO_HAND" = "SINGLE_HAND"
): { ok: true; vector: number[]; timestampMs: number } | ValidationFailure {
  if (!hands || !Array.isArray(hands) || hands.length === 0) {
    return {
      ok: false,
      reason: "MISSING_LANDMARKS",
      detail: "No hands present in input frame.",
    };
  }

  if (mode === "TWO_HAND") {
    const vector = extractTwoHandFeatureVector(hands);
    // Ensure at least one hand had non-zero values
    const hasAnySignal = vector.some((v) => v !== 0.0);
    if (!hasAnySignal) {
      return {
        ok: false,
        reason: "MISSING_LANDMARKS",
        detail: "No valid hands could be normalized in two-hand mode.",
      };
    }
    return {
      ok: true,
      vector,
      timestampMs,
    };
  }

  // SINGLE_HAND mode
  const singleResult = processFrame(hands, timestampMs);
  if (!singleResult.ok) {
    return singleResult;
  }

  return {
    ok: true,
    vector: singleResult.features.hands[0].vector,
    timestampMs,
  };
}

