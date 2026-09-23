import type { Landmark, HandLandmarks, Handedness } from "./landmarks";

export type RawHandLandmarks = HandLandmarks;

/** Landmarks translated relative to the wrist (index 0) and scale-normalized by wrist-to-middle-MCP distance. */
export interface NormalizedHandLandmarks {
  landmarks: Landmark[]; // exactly 21 entries, wrist-relative & scaled
  scaleFactor: number; // Euclidean distance divisor used (wrist to middle finger MCP)
}

export type HandMode = "SINGLE_HAND" | "TWO_HAND";
export const SINGLE_HAND_FEATURES = 63;
export const TWO_HAND_FEATURES = 126;

/** Flat, ordered, fixed-length numeric vector. Always length 63 for SINGLE_HAND or 126 for TWO_HAND. */
export type LandmarkFeatureVector = number[];

/** One video frame's worth of processed hands (0, 1, or 2). */
export interface FrameFeatures {
  hands: Array<{
    vector: LandmarkFeatureVector;
    handedness: Handedness;
    handednessScore: number;
    scaleFactor: number;
    rawLandmarks: Landmark[];
    normalizedLandmarks: Landmark[];
  }>;
  timestampMs: number;
}

export type RejectionReason =
  | "MISSING_LANDMARKS"
  | "WRONG_LANDMARK_COUNT"
  | "NON_FINITE_VALUE"
  | "MALFORMED_INPUT"
  | "DEGENERATE_SCALE";

/** Result type used instead of throwing exceptions on expected frame failure modes. */
export type ProcessingResult =
  | { ok: true; features: FrameFeatures }
  | { ok: false; reason: RejectionReason; detail: string };
