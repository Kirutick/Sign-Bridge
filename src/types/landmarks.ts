export interface Landmark {
  x: number; // normalized 0–1, image-relative
  y: number; // normalized 0–1, image-relative
  z: number; // normalized, relative depth (wrist-origin)
}

export type Handedness = "Left" | "Right" | "Unknown";

export interface HandLandmarks {
  landmarks: Landmark[]; // exactly 21 when a hand is detected
  handedness: Handedness;
  handednessScore: number; // MediaPipe's confidence for the label
}

export interface DetectionResult {
  hands: HandLandmarks[]; // 0, 1, or 2 entries
  timestampMs: number;
  inferenceTimeMs: number;
}

export type VisionDelegate = "GPU" | "CPU";

export type HandLandmarkerStatus =
  | "uninitialized"
  | "loading"
  | "ready"
  | "error";

export class HandLandmarkerError extends Error {
  constructor(
    public readonly code:
      | "WASM_LOAD_ERROR"
      | "MODEL_FETCH_ERROR"
      | "INITIALIZATION_ERROR"
      | "GPU_DELEGATE_ERROR",
    message: string,
    public readonly originalError?: unknown
  ) {
    super(message);
    this.name = "HandLandmarkerError";
  }
}
