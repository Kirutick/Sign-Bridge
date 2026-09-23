import {
  FilesetResolver,
  HandLandmarker as MediaPipeHandLandmarker,
  HandLandmarkerResult as MediaPipeResult,
} from "@mediapipe/tasks-vision";
import {
  DetectionResult,
  Handedness,
  HandLandmarks,
  HandLandmarkerError,
  VisionDelegate,
} from "../types/landmarks";

/**
 * Service encapsulating MediaPipe HandLandmarker initialized for video streaming.
 * Handles WASM loading, GPU-to-CPU fallback, frame inference, and result type mapping.
 */

let landmarkerInstance: MediaPipeHandLandmarker | null = null;
let imageLandmarkerInstance: MediaPipeHandLandmarker | null = null;
let currentDelegate: VisionDelegate = "GPU";
let lastMonotonicTimestamp = -1;

// Local asset path and fallback CDN URL for hand landmarker model
const LOCAL_MODEL_PATH = "/models/hand_landmarker.task";
const CDN_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const WASM_CDN_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

export interface InitOptions {
  forceCpu?: boolean;
  modelUrl?: string;
}

/**
 * Initializes the MediaPipe HandLandmarker.
 * Configured for 2 hands in VIDEO mode, with GPU acceleration attempt and CPU fallback.
 */
export async function initHandLandmarker(options: InitOptions = {}): Promise<VisionDelegate> {
  // Dispose existing instance if present
  if (landmarkerInstance) {
    disposeHandLandmarker();
  }
  lastMonotonicTimestamp = -1;

  let visionFiles: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
  try {
    // Load MediaPipe WASM binaries
    visionFiles = await FilesetResolver.forVisionTasks(WASM_CDN_URL);
  } catch (err) {
    throw new HandLandmarkerError(
      "WASM_LOAD_ERROR",
      "Failed to load MediaPipe WASM vision binaries from CDN.",
      err
    );
  }

  // Attempt GPU delegate first, with CPU fallback if GPU fails or forceCpu is requested
  const targetDelegate: VisionDelegate = options.forceCpu ? "CPU" : "GPU";

  try {
    landmarkerInstance = await createInstanceWithModel(
      visionFiles,
      targetDelegate,
      options.modelUrl
    );
    currentDelegate = targetDelegate;
    return currentDelegate;
  } catch (firstError) {
    if (targetDelegate === "GPU") {
      // GPU failed: fallback to CPU delegate
      console.warn("GPU delegate initialization failed. Retrying with CPU delegate...", firstError);
      try {
        landmarkerInstance = await createInstanceWithModel(
          visionFiles,
          "CPU",
          options.modelUrl
        );
        currentDelegate = "CPU";
        return currentDelegate;
      } catch (cpuError) {
        throw new HandLandmarkerError(
          "INITIALIZATION_ERROR",
          "Failed to initialize HandLandmarker on both GPU and CPU delegates.",
          cpuError
        );
      }
    } else {
      throw new HandLandmarkerError(
        "INITIALIZATION_ERROR",
        "Failed to initialize HandLandmarker with CPU delegate.",
        firstError
      );
    }
  }
}

/**
 * Initializes a separate MediaPipe HandLandmarker in IMAGE mode.
 * IMAGE mode does not require monotonic timestamps and is suitable
 * for processing individual uploaded images or video frames extracted offline.
 * Call this once after initHandLandmarker() if image inference is needed.
 */
export async function initImageHandLandmarker(options: InitOptions = {}): Promise<void> {
  if (imageLandmarkerInstance) {
    try { imageLandmarkerInstance.close(); } catch (_) {}
    imageLandmarkerInstance = null;
  }

  let visionFiles: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;
  try {
    visionFiles = await FilesetResolver.forVisionTasks(WASM_CDN_URL);
  } catch (err) {
    throw new HandLandmarkerError("WASM_LOAD_ERROR", "Failed to load MediaPipe WASM for image mode.", err);
  }

  const delegate: VisionDelegate = options.forceCpu ? "CPU" : "GPU";

  const imgOptions = {
    baseOptions: {
      modelAssetPath: options.modelUrl || LOCAL_MODEL_PATH,
      delegate,
    },
    runningMode: "IMAGE" as const,
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };

  try {
    imageLandmarkerInstance = await MediaPipeHandLandmarker.createFromOptions(visionFiles, imgOptions);
  } catch (err) {
    // CPU fallback
    try {
      imgOptions.baseOptions.modelAssetPath = CDN_MODEL_URL;
      imgOptions.baseOptions.delegate = "CPU";
      imageLandmarkerInstance = await MediaPipeHandLandmarker.createFromOptions(visionFiles, imgOptions);
    } catch (cpuErr) {
      throw new HandLandmarkerError("INITIALIZATION_ERROR", "Failed to init IMAGE mode HandLandmarker.", cpuErr);
    }
  }
}

/**
 * Performs hand landmark detection on a single image element (IMAGE running mode).
 * Does NOT require monotonic timestamps. Safe to call multiple times on different images.
 * Returns empty hands array if no landmarker is initialized — call initImageHandLandmarker() first.
 */
export function detectForImage(
  imageElement: HTMLImageElement | HTMLCanvasElement | ImageBitmap
): DetectionResult {
  if (!imageLandmarkerInstance) {
    return { hands: [], timestampMs: 0, inferenceTimeMs: 0 };
  }

  try {
    const startTime = performance.now();
    const rawResult = imageLandmarkerInstance.detect(imageElement as HTMLImageElement);
    const inferenceTimeMs = performance.now() - startTime;
    return mapRawResultToDetectionResult(rawResult, 0, inferenceTimeMs);
  } catch (err) {
    console.warn("MediaPipe detectForImage error:", err);
    return { hands: [], timestampMs: 0, inferenceTimeMs: 0 };
  }
}

/**
 * Disposes the IMAGE mode HandLandmarker instance.
 */
export function disposeImageHandLandmarker(): void {
  if (imageLandmarkerInstance) {
    try { imageLandmarkerInstance.close(); } catch (_) {}
    imageLandmarkerInstance = null;
  }
}

/**
 * Returns whether the IMAGE mode HandLandmarker is ready.
 */
export function isImageLandmarkerReady(): boolean {
  return imageLandmarkerInstance !== null;
}

/**
 * Helper to construct the HandLandmarker using local model file with CDN fallback.
 */
async function createInstanceWithModel(
  visionFiles: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>,
  delegate: VisionDelegate,
  customModelUrl?: string
): Promise<MediaPipeHandLandmarker> {
  const primaryUrl = customModelUrl || LOCAL_MODEL_PATH;
  
  const options = {
    baseOptions: {
      modelAssetPath: primaryUrl,
      delegate: delegate,
    },
    runningMode: "VIDEO" as const,
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };

  try {
    return await MediaPipeHandLandmarker.createFromOptions(visionFiles, options);
  } catch (localError) {
    // If using local model path and it fails, attempt fallback to CDN URL
    if (primaryUrl === LOCAL_MODEL_PATH) {
      console.warn("Local model file fetch failed. Falling back to official MediaPipe CDN...", localError);
      options.baseOptions.modelAssetPath = CDN_MODEL_URL;
      try {
        return await MediaPipeHandLandmarker.createFromOptions(visionFiles, options);
      } catch (cdnError) {
        throw new HandLandmarkerError(
          "MODEL_FETCH_ERROR",
          "Failed to fetch hand_landmarker.task model file from both local storage and CDN.",
          cdnError
        );
      }
    }
    throw localError;
  }
}

/**
 * Performs hand landmark detection on a single video frame.
 * Enforces strictly monotonic timestamps required by MediaPipe Tasks Vision.
 */
export function detectForVideo(
  video: HTMLVideoElement,
  timestampMs: number
): DetectionResult {
  if (!landmarkerInstance) {
    return {
      hands: [],
      timestampMs,
      inferenceTimeMs: 0,
    };
  }

  // Ensure strict monotonic increase required by MediaPipe Tasks Vision
  let safeTimestamp = timestampMs;
  if (safeTimestamp <= lastMonotonicTimestamp) {
    safeTimestamp = lastMonotonicTimestamp + 1;
  }
  lastMonotonicTimestamp = safeTimestamp;

  try {
    const startTime = performance.now();
    const rawResult: MediaPipeResult = landmarkerInstance.detectForVideo(video, safeTimestamp);
    const inferenceTimeMs = performance.now() - startTime;

    return mapRawResultToDetectionResult(rawResult, safeTimestamp, inferenceTimeMs);
  } catch (err) {
    console.warn("MediaPipe detectForVideo frame execution warning:", err);
    return {
      hands: [],
      timestampMs: safeTimestamp,
      inferenceTimeMs: 0,
    };
  }
}

/**
 * Disposes the current HandLandmarker instance and releases WebGL/WASM resources.
 */
export function disposeHandLandmarker(): void {
  if (landmarkerInstance) {
    try {
      landmarkerInstance.close();
    } catch (e) {
      console.warn("Error closing HandLandmarker instance:", e);
    }
    landmarkerInstance = null;
  }
  lastMonotonicTimestamp = -1;
}

/**
 * Returns the currently active delegate (GPU or CPU).
 */
export function getCurrentDelegate(): VisionDelegate {
  return currentDelegate;
}

export const getActiveDelegate = getCurrentDelegate;

/**
 * Maps raw MediaPipe Tasks Vision result to application Landmark schema types.
 */
function mapRawResultToDetectionResult(
  rawResult: MediaPipeResult,
  timestampMs: number,
  inferenceTimeMs: number
): DetectionResult {
  if (!rawResult.landmarks || rawResult.landmarks.length === 0) {
    return {
      hands: [],
      timestampMs,
      inferenceTimeMs,
    };
  }

  const mappedHands: HandLandmarks[] = [];

  for (let i = 0; i < rawResult.landmarks.length; i++) {
    const landmarks = rawResult.landmarks[i];
    if (landmarks.length !== 21) {
      continue; // Skip invalid landmark sets
    }

    let handedness: Handedness = "Unknown";
    let handednessScore = 1.0;

    if (rawResult.handednesses && rawResult.handednesses[i] && rawResult.handednesses[i][0]) {
      const h = rawResult.handednesses[i][0];
      const categoryName = h.categoryName;
      if (categoryName === "Right" || categoryName === "Left") {
        handedness = categoryName;
      }
      handednessScore = h.score ?? 1.0;
    }

    mappedHands.push({
      landmarks: landmarks.map((l) => ({
        x: l.x,
        y: l.y,
        z: l.z,
      })),
      handedness,
      handednessScore,
    });
  }

  return {
    hands: mappedHands,
    timestampMs,
    inferenceTimeMs,
  };
}
