import { RecordingConfig, createDefaultRecordingConfig } from "./recordingConfig";
import { RecordedSample, validateSample } from "./validateSample";
import type { HandLandmarks } from "../types/landmarks";
import { processFrame, NORMALIZATION_VERSION } from "../services/landmarkProcessor";

export type RecordingState =
  | "IDLE"
  | "COUNTDOWN"
  | "RECORDING"
  | "COMPLETED"
  | "CANCELLED"
  | "ERROR";

export interface FrameInput {
  hands: HandLandmarks[];
  timestampMs: number;
}

export interface RecordingStatePayload {
  state: RecordingState;
  validFrameCount: number;
  invalidFrameStreak: number;
  totalInvalidFrames: number;
  countdownRemaining: number;
  cancellationReason: string | null;
  sample: RecordedSample | null;
  error: string | null;
}

export class RecordingSession {
  private config: RecordingConfig;
  private state: RecordingState = "IDLE";
  private validFrameCount: number = 0;
  private invalidFrameStreak: number = 0;
  private totalInvalidFrames: number = 0;
  private countdownRemaining: number = 0;
  private cancellationReason: string | null = null;
  private errorDetails: string | null = null;
  private completedSample: RecordedSample | null = null;
  private sequenceBuffer: number[][] = [];
  private detectedHandedness: "Left" | "Right" | "Unknown" = "Unknown";
  private onStateChange?: (payload: RecordingStatePayload) => void;

  constructor(
    config?: Partial<RecordingConfig>,
    onStateChange?: (payload: RecordingStatePayload) => void
  ) {
    this.config = createDefaultRecordingConfig(config);
    this.onStateChange = onStateChange;
    this.countdownRemaining = this.config.countdownDuration;
  }

  public getState(): RecordingStatePayload {
    return {
      state: this.state,
      validFrameCount: this.validFrameCount,
      invalidFrameStreak: this.invalidFrameStreak,
      totalInvalidFrames: this.totalInvalidFrames,
      countdownRemaining: this.countdownRemaining,
      cancellationReason: this.cancellationReason,
      sample: this.completedSample,
      error: this.errorDetails,
    };
  }

  private notify() {
    if (this.onStateChange) {
      this.onStateChange(this.getState());
    }
  }

  public updateConfig(newConfig: Partial<RecordingConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.countdownRemaining = this.config.countdownDuration;
  }

  public startCountdown(): boolean {
    if (this.state !== "IDLE" && this.state !== "COMPLETED" && this.state !== "CANCELLED") {
      return false;
    }

    this.state = "COUNTDOWN";
    this.validFrameCount = 0;
    this.invalidFrameStreak = 0;
    this.totalInvalidFrames = 0;
    this.countdownRemaining = this.config.countdownDuration;
    this.cancellationReason = null;
    this.errorDetails = null;
    this.completedSample = null;
    this.sequenceBuffer = [];
    this.detectedHandedness = "Unknown";
    this.notify();
    return true;
  }

  public tickCountdown(): void {
    if (this.state !== "COUNTDOWN") return;

    if (this.countdownRemaining > 1) {
      this.countdownRemaining -= 1;
      this.notify();
    } else {
      this.countdownRemaining = 0;
      this.state = "RECORDING";
      this.notify();
    }
  }

  public cancel(reason: string = "User cancelled recording"): void {
    if (this.state === "IDLE") return;

    this.state = "CANCELLED";
    this.cancellationReason = reason;
    this.sequenceBuffer = []; // Discard partial sequence buffer (§4.2)
    this.notify();
  }

  public reset(): void {
    this.state = "IDLE";
    this.validFrameCount = 0;
    this.invalidFrameStreak = 0;
    this.totalInvalidFrames = 0;
    this.countdownRemaining = this.config.countdownDuration;
    this.cancellationReason = null;
    this.errorDetails = null;
    this.completedSample = null;
    this.sequenceBuffer = [];
    this.detectedHandedness = "Unknown";
    this.notify();
  }

  /**
   * Main Per-Frame Recording Engine Processing (§4).
   * Structural Invariant: Frame processing ONLY occurs during RECORDING state.
   */
  public onFrame(input: FrameInput): void {
    if (this.state !== "RECORDING") return;

    const { hands, timestampMs } = input;

    // 1. Hand Count Check (§4.1)
    if (!hands || hands.length !== this.config.requiredHandCount) {
      const reason =
        !hands || hands.length === 0
          ? "hand lost for too long"
          : "multiple hands detected";
      this.handleInvalidFrame(reason);
      return;
    }

    const primaryHand = hands[0];

    // Record handedness from MediaPipe result (§6)
    if (primaryHand.handedness) {
      this.detectedHandedness = primaryHand.handedness;
    }

    // 2. Landmark Normalization using Shared processFrame (§0.1, §4.1)
    const procResult = processFrame([primaryHand], timestampMs);

    if (!procResult.ok || !procResult.features.hands[0]) {
      this.handleInvalidFrame("landmark data invalid or scale degenerate");
      return;
    }

    const featureVector = procResult.features.hands[0].vector;

    // 3. Vector Numeric Quality Gate (§4.1)
    if (
      !featureVector ||
      featureVector.length !== this.config.featuresPerFrame ||
      featureVector.some((v) => typeof v !== "number" || Number.isNaN(v) || !Number.isFinite(v))
    ) {
      this.handleInvalidFrame("landmark feature vector contains NaN/Infinity");
      return;
    }

    // 4. Append Valid Frame to Sequence Buffer (§4.1)
    this.sequenceBuffer.push(featureVector);
    this.validFrameCount = this.sequenceBuffer.length;
    this.invalidFrameStreak = 0;
    this.notify();

    // 5. Sequence Completion Gate (§5)
    if (this.validFrameCount === this.config.sequenceLength) {
      this.finalizeSequence();
    }
  }

  private handleInvalidFrame(reasonCode: string): void {
    this.invalidFrameStreak += 1;
    this.totalInvalidFrames += 1;

    // Check consecutive invalid frame policy (§4.2)
    if (this.invalidFrameStreak >= this.config.maxConsecutiveInvalidFrames) {
      this.state = "CANCELLED";
      this.cancellationReason = `Recording cancelled: ${reasonCode}`;
      this.sequenceBuffer = []; // Discard partial sequence (§4.2)
      this.notify();
      return;
    }

    // Check total invalid ratio policy (§4.2)
    const totalAttempted = this.validFrameCount + this.totalInvalidFrames;
    if (
      totalAttempted >= 10 &&
      this.totalInvalidFrames / totalAttempted > this.config.maxInvalidRatio
    ) {
      this.state = "CANCELLED";
      this.cancellationReason = "Recording cancelled: excessive invalid frame flickering";
      this.sequenceBuffer = []; // Discard partial sequence (§4.2)
      this.notify();
      return;
    }

    this.notify();
  }

  private finalizeSequence(): void {
    // Structural assertion check (§5)
    if (this.sequenceBuffer.length !== this.config.sequenceLength) {
      this.state = "ERROR";
      this.errorDetails = `Assertion failure: buffer length ${this.sequenceBuffer.length} !== expected ${this.config.sequenceLength}`;
      this.sequenceBuffer = [];
      this.notify();
      return;
    }

    const sampleId = `sample_${this.config.label}_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;

    const candidateSample: RecordedSample = {
      id: sampleId,
      label: this.config.label,
      sequence: this.sequenceBuffer,
      sequenceLength: this.config.sequenceLength,
      featureCount: this.config.featuresPerFrame,
      timestamp: new Date().toISOString(),
      handedness: this.detectedHandedness,
      source: "webcam",
      datasetVersion: this.config.datasetVersion,
      normalizationVersion: NORMALIZATION_VERSION,
    };

    // Pre-Save Quality Gate Pass (§7)
    const valResult = validateSample(candidateSample, this.config);

    if (!valResult.valid) {
      this.state = "ERROR";
      this.errorDetails = `Sample quality gate validation failed: ${valResult.reason}`;
      this.sequenceBuffer = [];
      this.notify();
      return;
    }

    this.completedSample = candidateSample;
    this.state = "COMPLETED";
    this.notify();
  }
}
