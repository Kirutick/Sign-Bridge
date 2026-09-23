import { detectForVideo } from "./handLandmarker";
import { selectPrimaryHand } from "../system/selectPrimaryHand";
import { processFrame } from "./landmarkProcessor";
import { signClassifier } from "./signClassifier";
import { SequenceBuffer } from "./sequenceBuffer";
import { PredictionStabilizer } from "../recognition/predictionStabilizer";

export interface VideoInferenceToken {
  timeSec: number;
  label: string;
  confidence: number;
}

export interface VideoInferenceProgress {
  progressPercent: number;
  currentTimeSec: number;
  durationSec: number;
  currentCandidate: string | null;
  recognizedTokens: VideoInferenceToken[];
}

export interface VideoInferenceResult {
  tokens: VideoInferenceToken[];
  sentence: string;
  durationSec: number;
  fpsEstimated: number;
}

export class VideoInferenceService {
  private isProcessing: boolean = false;
  private abortRequested: boolean = false;

  public cancel(): void {
    this.abortRequested = true;
  }

  public isRunning(): boolean {
    return this.isProcessing;
  }

  /**
   * Runs offline inference on an uploaded video file.
   * Frame-by-frame: video -> MediaPipe -> landmark normalization -> 30-frame buffer -> LSTM -> stabilizer.
   */
  public async processVideoFile(
    file: File,
    onProgress?: (progress: VideoInferenceProgress) => void,
    onLandmarks?: (landmarks: any[]) => void
  ): Promise<VideoInferenceResult> {
    if (this.isProcessing) {
      throw new Error("Video inference already in progress.");
    }

    if (!signClassifier.isReady()) {
      throw new Error("ML Classifier model is not ready. Please wait for model load.");
    }

    this.isProcessing = true;
    this.abortRequested = false;

    const videoUrl = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.src = videoUrl;
    video.muted = true;
    video.playsInline = true;
    video.crossOrigin = "anonymous";

    try {
      // Wait for metadata
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error("Failed to load video metadata. Invalid format or codec."));
      });

      const duration = video.duration || 10;
      const buffer = new SequenceBuffer(30, 63);
      const stabilizer = new PredictionStabilizer({
        confidenceThreshold: 0.65,
        stabilityDurationMs: 300,
        duplicateSuppressionRequired: true,
      });

      const recognizedTokens: VideoInferenceToken[] = [];
      const frameIntervalMs = 33.33; // ~30 FPS
      const totalSteps = Math.floor((duration * 1000) / frameIntervalMs);

      for (let step = 0; step < totalSteps; step++) {
        if (this.abortRequested) {
          break;
        }

        const targetTimeSec = (step * frameIntervalMs) / 1000;
        video.currentTime = targetTimeSec;

        await new Promise<void>((res) => {
          video.onseeked = () => res();
        });

        const timestampMs = step * frameIntervalMs;
        const detection = detectForVideo(video, timestampMs);

        if (onLandmarks) {
          onLandmarks(detection.hands);
        }

        const selection = selectPrimaryHand(detection.hands);

        if (selection.primaryHand) {
          const result = processFrame([selection.primaryHand], timestampMs);
          if (result.ok) {
            buffer.push(result.features.hands[0].vector);

            if (buffer.isFull()) {
              const sequence = buffer.getSequence();
              const pred = await signClassifier.predict(sequence);

              const stabRes = stabilizer.processFrame({
                label: pred.label,
                confidence: pred.confidence,
                handDetected: true,
                timestampMs,
              });

              if (stabRes.committedTextDelta) {
                recognizedTokens.push({
                  timeSec: Math.round(targetTimeSec * 10) / 10,
                  label: stabRes.committedTextDelta,
                  confidence: pred.confidence,
                });
              }
            }
          } else {
            buffer.handleHandLoss();
            stabilizer.processFrame({
              label: null,
              confidence: null,
              handDetected: false,
              timestampMs,
            });
          }
        } else {
          buffer.handleHandLoss();
          stabilizer.processFrame({
            label: null,
            confidence: null,
            handDetected: false,
            timestampMs,
          });
        }

        if (onProgress && step % 3 === 0) {
          onProgress({
            progressPercent: Math.min(100, Math.round((step / totalSteps) * 100)),
            currentTimeSec: targetTimeSec,
            durationSec: duration,
            currentCandidate: stabilizer.processFrame({
              label: null,
              confidence: null,
              handDetected: false,
              timestampMs,
            }).currentCandidate,
            recognizedTokens: [...recognizedTokens],
          });
        }
      }

      const sentence = recognizedTokens.map((t) => t.label).join(" ");

      return {
        tokens: recognizedTokens,
        sentence,
        durationSec: duration,
        fpsEstimated: 30,
      };
    } finally {
      this.isProcessing = false;
      this.abortRequested = false;
      URL.revokeObjectURL(videoUrl);
      video.remove();
    }
  }
}

export const videoInferenceService = new VideoInferenceService();
