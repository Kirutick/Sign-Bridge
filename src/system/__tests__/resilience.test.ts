import { describe, it, expect, beforeEach } from "vitest";
import { CameraController } from "../cameraController";
import { MediaPipeLoader } from "../mediapipeLoader";
import { ModelLoader } from "../modelLoader";
import { selectPrimaryHand } from "../selectPrimaryHand";
import { PredictionStabilizer } from "../../recognition/predictionStabilizer";

describe("Phase 8 Failure & System Resilience Unit Tests (§3)", () => {
  describe("Camera Failure Recovery Tests", () => {
    it("handles camera state transitions cleanly on stopCamera()", () => {
      const camera = new CameraController();
      expect(camera.getStatus().state).toBe("CAMERA_IDLE");

      camera.stopCamera();
      expect(camera.getStatus().state).toBe("CAMERA_IDLE");
      expect(camera.getStatus().stream).toBeNull();
    });

    it("handles browser support guard when mediaDevices is unavailable", async () => {
      const camera = new CameraController();

      // Temporarily delete mediaDevices
      const originalMediaDevices = navigator.mediaDevices;
      Object.defineProperty(navigator, "mediaDevices", {
        value: undefined,
        configurable: true,
      });

      await camera.startCamera();
      const status = camera.getStatus();

      expect(status.state).toBe("CAMERA_UNAVAILABLE");
      expect(status.error).toContain("doesn't support");

      Object.defineProperty(navigator, "mediaDevices", {
        value: originalMediaDevices,
        configurable: true,
      });
    });
  });

  describe("MediaPipe Resilience Tests", () => {
    it("handles WASM load failures and sets load error status", () => {
      const loader = new MediaPipeLoader();
      loader.setPendingLoad();
      loader.setError("CDN network timeout fetching WASM task binary");

      const status = loader.getStatus();
      expect(status.state).toBe("MEDIAPIPE_LOAD_ERROR");
      expect(status.error).toBe("CDN network timeout fetching WASM task binary");
    });
  });

  describe("Model Loader Resilience & Classification Tests", () => {
    it("classifies network errors as recoverable", () => {
      const loader = new ModelLoader();
      loader.setError("NETWORK_ERROR", "Failed to fetch model.json over network");

      const status = loader.getStatus();
      expect(status.state).toBe("MODEL_ERROR");
      expect(status.errorType).toBe("NETWORK_ERROR");
      expect(status.isRecoverable).toBe(true);
    });

    it("classifies config shape mismatches as non-recoverable (§2.3)", () => {
      const loader = new ModelLoader();
      loader.setError("CONFIG_MISMATCH", "Expected input shape [30, 63] does not match model");

      const status = loader.getStatus();
      expect(status.state).toBe("MODEL_ERROR");
      expect(status.errorType).toBe("CONFIG_MISMATCH");
      expect(status.isRecoverable).toBe(false);
    });
  });

  describe("Multi-Hand Isolation Policy Tests (§6)", () => {
    it("guarantees zero landmark mixing when multiple hands are detected", () => {
      const handLeft = {
        landmarks: new Array(21).fill(null).map((_, i) => ({ x: 0.1 * i, y: 0.1, z: 0.0 })),
        handedness: "Left" as const,
        handednessScore: 0.90,
      };

      const handRight = {
        landmarks: new Array(21).fill(null).map((_, i) => ({ x: 0.5 * i, y: 0.5, z: 0.0 })),
        handedness: "Right" as const,
        handednessScore: 0.98,
      };

      const result = selectPrimaryHand([handLeft, handRight]);
      expect(result.subState).toBe("MULTIPLE_HANDS_DETECTED");
      expect(result.handCount).toBe(2);

      // Verify selected primary hand has unpolluted landmarks
      const primaryLandmarks = result.primaryHand!.landmarks;
      expect(primaryLandmarks.length).toBe(21);
      expect(primaryLandmarks[1].x).toBe(0.5); // Entirely handRight values
    });
  });

  describe("Stabilizer Low-Confidence & Interrupted Gesture Resilience Tests", () => {
    let stabilizer: PredictionStabilizer;

    beforeEach(() => {
      stabilizer = new PredictionStabilizer({ confidenceThreshold: 0.80 });
    });

    it("resets state without committing when hand leaves frame mid-sign (§3)", () => {
      // 5 frames of valid candidate "HELLO"
      for (let i = 0; i < 5; i++) {
        stabilizer.processFrame({ label: "HELLO", confidence: 0.90, handDetected: true, timestampMs: i * 33 });
      }

      // Hand leaves frame mid-sign
      const resInterrupted = stabilizer.processFrame({ label: null, confidence: null, handDetected: false, timestampMs: 200 });

      expect(resInterrupted.state).toBe("IDLE");
      expect(resInterrupted.statusMessage).toBe("No hand detected");
      expect(resInterrupted.committedTextDelta).toBeNull();
      expect(stabilizer.getFullCommittedText()).toEqual([]);
    });

    it("never commits when predictions are below confidence threshold", () => {
      for (let i = 0; i < 20; i++) {
        const res = stabilizer.processFrame({ label: "YES", confidence: 0.60, handDetected: true, timestampMs: i * 33 });
        expect(res.committedTextDelta).toBeNull();
      }
      expect(stabilizer.getFullCommittedText()).toEqual([]);
    });
  });
});
