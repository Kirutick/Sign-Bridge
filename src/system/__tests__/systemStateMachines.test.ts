import { describe, it, expect, beforeEach } from "vitest";
import { CameraController } from "../cameraController";
import { MediaPipeLoader } from "../mediapipeLoader";
import { ModelLoader } from "../modelLoader";
import { selectPrimaryHand } from "../selectPrimaryHand";
import { PerformanceMetricsTracker } from "../performanceMetrics";
import type { HandLandmarks } from "../../types/landmarks";

describe("Phase 7 Subsystem State Machines & Utilities Unit Tests (§13)", () => {
  describe("CameraController State Machine Unit Tests", () => {
    it("starts in CAMERA_IDLE state", () => {
      const controller = new CameraController();
      expect(controller.getStatus().state).toBe("CAMERA_IDLE");
      expect(controller.getStatus().stream).toBeNull();
    });

    it("stops active camera cleanly on stopCamera()", () => {
      const controller = new CameraController();
      controller.stopCamera();
      expect(controller.getStatus().state).toBe("CAMERA_IDLE");
    });
  });

  describe("MediaPipeLoader State Machine Unit Tests", () => {
    let loader: MediaPipeLoader;

    beforeEach(() => {
      loader = new MediaPipeLoader();
    });

    it("transitions through MEDIAPIPE_LOADING to MEDIAPIPE_READY", () => {
      expect(loader.getStatus().state).toBe("MEDIAPIPE_NOT_LOADED");

      loader.setPendingLoad();
      expect(loader.getStatus().state).toBe("MEDIAPIPE_LOADING");

      loader.setReady();
      expect(loader.getStatus().state).toBe("MEDIAPIPE_READY");
    });

    it("updates hand presence sub-states accurately", () => {
      loader.setReady();

      loader.updateHandCount(0);
      expect(loader.getStatus().subState).toBe("NO_HAND_DETECTED");

      loader.updateHandCount(1);
      expect(loader.getStatus().subState).toBe("HAND_DETECTED");

      loader.updateHandCount(2);
      expect(loader.getStatus().subState).toBe("MULTIPLE_HANDS_DETECTED");
    });

    it("handles load errors cleanly", () => {
      loader.setError("Network timeout fetching MediaPipe WASM");
      expect(loader.getStatus().state).toBe("MEDIAPIPE_LOAD_ERROR");
      expect(loader.getStatus().error).toBe("Network timeout fetching MediaPipe WASM");
    });
  });

  describe("ModelLoader State Machine Unit Tests", () => {
    let loader: ModelLoader;

    beforeEach(() => {
      loader = new ModelLoader();
    });

    it("transitions through MODEL_LOADING to MODEL_LOADED", () => {
      expect(loader.getStatus().state).toBe("MODEL_NOT_LOADED");

      loader.setPendingLoad();
      expect(loader.getStatus().state).toBe("MODEL_LOADING");

      loader.setLoaded();
      expect(loader.getStatus().state).toBe("MODEL_LOADED");
      expect(loader.getStatus().subState).toBe("COLLECTING_SEQUENCE");
    });

    it("classifies error types and sets recoverability flag (§2.3)", () => {
      loader.setError("NETWORK_ERROR", "Failed to fetch model.json");
      expect(loader.getStatus().errorType).toBe("NETWORK_ERROR");
      expect(loader.getStatus().isRecoverable).toBe(true);

      loader.setError("CONFIG_MISMATCH", "Expected input shape [30, 63] does not match model [30, 126]");
      expect(loader.getStatus().errorType).toBe("CONFIG_MISMATCH");
      expect(loader.getStatus().isRecoverable).toBe(false);
    });

    it("updates sequence buffer sub-states", () => {
      loader.setLoaded();

      loader.updateBufferSubState(false);
      expect(loader.getStatus().subState).toBe("COLLECTING_SEQUENCE");

      loader.updateBufferSubState(true);
      expect(loader.getStatus().subState).toBe("RECOGNIZING");
    });
  });

  describe("selectPrimaryHand Multi-Hand Selection Policy Tests (§6)", () => {
    const makeMockHand = (
      minX: number,
      minY: number,
      maxX: number,
      maxY: number,
      handedness: "Left" | "Right" = "Right"
    ): HandLandmarks => {
      const landmarks = new Array(21).fill(null).map((_, i) => ({
        x: i === 0 ? minX : i === 20 ? maxX : (minX + maxX) / 2,
        y: i === 0 ? minY : i === 20 ? maxY : (minY + maxY) / 2,
        z: 0.0,
      }));

      return {
        landmarks,
        handedness,
        handednessScore: 0.95,
      };
    };

    it("returns NO_HAND_DETECTED when empty array is passed", () => {
      const res = selectPrimaryHand([]);
      expect(res.subState).toBe("NO_HAND_DETECTED");
      expect(res.primaryHand).toBeNull();
    });

    it("returns HAND_DETECTED for single hand input", () => {
      const hand1 = makeMockHand(0.1, 0.1, 0.3, 0.3);
      const res = selectPrimaryHand([hand1]);

      expect(res.subState).toBe("HAND_DETECTED");
      expect(res.handCount).toBe(1);
      expect(res.primaryHand).not.toBeNull();
    });

    it("selects hand with largest bounding box area when multiple hands detected (§6)", () => {
      const smallHand = makeMockHand(0.1, 0.1, 0.2, 0.2, "Left"); // Area 0.01
      const largeHand = makeMockHand(0.4, 0.4, 0.8, 0.8, "Right"); // Area 0.16

      const res = selectPrimaryHand([smallHand, largeHand]);

      expect(res.subState).toBe("MULTIPLE_HANDS_DETECTED");
      expect(res.handCount).toBe(2);
      expect(res.primaryHand?.handedness).toBe("Right");
    });

    it("guarantees zero landmark merging between hands (Absolute Invariant §6)", () => {
      const hand1 = makeMockHand(0.1, 0.1, 0.2, 0.2, "Left");
      const hand2 = makeMockHand(0.5, 0.5, 0.9, 0.9, "Right");

      const res = selectPrimaryHand([hand1, hand2]);
      const primaryLandmarks = res.primaryHand!.landmarks;

      expect(primaryLandmarks.length).toBe(21);
      // Verify no landmarks from hand1 were mixed in
      expect(primaryLandmarks[0].x).toBe(0.5);
      expect(primaryLandmarks[20].x).toBe(0.9);
    });
  });

  describe("PerformanceMetricsTracker Rolling Buffer Tests (§7)", () => {
    it("computes accurate rolling average across metrics", () => {
      const tracker = new PerformanceMetricsTracker(5);

      tracker.recordCameraFps(30);
      tracker.recordCameraFps(30);
      tracker.recordCameraFps(30);

      expect(tracker.getSnapshot().cameraFps).toBe(30);

      tracker.recordMediaPipeLatency(10);
      tracker.recordMediaPipeLatency(20);

      expect(tracker.getSnapshot().mediapipeLatencyMs).toBe(15);
    });
  });
});
