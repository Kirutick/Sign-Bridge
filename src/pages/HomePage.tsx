import React, { useState, useEffect, useRef, useCallback } from "react";
import { CameraController, CameraStatusPayload } from "../system/cameraController";
import { MediaPipeLoader, MediaPipeStatusPayload } from "../system/mediapipeLoader";
import { ModelLoader, ModelStatusPayload } from "../system/modelLoader";
import { selectPrimaryHand } from "../system/selectPrimaryHand";
import { PerformanceMetricsTracker, PerformanceSnapshot } from "../system/performanceMetrics";

import {
  initHandLandmarker,
  detectForVideo,
  disposeHandLandmarker,
} from "../services/handLandmarker";
import { processFrame } from "../services/landmarkProcessor";
import { signClassifier, type RawPrediction } from "../services/signClassifier";
import { SequenceBuffer } from "../services/sequenceBuffer";
import { PredictionSmoother, type SignPrediction } from "../services/predictionSmoother";
import { useSignRecognition } from "../hooks/useSignRecognition";
import { DEFAULT_INFERENCE_CONFIG } from "../config/inference";

import { LandmarkCanvas } from "../components/LandmarkCanvas";
import { StatusChip } from "../components/StatusChip";
import { ErrorBanner } from "../components/ErrorBanner";
import { ConfidenceMeter } from "../components/ConfidenceMeter";
import { RecognitionStatusBadge } from "../components/RecognitionStatusBadge";
import { CommittedTextArea } from "../components/CommittedTextArea";
import { RecognitionControls } from "../components/RecognitionControls";
import { DebugPanel } from "../components/DebugPanel";
import { FeatureDebugPanel } from "../components/FeatureDebugPanel";
import { VideoInferenceModal } from "../components/VideoInferenceModal";
import { QuickSignGuide } from "../components/QuickSignGuide";

import type { HandLandmarks } from "../types/landmarks";
import type { ProcessingResult, RejectionReason } from "../types/features";
import type { SignLanguageKey } from "../types/languages";

interface HomePageProps {
  onPracticeSign: (label: string) => void;
  onOpenFullGuide: () => void;
}

export const HomePage: React.FC<HomePageProps> = ({ onPracticeSign, onOpenFullGuide }) => {
  // Video Inference Modal State
  const [isVideoModalOpen, setIsVideoModalOpen] = useState<boolean>(false);
  // Subsystem States
  const [cameraStatus, setCameraStatus] = useState<CameraStatusPayload>({
    state: "CAMERA_IDLE",
    stream: null,
    error: null,
  });
  const [cameraController] = useState(() => new CameraController((status) => setCameraStatus(status)));
  const [mediapipeStatus, setMediapipeStatus] = useState<MediaPipeStatusPayload>({ state: "MEDIAPIPE_NOT_LOADED", subState: "NO_HAND_DETECTED", handCount: 0, error: null });
  const [modelStatus, setModelStatus] = useState<ModelStatusPayload>({ state: "MODEL_NOT_LOADED", subState: "COLLECTING_SEQUENCE", errorType: null, errorMessage: null, error: null, isRecoverable: true });
  const [perfSnapshot, setPerfSnapshot] = useState<PerformanceSnapshot>(() => new PerformanceMetricsTracker(30).getSnapshot());

  const [mediapipeLoader] = useState(() => new MediaPipeLoader((status) => setMediapipeStatus(status)));
  const [modelLoader] = useState(() => new ModelLoader((status) => setModelStatus(status)));
  const [perfTracker] = useState(() => new PerformanceMetricsTracker(30));

  // Landmarks & Feature Extraction State
  const [hands, setHands] = useState<HandLandmarks[]>([]);
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null);
  const [rejectionCount, setRejectionCount] = useState<number>(0);
  const [lastRejectionReason, setLastRejectionReason] = useState<RejectionReason | null>(null);

  // Phase 5/6 Inference & Stabilization State
  const [rawPrediction, setRawPrediction] = useState<RawPrediction | null>(null);
  const [smoothedPrediction, setSmoothedPrediction] = useState<SignPrediction | null>(null);
  const [modelVersion, setModelVersion] = useState<string | null>(null);
  const [multiHandNotice, setMultiHandNotice] = useState<string | null>(null);

  // Developer Mode Toggle (§11)
  const [developerMode, setDeveloperMode] = useState<boolean>(false);
  const [showFeatureDebug, setShowFeatureDebug] = useState<boolean>(false);

  // Phase 6 Stabilization Hook (§2, §3, §7)
  const {
    committedTokens,
    isPaused,
    confidenceThreshold,
    currentResult,
    startRecognition,
    pauseRecognition,
    clearCommittedText,
    backspaceLastToken,
    undoLastToken,
    commitExplicitToken,
    updateThreshold,
    processPredictionFrame,
  } = useSignRecognition();

  // Performance FPS tracking
  const frameCountRef = useRef<number>(0);
  const lastFpsCalcRef = useRef<number>(performance.now());
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // Subsystem References
  const sequenceBufferRef = useRef<SequenceBuffer>(new SequenceBuffer(30, 63));
  const predictionSmootherRef = useRef<PredictionSmoother>(new PredictionSmoother(5, 3));
  const lastInferenceTimeRef = useRef<number>(0);
  const isInferencePendingRef = useRef<boolean>(false);

  // Language: Indian Sign Language (ISL) strictly
  const currentLanguage: SignLanguageKey = "isl";

  // 1. Cold Load ML Model & Init MediaPipe
  const setupModel = useCallback(async (langKey: SignLanguageKey = "isl") => {
    modelLoader.setPendingLoad();
    try {
      await signClassifier.loadModel(langKey);
      modelLoader.setLoaded();
      setModelVersion(signClassifier.getModelVersion());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load ML model.";
      if (msg.includes("compatibility mismatch")) {
        modelLoader.setError("CONFIG_MISMATCH", msg);
      } else if (msg.includes("fetch")) {
        modelLoader.setError("NETWORK_ERROR", msg);
      } else {
        modelLoader.setError("UNKNOWN", msg);
      }
    }
  }, [modelLoader]);

  const setupMediaPipe = useCallback(async () => {
    mediapipeLoader.setPendingLoad();
    try {
      await initHandLandmarker();
      mediapipeLoader.setReady();
    } catch (err: unknown) {
      mediapipeLoader.setError(
        err instanceof Error ? err.message : "Failed to initialize MediaPipe WASM binaries."
      );
    }
  }, [mediapipeLoader]);

  useEffect(() => {
    setupModel(currentLanguage);
    setupMediaPipe();
    return () => {
      disposeHandLandmarker();
      cameraController.stopCamera();
    };
  }, [setupModel, setupMediaPipe, cameraController]);

  // Bind camera stream to hidden video
  useEffect(() => {
    if (videoRef.current && cameraStatus.stream) {
      videoRef.current.srcObject = cameraStatus.stream;
      videoRef.current.play().catch(console.error);
    }
  }, [cameraStatus.stream]);

  // Camera Actions
  const handleStartCamera = async () => {
    const stream = await cameraController.startCamera();
    setCameraStatus(cameraController.getStatus());
    if (stream) {
      startRecognition();
    }
  };

  const handleStopCamera = () => {
    cameraController.stopCamera();
    setCameraStatus(cameraController.getStatus());
    pauseRecognition();
  };

  // Throttled Async Inference Runner (§5, §9, §7 Latency Instrumentation)
  const runAsyncInference = useCallback(
    async (frameStartTime: number) => {
      if (
        !signClassifier.isReady() ||
        !sequenceBufferRef.current.isFull() ||
        isInferencePendingRef.current ||
        isPaused
      ) {
        return;
      }

      const now = performance.now();
      if (now - lastInferenceTimeRef.current < DEFAULT_INFERENCE_CONFIG.inferenceCadenceMs) {
        return;
      }

      isInferencePendingRef.current = true;
      lastInferenceTimeRef.current = now;

      try {
        const sequenceCopy = sequenceBufferRef.current.getSequence();
        const lstmStart = performance.now();
        const rawPred = await signClassifier.predict(sequenceCopy);
        const lstmDuration = performance.now() - lstmStart;

        perfTracker.recordLstmLatency(lstmDuration);

        predictionSmootherRef.current.addPrediction(rawPred);
        const smoothedPred = predictionSmootherRef.current.getSmoothedPrediction(confidenceThreshold);

        setRawPrediction(rawPred);
        setSmoothedPrediction(smoothedPred);

        // Record Total End-to-End Latency (§7)
        const totalDuration = performance.now() - frameStartTime;
        perfTracker.recordTotalLatency(totalDuration);

        // Feed to Phase 6 Stabilization State Machine (§9)
        processPredictionFrame({
          label: smoothedPred.label,
          confidence: smoothedPred.confidence,
          handDetected: true,
          timestampMs: frameStartTime,
        });
      } catch (e) {
        console.warn("Async inference error:", e);
      } finally {
        isInferencePendingRef.current = false;
      }
    },
    [isPaused, confidenceThreshold, processPredictionFrame, perfTracker]
  );

  // Frame processing loop
  useEffect(() => {
    let running = true;

    const processLoop = () => {
      if (!running) return;

      const video = videoRef.current;
      if (
        video &&
        video.readyState >= 2 &&
        cameraStatus.state === "CAMERA_ACTIVE" &&
        mediapipeStatus.state === "MEDIAPIPE_READY"
      ) {
        const frameStartTime = performance.now();

        // 1. MediaPipe Inference
        const mpStart = performance.now();
        const detection = detectForVideo(video, frameStartTime);
        const mpDuration = performance.now() - mpStart;
        perfTracker.recordMediaPipeLatency(mpDuration);

        // 2. Multi-Hand Selection Policy (§6)
        const selection = selectPrimaryHand(detection.hands);
        mediapipeLoader.updateHandCount(selection.handCount);
        setHands(detection.hands);

        if (selection.subState === "MULTIPLE_HANDS_DETECTED") {
          setMultiHandNotice(`Multiple hands detected (${selection.handCount}) — using primary hand`);
        } else {
          setMultiHandNotice(null);
        }

        // 3. Feature Extraction & Normalization
        if (selection.primaryHand) {
          const result = processFrame([selection.primaryHand], frameStartTime);
          setProcessingResult(result);

          if (result.ok) {
            sequenceBufferRef.current.push(result.features.hands[0].vector);
            const isFull = sequenceBufferRef.current.isFull();
            modelLoader.updateBufferSubState(isFull);

            runAsyncInference(frameStartTime);
          } else {
            // Degenerate scale -> Hand loss policy
            setRejectionCount((prev) => prev + 1);
            setLastRejectionReason(result.reason);

            sequenceBufferRef.current.handleHandLoss();
            predictionSmootherRef.current.reset();
            modelLoader.updateBufferSubState(false);

            processPredictionFrame({
              label: null,
              confidence: null,
              handDetected: false,
              timestampMs: frameStartTime,
            });
          }
        } else {
          // No hand present -> Hand loss policy (§4)
          setProcessingResult(null);
          sequenceBufferRef.current.handleHandLoss();
          predictionSmootherRef.current.reset();
          modelLoader.updateBufferSubState(false);
          setRawPrediction(null);
          setSmoothedPrediction(null);

          processPredictionFrame({
            label: null,
            confidence: null,
            handDetected: false,
            timestampMs: frameStartTime,
          });
        }

        // 4. Camera FPS Instrumentation (§7)
        frameCountRef.current++;
        if (frameStartTime - lastFpsCalcRef.current >= 1000) {
          const currentFps = Math.round(
            (frameCountRef.current * 1000) / (frameStartTime - lastFpsCalcRef.current)
          );
          perfTracker.recordCameraFps(currentFps);
          setPerfSnapshot(perfTracker.getSnapshot());

          frameCountRef.current = 0;
          lastFpsCalcRef.current = frameStartTime;
        }
      }

      animFrameIdRef.current = requestAnimationFrame(processLoop);
    };

    if (cameraStatus.state === "CAMERA_ACTIVE" && mediapipeStatus.state === "MEDIAPIPE_READY") {
      animFrameIdRef.current = requestAnimationFrame(processLoop);
    }

    return () => {
      running = false;
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [
    cameraStatus.state,
    mediapipeStatus.state,
    runAsyncInference,
    processPredictionFrame,
    mediapipeLoader,
    modelLoader,
    perfTracker,
  ]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", alignItems: "center", width: "100%", maxWidth: "1000px", margin: "0 auto", paddingBottom: "40px" }}>
      {/* Dedicated Indian Sign Language (ISL) -> English Banner */}
      <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px 18px", background: "rgba(0, 242, 254, 0.08)", border: "1px solid var(--border-glow)", borderRadius: "var(--radius-sm)", width: "100%", justifyContent: "space-between", flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "1.2rem" }}>🇮🇳</span>
          <span style={{ fontWeight: 800, fontSize: "0.95rem", color: "var(--text-primary)" }}>
            Input: Indian Sign Language (ISL)
          </span>
          <span style={{ fontSize: "0.78rem", color: "var(--text-muted)", marginLeft: "4px" }}>
            (50-Class Vocabulary)
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: 600 }}>Output:</span>
          <span style={{ fontWeight: 800, fontSize: "0.85rem", color: "var(--accent-cyan)", background: "rgba(0, 242, 254, 0.15)", border: "1px solid rgba(0, 242, 254, 0.3)", padding: "3px 10px", borderRadius: "4px" }}>
            English Text & Speech (en-IN)
          </span>
        </div>
      </div>

      {/* Honest Prototype / Synthetic Demo Model Banner */}
      <div style={{ display: "flex", alignItems: "flex-start", gap: "10px", padding: "10px 16px", background: "rgba(245, 158, 11, 0.08)", border: "1px solid rgba(245, 158, 11, 0.35)", borderRadius: "var(--radius-sm)", width: "100%", color: "#fef3c7" }}>
        <span style={{ fontSize: "1.2rem", lineHeight: "1" }}>⚠️</span>
        <div style={{ fontSize: "0.82rem", lineHeight: "1.4" }}>
          <strong style={{ color: "#f59e0b" }}>Prototype Architecture with Demo Weights:</strong> The complete computer vision, landmark extraction, and temporal inference pipeline is fully functional on-device. However, current weights were trained on a synthetic demonstration dataset (17.75% synthetic accuracy; 0.0% real-world accuracy). For genuine ISL recognition, capture authentic signs in <strong>Dataset Studio</strong> or ingest verified ISL recordings.
        </div>
      </div>

      {/* Subsystem Error Banners (§5) */}
      {cameraStatus.error && (
        <ErrorBanner
          title="Camera Access Error"
          message={cameraStatus.error}
          onRetry={handleStartCamera}
          retryText="🔄 Enable / Retry Camera"
        />
      )}

      {mediapipeStatus.error && (
        <ErrorBanner
          title="MediaPipe Initialization Failure"
          message={mediapipeStatus.error}
          onRetry={setupMediaPipe}
          retryText="🔄 Retry MediaPipe"
        />
      )}

      {modelStatus.error && (
        <ErrorBanner
          title="ML Recognition Model Error"
          message={modelStatus.errorMessage || "Model error"}
          onRetry={modelStatus.isRecoverable ? setupModel : undefined}
          retryText="🔄 Retry Downloading Model"
        />
      )}

      {/* Multi-Hand Notification Banner (§6) */}
      {multiHandNotice && (
        <div style={{ width: "100%", padding: "10px 16px", background: "rgba(255, 209, 102, 0.12)", border: "1px solid #ffd166", borderRadius: "var(--radius-sm)", color: "#ffd166", fontSize: "0.85rem", fontWeight: 600 }}>
          ℹ️ {multiHandNotice}
        </div>
      )}

      {/* Status Chips Toolbar (§4) */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", width: "100%" }}>
        <StatusChip
          icon="📷"
          label="Camera"
          value={cameraStatus.state.replace("CAMERA_", "")}
          variant={cameraStatus.state === "CAMERA_ACTIVE" ? "active" : cameraStatus.error ? "error" : "idle"}
        />
        <StatusChip
          icon="✋"
          label="Hands"
          value={`${mediapipeStatus.handCount} (${mediapipeStatus.subState.replace("_DETECTED", "")})`}
          variant={mediapipeStatus.handCount > 0 ? "active" : "idle"}
        />
        <StatusChip
          icon="🤖"
          label="LSTM Model"
          value={`${modelStatus.state.replace("MODEL_", "")} [${modelStatus.subState}]`}
          variant={modelStatus.state === "MODEL_LOADED" ? "active" : modelStatus.error ? "error" : "warning"}
        />
        <StatusChip
          icon="⚡"
          label="Recognition"
          value={currentResult?.statusMessage || (isPaused ? "PAUSED" : "IDLE")}
          variant={currentResult?.currentCandidate ? "active" : "idle"}
        />
      </div>

      {/* Main Two-Panel Responsive Grid Layout (§3.2, §8) */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "20px", width: "100%" }}>
        
        {/* Panel 1: Camera Preview & Hand Tracking Canvas */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div style={{ position: "relative", width: "100%", aspectRatio: "4/3", background: "#080c14", borderRadius: "var(--radius-md)", overflow: "hidden", border: "1px solid var(--border-color)", boxShadow: "0 20px 40px rgba(0,0,0,0.5)" }}>
            <video
              ref={videoRef}
              playsInline
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: "scaleX(-1)",
                display: cameraStatus.state === "CAMERA_ACTIVE" ? "block" : "none",
              }}
              aria-label="Webcam live preview feed"
            />
            {cameraStatus.state === "CAMERA_ACTIVE" ? (
              <LandmarkCanvas
                hands={hands}
                width={videoRef.current?.videoWidth || 640}
                height={videoRef.current?.videoHeight || 480}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  transform: "scaleX(-1)",
                  pointerEvents: "none",
                }}
              />
            ) : (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", gap: "12px", color: "var(--text-muted)" }}>
                <span style={{ fontSize: "3rem" }}>📷</span>
                <span>Camera Stream Inactive</span>
                <button type="button" onClick={handleStartCamera} style={{ padding: "8px 16px", background: "var(--accent-cyan)", border: "none", borderRadius: "4px", color: "#000", fontWeight: 700, cursor: "pointer" }}>
                  Start Camera Feed
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Panel 2: Recognition Output & Candidate Display */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {/* Current Candidate Card */}
          <div style={{ padding: "20px 24px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)", backdropFilter: "var(--glass-backdrop)", display: "flex", flexDirection: "column", gap: "16px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "0.8rem", fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--text-secondary)" }}>
                Current Candidate Sign
              </span>
              <RecognitionStatusBadge
                statusMessage={currentResult?.statusMessage || (isPaused ? "Paused" : "Idle")}
                isPaused={isPaused}
              />
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 20px", background: "rgba(0, 0, 0, 0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)" }}>
              <span style={{ fontSize: "2.5rem", fontWeight: 900, color: currentResult?.currentCandidate ? "var(--accent-cyan)" : "var(--text-muted)" }}>
                {currentResult?.currentCandidate
                  ? currentResult.currentCandidate
                  : currentResult?.state === "LOW_CONFIDENCE"
                  ? "— (uncertain)"
                  : "—"}
              </span>

              {/* Confidence Meter */}
              <div style={{ width: "200px" }}>
                <ConfidenceMeter
                  confidence={smoothedPrediction?.confidence ?? null}
                  threshold={confidenceThreshold}
                />
              </div>
            </div>
          </div>

          {/* Recognized Committed Text Area & Sentence Builder */}
          <CommittedTextArea
            committedTokens={committedTokens}
            onClear={clearCommittedText}
            onBackspace={backspaceLastToken}
            onUndo={undoLastToken}
            onSpace={() => commitExplicitToken(" ")}
          />
        </div>
      </div>

      {/* Controls Bar (§3.3, §7) */}
      <div style={{ width: "100%" }}>
        <RecognitionControls
          isPaused={isPaused}
          onStart={handleStartCamera}
          onPause={handleStopCamera}
          onClearText={clearCommittedText}
          onBackspace={backspaceLastToken}
          confidenceThreshold={confidenceThreshold}
          onThresholdChange={updateThreshold}
          hasCommittedText={committedTokens.length > 0}
        />
      </div>

      {/* Auxiliary Tools & Developer Mode Toggle */}
      <div style={{ display: "flex", gap: "12px", width: "100%", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={() => setIsVideoModalOpen(true)}
          style={{ padding: "8px 16px", background: "rgba(16, 185, 129, 0.12)", border: "1px solid var(--accent-green)", borderRadius: "var(--radius-sm)", color: "#34d399", cursor: "pointer", fontWeight: 600, fontSize: "0.8rem", display: "flex", alignItems: "center", gap: "6px" }}
        >
          🎬 Upload Video for Offline Inference
        </button>

        <button
          type="button"
          onClick={() => setDeveloperMode(!developerMode)}
          style={{ padding: "8px 16px", background: developerMode ? "rgba(0, 242, 254, 0.15)" : "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", color: "var(--accent-cyan)", cursor: "pointer", fontWeight: 600, fontSize: "0.8rem" }}
        >
          {developerMode ? "🛠️ Hide Developer Mode" : "🛠️ Enable Developer Mode"}
        </button>
      </div>

      {/* Developer Mode Inspection Panel (§11) */}
      {developerMode && (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px", width: "100%" }}>
          <DebugPanel
            performance={perfSnapshot}
            cameraState={cameraStatus.state}
            mediapipeState={mediapipeStatus.state}
            handSubState={mediapipeStatus.subState}
            modelState={modelStatus.state}
            modelSubState={modelStatus.subState}
            stabilizerState={currentResult?.state || "IDLE"}
            modelVersion={modelVersion}
            rawPrediction={rawPrediction}
          />

          <button
            type="button"
            onClick={() => setShowFeatureDebug(!showFeatureDebug)}
            style={{ padding: "6px 12px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", color: "var(--text-secondary)", cursor: "pointer", fontSize: "0.75rem", alignSelf: "flex-start" }}
          >
            {showFeatureDebug ? "Hide Vector Inspector" : "Show Landmark Vector Inspector"}
          </button>

          {showFeatureDebug && (
            <FeatureDebugPanel
              processingResult={processingResult}
              rejectionCount={rejectionCount}
              lastRejectionReason={lastRejectionReason}
            />
          )}
        </div>
      )}

      {/* Video File Offline Inference Modal */}
      <VideoInferenceModal
        isOpen={isVideoModalOpen}
        onClose={() => setIsVideoModalOpen(false)}
        onAppendToSentence={(text) => commitExplicitToken(text)}
      />

      <QuickSignGuide
        onPracticeSign={onPracticeSign}
        onOpenFullGuide={onOpenFullGuide}
      />
    </div>
  );
};
