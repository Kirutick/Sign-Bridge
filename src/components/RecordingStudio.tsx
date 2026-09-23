import React, { useState, useEffect, useRef } from "react";
import { useCollectionStudio } from "../hooks/useCollectionStudio";
import { LabelManager } from "./LabelManager";
import { RecordingPanel } from "./RecordingPanel";
import { SampleReview } from "./SampleReview";
import { VariationReminder } from "./VariationReminder";
import { DatasetDashboard } from "./DatasetDashboard";
import { LandmarkCanvas } from "./LandmarkCanvas";
import { detectForVideo } from "../services/handLandmarker";
import { selectPrimaryHand } from "../system/selectPrimaryHand";
import { datasetStorage } from "../services/datasetStorage";
import type { HandLandmarks } from "../types/landmarks";
import styles from "../styles/RecordingStudio.module.css";

interface RecordingStudioProps {
  stream: MediaStream | null;
  isLandmarkerReady: boolean;
}

export const RecordingStudio: React.FC<RecordingStudioProps> = ({
  stream,
  isLandmarkerReady,
}) => {
  const {
    activeLabel,
    labels,
    sessionInfo,
    sessionState,
    duplicateWarning,
    createLabel,
    selectLabel,
    deleteLabel,
    startStudioRecording,
    cancelStudioRecording,
    keepSample,
    retrySample,
    deleteCurrentSample,
    processStudioFrame,
    refreshLabels: _refreshLabels,
  } = useCollectionStudio(20);

  const [currentHands, setCurrentHands] = useState<HandLandmarks[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // Bind video stream
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    if (stream) {
      video.srcObject = stream;
      video.play().catch(console.warn);
    } else {
      video.srcObject = null;
    }
  }, [stream]);

  // Frame Capture & Processing Loop
  useEffect(() => {
    let running = true;

    const processLoop = () => {
      if (!running) return;

      const video = videoRef.current;
      if (video && video.readyState >= 2 && stream && isLandmarkerReady) {
        const timestamp = performance.now();
        const detection = detectForVideo(video, timestamp);

        const selection = selectPrimaryHand(detection.hands);
        setCurrentHands(detection.hands);

        if (selection.primaryHand) {
          processStudioFrame([selection.primaryHand], timestamp);
        } else {
          processStudioFrame([], timestamp);
        }
      }

      animFrameIdRef.current = requestAnimationFrame(processLoop);
    };

    if (stream && isLandmarkerReady) {
      animFrameIdRef.current = requestAnimationFrame(processLoop);
    }

    return () => {
      running = false;
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [stream, isLandmarkerReady, processStudioFrame]);

  // Ergonomic Keyboard Shortcuts Handler (§4.3)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Do NOT trigger shortcuts when focused in text inputs (§4.3)
      const targetTag = (e.target as HTMLElement)?.tagName?.toLowerCase();
      if (targetTag === "input" || targetTag === "textarea" || targetTag === "select") {
        return;
      }

      if (e.code === "Space") {
        e.preventDefault();
        if (sessionState.state === "IDLE" || sessionState.state === "CANCELLED") {
          startStudioRecording();
        }
      } else if (e.code === "Enter") {
        e.preventDefault();
        if (sessionState.state === "COMPLETED") {
          keepSample();
        }
      } else if (e.code === "KeyR") {
        e.preventDefault();
        if (sessionState.state === "COMPLETED") {
          retrySample();
        }
      } else if (e.code === "Escape") {
        e.preventDefault();
        if (sessionState.state === "COUNTDOWN" || sessionState.state === "RECORDING") {
          cancelStudioRecording();
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [sessionState.state, startStudioRecording, keepSample, retrySample, cancelStudioRecording]);

  const handleExportJson = async () => {
    try {
      const exportData = await datasetStorage.exportAll();
      const jsonStr = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sign_bridge_studio_dataset_v1.0.0_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    }
  };

  // Determine state-driven border class for camera container (§8)
  let cameraBorderClass = styles.stateIdle;
  if (sessionState.state === "COUNTDOWN") cameraBorderClass = styles.stateCountdown;
  else if (sessionState.state === "RECORDING") cameraBorderClass = styles.stateRecording;
  else if (sessionState.state === "COMPLETED") cameraBorderClass = styles.stateCompleted;

  return (
    <div className={styles.container}>
      {/* Hidden Video Source */}
      <video ref={videoRef} playsInline muted style={{ display: "none" }} />

      {/* Top Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ fontSize: "1.5rem", fontWeight: 900, color: "var(--accent-cyan)" }}>
          🎙️ Dataset Recording Studio (Phase 8 / Data Collection 2)
        </h2>
        <button
          type="button"
          onClick={handleExportJson}
          style={{ padding: "8px 16px", background: "var(--accent-green)", border: "none", borderRadius: "var(--radius-sm)", color: "#000", fontWeight: 800, cursor: "pointer" }}
        >
          📥 Export Dataset JSON
        </button>
      </div>

      {/* Hotkey Guide (§4.3) */}
      <div className={styles.hotkeyNotice}>
        <span>⌨️ Hotkeys:</span>
        <span><strong>[Space]</strong> Record</span>
        <span><strong>[Enter]</strong> Keep</span>
        <span><strong>[R]</strong> Retry</span>
        <span><strong>[Esc]</strong> Cancel</span>
        <span style={{ marginLeft: "auto", fontSize: "0.75rem" }}>Session ID: {sessionInfo.id.substring(0, 16)}</span>
      </div>

      {/* Main Studio Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "20px", width: "100%" }}>
        
        {/* Left Column: Camera Preview & State Border */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          <div className={`${styles.cameraContainer} ${cameraBorderClass}`}>
            <video
              ref={videoRef}
              playsInline
              muted
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                transform: "scaleX(-1)",
                display: stream ? "block" : "none",
              }}
              aria-label="Studio webcam feed"
            />
            {stream ? (
              <LandmarkCanvas
                hands={currentHands}
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
              <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: "var(--text-muted)" }}>
                Camera stream inactive
              </div>
            )}
          </div>

          {/* Advisory Variation Reminders (§7) */}
          <VariationReminder />
        </div>

        {/* Right Column: Label Manager & Live Recording Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          <LabelManager
            labels={labels}
            activeLabelId={activeLabel?.id || null}
            onCreateLabel={createLabel}
            onSelectLabel={selectLabel}
            onDeleteLabel={deleteLabel}
          />

          {sessionState.state === "COMPLETED" && sessionState.sample ? (
            <SampleReview
              sample={sessionState.sample}
              duplicateWarning={duplicateWarning}
              onKeep={keepSample}
              onRetry={retrySample}
              onDelete={deleteCurrentSample}
            />
          ) : (
            <RecordingPanel
              sessionState={sessionState}
              duplicateWarning={duplicateWarning}
              currentLabel={activeLabel?.name || "UNLABELED"}
              totalCollectedForLabel={activeLabel?.sampleCount || 0}
            />
          )}
        </div>
      </div>

      {/* Dataset Dashboard (§11) */}
      <DatasetDashboard labels={labels} onSelectLabel={selectLabel} />
    </div>
  );
};
