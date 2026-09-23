import React, { useState, useEffect, useRef } from "react";
import { useRecordingSession } from "../hooks/useRecordingSession";
import { RecordingPanel } from "./RecordingPanel";
import { LandmarkCanvas } from "./LandmarkCanvas";
import { detectForVideo } from "../services/handLandmarker";
import { selectPrimaryHand } from "../system/selectPrimaryHand";
import { datasetStorage } from "../services/datasetStorage";
import type { HandLandmarks } from "../types/landmarks";

interface DatasetCollectorProps {
  stream: MediaStream | null;
  isLandmarkerReady: boolean;
  existingLabels: string[];
  onSampleSaved: () => void;
}

export const DatasetCollector: React.FC<DatasetCollectorProps> = ({
  stream,
  isLandmarkerReady,
  existingLabels,
  onSampleSaved,
}) => {
  const [rawLabelInput, setRawLabelInput] = useState<string>("WATER");
  const [currentHands, setCurrentHands] = useState<HandLandmarks[]>([]);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  const normalizedLabel = rawLabelInput.trim().toUpperCase().replace(/\s+/g, "_");

  // Phase 8 Recording Session Hook (§2, §4, §8, §9)
  const {
    sessionState,
    duplicateWarning,
    recordedSamples,
    startRecording,
    cancelRecording,
    resetSession,
    processRecordingFrame,
    loadSamples,
  } = useRecordingSession({ label: normalizedLabel });

  // Filter samples for current label
  const labelSamples = recordedSamples.filter((s) => s.label === normalizedLabel);

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

        // Multi-hand primary hand selection policy (§6)
        const selection = selectPrimaryHand(detection.hands);
        setCurrentHands(detection.hands);

        if (selection.primaryHand) {
          processRecordingFrame([selection.primaryHand], timestamp);
        } else {
          processRecordingFrame([], timestamp);
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
  }, [stream, isLandmarkerReady, processRecordingFrame]);

  // Notify parent on sample completion
  useEffect(() => {
    if (sessionState.state === "COMPLETED") {
      onSampleSaved();
    }
  }, [sessionState.state, onSampleSaved]);

  const handleDeleteSample = async (id: string) => {
    try {
      await datasetStorage.deleteSample(id);
      await loadSamples();
      onSampleSaved();
    } catch (err) {
      console.error("Failed to delete sample:", err);
    }
  };

  const handleExportJson = async () => {
    try {
      const exportData = await datasetStorage.exportAll();
      const jsonStr = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonStr], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `sign_bridge_dataset_v1.0.0_${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export failed:", err);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", width: "100%", maxWidth: "900px", margin: "0 auto" }}>
      {/* Title Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--accent-cyan)" }}>
          📹 Dataset Collector Engine (Phase 8 / Data Collection 1)
        </h2>
        <button
          type="button"
          onClick={handleExportJson}
          style={{ padding: "8px 16px", background: "var(--accent-green)", border: "none", borderRadius: "var(--radius-sm)", color: "#000", fontWeight: 700, cursor: "pointer" }}
        >
          📥 Export Dataset JSON
        </button>
      </div>

      {/* Label Input Controls */}
      <div style={{ display: "flex", gap: "16px", alignItems: "center", padding: "16px 20px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)", flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
          <span style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--accent-cyan)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Sign Language Input
          </span>
          <span style={{ fontSize: "0.9rem", fontWeight: 800, color: "var(--text-primary)" }}>
            🇮🇳 Indian Sign Language (ISL)
          </span>
        </div>

        <div style={{ width: "1px", height: "34px", background: "var(--border-color)" }} />

        <div style={{ display: "flex", flexDirection: "column", gap: "3px" }}>
          <label htmlFor="sign-label-input" style={{ fontSize: "0.72rem", fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Output Concept Label (English)
          </label>
          <input
            id="sign-label-input"
            type="text"
            value={rawLabelInput}
            onChange={(e) => setRawLabelInput(e.target.value)}
            placeholder="e.g. WATER, FOOD, HELP, HELLO"
            style={{ padding: "8px 14px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", color: "var(--text-primary)", fontWeight: 700, fontSize: "1rem" }}
          />
        </div>

        <div style={{ display: "flex", gap: "10px", marginLeft: "auto" }}>
          {sessionState.state === "IDLE" || sessionState.state === "COMPLETED" || sessionState.state === "CANCELLED" ? (
            <button
              type="button"
              onClick={() => startRecording(normalizedLabel)}
              disabled={!stream || !isLandmarkerReady || !normalizedLabel}
              style={{ padding: "10px 20px", background: "linear-gradient(135deg, #00f2fe 0%, #4facfe 100%)", border: "none", borderRadius: "var(--radius-sm)", color: "#000", fontWeight: 800, cursor: "pointer" }}
            >
              🔴 Record Sample
            </button>
          ) : (
            <button
              type="button"
              onClick={() => cancelRecording("User clicked cancel button")}
              style={{ padding: "10px 20px", background: "rgba(255, 75, 75, 0.2)", border: "1px solid var(--accent-coral)", borderRadius: "var(--radius-sm)", color: "var(--accent-coral)", fontWeight: 800, cursor: "pointer" }}
            >
              ⏹️ Cancel Session
            </button>
          )}

          <button
            type="button"
            onClick={resetSession}
            style={{ padding: "10px 14px", background: "rgba(255, 255, 255, 0.05)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", color: "var(--text-secondary)", fontWeight: 600, cursor: "pointer" }}
          >
            Reset Engine
          </button>
        </div>
      </div>

      {/* Main Grid: Video Stream & Live Recording Panel */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "20px", width: "100%" }}>
        
        {/* Video Preview & Landmarks */}
        <div style={{ position: "relative", width: "100%", aspectRatio: "4/3", background: "#080c14", borderRadius: "var(--radius-md)", overflow: "hidden", border: "1px solid var(--border-color)" }}>
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
            aria-label="Collector webcam feed"
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
              Camera feed inactive
            </div>
          )}
        </div>

        {/* Live Recording Panel Feedback (§8) */}
        <RecordingPanel
          sessionState={sessionState}
          duplicateWarning={duplicateWarning}
          currentLabel={normalizedLabel}
          totalCollectedForLabel={labelSamples.length}
        />
      </div>

      {/* Existing Samples Table */}
      <div style={{ padding: "20px 24px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>
            Stored Dataset Samples ({recordedSamples.length} total)
          </h3>
          <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Existing Labels: {existingLabels.join(", ") || "None"}
          </span>
        </div>

        <div style={{ maxHeight: "200px", overflowY: "auto" }}>
          {recordedSamples.length > 0 ? (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem", textAlign: "left" }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border-color)", color: "var(--text-secondary)" }}>
                  <th style={{ padding: "8px", textAlign: "left" }}>ID</th>
                  <th style={{ padding: "8px", textAlign: "left" }}>Label</th>
                  <th style={{ padding: "8px", textAlign: "left" }}>Shape</th>
                  <th style={{ padding: "8px", textAlign: "left" }}>Handedness</th>
                  <th style={{ padding: "8px", textAlign: "left" }}>Timestamp</th>
                  <th style={{ padding: "8px", textAlign: "right" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {recordedSamples.map((s) => (
                  <tr key={s.id} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <td style={{ padding: "8px", fontFamily: "monospace" }}>{s.id.substring(0, 18)}...</td>
                    <td style={{ padding: "8px", fontWeight: 700, color: "var(--accent-cyan)" }}>{s.label}</td>
                    <td style={{ padding: "8px" }}>[{s.sequenceLength}, {s.featureCount}]</td>
                    <td style={{ padding: "8px" }}>{s.handedness}</td>
                    <td style={{ padding: "8px", color: "var(--text-muted)" }}>{new Date(s.timestamp).toLocaleTimeString()}</td>
                    <td style={{ padding: "8px", textAlign: "right" }}>
                      <button
                        type="button"
                        onClick={() => handleDeleteSample(s.id)}
                        style={{ padding: "4px 8px", background: "rgba(255,75,75,0.15)", border: "1px solid var(--accent-coral)", borderRadius: "4px", color: "var(--accent-coral)", cursor: "pointer", fontSize: "0.75rem" }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ padding: "16px", color: "var(--text-muted)", fontStyle: "italic", textAlign: "center" }}>
              No recorded samples in IndexedDB storage yet.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
