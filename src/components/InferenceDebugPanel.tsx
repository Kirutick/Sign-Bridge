import React from "react";
import type { RawPrediction } from "../services/signClassifier";
import type { SignPrediction } from "../services/predictionSmoother";
import styles from "../styles/InferenceDebugPanel.module.css";

interface InferenceDebugPanelProps {
  isModelReady: boolean;
  modelVersion: string | null;
  bufferLength: number;
  sequenceLength: number;
  stalledCount: number;
  handDetected: boolean;
  rawPrediction: RawPrediction | null;
  smoothedPrediction: SignPrediction | null;
  threshold: number;
  onThresholdChange: (val: number) => void;
  fps: number;
}

export const InferenceDebugPanel: React.FC<InferenceDebugPanelProps> = ({
  isModelReady,
  modelVersion,
  bufferLength,
  sequenceLength,
  stalledCount,
  handDetected,
  rawPrediction,
  smoothedPrediction,
  threshold,
  onThresholdChange,
  fps,
}) => {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.title}>
          <span>🤖</span> Real-Time LSTM Inference Engine (Phase 5)
        </div>
        <div style={{ fontSize: "0.8rem", color: isModelReady ? "var(--accent-green)" : "var(--accent-coral)", fontWeight: 600 }}>
          {isModelReady ? `● Model Ready (${modelVersion || "Loaded"})` : "○ Model Loading / Error"}
        </div>
      </div>

      <div className={styles.grid}>
        {/* Sequence Buffer Fill */}
        <div className={styles.box}>
          <span className={styles.label}>Sequence Buffer</span>
          <span className={styles.val}>
            {bufferLength} / {sequenceLength}
          </span>
        </div>

        {/* Hand Status */}
        <div className={styles.box}>
          <span className={styles.label}>Hand Tracked</span>
          <span className={styles.val} style={{ color: handDetected ? "var(--accent-green)" : "var(--accent-coral)" }}>
            {handDetected ? "YES" : "NO"}
          </span>
        </div>

        {/* Stalled / Cleared Count */}
        <div className={styles.box}>
          <span className={styles.label}>Hand-Loss Clears</span>
          <span className={styles.val}>{stalledCount}</span>
        </div>

        {/* Inference Latency */}
        <div className={styles.box}>
          <span className={styles.label}>Inference Latency</span>
          <span className={styles.val}>
            {rawPrediction ? `${Math.round(rawPrediction.inferenceTimeMs)} ms` : "N/A"}
          </span>
        </div>

        {/* Camera FPS */}
        <div className={styles.box}>
          <span className={styles.label}>Camera Loop FPS</span>
          <span className={styles.val}>{fps}</span>
        </div>
      </div>

      {/* Threshold Slider (§7) */}
      <div className={styles.sliderRow}>
        <label className={styles.sliderLabel}>
          Confidence Threshold: <strong>{Math.round(threshold * 100)}%</strong> ({threshold.toFixed(2)})
        </label>
        <input
          type="range"
          min="0.50"
          max="0.95"
          step="0.05"
          value={threshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className={styles.slider}
        />
      </div>

      {/* Predictions Comparison (Raw vs Smoothed) */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
        <div className={styles.box}>
          <span className={styles.label}>Raw Single-Call Prediction</span>
          <span className={styles.val} style={{ color: "var(--accent-cyan)", fontSize: "1.2rem" }}>
            {rawPrediction ? `${rawPrediction.label} (${Math.round(rawPrediction.confidence * 100)}%)` : "Waiting..."}
          </span>
        </div>

        <div className={styles.box}>
          <span className={styles.label}>Smoothed & Thresholded Output</span>
          <span
            className={styles.val}
            style={{
              color: smoothedPrediction?.label ? "var(--accent-green)" : "var(--text-muted)",
              fontSize: "1.2rem",
            }}
          >
            {smoothedPrediction?.label
              ? `${smoothedPrediction.label} (${Math.round(smoothedPrediction.confidence * 100)}%)`
              : smoothedPrediction?.belowThreshold
              ? "No Confident Prediction"
              : "Waiting..."}
          </span>
        </div>
      </div>
    </div>
  );
};
