import React from "react";
import type { RecordingStatePayload } from "../dataCollection/recordingStateMachine";
import type { DuplicateCheckResult } from "../dataCollection/duplicateDetector";
import styles from "../styles/RecordingPanel.module.css";

interface RecordingPanelProps {
  sessionState: RecordingStatePayload;
  duplicateWarning: DuplicateCheckResult | null;
  currentLabel: string;
  totalCollectedForLabel: number;
}

export const RecordingPanel: React.FC<RecordingPanelProps> = ({
  sessionState,
  duplicateWarning,
  currentLabel,
  totalCollectedForLabel,
}) => {
  const { state, validFrameCount, countdownRemaining, cancellationReason, error } = sessionState;

  return (
    <div className={styles.panel} role="region" aria-label="Landmark Sequence Recording Feedback Panel">
      <div className={styles.header}>
        <div className={styles.title}>
          <span>📹</span> Recording Engine Status (§8)
        </div>
        <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
          Schema v1.0.0
        </div>
      </div>

      {/* Live Countdown Display */}
      {state === "COUNTDOWN" && (
        <div className={styles.countdownBox} role="timer" aria-live="assertive">
          <span>Get Ready: {countdownRemaining}s</span>
        </div>
      )}

      {/* Live Feedback Metrics Grid (§8) */}
      <div className={styles.grid}>
        <div className={styles.box}>
          <span className={styles.label}>Target Label</span>
          <span className={`${styles.value} ${styles.valueHighlight}`}>{currentLabel || "—"}</span>
        </div>

        <div className={styles.box}>
          <span className={styles.label}>Collected Samples</span>
          <span className={styles.value}>{totalCollectedForLabel}</span>
        </div>

        <div className={styles.box}>
          <span className={styles.label}>Sequence Frame Progress</span>
          <span className={styles.value}>
            {validFrameCount} / 30
          </span>
        </div>

        <div className={styles.box}>
          <span className={styles.label}>Hand Status</span>
          <span className={styles.value}>
            {state === "RECORDING" && validFrameCount > 0 ? "Detected" : "—"}
          </span>
        </div>
      </div>

      {/* Live Status Message & Accessible Live Region (§8) */}
      <div
        style={{
          padding: "12px 16px",
          background: "rgba(0, 0, 0, 0.4)",
          border: "1px solid var(--border-color)",
          borderRadius: "var(--radius-sm)",
          fontSize: "0.88rem",
          fontWeight: 700,
        }}
        aria-live="polite"
      >
        <span>Status: </span>
        <span
          style={{
            color:
              state === "COMPLETED"
                ? "var(--accent-green)"
                : state === "CANCELLED" || state === "ERROR"
                ? "var(--accent-coral)"
                : "var(--accent-cyan)",
          }}
        >
          {state === "COUNTDOWN"
            ? `Countdown: ${countdownRemaining}s...`
            : state === "RECORDING"
            ? `Recording... (${validFrameCount}/30 frames)`
            : state === "COMPLETED"
            ? "✓ Sample Successfully Recorded & Validated"
            : state === "CANCELLED"
            ? `Cancelled (${cancellationReason || "Unknown reason"})`
            : state === "ERROR"
            ? `Error: ${error || "Unknown system failure"}`
            : "Idle — Ready to Record"}
        </span>
      </div>

      {/* Non-Blocking Advisory Duplicate Warning (§9) */}
      {duplicateWarning && (
        <div className={styles.warningBox} role="alert">
          ⚠️ <strong>Advisory Duplicate Notice:</strong> This recorded sample is very similar to
          sample <code>{duplicateWarning.mostSimilarId}</code> for <strong>{currentLabel}</strong> (Distance Score: {duplicateWarning.similarityScore}).
          Recording was accepted, but consider varying hand speed or angle for dataset diversity.
        </div>
      )}
    </div>
  );
};
