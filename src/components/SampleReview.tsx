import React from "react";
import type { RecordedSample } from "../dataCollection/validateSample";
import type { DuplicateCheckResult } from "../dataCollection/duplicateDetector";
import styles from "../styles/SampleReview.module.css";

interface SampleReviewProps {
  sample: RecordedSample;
  duplicateWarning: DuplicateCheckResult | null;
  onKeep: () => void;
  onRetry: () => void;
  onDelete: () => void;
}

export const SampleReview: React.FC<SampleReviewProps> = ({
  sample,
  duplicateWarning,
  onKeep,
  onRetry,
  onDelete,
}) => {
  return (
    <div className={styles.container} role="dialog" aria-label="Sample Review Dialog">
      <div className={styles.title}>
        <span>✓</span> Sample Recorded & Validated (§9)
      </div>

      <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
        Recorded <strong>{sample.sequenceLength} frames</strong> ({sample.featureCount} features) for label{" "}
        <strong style={{ color: "var(--accent-cyan)" }}>{sample.label}</strong> (Handedness: {sample.handedness}).
      </div>

      {/* Advisory Duplicate Notice (§9) */}
      {duplicateWarning && (
        <div style={{ padding: "10px 14px", background: "rgba(255, 209, 102, 0.15)", border: "1px solid #ffd166", borderRadius: "var(--radius-sm)", color: "#ffd166", fontSize: "0.82rem", fontWeight: 600 }}>
          ⚠️ <strong>Advisory Duplicate Notice:</strong> This sample is very similar to existing sample{" "}
          <code>{duplicateWarning.mostSimilarId}</code> (Distance: {duplicateWarning.similarityScore}). Consider pressing <strong>Retry [R]</strong> if you want to record a varied gesture.
        </div>
      )}

      {/* Keep / Retry / Delete Actions (§9) */}
      <div className={styles.buttonRow}>
        <button type="button" className={styles.btnKeep} onClick={onKeep}>
          ✓ Keep Sample [Enter]
        </button>

        <button type="button" className={styles.btnRetry} onClick={onRetry}>
          🔄 Retry Sample [R]
        </button>

        <button type="button" className={styles.btnDelete} onClick={onDelete}>
          🗑 Delete Sample
        </button>
      </div>
    </div>
  );
};
