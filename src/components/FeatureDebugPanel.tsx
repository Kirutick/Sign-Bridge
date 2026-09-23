import React from "react";
import type { ProcessingResult, RejectionReason } from "../types/features";
import styles from "../styles/FeatureDebugPanel.module.css";

interface FeatureDebugPanelProps {
  processingResult: ProcessingResult | null;
  rejectionCount: number;
  lastRejectionReason: RejectionReason | null;
}

export const FeatureDebugPanel: React.FC<FeatureDebugPanelProps> = ({
  processingResult,
  rejectionCount,
  lastRejectionReason,
}) => {
  const isOk = processingResult?.ok === true;
  const hands = isOk ? processingResult.features.hands : [];

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.title}>
          <span>🧪</span> Feature Extraction Pipeline (Phase 2)
        </div>
      </div>

      <div className={styles.summaryGrid}>
        {/* Total Rejections Counter */}
        <div className={styles.summaryBox}>
          <span className={styles.label}>Rejected Frames</span>
          <span className={`${styles.val} ${rejectionCount > 0 ? styles.valInvalid : styles.valValid}`}>
            {rejectionCount}
          </span>
        </div>

        {/* Most Recent Rejection Reason */}
        <div className={styles.summaryBox}>
          <span className={styles.label}>Last Rejection Reason</span>
          <span className={styles.val} style={{ fontSize: "0.85rem" }}>
            {lastRejectionReason ? lastRejectionReason : "None (0)"}
          </span>
        </div>

        {/* Processed Hands Count */}
        <div className={styles.summaryBox}>
          <span className={styles.label}>Processed Hands</span>
          <span className={styles.val}>{hands.length}</span>
        </div>
      </div>

      {/* Rejection Notification Banner */}
      {!isOk && processingResult && (
        <div className={styles.rejectionBanner} role="status">
          <span>⚠️</span>
          <div>
            <strong>Frame Processing Rejection:</strong> {processingResult.reason} — {processingResult.detail}
          </div>
        </div>
      )}

      {/* Per-Hand Feature Details */}
      {hands.map((hand, idx) => {
        const rawLms = hand.rawLandmarks;
        const normLms = hand.normalizedLandmarks;
        const vec = hand.vector;
        const isVectorLengthValid = vec.length === 63;

        // Extract first 6 raw values (x0, y0, z0, x1, y1, z1)
        const first6Raw = rawLms.slice(0, 2).flatMap((lm) => [lm.x, lm.y, lm.z]);
        // Extract first 6 normalized values (tx0, ty0, tz0, tx1, ty1, tz1)
        const first6Norm = normLms.slice(0, 2).flatMap((lm) => [lm.x, lm.y, lm.z]);

        return (
          <div key={idx} className={styles.handSection}>
            <div className={styles.handHeader}>
              <span>
                Hand #{idx + 1}: {hand.handedness} ({Math.round(hand.handednessScore * 100)}%)
              </span>
              <span>Scale Factor: {hand.scaleFactor.toFixed(4)}</span>
            </div>

            <div style={{ display: "flex", gap: "16px", fontSize: "0.8rem" }}>
              <div>
                Raw Landmarks: <strong>{rawLms.length}</strong> (expected 21)
              </div>
              <div>
                Vector Length:{" "}
                <strong className={isVectorLengthValid ? styles.valValid : styles.valInvalid}>
                  {vec.length}
                </strong>{" "}
                (expected 63)
              </div>
            </div>

            {/* Side-by-Side Comparison of First 6 Values (§8) */}
            <div className={comparisonGridClass(styles.comparisonGrid)}>
              <div>
                <div className={styles.columnTitle}>First 6 Raw Values [0..1]</div>
                <div className={styles.vectorList}>
                  {first6Raw.map((val, i) => (
                    <div key={i} className={styles.vectorRow}>
                      <span className={styles.vectorKey}>
                        {i < 3 ? `Wrist ${["X", "Y", "Z"][i]}` : `ThumbCMC ${["X", "Y", "Z"][i - 3]}`}
                      </span>
                      <span className={styles.vectorVal}>{val.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className={styles.columnTitle}>First 6 Normalized Values (Wrist-Rel)</div>
                <div className={styles.vectorList}>
                  {first6Norm.map((val, i) => (
                    <div key={i} className={styles.vectorRow}>
                      <span className={styles.vectorKey}>
                        {i < 3 ? `Wrist ${["X", "Y", "Z"][i]}` : `ThumbCMC ${["X", "Y", "Z"][i - 3]}`}
                      </span>
                      <span className={styles.vectorVal}>{val.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

function comparisonGridClass(cls: string) {
  return cls;
}
