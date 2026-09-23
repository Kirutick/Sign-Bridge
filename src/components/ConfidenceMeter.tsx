import React from "react";
import styles from "../styles/ConfidenceMeter.module.css";

interface ConfidenceMeterProps {
  confidence: number | null; // 0 to 1
  threshold: number; // 0 to 1
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  confidence,
  threshold,
}) => {
  const confVal = Math.min(1.0, Math.max(0.0, confidence ?? 0));
  const confPct = Math.round(confVal * 100);
  const thresholdPct = Math.round(threshold * 100);

  let colorClass = styles.colorLow;
  if (confVal >= threshold) {
    colorClass = styles.colorHigh;
  } else if (confVal >= threshold - 0.10) {
    colorClass = styles.colorMedium;
  }

  return (
    <div className={styles.container}>
      <div className={styles.labelRow}>
        <span>Confidence Score</span>
        <span>
          {confidence !== null ? `${confPct}%` : "—"} (Threshold: {thresholdPct}%)
        </span>
      </div>

      <div className={styles.barTrack}>
        {/* Dynamic Threshold Line Marker */}
        <div
          className={styles.thresholdMarker}
          style={{ left: `${thresholdPct}%` }}
          title={`Threshold: ${thresholdPct}%`}
        />

        {/* Color-Coded Progress Bar */}
        <div
          className={`${styles.barFill} ${colorClass}`}
          style={{ width: `${confPct}%` }}
        />
      </div>
    </div>
  );
};
