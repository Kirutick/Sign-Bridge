import React from "react";
import styles from "../styles/RecognitionStatusBadge.module.css";

interface RecognitionStatusBadgeProps {
  statusMessage: string;
  isPaused: boolean;
}

export const RecognitionStatusBadge: React.FC<RecognitionStatusBadgeProps> = ({
  statusMessage,
  isPaused,
}) => {
  if (isPaused) {
    return (
      <div className={`${styles.badge} ${styles.statusPaused}`}>
        <span className={styles.dot} />
        <span>⏸ Paused</span>
      </div>
    );
  }

  let styleClass = styles.statusIdle;

  if (statusMessage === "Recognizing...") {
    styleClass = styles.statusRecognizing;
  } else if (statusMessage === "Holding...") {
    styleClass = styles.statusHolding;
  } else if (statusMessage === "Committed" || statusMessage.includes("Committed")) {
    styleClass = styles.statusCommitted;
  } else if (statusMessage === "Uncertain") {
    styleClass = styles.statusUncertain;
  }

  return (
    <div className={`${styles.badge} ${styleClass}`} role="status">
      <span className={styles.dot} />
      <span>{statusMessage}</span>
    </div>
  );
};
