import React from "react";
import styles from "../styles/RecognitionControls.module.css";

interface RecognitionControlsProps {
  isPaused: boolean;
  onStart: () => void;
  onPause: () => void;
  onClearText: () => void;
  onBackspace: () => void;
  confidenceThreshold: number;
  onThresholdChange: (val: number) => void;
  hasCommittedText: boolean;
}

export const RecognitionControls: React.FC<RecognitionControlsProps> = ({
  isPaused,
  onStart,
  onPause,
  onClearText,
  onBackspace,
  confidenceThreshold,
  onThresholdChange,
  hasCommittedText,
}) => {
  return (
    <div className={styles.bar} role="group" aria-label="Sign Language Recognition Controls">
      <div className={styles.buttonGroup}>
        {/* Start Recognition */}
        <button
          type="button"
          className={`${styles.btn} ${styles.btnStart}`}
          onClick={onStart}
          disabled={!isPaused}
          aria-label="Start Real-Time Recognition"
        >
          <span>▶</span> Start Recognition
        </button>

        {/* Pause Recognition */}
        <button
          type="button"
          className={`${styles.btn} ${styles.btnPause}`}
          onClick={onPause}
          disabled={isPaused}
          aria-label="Pause Real-Time Recognition"
        >
          <span>⏸</span> Pause Recognition
        </button>

        {/* Backspace Last Sign */}
        <button
          type="button"
          className={`${styles.btn} ${styles.btnSecondary}`}
          onClick={onBackspace}
          disabled={!hasCommittedText}
          aria-label="Backspace last committed sign token"
        >
          <span>⌫</span> Backspace Last Sign
        </button>

        {/* Clear Text */}
        <button
          type="button"
          className={`${styles.btn} ${styles.btnSecondary}`}
          onClick={onClearText}
          disabled={!hasCommittedText}
          aria-label="Clear all committed text"
        >
          <span>🗑</span> Clear Text
        </button>
      </div>

      {/* Live Confidence Threshold Slider (§7) */}
      <div className={styles.sliderBox}>
        <label htmlFor="confidence-threshold-input">
          Confidence Threshold: <strong>{Math.round(confidenceThreshold * 100)}%</strong>
        </label>
        <input
          id="confidence-threshold-input"
          type="range"
          min="0.50"
          max="0.95"
          step="0.05"
          value={confidenceThreshold}
          onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
          className={styles.slider}
          aria-label="Adjust Confidence Threshold"
        />
      </div>
    </div>
  );
};
