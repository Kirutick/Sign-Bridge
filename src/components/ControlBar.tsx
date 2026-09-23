import React from "react";
import { WebcamStatus } from "../hooks/useWebcam";
import styles from "../styles/ControlBar.module.css";

interface ControlBarProps {
  webcamStatus: WebcamStatus;
  onStart: () => void;
  onStop: () => void;
}

export const ControlBar: React.FC<ControlBarProps> = ({
  webcamStatus,
  onStart,
  onStop,
}) => {
  const isActive = webcamStatus === "active";
  const isRequesting = webcamStatus === "requesting";
  const isUnsupported = webcamStatus === "unsupported";

  return (
    <div className={styles.bar} role="group" aria-label="Camera controls">
      <button
        type="button"
        className={`${styles.btn} ${styles.btnPrimary}`}
        onClick={onStart}
        disabled={isActive || isRequesting || isUnsupported}
        aria-label="Start Camera Feed"
      >
        {isRequesting ? (
          <>
            <span className={styles.spinner} aria-hidden="true" />
            Requesting Access...
          </>
        ) : (
          <>
            <span>▶</span> Start Camera
          </>
        )}
      </button>

      <button
        type="button"
        className={`${styles.btn} ${styles.btnSecondary}`}
        onClick={onStop}
        disabled={!isActive && !isRequesting}
        aria-label="Stop Camera Feed"
      >
        <span>⏹</span> Stop Camera
      </button>
    </div>
  );
};
