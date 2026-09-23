import React from "react";
import { WebcamStatus } from "../hooks/useWebcam";
import { HandLandmarks, HandLandmarkerError } from "../types/landmarks";
import styles from "../styles/StatusPanel.module.css";

interface StatusPanelProps {
  webcamStatus: WebcamStatus;
  webcamError: string | null;
  isLandmarkerReady: boolean;
  landmarkerLoading: boolean;
  landmarkerError: HandLandmarkerError | null;
  onRetryLandmarker: () => void;
  hands: HandLandmarks[];
}

export const StatusPanel: React.FC<StatusPanelProps> = ({
  webcamStatus,
  webcamError,
  isLandmarkerReady,
  landmarkerLoading,
  landmarkerError,
  onRetryLandmarker,
  hands,
}) => {
  const hasHands = hands.length > 0;

  // Format camera status string
  const getCameraStatusText = () => {
    switch (webcamStatus) {
      case "active": return "Active Feed";
      case "requesting": return "Requesting Access...";
      case "denied": return "Access Denied";
      case "unsupported": return "Unsupported Context";
      case "error": return "Hardware Error";
      default: return "Inactive";
    }
  };

  const getCameraIndicatorClass = () => {
    switch (webcamStatus) {
      case "active": return styles.indicatorActive;
      case "requesting": return styles.indicatorWarning;
      case "denied":
      case "unsupported":
      case "error": return styles.indicatorError;
      default: return styles.indicatorIdle;
    }
  };

  // Format MediaPipe model status string
  const getModelStatusText = () => {
    if (landmarkerLoading) return "Loading WASM & Model...";
    if (landmarkerError) return "Load Failure";
    if (isLandmarkerReady) return "Ready (Loaded)";
    return "Uninitialized";
  };

  const getModelIndicatorClass = () => {
    if (landmarkerLoading) return styles.indicatorWarning;
    if (landmarkerError) return styles.indicatorError;
    if (isLandmarkerReady) return styles.indicatorActive;
    return styles.indicatorIdle;
  };

  // Construct dynamic screen-reader announcement string
  const liveAnnouncement = `Camera status: ${getCameraStatusText()}. MediaPipe model: ${getModelStatusText()}. Hand detected: ${hasHands ? `Yes (${hands.length} hands)` : "No"}.`;

  return (
    <div className={styles.panel}>
      <div className={styles.title}>
        <span>📊</span> System Status Overview
      </div>

      {/* Accessible Polite Announcement (§8) */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {liveAnnouncement}
      </div>

      <div className={styles.grid}>
        {/* Camera Status Card */}
        <div className={styles.statusCard}>
          <span className={styles.cardLabel}>Camera Status</span>
          <div className={styles.cardValue}>
            <span className={`${styles.indicator} ${getCameraIndicatorClass()}`} />
            {getCameraStatusText()}
          </div>
        </div>

        {/* MediaPipe Model Status Card */}
        <div className={styles.statusCard}>
          <span className={styles.cardLabel}>MediaPipe Model</span>
          <div className={styles.cardValue}>
            <span className={`${styles.indicator} ${getModelIndicatorClass()}`} />
            {getModelStatusText()}
          </div>
        </div>

        {/* Hand Detection Card */}
        <div className={styles.statusCard}>
          <span className={styles.cardLabel}>Hand Detection</span>
          <div className={styles.cardValue}>
            <span
              className={`${styles.indicator} ${
                hasHands ? styles.indicatorActive : styles.indicatorIdle
              }`}
            />
            {hasHands ? `Detected (${hands.length})` : "No hands in frame"}
          </div>
          {hasHands && (
            <div className={styles.handBadges}>
              {hands.map((h, idx) => (
                <span
                  key={idx}
                  className={`${styles.badge} ${
                    h.handedness === "Left" ? styles.badgeLeft : styles.badgeRight
                  }`}
                >
                  {h.handedness}: {Math.round(h.handednessScore * 100)}%
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Webcam Explicit Error Banner (§9) */}
      {webcamError && (
        <div className={styles.errorMessage} role="alert">
          <span>⚠️</span>
          <div>
            <strong>Camera Error:</strong> {webcamError}
          </div>
        </div>
      )}

      {/* MediaPipe Load Failure Banner (§9) */}
      {landmarkerError && (
        <div className={styles.errorMessage} role="alert">
          <span>⚠️</span>
          <div>
            <strong>MediaPipe Initialization Failure:</strong> {landmarkerError.message}
            <div>
              <button
                type="button"
                className={styles.retryBtn}
                onClick={onRetryLandmarker}
              >
                🔄 Retry MediaPipe Init
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
