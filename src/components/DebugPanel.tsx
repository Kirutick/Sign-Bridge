import React from "react";
import type { PerformanceSnapshot } from "../system/performanceMetrics";
import styles from "../styles/DebugPanel.module.css";

interface DebugPanelProps {
  performance: PerformanceSnapshot;
  cameraState: string;
  mediapipeState: string;
  handSubState: string;
  modelState: string;
  modelSubState: string;
  stabilizerState: string;
  modelVersion: string | null;
  rawPrediction?: { label: string; confidence: number; inferenceTimeMs: number } | null;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({
  performance,
  cameraState,
  mediapipeState,
  handSubState,
  modelState,
  modelSubState,
  stabilizerState,
  modelVersion,
}) => {
  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <div className={styles.title}>
          <span>🛠️</span> Developer Mode — Performance & Subsystems Inspector (§7, §11)
        </div>
        <div style={{ fontSize: "0.78rem", color: "var(--accent-cyan)", fontFamily: "monospace" }}>
          Model: {modelVersion || "v1_baseline"}
        </div>
      </div>

      {/* Performance Instrumentation Rolling Averages (§7) */}
      <div className={styles.metricsGrid}>
        <div className={styles.metricBox}>
          <span className={styles.metricLabel}>Camera Rolling FPS</span>
          <span className={styles.metricVal}>{performance.cameraFps} FPS</span>
        </div>

        <div className={styles.metricBox}>
          <span className={styles.metricLabel}>MediaPipe Latency</span>
          <span className={styles.metricVal}>{performance.mediapipeLatencyMs} ms</span>
        </div>

        <div className={styles.metricBox}>
          <span className={styles.metricLabel}>LSTM Forward Pass</span>
          <span className={styles.metricVal}>{performance.lstmLatencyMs} ms</span>
        </div>

        <div className={styles.metricBox}>
          <span className={styles.metricLabel}>End-to-End Latency</span>
          <span className={styles.metricVal}>{performance.totalLatencyMs} ms</span>
        </div>
      </div>

      {/* Subsystem Finite State Machines (§2) */}
      <div className={styles.subStatesRow}>
        <div className={styles.stateTag}>
          Camera: <strong>{cameraState}</strong>
        </div>
        <div className={styles.stateTag}>
          MediaPipe: <strong>{mediapipeState}</strong> ({handSubState})
        </div>
        <div className={styles.stateTag}>
          LSTM Model: <strong>{modelState}</strong> ({modelSubState})
        </div>
        <div className={styles.stateTag}>
          Stabilizer: <strong>{stabilizerState}</strong>
        </div>
      </div>
    </div>
  );
};
