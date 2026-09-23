import React from "react";
import styles from "../styles/AboutPage.module.css";

export const AboutPage: React.FC = () => {
  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>About & Methodology</h1>
        <p className={styles.subtitle}>
          Technical architecture, mathematical preprocessing contracts, and operational scope of Sign Bridge.
        </p>
      </div>

      {/* Mandatory Scope & Limitation Disclaimer */}
      <div className={styles.disclaimerBanner} role="alert">
        <strong style={{ fontSize: "1.05rem", display: "flex", alignItems: "center", gap: "8px" }}>
          ⚠️ Important Scope & Methodology Disclaimer
        </strong>
        <p style={{ fontSize: "0.9rem", lineHeight: 1.5 }}>
          <strong>This system recognizes a configured vocabulary of signs. It is not a complete unrestricted translation of natural sign language.</strong>
        </p>
        <p style={{ fontSize: "0.85rem", opacity: 0.9 }}>
          Sign Bridge is an experimental machine learning recognition engine designed specifically for Indian Sign Language (ISL) gestures within its configured 50-class vocabulary. It operates on structured temporal landmark vectors and does not perform continuous grammatical parsing or unbounded conversational discourse.
        </p>
      </div>

      {/* Scope & Language Invariant Card */}
      <div className={styles.card}>
        <div className={styles.sectionHeading}>
          <span>🇮🇳</span> Language Specification: Indian Sign Language (ISL) → English
        </div>
        <p style={{ color: "var(--text-primary)", fontSize: "1rem", fontWeight: 600, borderLeft: "3px solid var(--accent-cyan)", paddingLeft: "12px" }}>
          "Sign Bridge is an Indian Sign Language recognition architecture and prototype that uses computer vision and temporal machine learning to process configured ISL signs and convert them into English text and speech."
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px", color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "4px" }}>
          <p><strong>Strict Language Architecture:</strong></p>
          <ul style={{ paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "6px" }}>
            <li><strong>Supported Input:</strong> Indian Sign Language (ISL) exclusively. Hand gestures represent authentic ISL vocabulary concepts.</li>
            <li><strong>Supported Output:</strong> English text and English speech synthesis (with <code>en-IN</code> Indian English voice priority).</li>
            <li><strong>NOT Supported / Excluded:</strong> American Sign Language (ASL), British Sign Language (BSL), International Sign (IS), Hindi Sign Language, Hindi text, and Hindi speech.</li>
          </ul>
        </div>
      </div>

      {/* Forensic ML Reality Audit & Prototype Status Card */}
      <div className={styles.card} style={{ borderColor: "rgba(245, 158, 11, 0.4)", background: "rgba(245, 158, 11, 0.04)" }}>
        <div className={styles.sectionHeading} style={{ color: "#f59e0b" }}>
          <span>🔍</span> Machine Learning Reality & Prototype Status
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", lineHeight: 1.5 }}>
          In accordance with the latest ML Reality Audit, Sign Bridge maintains complete transparency regarding model weights and dataset provenance:
        </p>
        <ul style={{ paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "6px", color: "var(--text-secondary)", fontSize: "0.88rem" }}>
          <li><strong>Software Pipeline:</strong> The client-side MediaPipe vision tracking, canonical scale normalization ($2.22 \times 10^{-16}$ Python parity), rolling sequence buffer, TFJS execution engine ($2.38 \times 10^{-7}$ Python parity), and prediction stabilizer FSM are 100% complete and operational.</li>
          <li><strong>Current Weights Status:</strong> The current active model was trained on procedurally synthesized demonstration data using single-handed ASL kinematic approximations. It achieves 17.75% synthetic test accuracy and 0.0% real-world accuracy on native signers.</li>
          <li><strong>Path to Real ISL Production:</strong> Real human ISL recordings can be recorded directly in <strong>Dataset Studio</strong> or imported via <code>ml/scripts/ingest_real_isl_dataset.py</code> to train validated native models without architectural changes.</li>
        </ul>
      </div>

      {/* End-to-End Pipeline Architecture */}
      <div className={styles.card}>
        <div className={styles.sectionHeading}>
          <span>🔬</span> The End-to-End Recognition Pipeline
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          The entire application pipeline runs deterministically on client hardware inside the browser:
        </p>

        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>01</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>Camera Capture & Device FSM</span>
              <span className={styles.stepDesc}>
                Webcam stream acquired locally at 720p 30 FPS. Governed by a resilient finite state machine handling permission denials, disconnects, and auto-recovery.
              </span>
            </div>
          </div>

          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>02</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>MediaPipe HandLandmarker</span>
              <span className={styles.stepDesc}>
                Tracks 21 3D landmarks (x, y, z) per hand in real-time using WASM/WebGL acceleration. Primary hand selection isolates the dominant hand without vector merging.
              </span>
            </div>
          </div>

          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>03</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>Deterministic Preprocessing & Scale Normalization</span>
              <span className={styles.stepDesc}>
                Transforms absolute pixel coordinates into wrist-relative coordinates and normalizes scale by the Euclidean distance from Wrist (0) to Middle Finger MCP (9).
              </span>
            </div>
          </div>

          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>04</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>Rolling 30-Frame Sequence Buffer</span>
              <span className={styles.stepDesc}>
                Accumulates a continuous sliding window of shape <code>[30, 63]</code>. Clears atomically on tracking loss to maintain temporal sequence contiguity.
              </span>
            </div>
          </div>

          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>05</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>Deep 2-Layer LSTM Classifier</span>
              <span className={styles.stepDesc}>
                Evaluated by TensorFlow.js with GPU WebGL acceleration: LSTM(128) → Dropout → LSTM(64) → Dense(64) → Dense(50, Softmax).
              </span>
            </div>
          </div>

          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>06</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>Prediction Stabilization & Duplicate Suppression</span>
              <span className={styles.stepDesc}>
                6-state FSM with sliding temporal voting consensus, confidence gating, cooldown lockouts, and duplicate suppression preventing frame spam.
              </span>
            </div>
          </div>

          <div className={styles.pipelineStep}>
            <span className={styles.stepNumber}>07</span>
            <div className={styles.stepContent}>
              <span className={styles.stepTitle}>Sentence Buffer & Web Speech TTS</span>
              <span className={styles.stepDesc}>
                Accumulates stabilized tokens into coherent sentences with Space, Backspace, Undo, Clear, Save, and browser Web Speech API vocalization.
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Normalization Math Contract */}
      <div className={styles.card}>
        <div className={styles.sectionHeading}>
          <span>📐</span> Mathematical Normalization Contract
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          To prevent distance-from-camera distortion and bounding-box jitter, coordinates are normalized relative to anatomical anchors:
        </p>

        <div className={styles.mathBlock}>
          // 1. Translation: Wrist (Index 0) as Origin<br />
          tx_i = x_i - x_wrist<br />
          ty_i = y_i - y_wrist<br />
          tz_i = z_i - z_wrist<br /><br />
          // 2. Anatomical Scale: Distance to Middle MCP (Index 9)<br />
          scaleFactor = sqrt(tx_9^2 + ty_9^2 + tz_9^2)<br /><br />
          // 3. Normalized Feature Vector (63 values)<br />
          v_i = [tx_i / scaleFactor, ty_i / scaleFactor, tz_i / scaleFactor]<br />
          Output: [v_0, v_1, ..., v_20] (Length: 63)
        </div>
      </div>

      {/* Privacy Guarantee */}
      <div className={styles.card}>
        <div className={styles.sectionHeading}>
          <span>🔒</span> Privacy & Local Computation Guarantee
        </div>
        <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          Sign Bridge is built on strict local-first privacy principles:
        </p>
        <ul style={{ paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "8px", color: "var(--text-secondary)", fontSize: "0.9rem" }}>
          <li>All camera frames are processed exclusively in browser memory.</li>
          <li>No video frames, camera screenshots, or landmark arrays are uploaded to any server.</li>
          <li>Recorded training samples are stored locally in your browser's IndexedDB.</li>
          <li>Speech synthesis uses the on-device browser Web Speech API.</li>
        </ul>
      </div>
    </div>
  );
};
