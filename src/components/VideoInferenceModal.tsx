import React, { useState, useRef } from "react";
import { videoInferenceService, VideoInferenceToken } from "../services/videoInferenceService";
import { speechService } from "../services/speechService";
import { LandmarkCanvas } from "./LandmarkCanvas";
import styles from "../styles/VideoInferenceModal.module.css";

interface VideoInferenceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAppendToSentence?: (text: string) => void;
}

export const VideoInferenceModal: React.FC<VideoInferenceModalProps> = ({
  isOpen,
  onClose,
  onAppendToSentence,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);
  const [tokens, setTokens] = useState<VideoInferenceToken[]>([]);
  const [transcriptionSentence, setTranscriptionSentence] = useState<string>("");
  const [hands, setHands] = useState<any[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const videoElRef = useRef<HTMLVideoElement | null>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("video/")) {
      setErrorMsg("Please select a valid video file (.mp4, .webm, .mov).");
      return;
    }

    setErrorMsg(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setVideoPreviewUrl(url);
    setTokens([]);
    setTranscriptionSentence("");
    setProgressPercent(0);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    setErrorMsg(null);
    setTokens([]);
    setTranscriptionSentence("");

    try {
      const result = await videoInferenceService.processVideoFile(
        selectedFile,
        (progress) => {
          setProgressPercent(progress.progressPercent);
          setCurrentTime(progress.currentTimeSec);
          setDuration(progress.durationSec);
          setTokens([...progress.recognizedTokens]);
        },
        (detectedHands) => {
          setHands(detectedHands);
        }
      );

      setTokens(result.tokens);
      setTranscriptionSentence(result.sentence);
      setProgressPercent(100);
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : "Video inference failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleCancel = () => {
    videoInferenceService.cancel();
    setIsProcessing(false);
  };

  const handleClose = () => {
    if (isProcessing) {
      handleCancel();
    }
    if (videoPreviewUrl) {
      URL.revokeObjectURL(videoPreviewUrl);
    }
    onClose();
  };

  const handleAppend = () => {
    if (transcriptionSentence && onAppendToSentence) {
      onAppendToSentence(transcriptionSentence);
      handleClose();
    }
  };

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-labelledby="video-modal-title">
      <div className={styles.modal}>
        <div className={styles.header}>
          <div id="video-modal-title" className={styles.title}>
            <span>🎬</span> Video File ISL Recognition
          </div>
          <button type="button" className={styles.closeBtn} onClick={handleClose} aria-label="Close modal">
            ✕
          </button>
        </div>

        {errorMsg && (
          <div style={{ padding: "10px", background: "rgba(239, 68, 68, 0.15)", border: "1px solid #ef4444", borderRadius: "6px", color: "#fca5a5", fontSize: "0.85rem" }}>
            ⚠️ {errorMsg}
          </div>
        )}

        {/* File Dropzone */}
        {!videoPreviewUrl ? (
          <div
            className={styles.dropzone}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <div style={{ fontSize: "2rem", marginBottom: "8px" }}>📁</div>
            <p style={{ fontWeight: 600, color: "var(--text-primary)" }}>
              Click to select a video file (.mp4, .webm)
            </p>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "4px" }}>
              Processes client-side: video is never uploaded to any server.
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            <div className={styles.previewContainer}>
              <video
                ref={videoElRef}
                src={videoPreviewUrl}
                className={styles.previewVideo}
                controls={!isProcessing}
                playsInline
                muted
              />
              {hands.length > 0 && (
                <LandmarkCanvas
                  hands={hands}
                  width={640}
                  height={360}
                  style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
                />
              )}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              <span>File: <strong>{selectedFile?.name}</strong></span>
              <span>Time: {currentTime.toFixed(1)}s / {duration.toFixed(1)}s</span>
            </div>

            {/* Progress Bar */}
            <div className={styles.progressBarContainer}>
              <div
                className={styles.progressBarFill}
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        )}

        {/* Live Transcription Log */}
        {tokens.length > 0 && (
          <div className={styles.resultsBox}>
            <strong style={{ fontSize: "0.85rem", color: "var(--accent-green)" }}>
              Recognized Signs Timeline:
            </strong>
            {tokens.map((t, idx) => (
              <div key={idx} className={styles.tokenRow}>
                <span>⏱️ {t.timeSec}s: <strong>{t.label}</strong></span>
                <span style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                  {Math.round(t.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Full Transcription Sentence */}
        {transcriptionSentence && (
          <div style={{ padding: "12px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid var(--accent-green)", borderRadius: "6px" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Full Transcription:</span>
            <p style={{ fontSize: "1.2rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "4px" }}>
              {transcriptionSentence}
            </p>
          </div>
        )}

        {/* Actions Bar */}
        <div className={styles.btnRow}>
          {videoPreviewUrl && !isProcessing && progressPercent === 0 && (
            <button
              type="button"
              className={`${styles.btn} ${styles.btnPrimary}`}
              onClick={handleStartAnalysis}
            >
              ▶ Start Video Analysis
            </button>
          )}

          {isProcessing && (
            <button
              type="button"
              className={styles.btn}
              onClick={handleCancel}
              style={{ color: "#f87171" }}
            >
              ⏹ Cancel Analysis
            </button>
          )}

          {transcriptionSentence && (
            <>
              <button
                type="button"
                className={styles.btn}
                onClick={() => speechService.speak(transcriptionSentence)}
              >
                🔊 Speak
              </button>
              {onAppendToSentence && (
                <button
                  type="button"
                  className={`${styles.btn} ${styles.btnPrimary}`}
                  onClick={handleAppend}
                >
                  ➕ Append to Sentence Buffer
                </button>
              )}
            </>
          )}

          <button type="button" className={styles.btn} onClick={handleClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
