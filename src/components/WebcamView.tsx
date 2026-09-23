import React, { useEffect, useRef, useState, useCallback } from "react";
import { detectForVideo } from "../services/handLandmarker";
import { DetectionResult, HandLandmarks } from "../types/landmarks";
import { FpsCounter } from "../utils/fpsCounter";
import { LandmarkCanvas } from "./LandmarkCanvas";
import styles from "../styles/WebcamView.module.css";

interface WebcamViewProps {
  stream: MediaStream | null;
  isLandmarkerReady: boolean;
  onDetectionResult: (result: DetectionResult, fps: number) => void;
}

export const WebcamView: React.FC<WebcamViewProps> = ({
  stream,
  isLandmarkerReady,
  onDetectionResult,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({
    width: 640,
    height: 360,
  });
  const [currentHands, setCurrentHands] = useState<HandLandmarks[]>([]);
  const animFrameIdRef = useRef<number | null>(null);
  const fpsCounterRef = useRef<FpsCounter>(new FpsCounter(30));

  // Attach stream to video element
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (stream) {
      video.srcObject = stream;
      video.play().catch((err) => console.warn("Video play interrupted:", err));
    } else {
      video.srcObject = null;
      setCurrentHands([]);
    }
  }, [stream]);

  // Main animation frame inference loop (§7)
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    if (
      video &&
      video.readyState >= 2 &&
      !video.paused &&
      !video.ended &&
      isLandmarkerReady
    ) {
      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 360;

      if (vw !== dimensions.width || vh !== dimensions.height) {
        setDimensions({ width: vw, height: vh });
      }

      const timestampMs = performance.now();
      const result = detectForVideo(video, timestampMs);
      const currentFps = fpsCounterRef.current.tick();

      setCurrentHands(result.hands);
      onDetectionResult(result, currentFps);
    }

    animFrameIdRef.current = requestAnimationFrame(processFrame);
  }, [isLandmarkerReady, dimensions, onDetectionResult]);

  useEffect(() => {
    if (stream && isLandmarkerReady) {
      fpsCounterRef.current.reset();
      animFrameIdRef.current = requestAnimationFrame(processFrame);
    } else {
      if (animFrameIdRef.current !== null) {
        cancelAnimationFrame(animFrameIdRef.current);
        animFrameIdRef.current = null;
      }
    }

    return () => {
      if (animFrameIdRef.current !== null) {
        cancelAnimationFrame(animFrameIdRef.current);
        animFrameIdRef.current = null;
      }
    };
  }, [stream, isLandmarkerReady, processFrame]);

  return (
    <div className={styles.container}>
      {stream ? (
        <div className={styles.videoWrapper}>
          <div className={styles.liveBadge}>
            <span className={styles.liveDot} /> LIVE CAMERA
          </div>
          <video
            ref={videoRef}
            playsInline
            muted
            className={styles.video}
            aria-label="Webcam live preview feed"
          />
          <LandmarkCanvas
            hands={currentHands}
            width={dimensions.width}
            height={dimensions.height}
            className={styles.canvas}
          />
        </div>
      ) : (
        <div className={styles.placeholder} role="region" aria-label="Webcam inactive">
          <div className={styles.placeholderIcon}>📷</div>
          <div className={styles.placeholderTitle}>Camera Preview Inactive</div>
          <div className={styles.placeholderSubtitle}>
            Click <strong>Start Camera</strong> below to enable real-time hand landmark tracking.
          </div>
        </div>
      )}
    </div>
  );
};
