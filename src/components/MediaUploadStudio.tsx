import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  initImageHandLandmarker,
  detectForImage,
  detectForVideo,
} from "../services/handLandmarker";
import { selectPrimaryHand } from "../system/selectPrimaryHand";
import { processFrame } from "../services/landmarkProcessor";
import { datasetStorage } from "../services/datasetStorage";
import { NORMALIZATION_VERSION } from "../services/landmarkProcessor";
import type { HandLandmarks } from "../types/landmarks";

// ─── ISL constants ────────────────────────────────────────────────────────────
const SIGN_LANGUAGE = "Indian Sign Language";
const SIGN_LANGUAGE_CODE = "ISL";
const OUTPUT_LANGUAGE = "English";
const OUTPUT_LANGUAGE_CODE = "en";
const SEQUENCE_LENGTH = 30;
const FEATURE_COUNT = 63;
const SLIDE_STRIDE = 10; // frames between windows

// ─── Hand connections for landmark drawing ────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

interface ProcessingStats {
  framesTotal: number;
  handsDetected: number;
  framesValid: number;
  framesRejected: number;
  sequencesGenerated: number;
  samplesSaved: number;
  rejectionReasons: Record<string, number>;
}

interface PreviewFrame {
  imageDataUrl: string;
  hands: HandLandmarks[];
  valid: boolean;
}

type SourceType = "image_upload" | "video_upload" | "screen_recording";

interface MediaUploadStudioProps {
  onSampleSaved?: () => void;
}

export const MediaUploadStudio: React.FC<MediaUploadStudioProps> = ({ onSampleSaved }) => {
  const [label, setLabel] = useState("WATER");
  const [signerLabel, setSignerLabel] = useState("signer_01");
  const [mirrorLeft, setMirrorLeft] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isLandmarkerLoading, setIsLandmarkerLoading] = useState(false);
  const [landmarkerReady, setLandmarkerReady] = useState(false);
  const [stats, setStats] = useState<ProcessingStats | null>(null);
  const [statusMsg, setStatusMsg] = useState<string>("");
  const [previewFrames, setPreviewFrames] = useState<PreviewFrame[]>([]);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  const normalizedLabel = label.trim().toUpperCase().replace(/\s+/g, "_");

  // Initialize IMAGE landmarker on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setIsLandmarkerLoading(true);
      try {
        await initImageHandLandmarker();
        if (!cancelled) setLandmarkerReady(true);
      } catch (err) {
        if (!cancelled) setError("Failed to load MediaPipe IMAGE model: " + String(err));
      } finally {
        if (!cancelled) setIsLandmarkerLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Draw landmarks on preview canvas
  useEffect(() => {
    if (!previewFrames.length) return;
    const frame = previewFrames[previewIndex];
    const canvas = previewCanvasRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      frame.hands.forEach((hand) => {
        const color = frame.valid ? "#22c55e" : "#ef4444";
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.fillStyle = color;
        HAND_CONNECTIONS.forEach(([a, b]) => {
          const la = hand.landmarks[a];
          const lb = hand.landmarks[b];
          if (!la || !lb) return;
          ctx.beginPath();
          ctx.moveTo(la.x * img.width, la.y * img.height);
          ctx.lineTo(lb.x * img.width, lb.y * img.height);
          ctx.stroke();
        });
        hand.landmarks.forEach((lm) => {
          ctx.beginPath();
          ctx.arc(lm.x * img.width, lm.y * img.height, 4, 0, Math.PI * 2);
          ctx.fill();
        });
      });
      if (!frame.valid && !frame.hands.length) {
        ctx.fillStyle = "rgba(239,68,68,0.25)";
        ctx.fillRect(0, 0, img.width, img.height);
        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 20px sans-serif";
        ctx.fillText("No hand detected", 16, 36);
      }
    };
    img.src = frame.imageDataUrl;
  }, [previewFrames, previewIndex]);

  // ─── Mirror utility (TypeScript, matches Python) ────────────────────────
  function mirrorLandmarksX(hands: HandLandmarks[]): HandLandmarks[] {
    return hands.map((h) => ({
      ...h,
      landmarks: h.landmarks.map((lm) => ({ ...lm, x: 1 - lm.x })),
    }));
  }

  // ─── Frame extraction from video element ────────────────────────────────
  async function extractVideoFrames(
    file: File
  ): Promise<{ sequences: number[][][]; usedHandedness: string[]; stats: ProcessingStats }> {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const video = document.createElement("video");
      video.src = url;
      video.muted = true;
      video.playsInline = true;
      video.crossOrigin = "anonymous";

      const st: ProcessingStats = {
        framesTotal: 0, handsDetected: 0, framesValid: 0,
        framesRejected: 0, sequencesGenerated: 0, samplesSaved: 0,
        rejectionReasons: {},
      };

      const validFrameVectors: Array<{ vec: number[]; handedness: string }> = [];
      const previewBatch: PreviewFrame[] = [];

      video.onloadedmetadata = async () => {
        const duration = video.duration || 10;
        const fps = 30;
        const totalSteps = Math.floor(duration * fps);
        st.framesTotal = totalSteps;

        for (let step = 0; step < totalSteps; step++) {
          const t = step / fps;
          video.currentTime = t;
          await new Promise<void>((res) => { video.onseeked = () => res(); });

          const ts = step * (1000 / fps);
          const det = detectForVideo(video, ts);
          const sel = selectPrimaryHand(det.hands);

          // Build preview frame (sample every 15th frame)
          if (step % 15 === 0 && previewBatch.length < 20) {
            const offscreen = document.createElement("canvas");
            offscreen.width = video.videoWidth;
            offscreen.height = video.videoHeight;
            const ctx = offscreen.getContext("2d");
            ctx?.drawImage(video, 0, 0);
            previewBatch.push({
              imageDataUrl: offscreen.toDataURL("image/jpeg", 0.7),
              hands: sel.primaryHand ? [sel.primaryHand] : [],
              valid: !!sel.primaryHand,
            });
          }

          if (!sel.primaryHand) {
            st.framesRejected++;
            const reason = "No hand detected";
            st.rejectionReasons[reason] = (st.rejectionReasons[reason] || 0) + 1;
            continue;
          }
          st.handsDetected++;

          let handsForProcessing = [sel.primaryHand];
          const shouldMirror = mirrorLeft && sel.primaryHand.handedness === "Left";
          if (shouldMirror) {
            handsForProcessing = mirrorLandmarksX(handsForProcessing);
          }

          const res = processFrame(handsForProcessing, ts);
          if (!res.ok) {
            st.framesRejected++;
            st.rejectionReasons[res.reason] = (st.rejectionReasons[res.reason] || 0) + 1;
            continue;
          }

          st.framesValid++;
          validFrameVectors.push({
            vec: res.features.hands[0].vector,
            handedness: shouldMirror ? "Right" : (sel.primaryHand.handedness || "Unknown"),
          });
        }

        URL.revokeObjectURL(url);
        video.remove();

        // Sliding-window sequence extraction
        const sequences: number[][][] = [];
        const usedHandedness: string[] = [];
        let pos = 0;
        while (pos + SEQUENCE_LENGTH <= validFrameVectors.length) {
          const window = validFrameVectors.slice(pos, pos + SEQUENCE_LENGTH);
          sequences.push(window.map((f) => f.vec));
          usedHandedness.push(window[0].handedness);
          pos += SLIDE_STRIDE;
        }
        st.sequencesGenerated = sequences.length;
        setPreviewFrames(previewBatch);
        resolve({ sequences, usedHandedness, stats: st });
      };

      video.onerror = () => reject(new Error("Could not decode video file"));
    });
  }

  // ─── Process image file ────────────────────────────────────────────────
  async function processImageFile(file: File): Promise<{ sequence: number[][], handedness: string } | null> {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = async () => {
        const offscreen = document.createElement("canvas");
        offscreen.width = img.width;
        offscreen.height = img.height;
        const ctx = offscreen.getContext("2d");
        ctx?.drawImage(img, 0, 0);

        const det = detectForImage(offscreen);
        const sel = selectPrimaryHand(det.hands);

        const previewFrame: PreviewFrame = {
          imageDataUrl: offscreen.toDataURL("image/jpeg", 0.8),
          hands: sel.primaryHand ? [sel.primaryHand] : [],
          valid: !!sel.primaryHand,
        };
        setPreviewFrames([previewFrame]);
        setPreviewIndex(0);

        if (!sel.primaryHand) { URL.revokeObjectURL(url); resolve(null); return; }

        let handsForProcessing = [sel.primaryHand];
        const shouldMirror = mirrorLeft && sel.primaryHand.handedness === "Left";
        if (shouldMirror) handsForProcessing = mirrorLandmarksX(handsForProcessing);

        const res = processFrame(handsForProcessing, 0);
        URL.revokeObjectURL(url);
        if (!res.ok) { resolve(null); return; }

        const vec = res.features.hands[0].vector;
        // Replicate 30× for static signs
        const sequence: number[][] = Array.from({ length: SEQUENCE_LENGTH }, () => [...vec]);
        const handedness = shouldMirror ? "Right" : (sel.primaryHand.handedness || "Unknown");
        resolve({ sequence, handedness });
      };
      img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
      img.src = url;
    });
  }

  // ─── Save samples to IndexedDB ────────────────────────────────────────
  async function saveSamplesToStorage(
    sequences: number[][][],
    handednessList: string[],
    sourceType: SourceType,
    sourceFileName: string
  ): Promise<number> {
    let saved = 0;
    for (let i = 0; i < sequences.length; i++) {
      const seq = sequences[i];
      if (seq.length !== SEQUENCE_LENGTH) continue;
      if (seq.some((row) => row.length !== FEATURE_COUNT)) continue;

      await datasetStorage.saveSample({
        label: normalizedLabel,
        sequence: seq,
        sequenceLength: SEQUENCE_LENGTH,
        featureCount: FEATURE_COUNT,
        handedness: [handednessList[i] || "Unknown"],
        source: sourceType,
        datasetVersion: "v1",
        normalizationVersion: NORMALIZATION_VERSION,
        signLanguage: SIGN_LANGUAGE,
        signLanguageCode: SIGN_LANGUAGE_CODE,
        outputLanguage: OUTPUT_LANGUAGE,
        outputLanguageCode: OUTPUT_LANGUAGE_CODE,
        isSynthetic: false,
        metadata: {
          signerId: signerLabel,
          sourceFile: sourceFileName,
          sourceType,
          isStaticSign: sourceType === "image_upload",
        },
      } as any);
      saved++;
    }
    return saved;
  }

  // ─── Handle Upload Image ───────────────────────────────────────────────
  const handleImageUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !normalizedLabel) return;
    e.target.value = "";
    setError(null);
    setStats(null);
    setIsProcessing(true);
    setStatusMsg("Detecting hand landmarks in image…");

    try {
      const result = await processImageFile(file);
      const st: ProcessingStats = {
        framesTotal: 1, handsDetected: result ? 1 : 0,
        framesValid: result ? 1 : 0, framesRejected: result ? 0 : 1,
        sequencesGenerated: result ? 1 : 0, samplesSaved: 0,
        rejectionReasons: result ? {} : { "No hand detected": 1 },
      };

      if (result) {
        setStatusMsg("Saving to dataset storage…");
        const saved = await saveSamplesToStorage(
          [result.sequence], [result.handedness], "image_upload", file.name
        );
        st.samplesSaved = saved;
        setStatusMsg(`✅ Saved 1 static-sign sample for "${normalizedLabel}"`);
        onSampleSaved?.();
      } else {
        setStatusMsg("❌ No hand detected in image. Try a clearer photo.");
      }
      setStats(st);
    } catch (err) {
      setError("Image processing failed: " + String(err));
    } finally {
      setIsProcessing(false);
    }
  }, [normalizedLabel, mirrorLeft, signerLabel]);

  // ─── Handle Upload Video / Screen Recording ────────────────────────────
  const handleVideoUpload = useCallback(async (
    e: React.ChangeEvent<HTMLInputElement>,
    sourceType: SourceType
  ) => {
    const file = e.target.files?.[0];
    if (!file || !normalizedLabel) return;
    e.target.value = "";
    setError(null);
    setStats(null);
    setIsProcessing(true);
    setStatusMsg("Extracting frames and detecting landmarks…");

    try {
      const { sequences, usedHandedness, stats: extractStats } = await extractVideoFrames(file);

      if (sequences.length === 0) {
        setStatusMsg("❌ No valid 30-frame sequences extracted. Ensure hands are visible.");
        setStats(extractStats);
        return;
      }

      setStatusMsg("Saving sequences to dataset storage…");
      const saved = await saveSamplesToStorage(sequences, usedHandedness, sourceType, file.name);
      extractStats.samplesSaved = saved;
      setStats(extractStats);
      setStatusMsg(`✅ Saved ${saved} sequence(s) for "${normalizedLabel}"`);
      onSampleSaved?.();
    } catch (err) {
      setError("Video processing failed: " + String(err));
    } finally {
      setIsProcessing(false);
    }
  }, [normalizedLabel, mirrorLeft, signerLabel]);

  // ─── Styles ────────────────────────────────────────────────────────────
  const card: React.CSSProperties = {
    background: "var(--bg-card)", border: "1px solid var(--border-color)",
    borderRadius: "var(--radius-md)", padding: "20px",
  };
  const statBox: React.CSSProperties = {
    background: "rgba(0,0,0,0.3)", borderRadius: "var(--radius-sm)", padding: "12px",
    textAlign: "center",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", width: "100%", maxWidth: "900px", margin: "0 auto" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--accent-cyan)" }}>
          📁 Dataset Studio — Upload Media
        </h2>
        {isLandmarkerLoading && (
          <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>⏳ Loading MediaPipe IMAGE model…</span>
        )}
        {landmarkerReady && !isLandmarkerLoading && (
          <span style={{ fontSize: "0.8rem", color: "var(--accent-green)" }}>✅ MediaPipe Ready</span>
        )}
      </div>

      {/* ISL Lock + Label */}
      <div style={{ ...card, display: "flex", gap: "20px", flexWrap: "wrap", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: "0.7rem", color: "var(--accent-cyan)", fontWeight: 700, textTransform: "uppercase" }}>Sign Language</div>
          <div style={{ fontSize: "0.95rem", fontWeight: 800, color: "var(--text-primary)" }}>🇮🇳 Indian Sign Language (ISL) — LOCKED</div>
        </div>
        <div style={{ width: 1, height: 36, background: "var(--border-color)" }} />
        <div style={{ flex: 1, minWidth: 180 }}>
          <label style={{ fontSize: "0.7rem", color: "var(--text-secondary)", fontWeight: 700, textTransform: "uppercase" }}>
            English Output Label
          </label>
          <input
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="WATER, FOOD, HELP, A, B…"
            disabled={isProcessing}
            style={{ display: "block", width: "100%", padding: "8px 12px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", color: "var(--text-primary)", fontWeight: 700, fontSize: "1rem", marginTop: 4 }}
          />
        </div>
        <div style={{ minWidth: 140 }}>
          <label style={{ fontSize: "0.7rem", color: "var(--text-secondary)", fontWeight: 700, textTransform: "uppercase" }}>
            Signer ID
          </label>
          <input
            type="text"
            value={signerLabel}
            onChange={(e) => setSignerLabel(e.target.value)}
            disabled={isProcessing}
            style={{ display: "block", width: "100%", padding: "8px 12px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", color: "var(--text-primary)", fontWeight: 600, fontSize: "0.9rem", marginTop: 4 }}
          />
        </div>
        <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", userSelect: "none" }}>
          <input
            type="checkbox"
            checked={mirrorLeft}
            onChange={(e) => setMirrorLeft(e.target.checked)}
            disabled={isProcessing}
            style={{ width: 16, height: 16 }}
          />
          <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>Mirror left hand → right</span>
        </label>
      </div>

      {/* Upload Buttons */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
        {/* Upload Image */}
        <label style={{ ...card, display: "flex", flexDirection: "column", alignItems: "center", gap: 12, cursor: !landmarkerReady || isProcessing ? "not-allowed" : "pointer", opacity: (!landmarkerReady || isProcessing) ? 0.5 : 1, transition: "opacity 0.2s", textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem" }}>🖼️</div>
          <div style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>Upload Image</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
            .jpg .jpeg .png .webp<br />
            Static signs — single frame replicated ×30
          </div>
          <input
            type="file" accept=".jpg,.jpeg,.png,.webp" style={{ display: "none" }}
            disabled={!landmarkerReady || isProcessing || !normalizedLabel}
            onChange={handleImageUpload}
          />
        </label>

        {/* Upload Video */}
        <label style={{ ...card, display: "flex", flexDirection: "column", alignItems: "center", gap: 12, cursor: !landmarkerReady || isProcessing ? "not-allowed" : "pointer", opacity: (!landmarkerReady || isProcessing) ? 0.5 : 1, transition: "opacity 0.2s", textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem" }}>🎬</div>
          <div style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>Upload Video</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
            .mp4 .webm .mov .mkv<br />
            Dynamic signs — 30-frame sliding windows
          </div>
          <input
            type="file" accept=".mp4,.webm,.mov,.mkv" style={{ display: "none" }}
            disabled={!landmarkerReady || isProcessing || !normalizedLabel}
            onChange={(e) => handleVideoUpload(e, "video_upload")}
          />
        </label>

        {/* Screen Recording */}
        <label style={{ ...card, display: "flex", flexDirection: "column", alignItems: "center", gap: 12, cursor: !landmarkerReady || isProcessing ? "not-allowed" : "pointer", opacity: (!landmarkerReady || isProcessing) ? 0.5 : 1, transition: "opacity 0.2s", textAlign: "center" }}>
          <div style={{ fontSize: "2.5rem" }}>🖥️</div>
          <div style={{ fontWeight: 700, color: "var(--accent-cyan)" }}>Import Screen Recording</div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
            .mp4 .webm .mov .mkv<br />
            Sign videos recorded from screen
          </div>
          <input
            type="file" accept=".mp4,.webm,.mov,.mkv" style={{ display: "none" }}
            disabled={!landmarkerReady || isProcessing || !normalizedLabel}
            onChange={(e) => handleVideoUpload(e, "screen_recording")}
          />
        </label>
      </div>

      {/* Status / Progress */}
      {(isProcessing || statusMsg) && (
        <div style={{ ...card, display: "flex", alignItems: "center", gap: 12 }}>
          {isProcessing && (
            <div style={{ width: 18, height: 18, border: "3px solid var(--accent-cyan)", borderTopColor: "transparent", borderRadius: "50%", animation: "spin 0.8s linear infinite", flexShrink: 0 }} />
          )}
          <span style={{ fontSize: "0.92rem", color: isProcessing ? "var(--text-primary)" : "var(--accent-green)", fontWeight: 600 }}>
            {statusMsg}
          </span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.15)", border: "1px solid #ef4444", borderRadius: "var(--radius-sm)", color: "#fca5a5" }}>
          ⚠️ {error}
        </div>
      )}

      {/* Processing Stats */}
      {stats && (
        <div style={{ ...card }}>
          <div style={{ fontWeight: 700, fontSize: "0.9rem", color: "var(--accent-cyan)", marginBottom: 12 }}>📊 Processing Report</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10 }}>
            {[
              { label: "Frames", value: stats.framesTotal },
              { label: "Hands Found", value: stats.handsDetected },
              { label: "Valid Frames", value: stats.framesValid, ok: stats.framesValid > 0 },
              { label: "Rejected", value: stats.framesRejected, bad: stats.framesRejected > 0 },
              { label: "Sequences", value: stats.sequencesGenerated, ok: stats.sequencesGenerated > 0 },
              { label: "Saved", value: stats.samplesSaved, ok: stats.samplesSaved > 0 },
            ].map((item) => (
              <div key={item.label} style={statBox}>
                <div style={{ fontSize: "1.4rem", fontWeight: 800, color: item.ok ? "var(--accent-green)" : item.bad ? "#ef4444" : "var(--text-primary)" }}>
                  {item.value}
                </div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", textTransform: "uppercase", marginTop: 2 }}>{item.label}</div>
              </div>
            ))}
          </div>
          {Object.keys(stats.rejectionReasons).length > 0 && (
            <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(239,68,68,0.08)", borderRadius: "var(--radius-sm)" }}>
              <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "#fca5a5", marginBottom: 6 }}>Rejection Reasons:</div>
              {Object.entries(stats.rejectionReasons).map(([reason, count]) => (
                <div key={reason} style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>• {reason}: {count} frame(s)</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Landmark Overlay Preview */}
      {previewFrames.length > 0 && (
        <div style={{ ...card }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div style={{ fontWeight: 700, fontSize: "0.9rem", color: "var(--accent-cyan)" }}>
              👁️ Visual Landmark Verification — <span style={{ color: previewFrames[previewIndex]?.valid ? "var(--accent-green)" : "#ef4444" }}>
                {previewFrames[previewIndex]?.valid ? "GREEN = Valid Detection" : "RED = No Hand Detected"}
              </span>
            </div>
            {previewFrames.length > 1 && (
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button type="button" onClick={() => setPreviewIndex(Math.max(0, previewIndex - 1))} disabled={previewIndex === 0}
                  style={{ padding: "4px 10px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: 4, color: "var(--text-primary)", cursor: "pointer" }}>◀</button>
                <span style={{ fontSize: "0.82rem", color: "var(--text-secondary)" }}>
                  Frame {previewIndex + 1} / {previewFrames.length}
                </span>
                <button type="button" onClick={() => setPreviewIndex(Math.min(previewFrames.length - 1, previewIndex + 1))} disabled={previewIndex === previewFrames.length - 1}
                  style={{ padding: "4px 10px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: 4, color: "var(--text-primary)", cursor: "pointer" }}>▶</button>
              </div>
            )}
          </div>
          <canvas
            ref={previewCanvasRef}
            style={{ width: "100%", maxHeight: 360, objectFit: "contain", borderRadius: "var(--radius-sm)", border: `2px solid ${previewFrames[previewIndex]?.valid ? "var(--accent-green)" : "#ef4444"}` }}
          />
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};
