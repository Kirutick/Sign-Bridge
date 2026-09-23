import React, { useState, useEffect, useRef } from "react";
import { CameraController } from "../system/cameraController";
import { initHandLandmarker, detectForVideo, initImageHandLandmarker, detectForImage, disposeImageHandLandmarker } from "../services/handLandmarker";
import { selectPrimaryHand } from "../system/selectPrimaryHand";
import { processFrame } from "../services/landmarkProcessor";
import { signClassifier, RawPrediction } from "../services/signClassifier";
import { SequenceBuffer } from "../services/sequenceBuffer";
import { LandmarkCanvas } from "../components/LandmarkCanvas";
import { speechService } from "../services/speechService";
import type { HandLandmarks } from "../types/landmarks";
import islGuideData from "../data/isl_sign_guide.json";

interface PracticePageProps {
  targetSign?: string | null;
}

type PracticeMode = "LEARNING" | "QUIZ";
type MediaMode = "CAMERA" | "IMAGE" | "VIDEO";
type EvaluationState = "WAITING" | "HOLDING" | "CORRECT" | "TRY_AGAIN";

export const PracticePage: React.FC<PracticePageProps> = ({ targetSign: initialTarget }) => {
  const [targetSign, setTargetSign] = useState<string>(initialTarget || "A");
  const [practiceMode, setPracticeMode] = useState<PracticeMode>("LEARNING");
  const [mediaMode, setMediaMode] = useState<MediaMode>("CAMERA");
  const [evalState, setEvalState] = useState<EvaluationState>("WAITING");
  const [detectedPrediction, setDetectedPrediction] = useState<RawPrediction | null>(null);
  const [holdProgress, setHoldProgress] = useState<number>(0);
  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [hands, setHands] = useState<HandLandmarks[]>([]);
  
  const [uploadedMedia, setUploadedMedia] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const cameraControllerRef = useRef<CameraController>(new CameraController());
  const sequenceBufferRef = useRef<SequenceBuffer>(new SequenceBuffer(30, 63));
  const holdStartRef = useRef<number | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  useEffect(() => {
    if (initialTarget) {
      setTargetSign(initialTarget);
    }
  }, [initialTarget]);

  useEffect(() => {
    async function init() {
      try {
        await initHandLandmarker();
        await initImageHandLandmarker();
        await signClassifier.loadModel();
      } catch (err) {
        console.error("Failed to init models:", err);
      }
    }
    init();
    return () => {
      stopCamera();
      disposeImageHandLandmarker();
      if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
    };
  }, []);

  const stopCamera = () => {
    cameraControllerRef.current.stopCamera();
    setCameraActive(false);
    if (animFrameIdRef.current) cancelAnimationFrame(animFrameIdRef.current);
  };

  const startCamera = async () => {
    setMediaMode("CAMERA");
    setUploadedMedia(null);
    try {
      const stream = await cameraControllerRef.current.startCamera();
      if (videoRef.current && stream) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setCameraActive(true);
          startInferenceLoop();
        };
      }
    } catch (err) {
      console.error("Camera error:", err);
    }
  };

  const handleMediaUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    stopCamera();
    const url = URL.createObjectURL(file);
    setUploadedMedia(url);
    if (file.type.startsWith("image/")) {
      setMediaMode("IMAGE");
    } else if (file.type.startsWith("video/")) {
      setMediaMode("VIDEO");
    }
    setEvalState("WAITING");
    setDetectedPrediction(null);
  };

  const processUploadedImage = async () => {
    if (!imgRef.current) return;
    try {
      const result = await detectForImage(imgRef.current);
      if (result.hands && result.hands.length > 0) {
        const selection = selectPrimaryHand(result.hands);
        if (selection.primaryHand) {
          const hand = selection.primaryHand;
          setHands([hand]);
          const procRes = processFrame([hand], 1);
          if (procRes.ok && procRes.features.hands.length > 0) {
            const featureVector = procRes.features.hands[0].vector;
            sequenceBufferRef.current.clear();
            for (let i = 0; i < 30; i++) sequenceBufferRef.current.push(featureVector);
            const seq = sequenceBufferRef.current.getSequence();
            if (seq) {
              const pred = await signClassifier.predict(seq);
              setDetectedPrediction(pred);
              evaluatePrediction(pred);
            }
          }
        }
      } else {
        setHands([]);
        setDetectedPrediction(null);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const startInferenceLoop = () => {
    const loop = async () => {
      if (!videoRef.current || videoRef.current.readyState < 2) {
        animFrameIdRef.current = requestAnimationFrame(loop);
        return;
      }
      const now = performance.now();
      const results = detectForVideo(videoRef.current, now);

      if (results.hands && results.hands.length > 0) {
        const selection = selectPrimaryHand(results.hands);
        if (selection.primaryHand) {
          const primary = selection.primaryHand;
          setHands([primary]);
          
          const procRes = processFrame([primary], now);
          if (procRes.ok && procRes.features.hands.length > 0) {
            const fv = procRes.features.hands[0].vector;
            sequenceBufferRef.current.push(fv);
            if (sequenceBufferRef.current.isFull()) {
              const seq = sequenceBufferRef.current.getSequence();
              if (seq) {
                const pred = await signClassifier.predict(seq);
                setDetectedPrediction(pred);
                handleContinuousEvaluation(pred, now);
              }
            }
          }
        }
      } else {
        setHands([]);
        setDetectedPrediction(null);
        holdStartRef.current = null;
        setHoldProgress(0);
        if (evalState !== "CORRECT") setEvalState("WAITING");
      }
      animFrameIdRef.current = requestAnimationFrame(loop);
    };
    animFrameIdRef.current = requestAnimationFrame(loop);
  };

  const evaluatePrediction = (pred: RawPrediction) => {
    if (pred.label === targetSign && pred.confidence > 0.6) {
      setEvalState("CORRECT");
      speechService.speak("Correct! " + targetSign.replace(/_/g, " "));
    } else {
      setEvalState("TRY_AGAIN");
    }
  };

  const handleContinuousEvaluation = (pred: RawPrediction, now: number) => {
    if (evalState === "CORRECT") return;
    
    if (pred.label === targetSign && pred.confidence > 0.6) {
      if (!holdStartRef.current) {
        holdStartRef.current = now;
        setEvalState("HOLDING");
      }
      const elapsed = now - holdStartRef.current;
      const prog = Math.min(100, (elapsed / 1000) * 100);
      setHoldProgress(prog);
      
      if (elapsed > 1000) {
        setEvalState("CORRECT");
        speechService.speak("Correct! " + targetSign.replace(/_/g, " "));
        setTimeout(() => {
          setEvalState("WAITING");
          setHoldProgress(0);
          holdStartRef.current = null;
        }, 2000);
      }
    } else {
      holdStartRef.current = null;
      setHoldProgress(0);
      setEvalState("TRY_AGAIN");
    }
  };

  const targetSignInfo = islGuideData.signs.find(s => s.label === targetSign);

  return (
    <div style={{ width: '100%', maxWidth: '1080px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 style={{ margin: 0, color: 'var(--text-primary)' }}>Practice Studio</h1>
          <p style={{ margin: 0, color: 'var(--text-secondary)' }}>Master ISL signs with real-time AI feedback.</p>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={() => setPracticeMode("LEARNING")} style={{ padding: '8px 16px', background: practiceMode === "LEARNING" ? 'var(--accent-blue)' : 'var(--bg-card)', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Learning Mode</button>
          <button onClick={() => setPracticeMode("QUIZ")} style={{ padding: '8px 16px', background: practiceMode === "QUIZ" ? 'var(--accent-purple)' : 'var(--bg-card)', color: '#fff', border: 'none', borderRadius: '4px', cursor: 'pointer' }}>Quiz Mode</button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '20px' }}>
        {/* LEFT PANEL - Target Sign */}
        <div style={{ flex: '1', background: 'var(--bg-card)', padding: '20px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div>
            <label style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Select Target Sign</label>
            <select value={targetSign} onChange={(e) => setTargetSign(e.target.value)} style={{ width: '100%', padding: '10px', marginTop: '4px', background: 'rgba(0,0,0,0.3)', color: '#fff', border: '1px solid var(--border-color)', borderRadius: '4px' }}>
              {islGuideData.signs.map(s => <option key={s.label} value={s.label}>{s.displayName} ({s.signType})</option>)}
            </select>
          </div>

          <div style={{ textAlign: 'center', padding: '20px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
             <h2 style={{ fontSize: '1.2rem', color: 'var(--text-secondary)', margin: '0 0 8px 0' }}>TARGET</h2>
             {practiceMode === "QUIZ" ? (
               <div style={{ fontSize: '3rem', fontWeight: 900, color: 'var(--text-primary)' }}>???</div>
             ) : (
               <>
                 <div style={{ fontSize: '2.5rem', fontWeight: 900, color: 'var(--accent-cyan)' }}>{targetSignInfo?.displayName}</div>
                 {targetSignInfo?.reference ? (
                   <img src={targetSignInfo.reference} alt="reference" style={{ maxHeight: '150px', marginTop: '16px', borderRadius: '4px' }} />
                 ) : (
                   <div style={{ marginTop: '16px', color: 'var(--text-muted)' }}>Reference unavailable</div>
                 )}
               </>
             )}
          </div>

          {practiceMode === "LEARNING" && targetSignInfo?.instructions && targetSignInfo.instructions.length > 0 && (
            <div>
              <h3 style={{ fontSize: '0.9rem', color: 'var(--accent-cyan)' }}>Instructions:</h3>
              <ol style={{ margin: 0, paddingLeft: '20px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                {targetSignInfo.instructions.map((step, idx) => <li key={idx}>{step}</li>)}
              </ol>
            </div>
          )}
        </div>

        {/* RIGHT PANEL - Live Feedback */}
        <div style={{ flex: '2', background: 'var(--bg-card)', padding: '20px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
             <h2 style={{ margin: 0, fontSize: '1.2rem' }}>Your Performance</h2>
             <div style={{ display: 'flex', gap: '8px' }}>
               <button onClick={startCamera} style={{ padding: '6px 12px', background: mediaMode === 'CAMERA' ? 'var(--accent-blue)' : 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-color)', borderRadius: '4px', cursor: 'pointer' }}>📹 Webcam</button>
               <label style={{ padding: '6px 12px', background: mediaMode !== 'CAMERA' ? 'var(--accent-blue)' : 'var(--bg-card)', color: '#fff', border: '1px solid var(--border-color)', borderRadius: '4px', cursor: 'pointer' }}>
                 📁 Upload Media
                 <input type="file" accept="image/*,video/*" style={{ display: 'none' }} onChange={handleMediaUpload} />
               </label>
             </div>
          </div>

          <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', background: '#000', borderRadius: '8px', overflow: 'hidden' }}>
            {mediaMode === "CAMERA" ? (
              cameraActive ? (
                <>
                  <video ref={videoRef} style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} playsInline muted />
                  <LandmarkCanvas hands={hands} width={640} height={480} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', transform: 'scaleX(-1)' }} />
                </>
              ) : (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <button onClick={startCamera} style={{ padding: '12px 24px', fontSize: '1.1rem', background: 'var(--accent-blue)', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer' }}>Start Camera</button>
                </div>
              )
            ) : uploadedMedia ? (
              mediaMode === "IMAGE" ? (
                <>
                  <img ref={imgRef} src={uploadedMedia} onLoad={processUploadedImage} style={{ width: '100%', height: '100%', objectFit: 'contain' }} alt="Uploaded" />
                  <LandmarkCanvas hands={hands} width={640} height={480} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }} />
                </>
              ) : (
                <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', color: '#fff' }}>
                  <video src={uploadedMedia} controls style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  <div style={{ position: 'absolute', top: 10, background: 'rgba(0,0,0,0.7)', padding: '8px', borderRadius: '4px' }}>Video practice processing coming soon</div>
                </div>
              )
            ) : null}
          </div>

          {/* Feedback Section */}
          <div style={{ display: 'flex', gap: '20px', background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>TARGET</div>
              <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>{practiceMode === "QUIZ" ? "???" : targetSignInfo?.displayName}</div>
            </div>
            
            <div style={{ width: '1px', background: 'rgba(255,255,255,0.1)' }}></div>

            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>DETECTED</div>
              {detectedPrediction ? (
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: detectedPrediction.label === targetSign ? '#10b981' : '#f59e0b' }}>
                  {detectedPrediction.label.replace(/_/g, " ")}
                </div>
              ) : (
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-muted)' }}>Waiting...</div>
              )}
            </div>

            <div style={{ width: '1px', background: 'rgba(255,255,255,0.1)' }}></div>

            <div style={{ flex: 1 }}>
              <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '4px' }}>CONFIDENCE</div>
              {detectedPrediction ? (
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: detectedPrediction.confidence > 0.6 ? '#10b981' : '#f59e0b' }}>
                  {(detectedPrediction.confidence * 100).toFixed(1)}%
                </div>
              ) : (
                <div style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-muted)' }}>--</div>
              )}
            </div>
          </div>

          {/* Hold Progress Bar for continuous video */}
          {mediaMode === "CAMERA" && (
            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ width: `${holdProgress}%`, height: '100%', background: '#10b981', transition: 'width 0.1s linear' }}></div>
            </div>
          )}

          {evalState === "CORRECT" && (
             <div style={{ padding: '12px', background: 'rgba(16, 185, 129, 0.15)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.4)', borderRadius: '4px', textAlign: 'center', fontWeight: 800 }}>
               ✅ Excellent! Sign Mastered.
               {practiceMode === "QUIZ" && <div style={{ fontSize: '0.9rem', fontWeight: 400, marginTop: '4px' }}>It was: {targetSignInfo?.displayName}</div>}
             </div>
          )}

        </div>
      </div>
    </div>
  );
};
