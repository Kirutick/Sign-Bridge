import React, { useState, useEffect } from "react";
import type { RecordedSample } from "../dataCollection/validateSample";
import type { OutlierFlag } from "../dataPipeline/outlierDetection";
import type { HandLandmarks } from "../types/landmarks";
import { reviewStore } from "../dataPipeline/reviewStore";
import { LandmarkCanvas } from "./LandmarkCanvas";

interface DataReviewViewerProps {
  samples: RecordedSample[];
  flags: OutlierFlag[];
  onClose: () => void;
}

export const DataReviewViewer: React.FC<DataReviewViewerProps> = ({ samples, flags, onClose }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Filter down to flagged samples that haven't been reviewed yet (optional, but good UX)
  // For simplicity, we just iterate through all provided flags.
  const sampleMap = new Map<string, RecordedSample>();
  samples.forEach(s => sampleMap.set(s.id, s));

  const validFlags = flags.filter(f => sampleMap.has(f.sampleId));

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setFrameIndex((prev) => (prev + 1) % 30);
      }, 50); // 20 fps
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  if (validFlags.length === 0) {
    return (
      <div className="p-4 bg-gray-900 border border-gray-700 rounded-lg text-white">
        <h2 className="text-xl font-bold mb-4 text-green-400">All Clear</h2>
        <p>No outlier flags require review.</p>
        <button className="mt-4 px-4 py-2 bg-blue-600 rounded" onClick={onClose}>Close</button>
      </div>
    );
  }

  const currentFlag = validFlags[currentIndex];
  const sample = sampleMap.get(currentFlag.sampleId)!;

  const handleMark = (action: "MARK_VALID" | "MARK_EXCLUDED") => {
    reviewStore.logAction(sample.id, action);
    if (currentIndex < validFlags.length - 1) {
      setCurrentIndex(currentIndex + 1);
      setFrameIndex(0);
    } else {
      onClose(); // Finished
    }
  };

  // Convert 63-feature normalized frame back into pseudo landmarks for rendering
  const currentFrameRaw = sample.sequence[frameIndex] || [];
  const fakeLandmarks = [];
  if (currentFrameRaw.length === 63) {
    for (let i = 0; i < 21; i++) {
      // Un-normalize loosely to fit in a 300x300 canvas. 
      // The features are centered at (0,0,0) wrist, with values typically in [-5, 5].
      // So we scale by 20 and offset by 150.
      fakeLandmarks.push({
        x: (currentFrameRaw[i * 3] * 20 + 150) / 300,
        y: (currentFrameRaw[i * 3 + 1] * 20 + 150) / 300,
        z: currentFrameRaw[i * 3 + 2] / 5, // z doesn't matter much for 2D rendering
      });
    }
  }

  const hands: HandLandmarks[] = fakeLandmarks.length === 21 ? [{ landmarks: fakeLandmarks, handedness: "Unknown", handednessScore: 1 } as HandLandmarks] : [];

  return (
    <div className="p-6 bg-gray-900 border border-red-900 rounded-lg text-white max-w-2xl mx-auto flex flex-col gap-4">
      <div className="flex justify-between items-center border-b border-gray-700 pb-2">
        <h2 className="text-xl font-bold text-red-400">Review Required ({currentIndex + 1} / {validFlags.length})</h2>
        <button onClick={onClose} className="text-gray-400 hover:text-white">✕</button>
      </div>

      <div className="bg-gray-800 p-4 rounded text-sm">
        <div className="flex justify-between mb-2">
          <span className="text-gray-400">Sample ID:</span>
          <span className="font-mono text-xs">{currentFlag.sampleId}</span>
        </div>
        <div className="flex justify-between mb-2">
          <span className="text-gray-400">Class Label:</span>
          <span className="font-bold text-yellow-400">{currentFlag.label}</span>
        </div>
        <div className="flex justify-between mb-2">
          <span className="text-gray-400">Flag Type:</span>
          <span className="font-bold text-red-400">{currentFlag.heuristic}</span>
        </div>
        <div className="mt-2 text-gray-300 italic border-l-2 border-red-500 pl-2">
          {currentFlag.details}
        </div>
      </div>

      <div className="flex justify-center bg-black rounded-lg overflow-hidden border border-gray-700 relative">
        <LandmarkCanvas hands={hands} width={300} height={300} />
        
        <div className="absolute bottom-2 left-2 right-2 flex justify-between text-xs text-gray-400">
          <span>Frame: {frameIndex + 1}/30</span>
          <button 
            onClick={() => setIsPlaying(!isPlaying)}
            className="bg-gray-800 px-2 rounded hover:bg-gray-700"
          >
            {isPlaying ? "Pause" : "Play"}
          </button>
        </div>
        
        <input 
          type="range" 
          min="0" max="29" 
          value={frameIndex} 
          onChange={(e) => {
            setFrameIndex(parseInt(e.target.value, 10));
            setIsPlaying(false);
          }}
          className="absolute bottom-0 w-full opacity-50 hover:opacity-100" 
        />
      </div>

      <div className="flex gap-4 mt-2">
        <button 
          onClick={() => handleMark("MARK_EXCLUDED")}
          className="flex-1 bg-red-600 hover:bg-red-500 text-white font-bold py-3 px-4 rounded transition-colors"
        >
          Exclude from Dataset
        </button>
        <button 
          onClick={() => handleMark("MARK_VALID")}
          className="flex-1 bg-green-600 hover:bg-green-500 text-white font-bold py-3 px-4 rounded transition-colors"
        >
          Mark as Valid (Ignore Flag)
        </button>
      </div>
    </div>
  );
};
