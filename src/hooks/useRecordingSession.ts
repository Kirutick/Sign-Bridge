import { useState, useEffect, useRef, useCallback } from "react";
import { RecordingSession, RecordingStatePayload } from "../dataCollection/recordingStateMachine";
import { RecordingConfig } from "../dataCollection/recordingConfig";
import { RecordedSample } from "../dataCollection/validateSample";
import { checkForNearDuplicate, DuplicateCheckResult } from "../dataCollection/duplicateDetector";
import { datasetStorage } from "../services/datasetStorage";
import type { DatasetSample } from "../types/dataset";
import type { HandLandmarks } from "../types/landmarks";

export interface UseRecordingSessionResult {
  sessionState: RecordingStatePayload;
  duplicateWarning: DuplicateCheckResult | null;
  recordedSamples: RecordedSample[];
  startRecording: (label: string) => void;
  cancelRecording: (reason?: string) => void;
  resetSession: () => void;
  processRecordingFrame: (hands: HandLandmarks[], timestampMs: number) => void;
  loadSamples: () => Promise<void>;
}

export function useRecordingSession(
  initialConfig?: Partial<RecordingConfig>
): UseRecordingSessionResult {
  const sessionRef = useRef<RecordingSession | null>(null);
  const [sessionState, setSessionState] = useState<RecordingStatePayload>(() => {
    const session = new RecordingSession(initialConfig, (newState) => {
      setSessionState(newState);
    });
    sessionRef.current = session;
    return session.getState();
  });

  const [duplicateWarning, setDuplicateWarning] = useState<DuplicateCheckResult | null>(null);
  const [recordedSamples, setRecordedSamples] = useState<RecordedSample[]>([]);

  // Load existing dataset samples from IndexedDB
  const loadSamples = useCallback(async () => {
    try {
      const dbSamples: DatasetSample[] = await datasetStorage.listSamples();
      const formatted: RecordedSample[] = dbSamples.map((s) => {
        const handVal = Array.isArray(s.handedness) ? s.handedness[0] : s.handedness;
        return {
          id: s.id,
          label: s.label,
          sequence: s.sequence,
          sequenceLength: s.sequence.length,
          featureCount: s.sequence[0]?.length || 63,
          timestamp: s.createdAt,
          handedness: (handVal as "Left" | "Right" | "Unknown") || "Unknown",
          source: "webcam",
          datasetVersion: "1.0.0",
          normalizationVersion: (s as DatasetSample & { normalizationVersion?: string }).normalizationVersion ?? "unknown",
        };
      });
      setRecordedSamples(formatted);
    } catch (err) {
      console.warn("Error loading stored samples:", err);
    }
  }, []);

  useEffect(() => {
    loadSamples();
  }, [loadSamples]);

  // Handle Countdown Ticker (§2)
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    if (sessionState.state === "COUNTDOWN") {
      timer = setInterval(() => {
        sessionRef.current?.tickCountdown();
      }, 1000);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [sessionState.state]);

  // Handle Session Completion & Non-Blocking Advisory Duplicate Check (§9)
  useEffect(() => {
    if (sessionState.state === "COMPLETED" && sessionState.sample) {
      const completed = sessionState.sample;

      // 1. Run Non-Blocking Advisory Duplicate Detection (§9)
      const dupResult = checkForNearDuplicate(completed, recordedSamples, 0.05);
      setDuplicateWarning(dupResult.isDuplicate ? dupResult : null);

      // 2. Persist to IndexedDB
      const handVal: ("Left" | "Right")[] | undefined =
        completed.handedness === "Left" || completed.handedness === "Right"
          ? [completed.handedness]
          : undefined;

      datasetStorage.addSample({
        id: completed.id,
        label: completed.label,
        sequence: completed.sequence,
        sequenceLength: completed.sequenceLength,
        featureCount: completed.featureCount,
        createdAt: completed.timestamp,
        handedness: handVal,
      })
        .then(() => loadSamples())
        .catch(console.error);
    }
  }, [sessionState.state, sessionState.sample, recordedSamples, loadSamples]);

  const startRecording = useCallback((label: string) => {
    setDuplicateWarning(null);
    sessionRef.current?.updateConfig({ label });
    sessionRef.current?.startCountdown();
  }, []);

  const cancelRecording = useCallback((reason?: string) => {
    sessionRef.current?.cancel(reason);
  }, []);

  const resetSession = useCallback(() => {
    setDuplicateWarning(null);
    sessionRef.current?.reset();
  }, []);

  const processRecordingFrame = useCallback((hands: HandLandmarks[], timestampMs: number) => {
    sessionRef.current?.onFrame({ hands, timestampMs });
  }, []);

  return {
    sessionState,
    duplicateWarning,
    recordedSamples,
    startRecording,
    cancelRecording,
    resetSession,
    processRecordingFrame,
    loadSamples,
  };
}
