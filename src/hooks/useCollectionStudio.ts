import { useState, useEffect, useRef, useCallback } from "react";
import { LabelStore, SignLabel } from "../dataCollection/labelStore";
import { SessionTracker, CollectionSession } from "../dataCollection/sessionTracker";
import { useRecordingSession } from "./useRecordingSession";
import { datasetStorage } from "../services/datasetStorage";
import type { HandLandmarks } from "../types/landmarks";

export interface UseCollectionStudioResult {
  activeLabel: SignLabel | null;
  labels: SignLabel[];
  sessionInfo: CollectionSession;
  sessionState: ReturnType<typeof useRecordingSession>["sessionState"];
  duplicateWarning: ReturnType<typeof useRecordingSession>["duplicateWarning"];
  createLabel: (name: string, target?: number) => SignLabel;
  selectLabel: (labelId: string) => void;
  deleteLabel: (labelId: string) => Promise<number>;
  startStudioRecording: () => void;
  cancelStudioRecording: () => void;
  keepSample: () => void;
  retrySample: () => Promise<void>;
  deleteCurrentSample: () => Promise<void>;
  processStudioFrame: (hands: HandLandmarks[], timestampMs: number) => void;
  refreshLabels: () => Promise<void>;
}

export function useCollectionStudio(
  initialGlobalTarget: number = 20
): UseCollectionStudioResult {
  const labelStoreRef = useRef<LabelStore>(new LabelStore(initialGlobalTarget));
  const sessionTrackerRef = useRef<SessionTracker>(new SessionTracker());

  const [labels, setLabels] = useState<SignLabel[]>([]);
  const [activeLabelId, setActiveLabelId] = useState<string | null>(null);
  const [sessionInfo, setSessionInfo] = useState<CollectionSession>(() => sessionTrackerRef.current.getSession());

  const activeLabel = labels.find((l) => l.id === activeLabelId) || labels[0] || null;

  const recording = useRecordingSession({
    label: activeLabel ? activeLabel.name : "HELLO",
  });

  const refreshLabels = useCallback(async () => {
    const list = await labelStoreRef.current.syncWithStorage();
    setLabels(list);
    setSessionInfo(sessionTrackerRef.current.getSession());
  }, []);

  useEffect(() => {
    refreshLabels();
  }, [refreshLabels]);

  const createLabel = useCallback((name: string, target?: number): SignLabel => {
    const created = labelStoreRef.current.createLabel(name, target);
    setActiveLabelId(created.id);
    refreshLabels();
    return created;
  }, [refreshLabels]);

  const selectLabel = useCallback((labelId: string) => {
    setActiveLabelId(labelId);
  }, []);

  const deleteLabel = useCallback(async (labelId: string): Promise<number> => {
    const deletedCount = await labelStoreRef.current.deleteLabel(labelId);
    if (activeLabelId === labelId) {
      setActiveLabelId(null);
    }
    await refreshLabels();
    return deletedCount;
  }, [activeLabelId, refreshLabels]);

  const startStudioRecording = useCallback(() => {
    if (activeLabel) {
      recording.startRecording(activeLabel.name);
    }
  }, [activeLabel, recording]);

  const cancelStudioRecording = useCallback(() => {
    recording.cancelRecording("User pressed cancel/Esc");
  }, [recording]);

  // Keep Sample -> Confirms and returns to Ready state for same label (§9)
  const keepSample = useCallback(() => {
    if (activeLabel) {
      sessionTrackerRef.current.recordSampleSaved(activeLabel.id);
    }
    recording.resetSession();
    refreshLabels();
  }, [activeLabel, recording, refreshLabels]);

  // Delete Current Sample -> Deletes persisted sample from storage (§9)
  const deleteCurrentSample = useCallback(async () => {
    if (recording.sessionState.sample) {
      const sampleId = recording.sessionState.sample.id;
      await datasetStorage.deleteSample(sampleId);
      if (activeLabel) {
        sessionTrackerRef.current.recordSampleDeleted(activeLabel.id);
      }
      recording.resetSession();
      await refreshLabels();
    }
  }, [recording, activeLabel, refreshLabels]);

  // Retry Sample -> Deletes sample and immediately re-enters Ready state for same label (§9)
  const retrySample = useCallback(async () => {
    if (recording.sessionState.sample) {
      const sampleId = recording.sessionState.sample.id;
      await datasetStorage.deleteSample(sampleId);
      if (activeLabel) {
        sessionTrackerRef.current.recordSampleDeleted(activeLabel.id);
      }
    }
    recording.resetSession();
    await refreshLabels();
    if (activeLabel) {
      recording.startRecording(activeLabel.name);
    }
  }, [recording, activeLabel, refreshLabels]);

  return {
    activeLabel,
    labels,
    sessionInfo,
    sessionState: recording.sessionState,
    duplicateWarning: recording.duplicateWarning,
    createLabel,
    selectLabel,
    deleteLabel,
    startStudioRecording,
    cancelStudioRecording,
    keepSample,
    retrySample,
    deleteCurrentSample,
    processStudioFrame: recording.processRecordingFrame,
    refreshLabels,
  };
}
