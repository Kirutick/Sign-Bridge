import { useState, useRef, useCallback } from "react";
import {
  PredictionStabilizer,
  StabilizationResult,
  FrameInput,
  StabilizationConfig,
} from "../recognition/predictionStabilizer";
import { DEFAULT_INFERENCE_CONFIG } from "../config/inference";

export interface UseSignRecognitionReturn {
  committedTokens: string[];
  committedString: string;
  isPaused: boolean;
  confidenceThreshold: number;
  currentResult: StabilizationResult | null;
  startRecognition: () => void;
  pauseRecognition: () => void;
  clearCommittedText: () => void;
  backspaceLastToken: () => void;
  undoLastToken: () => void;
  commitExplicitToken: (token: string) => void;
  updateThreshold: (newThreshold: number) => void;
  processPredictionFrame: (input: FrameInput) => StabilizationResult | null;
}

export function useSignRecognition(
  initialConfig: Partial<StabilizationConfig> = {}
): UseSignRecognitionReturn {
  const stabilizerRef = useRef<PredictionStabilizer>(
    new PredictionStabilizer({
      confidenceThreshold: DEFAULT_INFERENCE_CONFIG.confidenceThreshold,
      ...initialConfig,
    })
  );

  const [committedTokens, setCommittedTokens] = useState<string[]>([]);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [confidenceThreshold, setConfidenceThresholdState] = useState<number>(
    DEFAULT_INFERENCE_CONFIG.confidenceThreshold
  );
  const [currentResult, setCurrentResult] = useState<StabilizationResult | null>(null);

  const startRecognition = useCallback(() => {
    setIsPaused(false);
  }, []);

  const pauseRecognition = useCallback(() => {
    setIsPaused(true);
    stabilizerRef.current.reset(); // Pause signal resets state machine (§4)
    setCurrentResult(null);
  }, []);

  const clearCommittedText = useCallback(() => {
    stabilizerRef.current.clearCommittedText();
    setCommittedTokens([]);
  }, []);

  const backspaceLastToken = useCallback(() => {
    stabilizerRef.current.backspaceCommittedText();
    setCommittedTokens(stabilizerRef.current.getFullCommittedText());
  }, []);

  const undoLastToken = useCallback(() => {
    stabilizerRef.current.backspaceCommittedText();
    setCommittedTokens(stabilizerRef.current.getFullCommittedText());
  }, []);

  const commitExplicitToken = useCallback((token: string) => {
    if (!token) return;
    const current = stabilizerRef.current.getFullCommittedText();
    current.push(token);
    // synchronize stabilizer
    stabilizerRef.current.clearCommittedText();
    for (const t of current) {
      // push into stabilizer's internal array
      (stabilizerRef.current as any).committedText.push(t);
    }
    setCommittedTokens([...current]);
  }, []);

  const updateThreshold = useCallback((newThreshold: number) => {
    setConfidenceThresholdState(newThreshold);
    stabilizerRef.current.updateConfig({ confidenceThreshold: newThreshold });
  }, []);

  const processPredictionFrame = useCallback(
    (input: FrameInput): StabilizationResult | null => {
      if (isPaused) return null;

      const result = stabilizerRef.current.processFrame(input);
      setCurrentResult(result);

      if (result.committedTextDelta) {
        setCommittedTokens(result.committedText);
      }

      return result;
    },
    [isPaused]
  );

  const committedString = committedTokens.join(" ");

  return {
    committedTokens,
    committedString,
    isPaused,
    confidenceThreshold,
    currentResult,
    startRecognition,
    pauseRecognition,
    clearCommittedText,
    backspaceLastToken,
    undoLastToken,
    commitExplicitToken,
    updateThreshold,
    processPredictionFrame,
  };
}
