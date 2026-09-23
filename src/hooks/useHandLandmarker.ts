import { useState, useEffect, useCallback, useRef } from "react";
import {
  initHandLandmarker,
  disposeHandLandmarker,
  getActiveDelegate,
} from "../services/handLandmarker";
import {
  VisionDelegate,
  HandLandmarkerError,
} from "../types/landmarks";

export interface UseHandLandmarkerReturn {
  isReady: boolean;
  isLoading: boolean;
  error: HandLandmarkerError | null;
  activeDelegate: VisionDelegate;
  reinitialize: (forceCpu?: boolean) => Promise<void>;
}

export function useHandLandmarker(): UseHandLandmarkerReturn {
  const [isReady, setIsReady] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<HandLandmarkerError | null>(null);
  const [activeDelegate, setActiveDelegate] = useState<VisionDelegate>("GPU");

  const isMountedRef = useRef<boolean>(true);

  const init = useCallback(async (forceCpu: boolean = false) => {
    setIsLoading(true);
    setError(null);

    try {
      const delegateUsed = await initHandLandmarker({ forceCpu });
      if (isMountedRef.current) {
        setActiveDelegate(delegateUsed);
        setIsReady(true);
        setIsLoading(false);
      }
    } catch (err: unknown) {
      if (isMountedRef.current) {
        setIsReady(false);
        setIsLoading(false);
        if (err instanceof HandLandmarkerError) {
          setError(err);
        } else {
          setError(
            new HandLandmarkerError(
              "INITIALIZATION_ERROR",
              err instanceof Error ? err.message : "Failed to load hand landmarker.",
              err
            )
          );
        }
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    init();

    return () => {
      isMountedRef.current = false;
      disposeHandLandmarker();
    };
  }, [init]);

  const reinitialize = useCallback(
    async (forceCpu?: boolean) => {
      await init(forceCpu ?? activeDelegate === "CPU");
    },
    [init, activeDelegate]
  );

  return {
    isReady,
    isLoading,
    error,
    activeDelegate: getActiveDelegate(),
    reinitialize,
  };
}
