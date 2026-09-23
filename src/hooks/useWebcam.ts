import { useState, useCallback, useEffect, useRef } from "react";

export type WebcamStatus =
  | "idle"
  | "requesting"
  | "active"
  | "denied"
  | "unsupported"
  | "error";

export interface UseWebcamReturn {
  stream: MediaStream | null;
  status: WebcamStatus;
  error: string | null;
  start: () => Promise<void>;
  stop: () => void;
}

export function useWebcam(): UseWebcamReturn {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [error, setError] = useState<string | null>(null);

  const activeStreamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    if (activeStreamRef.current) {
      activeStreamRef.current.getTracks().forEach((track) => {
        track.stop();
      });
      activeStreamRef.current = null;
    }
    setStream(null);
    setStatus("idle");
    setError(null);
  }, []);

  const start = useCallback(async () => {
    // Check browser support and secure context requirements
    if (
      typeof window === "undefined" ||
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setStatus("unsupported");
      setError(
        "Camera access is unsupported in this environment. Ensure you are using a modern browser over HTTPS or localhost."
      );
      return;
    }

    // Stop existing stream before starting a new one
    if (activeStreamRef.current) {
      stop();
    }

    setStatus("requesting");
    setError(null);

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      // Handle unexpected track ending (e.g. webcam unplugged)
      mediaStream.getVideoTracks().forEach((track) => {
        track.onended = () => {
          setStatus("idle");
          setError("Camera device stream ended unexpectedly (disconnected).");
          stop();
        };
      });

      activeStreamRef.current = mediaStream;
      setStream(mediaStream);
      setStatus("active");
    } catch (err: unknown) {
      if (err instanceof DOMException) {
        if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
          setStatus("denied");
          setError(
            "Camera access was denied. Enable camera permissions in your browser's site settings and click Start Camera to try again."
          );
          return;
        } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
          setStatus("error");
          setError("No camera hardware found on this device.");
          return;
        } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
          setStatus("error");
          setError(
            "Camera hardware is currently in use by another application or operating system process."
          );
          return;
        }
      }

      setStatus("error");
      setError(
        err instanceof Error ? err.message : "Failed to start camera due to an unknown error."
      );
    }
  }, [stop]);

  // Clean up media tracks on unmount
  useEffect(() => {
    return () => {
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach((track) => track.stop());
        activeStreamRef.current = null;
      }
    };
  }, []);

  return { stream, status, error, start, stop };
}
