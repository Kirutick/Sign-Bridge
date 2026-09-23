export type CameraState =
  | "CAMERA_IDLE"
  | "CAMERA_REQUESTING_PERMISSION"
  | "CAMERA_ACTIVE"
  | "CAMERA_PERMISSION_DENIED"
  | "CAMERA_UNAVAILABLE"
  | "CAMERA_DISCONNECTED";

export interface CameraStatusPayload {
  state: CameraState;
  stream: MediaStream | null;
  error: string | null;
}

export class CameraController {
  private state: CameraState = "CAMERA_IDLE";
  private activeStream: MediaStream | null = null;
  private errorMessage: string | null = null;
  private onStatusChange?: (payload: CameraStatusPayload) => void;

  constructor(onStatusChange?: (payload: CameraStatusPayload) => void) {
    this.onStatusChange = onStatusChange;
  }

  public getStatus(): CameraStatusPayload {
    return {
      state: this.state,
      stream: this.activeStream,
      error: this.errorMessage,
    };
  }

  private updateState(newState: CameraState, stream: MediaStream | null = null, error: string | null = null) {
    this.state = newState;
    this.activeStream = stream;
    this.errorMessage = error;
    if (this.onStatusChange) {
      this.onStatusChange(this.getStatus());
    }
  }

  /**
   * Stops active camera tracks and releases hardware resources (§2.1).
   */
  public stopCamera(): void {
    if (this.activeStream) {
      this.activeStream.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (e) {
          console.warn("Error stopping camera track:", e);
        }
      });
      this.activeStream = null;
    }
    this.updateState("CAMERA_IDLE", null, null);
  }

  /**
   * Requests camera permissions and initiates stream (§2.1, §5).
   * Implements fallback from high-res constraints to basic video constraint.
   */
  public async startCamera(): Promise<MediaStream | null> {
    if (
      typeof window === "undefined" ||
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      this.updateState(
        "CAMERA_UNAVAILABLE",
        null,
        "Your browser doesn't support the camera features Sign Bridge needs. Try Chrome, Edge, or Firefox."
      );
      return null;
    }

    this.stopCamera();
    this.updateState("CAMERA_REQUESTING_PERMISSION", null, null);

    let mediaStream: MediaStream | null = null;

    try {
      // Primary attempt: 720p facing user
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
    } catch (primaryErr: unknown) {
      // Fallback attempt: basic generic video constraint
      try {
        console.warn("Primary camera constraints failed, attempting fallback to generic video:", primaryErr);
        mediaStream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
      } catch (err: unknown) {
        if (err instanceof DOMException) {
          if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
            this.updateState(
              "CAMERA_PERMISSION_DENIED",
              null,
              "Camera access was denied. Sign Bridge needs your camera to recognize signs — enable it in your browser's site settings and try again."
            );
            return null;
          } else if (
            err.name === "NotFoundError" ||
            err.name === "DevicesNotFoundError" ||
            err.name === "NotReadableError" ||
            err.name === "TrackStartError"
          ) {
            this.updateState(
              "CAMERA_UNAVAILABLE",
              null,
              "Camera is unavailable or in use by another application. Close other camera apps and try again."
            );
            return null;
          } else if (err.name === "OverconstrainedError") {
            this.updateState(
              "CAMERA_UNAVAILABLE",
              null,
              "The requested camera resolution is not supported by your camera hardware."
            );
            return null;
          }
        }
        this.updateState(
          "CAMERA_UNAVAILABLE",
          null,
          err instanceof Error ? err.message : "Unable to access your camera. Check device permissions."
        );
        return null;
      }
    }

    if (mediaStream) {
      // Handle unexpected disconnects (e.g. camera unplugged mid-session §5)
      mediaStream.getVideoTracks().forEach((track) => {
        track.onended = () => {
          this.updateState(
            "CAMERA_DISCONNECTED",
            null,
            "Your camera was disconnected. Reconnect it and click Start Camera to resume."
          );
        };
      });

      this.updateState("CAMERA_ACTIVE", mediaStream, null);
      return mediaStream;
    }

    return null;
  }
}
