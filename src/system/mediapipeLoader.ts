export type MediaPipeState =
  | "MEDIAPIPE_NOT_LOADED"
  | "MEDIAPIPE_LOADING"
  | "MEDIAPIPE_READY"
  | "MEDIAPIPE_LOAD_ERROR";

export type HandPresenceSubState =
  | "NO_HAND_DETECTED"
  | "HAND_DETECTED"
  | "MULTIPLE_HANDS_DETECTED";

export interface MediaPipeStatusPayload {
  state: MediaPipeState;
  subState: HandPresenceSubState;
  handCount: number;
  error: string | null;
}

export class MediaPipeLoader {
  private state: MediaPipeState = "MEDIAPIPE_NOT_LOADED";
  private subState: HandPresenceSubState = "NO_HAND_DETECTED";
  private handCount: number = 0;
  private errorMessage: string | null = null;
  private onStatusChange?: (payload: MediaPipeStatusPayload) => void;

  constructor(onStatusChange?: (payload: MediaPipeStatusPayload) => void) {
    this.onStatusChange = onStatusChange;
  }

  public getStatus(): MediaPipeStatusPayload {
    return {
      state: this.state,
      subState: this.subState,
      handCount: this.handCount,
      error: this.errorMessage,
    };
  }

  private notify() {
    if (this.onStatusChange) {
      this.onStatusChange(this.getStatus());
    }
  }

  public setPendingLoad(): void {
    this.state = "MEDIAPIPE_LOADING";
    this.errorMessage = null;
    this.notify();
  }

  public setReady(): void {
    this.state = "MEDIAPIPE_READY";
    this.errorMessage = null;
    this.notify();
  }

  public setError(msg: string): void {
    this.state = "MEDIAPIPE_LOAD_ERROR";
    this.errorMessage = msg;
    this.notify();
  }

  public updateHandCount(count: number): void {
    this.handCount = count;
    if (count === 0) {
      this.subState = "NO_HAND_DETECTED";
    } else if (count === 1) {
      this.subState = "HAND_DETECTED";
    } else {
      this.subState = "MULTIPLE_HANDS_DETECTED";
    }
    this.notify();
  }
}
