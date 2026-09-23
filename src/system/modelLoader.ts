export type ModelState =
  | "MODEL_NOT_LOADED"
  | "MODEL_LOADING"
  | "MODEL_LOADED"
  | "MODEL_ERROR";

export type ModelSubState = "COLLECTING_SEQUENCE" | "RECOGNIZING";

export type ModelErrorType =
  | "NETWORK_ERROR"
  | "CORRUPT_MODEL"
  | "CONFIG_MISMATCH"
  | "UNKNOWN";

export interface ModelStatusPayload {
  state: ModelState;
  subState: ModelSubState;
  errorType: ModelErrorType | null;
  errorMessage: string | null;
  error: string | null;
  isRecoverable: boolean;
}

export class ModelLoader {
  private state: ModelState = "MODEL_NOT_LOADED";
  private subState: ModelSubState = "COLLECTING_SEQUENCE";
  private errorType: ModelErrorType | null = null;
  private errorMessage: string | null = null;
  private isRecoverable: boolean = true;
  private onStatusChange?: (payload: ModelStatusPayload) => void;

  constructor(onStatusChange?: (payload: ModelStatusPayload) => void) {
    this.onStatusChange = onStatusChange;
  }

  public getStatus(): ModelStatusPayload {
    return {
      state: this.state,
      subState: this.subState,
      errorType: this.errorType,
      errorMessage: this.errorMessage,
      error: this.errorMessage,
      isRecoverable: this.isRecoverable,
    };
  }

  private notify() {
    if (this.onStatusChange) {
      this.onStatusChange(this.getStatus());
    }
  }

  public setPendingLoad(): void {
    this.state = "MODEL_LOADING";
    this.errorMessage = null;
    this.errorType = null;
    this.notify();
  }

  public setLoaded(): void {
    this.state = "MODEL_LOADED";
    this.subState = "COLLECTING_SEQUENCE";
    this.errorMessage = null;
    this.errorType = null;
    this.notify();
  }

  public setError(type: ModelErrorType, message: string): void {
    this.state = "MODEL_ERROR";
    this.errorType = type;
    this.errorMessage = message;
    this.isRecoverable = type === "NETWORK_ERROR" || type === "UNKNOWN";
    this.notify();
  }

  public updateBufferSubState(isBufferFull: boolean): void {
    if (this.state === "MODEL_LOADED") {
      this.subState = isBufferFull ? "RECOGNIZING" : "COLLECTING_SEQUENCE";
      this.notify();
    }
  }
}
