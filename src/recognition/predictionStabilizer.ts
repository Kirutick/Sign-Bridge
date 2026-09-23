export type StabilizerState =
  | "IDLE"
  | "UNSTABLE"
  | "CANDIDATE"
  | "STABLE_COMMITTED"
  | "COOLDOWN_WAIT_FOR_CHANGE"
  | "LOW_CONFIDENCE";

export interface StabilizationConfig {
  confidenceThreshold: number; // default 0.80
  votingWindowFrames: number; // default 10
  votingMajorityRatio: number; // default 0.70
  stabilityDurationMs: number; // default 600 ms
  pauseDurationMs: number; // default 500 ms
  movementThreshold: number; // default 0.01
  lowConfidenceGraceFrames: number; // default 5
  duplicateSuppressionRequired: boolean; // default true
}

export const DEFAULT_STABILIZATION_CONFIG: StabilizationConfig = {
  confidenceThreshold: 0.80,
  votingWindowFrames: 10,
  votingMajorityRatio: 0.70,
  stabilityDurationMs: 600,
  pauseDurationMs: 500,
  movementThreshold: 0.01,
  lowConfidenceGraceFrames: 5,
  duplicateSuppressionRequired: true,
};

export interface FrameInput {
  label: string | null;
  confidence: number | null;
  handDetected: boolean;
  landmarks?: Array<{ x: number; y: number; z: number }>;
  timestampMs: number;
}

export interface StabilizationResult {
  state: StabilizerState;
  currentCandidate: string | null;
  currentConfidence: number | null;
  committedTextDelta: string | null; // Non-null ONLY on the exact frame a commit occurs
  statusMessage: string;
  committedText: string[];
}

/**
 * Pure, framework-agnostic prediction stabilizer state machine (§2.4).
 */
export class PredictionStabilizer {
  private config: StabilizationConfig;
  private state: StabilizerState = "IDLE";
  private committedText: string[] = [];

  // Temporal voting window buffer
  private windowInputs: Array<{ label: string; confidence: number }> = [];

  // Candidate tracking state
  private candidateLabel: string | null = null;
  private candidateStartTimeMs: number | null = null;
  private lastCommittedLabel: string | null = null;

  // Frame Counters
  private lowConfidenceFrameCount: number = 0;
  private noHandFrameCount: number = 0;
  private lowMovementFrameCount: number = 0;
  private lastLandmarks: Array<{ x: number; y: number; z: number }> | null = null;

  constructor(config: Partial<StabilizationConfig> = {}) {
    this.config = { ...DEFAULT_STABILIZATION_CONFIG, ...config };
  }

  public getConfig(): StabilizationConfig {
    return { ...this.config };
  }

  public updateConfig(newConfig: Partial<StabilizationConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  public reset(): void {
    this.state = "IDLE";
    this.windowInputs = [];
    this.candidateLabel = null;
    this.candidateStartTimeMs = null;
    this.lastCommittedLabel = null;
    this.lowConfidenceFrameCount = 0;
    this.noHandFrameCount = 0;
    this.lowMovementFrameCount = 0;
    this.lastLandmarks = null;
  }

  public clearCommittedText(): void {
    this.committedText = [];
  }

  public backspaceCommittedText(): void {
    if (this.committedText.length > 0) {
      this.committedText.pop();
    }
  }

  public getFullCommittedText(): string[] {
    return [...this.committedText];
  }

  public getFullCommittedString(): string {
    return this.committedText.join(" ");
  }

  /**
   * Processes a single frame input and advances the state machine (§2.2).
   */
  public processFrame(input: FrameInput): StabilizationResult {
    const { label, confidence, handDetected, landmarks, timestampMs } = input;
    let committedTextDelta: string | null = null;

    // 1. Hand Loss & Movement Audit
    if (!handDetected) {
      this.noHandFrameCount++;
      this.windowInputs = [];
    } else {
      this.noHandFrameCount = 0;
    }

    // Measure movement displacement if landmarks present
    let isLowMovement = false;
    if (handDetected && landmarks && this.lastLandmarks && landmarks.length === 21) {
      let totalDist = 0;
      for (let i = 0; i < 21; i++) {
        const dx = landmarks[i].x - this.lastLandmarks[i].x;
        const dy = landmarks[i].y - this.lastLandmarks[i].y;
        const dz = landmarks[i].z - this.lastLandmarks[i].z;
        totalDist += Math.hypot(dx, dy, dz);
      }
      const meanDisplacement = totalDist / 21.0;
      isLowMovement = meanDisplacement < this.config.movementThreshold;
    }

    if (handDetected && landmarks) {
      this.lastLandmarks = landmarks;
    }

    if (isLowMovement) {
      this.lowMovementFrameCount++;
    } else {
      this.lowMovementFrameCount = 0;
    }

    // 2. Confidence Evaluation
    const isValidConfidence =
      handDetected &&
      label !== null &&
      confidence !== null &&
      confidence >= this.config.confidenceThreshold;

    if (!isValidConfidence && handDetected) {
      this.lowConfidenceFrameCount++;
    } else if (isValidConfidence) {
      this.lowConfidenceFrameCount = 0;
    }

    // Check Pause Condition (§4): 500ms no-hand or sustained low-confidence
    const isPauseConditionMet =
      this.noHandFrameCount >= 15 ||
      this.lowConfidenceFrameCount >= 15;

    // 3. State Machine Transitions (§2.2)
    switch (this.state) {
      case "IDLE": {
        if (isValidConfidence) {
          this.state = "UNSTABLE";
        } else if (handDetected && this.lowConfidenceFrameCount >= 1) {
          this.state = "LOW_CONFIDENCE";
        }
        break;
      }

      case "UNSTABLE": {
        if (isPauseConditionMet || !handDetected) {
          this.state = "IDLE";
          this.candidateLabel = null;
          break;
        }

        if (isValidConfidence && label) {
          this.windowInputs.push({ label, confidence: confidence! });
          if (this.windowInputs.length > this.config.votingWindowFrames) {
            this.windowInputs.shift();
          }

          // Check temporal majority voting (§2.2)
          const majorityCandidate = this.evaluateMajorityCandidate();
          if (majorityCandidate) {
            this.candidateLabel = majorityCandidate;
            this.candidateStartTimeMs = timestampMs;
            this.state = "CANDIDATE";
          }
        } else if (this.lowConfidenceFrameCount > this.config.lowConfidenceGraceFrames) {
          this.state = "LOW_CONFIDENCE";
        }
        break;
      }

      case "CANDIDATE": {
        if (!handDetected) {
          this.state = "IDLE";
          this.candidateLabel = null;
          break;
        }

        if (this.lowConfidenceFrameCount > this.config.lowConfidenceGraceFrames) {
          this.state = "LOW_CONFIDENCE";
          break;
        }

        if (isValidConfidence && label) {
          // If label changes mid-accumulation, restart accumulation on new label
          if (label !== this.candidateLabel) {
            this.candidateLabel = label;
            this.candidateStartTimeMs = timestampMs;
            this.windowInputs = [{ label, confidence: confidence! }];
            this.state = "UNSTABLE";
            break;
          }

          // Check stability duration holding (§2.2)
          const elapsed = timestampMs - (this.candidateStartTimeMs || timestampMs);
          if (elapsed >= this.config.stabilityDurationMs) {
            // Duplicate Suppression Check (§2.2, §8):
            // Commit of label X is ONLY permitted if not currently in cooldown for label X
            if (
              !this.config.duplicateSuppressionRequired ||
              this.candidateLabel !== this.lastCommittedLabel
            ) {
              this.state = "STABLE_COMMITTED";
            } else {
              // Same label held continuously -> move to COOLDOWN without re-committing
              this.state = "COOLDOWN_WAIT_FOR_CHANGE";
            }
          }
        }
        break;
      }

      case "STABLE_COMMITTED": {
        // Execute Commit exactly once on transition (§3, §8)
        if (this.candidateLabel) {
          committedTextDelta = this.candidateLabel;
          this.committedText.push(this.candidateLabel);
          this.lastCommittedLabel = this.candidateLabel;
        }
        // Immediately transition to COOLDOWN_WAIT_FOR_CHANGE
        this.state = "COOLDOWN_WAIT_FOR_CHANGE";
        break;
      }

      case "COOLDOWN_WAIT_FOR_CHANGE": {
        if (isPauseConditionMet || !handDetected) {
          this.state = "IDLE";
          this.lastCommittedLabel = null; // Unlocks duplicate commitment on return (§4)
          this.candidateLabel = null;
          break;
        }

        if (isValidConfidence && label && label !== this.lastCommittedLabel) {
          // Different sign label performed -> unlock cooldown immediately
          this.candidateLabel = label;
          this.candidateStartTimeMs = timestampMs;
          this.windowInputs = [{ label, confidence: confidence! }];
          this.state = "UNSTABLE";
        }
        break;
      }

      case "LOW_CONFIDENCE": {
        if (isValidConfidence) {
          this.state = "UNSTABLE";
        } else if (isPauseConditionMet || !handDetected) {
          this.state = "IDLE";
          this.candidateLabel = null;
        }
        break;
      }
    }

    // Generate human-readable status message (§6)
    const statusMessage = this.computeStatusMessage(handDetected);

    return {
      state: this.state,
      currentCandidate: this.state === "LOW_CONFIDENCE" ? null : this.candidateLabel,
      currentConfidence: confidence,
      committedTextDelta,
      statusMessage,
      committedText: [...this.committedText],
    };
  }

  /**
   * Evaluates temporal majority candidate from voting window buffer.
   */
  private evaluateMajorityCandidate(): string | null {
    if (this.windowInputs.length < Math.ceil(this.config.votingWindowFrames * 0.5)) {
      return null;
    }

    const counts = new Map<string, number>();
    for (const item of this.windowInputs) {
      counts.set(item.label, (counts.get(item.label) || 0) + 1);
    }

    const requiredVotes = Math.ceil(
      this.windowInputs.length * this.config.votingMajorityRatio
    );

    let winningLabel: string | null = null;
    counts.forEach((cnt, lbl) => {
      if (cnt >= requiredVotes) {
        winningLabel = lbl;
      }
    });

    return winningLabel;
  }

  /**
   * Computes status badge message for UI (§6).
   */
  private computeStatusMessage(handDetected: boolean): string {
    if (!handDetected) return "No hand detected";

    switch (this.state) {
      case "IDLE":
        return "Idle";
      case "UNSTABLE":
        return "Recognizing...";
      case "CANDIDATE":
        return "Holding...";
      case "STABLE_COMMITTED":
        return "Committed";
      case "COOLDOWN_WAIT_FOR_CHANGE":
        return "Sign Committed (Waiting for transition)";
      case "LOW_CONFIDENCE":
        return "Uncertain";
      default:
        return "Idle";
    }
  }
}
