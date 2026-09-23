import type { RawPrediction } from "./signClassifier";
import { DEFAULT_INFERENCE_CONFIG } from "../config/inference";

export interface SignPrediction {
  label: string | null;
  confidence: number;
  timestamp: string;
  probabilities?: Record<string, number>;
  belowThreshold: boolean;
  smoothed: boolean;
}

export class PredictionSmoother {
  private window: RawPrediction[] = [];
  private windowSize: number;
  private minAgreementCount: number;

  constructor(
    windowSize: number = DEFAULT_INFERENCE_CONFIG.smootherWindowSize,
    minAgreementCount: number = DEFAULT_INFERENCE_CONFIG.minAgreementCount
  ) {
    this.windowSize = windowSize;
    this.minAgreementCount = minAgreementCount;
  }

  /**
   * Adds a new raw prediction from signClassifier to the rolling window.
   */
  public addPrediction(pred: RawPrediction): void {
    this.window.push(pred);
    if (this.window.length > this.windowSize) {
      this.window.shift(); // Drop oldest prediction
    }
  }

  /**
   * Computes smoothed prediction using majority voting, confidence averaging, and threshold filtering (§8).
   */
  public getSmoothedPrediction(
    threshold: number = DEFAULT_INFERENCE_CONFIG.confidenceThreshold
  ): SignPrediction {
    const timestamp = new Date().toISOString();

    if (this.window.length === 0) {
      return {
        label: null,
        confidence: 0,
        timestamp,
        belowThreshold: true,
        smoothed: true,
      };
    }

    // 1. Count label frequencies in current window
    const labelCounts = new Map<string, { count: number; totalConfidence: number }>();

    for (const pred of this.window) {
      const lbl = pred.label;
      if (!labelCounts.has(lbl)) {
        labelCounts.set(lbl, { count: 0, totalConfidence: 0 });
      }
      const entry = labelCounts.get(lbl)!;
      entry.count++;
      entry.totalConfidence += pred.confidence;
    }

    // 2. Find majority label
    let majorityLabel: string | null = null;
    let maxCount = 0;
    let majorityAvgConfidence = 0;

    labelCounts.forEach((val, lbl) => {
      if (val.count > maxCount) {
        maxCount = val.count;
        majorityLabel = lbl;
        majorityAvgConfidence = val.totalConfidence / val.count;
      }
    });

    // 3. Minimum agreement count requirement (§8)
    if (maxCount < this.minAgreementCount || !majorityLabel) {
      return {
        label: null,
        confidence: majorityAvgConfidence,
        timestamp,
        belowThreshold: true,
        smoothed: true,
      };
    }

    // 4. Confidence Threshold Check (§7)
    const isBelowThreshold = majorityAvgConfidence < threshold;

    return {
      label: isBelowThreshold ? null : majorityLabel,
      confidence: majorityAvgConfidence,
      timestamp,
      belowThreshold: isBelowThreshold,
      smoothed: true,
    };
  }

  /**
   * Resets the smoother window (called on hand-loss buffer clears §8).
   */
  public reset(): void {
    this.window = [];
  }
}
