import * as tf from "@tensorflow/tfjs";
import { SignLanguageKey, SIGN_LANGUAGES } from "../types/languages";

export interface RawPrediction {
  probabilities: number[];
  labelId: number;
  label: string;
  confidence: number;
  inferenceTimeMs: number;
  timestamp: string;
}

export interface FeatureSpec {
  featureCount: number;
  sequenceLength: number;
  datasetVersion: string;
  modelVersion: string;
  language?: string;
}

export interface LabelMapData {
  language?: string;
  language_code?: string;
  label_to_index: Record<string, number>;
  index_to_label: Record<string, string>;
  num_classes: number;
}

/** Expected constants from live landmarkProcessor pipeline */
const LIVE_PIPELINE_FEATURE_COUNT = 63;
const LIVE_PIPELINE_SEQUENCE_LENGTH = 30;

export class SignClassifierService {
  private model: tf.LayersModel | null = null;
  private featureSpec: FeatureSpec | null = null;
  private labelMap: LabelMapData | null = null;
  private isLoaded: boolean = false;
  private currentLanguage: SignLanguageKey = "isl";

  /**
   * Loads tfjs model, label_map.json, and feature_spec.json for the specified sign language.
   * Performs automated compatibility check against live pipeline constants (§3c).
   */
  public async loadModel(languageKey: SignLanguageKey = "isl"): Promise<void> {
    this.isLoaded = false;
    this.dispose();
    this.currentLanguage = languageKey;

    const langMeta = SIGN_LANGUAGES[languageKey] || SIGN_LANGUAGES.isl;
    const baseUrl = langMeta.modelDir;

    try {
      // 1. Fetch feature_spec.json
      const specRes = await fetch(`${baseUrl}/feature_spec.json`);
      if (!specRes.ok) {
        throw new Error(`Failed to fetch feature_spec.json from ${baseUrl} (HTTP ${specRes.status}).`);
      }
      this.featureSpec = await specRes.json();

      // 2. Automated Compatibility Check (§3c)
      if (
        this.featureSpec?.featureCount !== LIVE_PIPELINE_FEATURE_COUNT ||
        this.featureSpec?.sequenceLength !== LIVE_PIPELINE_SEQUENCE_LENGTH
      ) {
        throw new Error(
          `Model compatibility mismatch! Model expects (${this.featureSpec?.sequenceLength}x${this.featureSpec?.featureCount}), ` +
          `but live preprocessing pipeline produces (${LIVE_PIPELINE_SEQUENCE_LENGTH}x${LIVE_PIPELINE_FEATURE_COUNT}).`
        );
      }

      // 3. Fetch label_map.json
      const labelRes = await fetch(`${baseUrl}/label_map.json`);
      if (!labelRes.ok) {
        throw new Error(`Failed to fetch label_map.json from ${baseUrl} (HTTP ${labelRes.status}).`);
      }
      this.labelMap = await labelRes.json();

      // 4. Load TFJS Layers Model
      this.model = await tf.loadLayersModel(`${baseUrl}/model.json`);
      this.isLoaded = true;

      console.log(`✅ SignClassifier [${langMeta.name}] model loaded: ${this.featureSpec.modelVersion}`);
    } catch (err: unknown) {
      this.isLoaded = false;
      this.dispose();
      throw new Error(
        `SignClassifier [${langMeta.name}] load failure: ${err instanceof Error ? err.message : "Unknown error"}`
      );
    }
  }

  public getCurrentLanguage(): SignLanguageKey {
    return this.currentLanguage;
  }

  public getVocabularyList(): string[] {
    if (!this.labelMap) return [];
    return Object.keys(this.labelMap.label_to_index || {});
  }

  public isReady(): boolean {
    return this.isLoaded && this.model !== null && this.labelMap !== null;
  }

  public getModelVersion(): string | null {
    return this.featureSpec?.modelVersion || null;
  }

  public getExpectedInputShape(): { sequenceLength: number; featureCount: number } {
    return {
      sequenceLength: this.featureSpec?.sequenceLength || LIVE_PIPELINE_SEQUENCE_LENGTH,
      featureCount: this.featureSpec?.featureCount || LIVE_PIPELINE_FEATURE_COUNT,
    };
  }

  /**
   * Executes Keras LSTM forward pass for sequence matrix [30, 63].
   * Shape sent to model: [1, 30, 63] (batch dimension added).
   * Wrapped in tf.tidy() to prevent GPU memory leaks (§3e).
   */
  public async predict(sequence: number[][]): Promise<RawPrediction> {
    if (!this.isReady() || !this.model || !this.labelMap) {
      throw new Error("SignClassifier is not ready for prediction.");
    }

    const { sequenceLength, featureCount } = this.getExpectedInputShape();

    // Shape Validation (§3d)
    if (!Array.isArray(sequence) || sequence.length !== sequenceLength) {
      throw new Error(
        `Invalid prediction sequence length (${sequence?.length || 0}). Expected exactly ${sequenceLength}.`
      );
    }

    for (let i = 0; i < sequence.length; i++) {
      if (!Array.isArray(sequence[i]) || sequence[i].length !== featureCount) {
        throw new Error(
          `Invalid feature vector length at index ${i} (${sequence[i]?.length}). Expected ${featureCount}.`
        );
      }
    }

    const startTime = performance.now();
    const timestamp = new Date().toISOString();

    // Wrapped in tf.tidy() for automatic GPU tensor cleanup (§3e)
    const probabilities = tf.tidy(() => {
      // 2D matrix [30, 63] -> 3D tensor [1, 30, 63]
      const inputTensor = tf.tensor3d([sequence], [1, sequenceLength, featureCount]);
      const outputTensor = this.model!.predict(inputTensor) as tf.Tensor;
      return Array.from(outputTensor.dataSync());
    });

    const inferenceTimeMs = performance.now() - startTime;

    // Find max probability index
    let maxIdx = 0;
    let maxProb = 0;
    for (let i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIdx = i;
      }
    }

    const labelStr = this.labelMap.index_to_label[String(maxIdx)] || `Class_${maxIdx}`;

    return {
      probabilities,
      labelId: maxIdx,
      label: labelStr,
      confidence: maxProb,
      inferenceTimeMs,
      timestamp,
    };
  }

  public dispose(): void {
    if (this.model) {
      try {
        this.model.dispose();
      } catch (e) {
        console.warn("Error disposing tfjs model:", e);
      }
      this.model = null;
    }
    this.isLoaded = false;
  }
}

export const signClassifier = new SignClassifierService();
