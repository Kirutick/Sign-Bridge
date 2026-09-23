export interface InferenceConfig {
  confidenceThreshold: number; // default 0.80, range 0.50 - 0.95
  inferenceCadenceMs: number; // default 250 ms, throttles predict() calls
  smootherWindowSize: number; // default 5 predictions
  minAgreementCount: number; // default 3 matching predictions required
}

export const DEFAULT_INFERENCE_CONFIG: InferenceConfig = {
  confidenceThreshold: 0.80,
  inferenceCadenceMs: 250,
  smootherWindowSize: 5,
  minAgreementCount: 3,
};
