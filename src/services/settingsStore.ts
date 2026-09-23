export interface AppSettings {
  confidenceThreshold: number;
  stabilityFrames: number;
  cooldownDurationMs: number;
  sequenceLength: number;
  handMode: "SINGLE_HAND" | "TWO_HAND";
  speechEnabled: boolean;
  speechRate: number;
  speechVoiceURI: string;
  debugMode: boolean;
}

export const DEFAULT_SETTINGS: AppSettings = {
  confidenceThreshold: 0.70,
  stabilityFrames: 15,
  cooldownDurationMs: 800,
  sequenceLength: 30,
  handMode: "SINGLE_HAND",
  speechEnabled: true,
  speechRate: 1.0,
  speechVoiceURI: "",
  debugMode: false,
};

const SETTINGS_KEY = "signbridge_app_settings_v1";

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function saveSettings(settings: AppSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (e) {
    console.warn("Could not persist settings to localStorage:", e);
  }
}
