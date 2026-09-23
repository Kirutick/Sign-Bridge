import React, { useState, useEffect } from "react";
import { AppSettings, DEFAULT_SETTINGS, loadSettings, saveSettings } from "../services/settingsStore";
import { speechService } from "../services/speechService";
import styles from "../styles/SettingsPage.module.css";

interface SettingsPageProps {
  onSettingsChanged?: (settings: AppSettings) => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ onSettingsChanged }) => {
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [savedBanner, setSavedBanner] = useState<boolean>(false);

  useEffect(() => {
    if (speechService.isSupported()) {
      setVoices(speechService.getVoices());
    }
  }, []);

  const handleChange = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleSave = () => {
    saveSettings(settings);
    if (onSettingsChanged) {
      onSettingsChanged(settings);
    }
    setSavedBanner(true);
    setTimeout(() => setSavedBanner(false), 2500);
  };

  const handleReset = () => {
    if (confirm("Reset all settings to default values?")) {
      setSettings(DEFAULT_SETTINGS);
      saveSettings(DEFAULT_SETTINGS);
      if (onSettingsChanged) {
        onSettingsChanged(DEFAULT_SETTINGS);
      }
      setSavedBanner(true);
      setTimeout(() => setSavedBanner(false), 2500);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>System Settings & Configuration</h1>
        <p className={styles.subtitle}>
          Fine-tune the neural inference pipeline, temporal stabilization parameters, and speech engine.
        </p>
      </div>

      {savedBanner && (
        <div style={{ padding: "12px 16px", background: "rgba(16, 185, 129, 0.15)", border: "1px solid var(--accent-green)", borderRadius: "8px", color: "#34d399", textAlign: "center", fontWeight: 600 }}>
          ✓ Settings saved successfully!
        </div>
      )}

      {/* Recognition & Stability Section */}
      <div className={styles.card}>
        <div className={styles.sectionTitle}>
          <span>🎯</span> Recognition & Temporal Stabilization
        </div>

        {/* Confidence Threshold */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Confidence Gate Threshold</span>
            <span className={styles.settingDesc}>
              Minimum prediction probability required to trigger candidate state (Default: 70%).
            </span>
          </div>
          <div className={styles.sliderWrapper}>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.05"
              value={settings.confidenceThreshold}
              onChange={(e) => handleChange("confidenceThreshold", parseFloat(e.target.value))}
              className={styles.slider}
            />
            <span style={{ fontWeight: 700, minWidth: "40px", color: "var(--accent-cyan)" }}>
              {Math.round(settings.confidenceThreshold * 100)}%
            </span>
          </div>
        </div>

        {/* Stability Frames */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Temporal Voting Window</span>
            <span className={styles.settingDesc}>
              Number of consecutive candidate frames required for consensus (Default: 15 frames).
            </span>
          </div>
          <input
            type="number"
            min="5"
            max="30"
            className={styles.inputControl}
            value={settings.stabilityFrames}
            onChange={(e) => handleChange("stabilityFrames", parseInt(e.target.value) || 15)}
          />
        </div>

        {/* Cooldown Duration */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Duplicate Cooldown (ms)</span>
            <span className={styles.settingDesc}>
              Holding duration lockout before an identical sign can be re-committed (Default: 800ms).
            </span>
          </div>
          <input
            type="number"
            min="200"
            max="3000"
            step="100"
            className={styles.inputControl}
            value={settings.cooldownDurationMs}
            onChange={(e) => handleChange("cooldownDurationMs", parseInt(e.target.value) || 800)}
          />
        </div>

        {/* Temporal Rolling Buffer Length */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Rolling Sequence Buffer Length</span>
            <span className={styles.settingDesc}>
              Temporal frames per gesture window (Default & canonical model: 30 frames $\approx$ 1.0s).
            </span>
          </div>
          <input
            type="number"
            min="10"
            max="60"
            className={styles.inputControl}
            value={settings.sequenceLength}
            disabled // fixed to model contract (30)
            title="Locked to 30 frames to match active LSTM model contract"
          />
        </div>

        {/* Hand Mode */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Hand Tracking Mode</span>
            <span className={styles.settingDesc}>
              Single-hand mode extracts 63 features (21×3); Two-hand mode extracts 126 features (21×3×2).
            </span>
          </div>
          <select
            className={styles.inputControl}
            value={settings.handMode}
            onChange={(e) => handleChange("handMode", e.target.value as "SINGLE_HAND" | "TWO_HAND")}
          >
            <option value="SINGLE_HAND">Single Hand (63 Features - ISL Default)</option>
            <option value="TWO_HAND">Two Hands (126 Features - Experimental)</option>
          </select>
        </div>
      </div>

      {/* Text to Speech Section */}
      <div className={styles.card}>
        <div className={styles.sectionTitle}>
          <span>🔊</span> Speech Synthesis (Text-to-Speech)
        </div>

        {/* Speech Enabled */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Enable Text-to-Speech</span>
            <span className={styles.settingDesc}>
              Read recognized sentences aloud automatically or upon button press.
            </span>
          </div>
          <label className={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={settings.speechEnabled}
              onChange={(e) => handleChange("speechEnabled", e.target.checked)}
            />
            <span className={styles.sliderToggle} />
          </label>
        </div>

        {/* Speech Rate */}
        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Speech Rate Speed</span>
            <span className={styles.settingDesc}>
              Playback speed multiplier (0.5x to 2.0x).
            </span>
          </div>
          <div className={styles.sliderWrapper}>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.1"
              value={settings.speechRate}
              onChange={(e) => handleChange("speechRate", parseFloat(e.target.value))}
              className={styles.slider}
            />
            <span style={{ fontWeight: 700, minWidth: "40px", color: "var(--accent-cyan)" }}>
              {settings.speechRate.toFixed(1)}x
            </span>
          </div>
        </div>

        {/* Voice Selection */}
        {voices.length > 0 && (
          <div className={styles.settingRow}>
            <div className={styles.settingMeta}>
              <span className={styles.settingLabel}>Synthesis Voice</span>
              <span className={styles.settingDesc}>
                Select preferred browser voice language accent.
              </span>
            </div>
            <select
              className={styles.inputControl}
              value={settings.speechVoiceURI}
              onChange={(e) => handleChange("speechVoiceURI", e.target.value)}
              style={{ maxWidth: "250px" }}
            >
              <option value="">Default (Auto-select)</option>
              {voices.map((v) => (
                <option key={v.voiceURI} value={v.voiceURI}>
                  {v.name} ({v.lang})
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Developer Debug Section */}
      <div className={styles.card}>
        <div className={styles.sectionTitle}>
          <span>🛠️</span> Developer & Diagnostics
        </div>

        <div className={styles.settingRow}>
          <div className={styles.settingMeta}>
            <span className={styles.settingLabel}>Telemetry & Debug Overlays</span>
            <span className={styles.settingDesc}>
              Displays live FPS, MediaPipe latency, LSTM forward pass time, and probability vectors.
            </span>
          </div>
          <label className={styles.toggleSwitch}>
            <input
              type="checkbox"
              checked={settings.debugMode}
              onChange={(e) => handleChange("debugMode", e.target.checked)}
            />
            <span className={styles.sliderToggle} />
          </label>
        </div>
      </div>

      {/* Action Buttons */}
      <div className={styles.actionsBar}>
        <button type="button" className={styles.btn} onClick={handleReset}>
          Reset to Defaults
        </button>
        <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={handleSave}>
          Save Settings
        </button>
      </div>
    </div>
  );
};
