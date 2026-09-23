import React from "react";
import styles from "../styles/AppHeader.module.css";

export type NavTabId = "recognize" | "learn" | "practice" | "studio" | "training" | "settings" | "about";

interface AppHeaderProps {
  activeTab: NavTabId;
  onTabChange: (tab: NavTabId) => void;
}

export const AppHeader: React.FC<AppHeaderProps> = ({ activeTab, onTabChange }) => {
  return (
    <header className={styles.header}>
      <div className={styles.topRow}>
        <div className={styles.brand}>
          <span className={styles.logo} role="img" aria-label="Sign Bridge Logo">
            🤟
          </span>
          <div>
            <h1 className={styles.title}>SIGN BRIDGE</h1>
            <span className={styles.subtitle}>Indian Sign Language (ISL) Recognition Architecture → English Speech & Text</span>
          </div>
        </div>

        {/* Language, Prototype Status & Client-Side Privacy Badges */}
        <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
          <div className={styles.privacyBadge} style={{ background: "rgba(245, 158, 11, 0.12)", borderColor: "rgba(245, 158, 11, 0.4)", color: "#f59e0b" }} title="Current weights run on synthetic demo kinematics. Real ISL data required for production.">
            <span>⚠️</span>
            <span>Prototype (Demo Weights)</span>
          </div>
          <div className={styles.privacyBadge} style={{ background: "rgba(0, 242, 254, 0.1)", borderColor: "rgba(0, 242, 254, 0.3)", color: "var(--accent-cyan)" }}>
            <span>🇮🇳</span>
            <span>ISL Input Only</span>
          </div>
          <div className={styles.privacyBadge} style={{ background: "rgba(79, 172, 254, 0.1)", borderColor: "rgba(79, 172, 254, 0.3)", color: "var(--accent-blue)" }}>
            <span>🇬🇧</span>
            <span>English Output (en-IN)</span>
          </div>
          <div className={styles.privacyBadge} role="note">
            <span>🔒</span>
            <span>100% On-Device Processing</span>
          </div>
        </div>
      </div>

      {/* Top Navigation Tabs */}
      <nav className={styles.navBar} aria-label="Main Navigation">
        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "recognize" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("recognize")}
        >
          <span>🏠</span> HOME
        </button>

        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "learn" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("learn")}
        >
          <span>📚</span> LEARN ISL
        </button>

        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "practice" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("practice")}
        >
          <span>🎯</span> PRACTICE
        </button>

        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "studio" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("studio")}
        >
          <span>🗄️</span> DATASET
        </button>

        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "training" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("training")}
        >
          <span>🚀</span> TRAINING
        </button>

        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "settings" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("settings")}
        >
          <span>⚙️</span> Settings
        </button>

        <button
          type="button"
          className={`${styles.navTab} ${activeTab === "about" ? styles.navTabActive : ""}`}
          onClick={() => onTabChange("about")}
        >
          <span>📖</span> ABOUT
        </button>
      </nav>
    </header>
  );
};
