import React from "react";
import { SignLanguageKey, SIGN_LANGUAGES } from "../types/languages";
import styles from "../styles/LanguageSelector.module.css";

interface LanguageSelectorProps {
  currentLanguage: SignLanguageKey;
  isLoading: boolean;
  onSelectLanguage: (key: SignLanguageKey) => void;
}

export const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  currentLanguage,
  isLoading,
  onSelectLanguage,
}) => {
  const activeMeta = SIGN_LANGUAGES[currentLanguage];

  return (
    <div className={styles.container} role="region" aria-label="Sign Language Selection">
      <div className={styles.selectorRow}>
        {(Object.keys(SIGN_LANGUAGES) as SignLanguageKey[]).map((key) => {
          const lang = SIGN_LANGUAGES[key];
          const isActive = key === currentLanguage;

          return (
            <button
              key={key}
              type="button"
              disabled={isLoading}
              className={`${styles.langButton} ${isActive ? styles.active : ""}`}
              onClick={() => onSelectLanguage(key)}
              title={lang.description}
            >
              <span className={styles.flag}>{lang.flagEmoji}</span>
              <span>{lang.name}</span>
              <span className={styles.badge}>{lang.vocabularyCount} signs</span>
            </button>
          );
        })}
      </div>

      <div className={styles.langDetails}>
        <span>Region: <strong>{activeMeta.region}</strong></span>
        <span>•</span>
        <span>
          Key Vocabulary: <span className={styles.vocabTag}>{activeMeta.featuredVocabulary.slice(0, 5).join(", ")}...</span>
        </span>
      </div>
    </div>
  );
};
