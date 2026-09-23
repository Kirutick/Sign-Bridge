import React, { useState, useRef, useEffect } from "react";
import { speechService } from "../services/speechService";
import styles from "../styles/CommittedTextArea.module.css";

interface CommittedTextAreaProps {
  committedTokens: string[];
  onClear?: () => void;
  onBackspace?: () => void;
  onUndo?: () => void;
  onSpace?: () => void;
  onCommitSign?: (sign: string) => void;
}

interface SavedSentence {
  id: string;
  text: string;
  timestamp: string;
}

export const CommittedTextArea: React.FC<CommittedTextAreaProps> = ({
  committedTokens,
  onClear,
  onBackspace,
  onUndo,
  onSpace,
}) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [savedSentences, setSavedSentences] = useState<SavedSentence[]>(() => {
    try {
      const stored = localStorage.getItem("signbridge_saved_sentences");
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });
  const [showHistory, setShowHistory] = useState<boolean>(false);
  const textDisplayRef = useRef<HTMLDivElement | null>(null);

  const fullText = committedTokens.join(" ");

  // Auto-scroll to bottom as new committed tokens arrive
  useEffect(() => {
    if (textDisplayRef.current) {
      textDisplayRef.current.scrollTop = textDisplayRef.current.scrollHeight;
    }
  }, [committedTokens]);

  const handleCopy = async () => {
    if (!fullText) return;
    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.warn("Copy failed:", e);
    }
  };

  const handleSpeak = async () => {
    if (!fullText || isSpeaking) return;
    setIsSpeaking(true);
    try {
      await speechService.speak(fullText);
    } catch (e) {
      console.warn("Speech error:", e);
    } finally {
      setIsSpeaking(false);
    }
  };

  const handleStopSpeech = () => {
    speechService.stop();
    setIsSpeaking(false);
  };

  const handleSaveSentence = () => {
    if (!fullText.trim()) return;
    const newEntry: SavedSentence = {
      id: Date.now().toString(),
      text: fullText.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    const updated = [newEntry, ...savedSentences.slice(0, 19)]; // keep latest 20
    setSavedSentences(updated);
    try {
      localStorage.setItem("signbridge_saved_sentences", JSON.stringify(updated));
    } catch (e) {
      console.warn("Error saving sentence to localStorage:", e);
    }
  };

  const handleDeleteSavedSentence = (id: string) => {
    const updated = savedSentences.filter((s) => s.id !== id);
    setSavedSentences(updated);
    try {
      localStorage.setItem("signbridge_saved_sentences", JSON.stringify(updated));
    } catch (e) {
      console.warn("Error deleting saved sentence:", e);
    }
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.title}>
          <span>💬</span> Recognized ISL Sentence Buffer
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {savedSentences.length > 0 && (
            <button
              type="button"
              className={styles.toolBtn}
              onClick={() => setShowHistory(!showHistory)}
              title="Toggle Saved Sentences History"
              aria-label="Toggle Saved Sentences History"
            >
              📜 History ({savedSentences.length})
            </button>
          )}
          {fullText && (
            <button
              type="button"
              className={styles.copyBtn}
              onClick={handleCopy}
              aria-label="Copy recognized sentence to clipboard"
            >
              {copied ? "✓ Copied!" : "📋 Copy"}
            </button>
          )}
        </div>
      </div>

      {/* Accessible Live Region */}
      <div
        ref={textDisplayRef}
        className={styles.textDisplay}
        aria-live="polite"
        aria-atomic="true"
        role="region"
        aria-label="Committed Sign Language Text Output"
      >
        {fullText ? (
          <span>{fullText}</span>
        ) : (
          <span className={styles.emptyState}>
            Signs recognized from camera will form a sentence here...
          </span>
        )}
      </div>

      {/* Sentence Controls Bar */}
      <div className={styles.controlsBar}>
        <div className={styles.actionGroup}>
          <button
            type="button"
            className={`${styles.actionBtn} ${styles.btnPrimary}`}
            onClick={isSpeaking ? handleStopSpeech : handleSpeak}
            disabled={!fullText}
            title="Read sentence aloud using Web Speech API"
            aria-label="Speak sentence aloud"
          >
            {isSpeaking ? "⏹ Stop Speech" : "🔊 Speak Sentence"}
          </button>

          {onSpace && (
            <button
              type="button"
              className={styles.actionBtn}
              onClick={onSpace}
              title="Add space / word separator"
              aria-label="Add space separator"
            >
              ␣ Space
            </button>
          )}

          {onUndo && (
            <button
              type="button"
              className={styles.actionBtn}
              onClick={onUndo}
              disabled={committedTokens.length === 0}
              title="Undo last recognized sign"
              aria-label="Undo last sign"
            >
              ↩ Undo
            </button>
          )}

          {onBackspace && (
            <button
              type="button"
              className={styles.actionBtn}
              onClick={onBackspace}
              disabled={committedTokens.length === 0}
              title="Backspace last sign"
              aria-label="Backspace last sign"
            >
              ⌫ Backspace
            </button>
          )}

          {onClear && (
            <button
              type="button"
              className={`${styles.actionBtn} ${styles.btnDanger}`}
              onClick={onClear}
              disabled={committedTokens.length === 0}
              title="Clear entire sentence"
              aria-label="Clear entire sentence"
            >
              🗑 Clear
            </button>
          )}

          <button
            type="button"
            className={styles.actionBtn}
            onClick={handleSaveSentence}
            disabled={!fullText.trim()}
            title="Save current sentence to history"
            aria-label="Save current sentence"
          >
            💾 Save Sentence
          </button>
        </div>
      </div>

      {/* Saved Sentences History Drawer */}
      {showHistory && (
        <div className={styles.historyDrawer}>
          <div className={styles.historyHeader}>
            <strong>Saved Sentences History</strong>
            <button
              type="button"
              onClick={() => setShowHistory(false)}
              className={styles.closeBtn}
              aria-label="Close history"
            >
              ✕
            </button>
          </div>
          <div className={styles.historyList}>
            {savedSentences.map((item) => (
              <div key={item.id} className={styles.historyItem}>
                <div className={styles.historyText}>
                  <span>{item.text}</span>
                  <small className={styles.historyTime}>{item.timestamp}</small>
                </div>
                <div style={{ display: "flex", gap: "6px" }}>
                  <button
                    type="button"
                    className={styles.historyActionBtn}
                    onClick={() => speechService.speak(item.text)}
                    title="Speak this saved sentence"
                    aria-label="Speak saved sentence"
                  >
                    🔊
                  </button>
                  <button
                    type="button"
                    className={styles.historyActionBtn}
                    onClick={() => handleDeleteSavedSentence(item.id)}
                    title="Delete this saved sentence"
                    aria-label="Delete saved sentence"
                  >
                    ✕
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className={styles.footer}>
        <span>
          Word Count: <strong>{committedTokens.length}</strong> signs
        </span>
        <span>Deterministic 30-Frame Stabilized ISL Output</span>
      </div>
    </div>
  );
};
