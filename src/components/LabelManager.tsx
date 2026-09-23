import React, { useState } from "react";
import type { SignLabel } from "../dataCollection/labelStore";
import styles from "../styles/LabelManager.module.css";

interface LabelManagerProps {
  labels: SignLabel[];
  activeLabelId: string | null;
  onCreateLabel: (name: string, target?: number) => void;
  onSelectLabel: (id: string) => void;
  onDeleteLabel: (id: string) => Promise<number>;
}

export const LabelManager: React.FC<LabelManagerProps> = ({
  labels,
  activeLabelId,
  onCreateLabel,
  onSelectLabel,
  onDeleteLabel,
}) => {
  const [newLabelInput, setNewLabelInput] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"name" | "sampleCountAsc">("name");

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);
    try {
      onCreateLabel(newLabelInput);
      setNewLabelInput("");
    } catch (err: unknown) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to create label.");
    }
  };

  const handleDeleteWithConfirmation = async (label: SignLabel, e: React.MouseEvent) => {
    e.stopPropagation();
    // Destructive confirmation naming exact sample count (§2.2, §15)
    const confirmed = window.confirm(
      `Are you sure you want to delete label "${label.name}" and all ${label.sampleCount} recorded samples? This cannot be undone.`
    );
    if (confirmed) {
      await onDeleteLabel(label.id);
    }
  };

  const sortedLabels = [...labels].sort((a, b) => {
    if (sortBy === "sampleCountAsc") {
      return a.sampleCount - b.sampleCount;
    }
    return a.name.localeCompare(b.name);
  });

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.title}>🏷️ Label Management Studio (§2)</div>
        <div style={{ display: "flex", gap: "8px", fontSize: "0.78rem" }}>
          <button
            type="button"
            onClick={() => setSortBy("name")}
            style={{ background: "none", border: "none", color: sortBy === "name" ? "var(--accent-cyan)" : "var(--text-muted)", cursor: "pointer", fontWeight: 700 }}
          >
            Name A-Z
          </button>
          <span>|</span>
          <button
            type="button"
            onClick={() => setSortBy("sampleCountAsc")}
            style={{ background: "none", border: "none", color: sortBy === "sampleCountAsc" ? "var(--accent-cyan)" : "var(--text-muted)", cursor: "pointer", fontWeight: 700 }}
          >
            Needs Samples ↑
          </button>
        </div>
      </div>

      {/* Create Label Input */}
      <form onSubmit={handleCreate} className={styles.createRow}>
        <input
          type="text"
          value={newLabelInput}
          onChange={(e) => setNewLabelInput(e.target.value)}
          placeholder="New Label Name (e.g. HELLO)"
          className={styles.input}
        />
        <button type="submit" className={styles.createBtn}>
          + Add Label
        </button>
      </form>

      {errorMessage && <div className={styles.errorText}>⚠️ {errorMessage}</div>}

      {/* Label Selection List */}
      <div className={styles.list}>
        {sortedLabels.map((lbl) => {
          const isActive = lbl.id === activeLabelId;
          const isUnderTarget = lbl.sampleCount < lbl.targetCount;

          return (
            <div
              key={lbl.id}
              className={`${styles.item} ${isActive ? styles.activeItem : ""}`}
              onClick={() => onSelectLabel(lbl.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") onSelectLabel(lbl.id); }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <span style={{ fontWeight: 800, color: isActive ? "var(--accent-cyan)" : "var(--text-primary)" }}>
                  {lbl.name}
                </span>
                <span style={{ fontSize: "0.75rem", color: isUnderTarget ? "#ffd166" : "var(--accent-green)", fontWeight: 600 }}>
                  ({lbl.sampleCount} / {lbl.targetCount})
                </span>
              </div>

              <button
                type="button"
                className={styles.deleteBtn}
                onClick={(e) => handleDeleteWithConfirmation(lbl, e)}
                title={`Delete ${lbl.name} and its ${lbl.sampleCount} samples`}
              >
                🗑 Delete
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
