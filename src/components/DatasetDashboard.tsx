import React from "react";
import type { SignLabel } from "../dataCollection/labelStore";
import styles from "../styles/DatasetDashboard.module.css";

interface DatasetDashboardProps {
  labels: SignLabel[];
  onSelectLabel?: (id: string) => void;
}

export const DatasetDashboard: React.FC<DatasetDashboardProps> = ({
  labels,
  onSelectLabel,
}) => {
  const totalClasses = labels.length;
  const totalSamples = labels.reduce((acc, l) => acc + l.sampleCount, 0);
  const avgSamples = totalClasses > 0 ? Math.round(totalSamples / totalClasses) : 0;

  // Imbalance threshold: classes under 50% of average sample count (§11)
  const imbalanceThreshold = Math.max(1, Math.round(avgSamples * 0.5));

  // Sorted worst-covered-first (fewest samples relative to target §11)
  const sortedLabels = [...labels].sort((a, b) => {
    const deltaA = a.sampleCount - a.targetCount;
    const deltaB = b.sampleCount - b.targetCount;
    return deltaA - deltaB;
  });

  return (
    <div className={styles.container}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: "1rem", fontWeight: 800, color: "var(--accent-cyan)", textTransform: "uppercase" }}>
          📊 Dataset Progress Dashboard (§11)
        </h3>
        <span style={{ fontSize: "0.78rem", color: "var(--text-secondary)" }}>
          Imbalance Alert Threshold: &lt; {imbalanceThreshold} samples
        </span>
      </div>

      {/* Summary Metrics Cards */}
      <div className={styles.summaryRow}>
        <div className={styles.card}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>Total Classes</span>
          <span style={{ fontSize: "1.4rem", fontWeight: 800 }}>{totalClasses}</span>
        </div>

        <div className={styles.card}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>Total Samples</span>
          <span style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--accent-green)" }}>{totalSamples}</span>
        </div>

        <div className={styles.card}>
          <span style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>Average / Class</span>
          <span style={{ fontSize: "1.4rem", fontWeight: 800 }}>{avgSamples}</span>
        </div>
      </div>

      {/* Per-Class Progress Table (Sorted Worst-Covered First §11) */}
      <div style={{ maxHeight: "240px", overflowY: "auto" }}>
        <table className={styles.table}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-color)", color: "var(--text-secondary)" }}>
              <th style={{ padding: "8px" }}>Class Label</th>
              <th style={{ padding: "8px" }}>Count / Target</th>
              <th style={{ padding: "8px" }}>Delta</th>
              <th style={{ padding: "8px" }}>Balance Status</th>
            </tr>
          </thead>
          <tbody>
            {sortedLabels.map((lbl) => {
              const delta = lbl.sampleCount - lbl.targetCount;
              const isImbalanced = totalSamples > 10 && lbl.sampleCount < imbalanceThreshold;

              return (
                <tr
                  key={lbl.id}
                  className={isImbalanced ? styles.imbalancedRow : ""}
                  style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", cursor: onSelectLabel ? "pointer" : "default" }}
                  onClick={() => onSelectLabel && onSelectLabel(lbl.id)}
                >
                  <td style={{ padding: "8px", fontWeight: 700, color: "var(--accent-cyan)" }}>{lbl.name}</td>
                  <td style={{ padding: "8px" }}>
                    <strong>{lbl.sampleCount}</strong> / {lbl.targetCount}
                  </td>
                  <td style={{ padding: "8px", color: delta >= 0 ? "var(--accent-green)" : "#ffd166" }}>
                    {delta >= 0 ? `+${delta}` : delta}
                  </td>
                  <td style={{ padding: "8px" }}>
                    {isImbalanced ? (
                      <span className={styles.flagChip}>⚠️ Needs Data</span>
                    ) : lbl.sampleCount >= lbl.targetCount ? (
                      <span style={{ color: "var(--accent-green)", fontWeight: 600 }}>✓ Target Met</span>
                    ) : (
                      <span style={{ color: "var(--text-secondary)" }}>In Progress</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
