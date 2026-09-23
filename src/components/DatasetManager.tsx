import React, { useState, useEffect, useRef } from "react";
import { datasetStorage } from "../services/datasetStorage";
import type { LabelInfo, DatasetSample, ImportResult } from "../types/dataset";
import styles from "../styles/DatasetManager.module.css";

interface DatasetManagerProps {
  onRefreshTrigger?: () => void;
}

export const DatasetManager: React.FC<DatasetManagerProps> = ({ onRefreshTrigger }) => {
  const [labels, setLabels] = useState<LabelInfo[]>([]);
  const [totalSamplesCount, setTotalSamplesCount] = useState<number>(0);
  const [expandedLabel, setExpandedLabel] = useState<string | null>(null);
  const [sampleList, setSampleList] = useState<DatasetSample[]>([]);
  const [importResultModal, setImportResultModal] = useState<ImportResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const refreshData = async () => {
    const lbls = await datasetStorage.listLabels();
    setLabels(lbls);

    const total = lbls.reduce((sum, l) => sum + l.count, 0);
    setTotalSamplesCount(total);

    if (onRefreshTrigger) {
      onRefreshTrigger();
    }
  };

  useEffect(() => {
    refreshData();
  }, []);

  const handleExpandLabel = async (lbl: string) => {
    if (expandedLabel === lbl) {
      setExpandedLabel(null);
      setSampleList([]);
    } else {
      setExpandedLabel(lbl);
      const samples = await datasetStorage.listSamples(lbl);
      setSampleList(samples);
    }
  };

  const handleDeleteSample = async (id: string, label: string) => {
    if (confirm("Are you sure you want to delete this sample?")) {
      await datasetStorage.deleteSample(id);
      const samples = await datasetStorage.listSamples(label);
      setSampleList(samples);
      await refreshData();
    }
  };

  const handleDeleteLabel = async (lbl: string, count: number) => {
    if (confirm(`Are you sure you want to delete label '${lbl}'? All ${count} samples will be permanently lost.`)) {
      await datasetStorage.deleteLabel(lbl);
      if (expandedLabel === lbl) {
        setExpandedLabel(null);
        setSampleList([]);
      }
      await refreshData();
    }
  };

  const handleClearAll = async () => {
    if (
      confirm("⚠️ CAUTION: Are you sure you want to CLEAR THE ENTIRE DATASET? All recorded samples across all labels will be permanently deleted.")
    ) {
      await datasetStorage.clearAll();
      setExpandedLabel(null);
      setSampleList([]);
      await refreshData();
    }
  };

  const handleExportAll = async (targetLabel?: string) => {
    const exportData = await datasetStorage.exportAll(targetLabel);
    const jsonStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const filename = targetLabel
      ? `sign_bridge_${targetLabel.toLowerCase()}_v1.json`
      : `sign_bridge_dataset_v1.json`;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const json = JSON.parse(text);

      const result = await datasetStorage.importAll(json, { skipInvalid: true });
      setImportResultModal(result);
      await refreshData();
    } catch (err: unknown) {
      alert(`Import Failed: File is not a valid JSON document (${err instanceof Error ? err.message : "Unknown error"}).`);
    }

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Estimate total storage size (§7b): ~63 floats * 4 bytes * 30 frames = ~7.5KB per sample
  const estimatedSizeBytes = totalSamplesCount * 30 * 63 * 4;
  const estimatedSizeKB = (estimatedSizeBytes / 1024).toFixed(1);
  const estimatedSizeMB = (estimatedSizeBytes / (1024 * 1024)).toFixed(2);

  return (
    <div className={styles.container}>
      {/* Overview Statistics Card */}
      <div className={styles.card}>
        <div className={styles.header}>
          <div className={styles.title}>
            <span>📁</span> Dataset Management Dashboard
          </div>
          <div className={styles.actionRow}>
            <button type="button" className={styles.btn} onClick={() => handleExportAll()}>
              📥 Export Full Dataset (JSON)
            </button>
            <button
              type="button"
              className={styles.btn}
              onClick={() => fileInputRef.current?.click()}
            >
              📤 Import Dataset (JSON)
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleImportFile}
              style={{ display: "none" }}
            />
            <button
              type="button"
              className={styles.btnDanger}
              onClick={handleClearAll}
            >
              🗑️ Clear Entire Dataset
            </button>
          </div>
        </div>

        <div className={styles.statsBar}>
          <div className={styles.statBox}>
            <span className={styles.statLabel}>Sign Classes (Labels)</span>
            <span className={styles.statVal}>{labels.length}</span>
          </div>
          <div className={styles.statBox}>
            <span className={styles.statLabel}>Total Sequence Samples</span>
            <span className={styles.statVal}>{totalSamplesCount}</span>
          </div>
          <div className={styles.statBox}>
            <span className={styles.statLabel}>Total Frames Captured</span>
            <span className={styles.statVal}>{totalSamplesCount * 30}</span>
          </div>
          <div className={styles.statBox}>
            <span className={styles.statLabel}>Estimated Data Size</span>
            <span className={styles.statVal}>
              {estimatedSizeBytes > 1024 * 1024 ? `${estimatedSizeMB} MB` : `${estimatedSizeKB} KB`}
            </span>
          </div>
        </div>

        {/* Label Table List (§7) */}
        {labels.length > 0 ? (
          <table className={styles.labelTable}>
            <thead>
              <tr>
                <th>Class Label</th>
                <th>Samples Count</th>
                <th>Performer Variation (§11a)</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {labels.map((item) => (
                <React.Fragment key={item.label}>
                  <tr>
                    <td>
                      <strong style={{ color: "var(--accent-cyan)" }}>{item.label}</strong>
                    </td>
                    <td>{item.count} samples</td>
                    <td>
                      {item.performersCount <= 1 ? (
                        <span className={styles.performerWarning}>
                          ⚠️ {item.count} samples, only {item.performersCount} performer
                        </span>
                      ) : (
                        <span style={{ color: "var(--accent-green)", fontSize: "0.8rem", fontWeight: 600 }}>
                          ✅ {item.performersCount} distinct performers
                        </span>
                      )}
                    </td>
                    <td>
                      <div style={{ display: "flex", gap: "8px" }}>
                        <button
                          type="button"
                          className={styles.btn}
                          style={{ padding: "4px 10px", fontSize: "0.78rem" }}
                          onClick={() => handleExpandLabel(item.label)}
                        >
                          {expandedLabel === item.label ? "Hide Samples" : "Inspect Samples"}
                        </button>
                        <button
                          type="button"
                          className={styles.btn}
                          style={{ padding: "4px 10px", fontSize: "0.78rem" }}
                          onClick={() => handleExportAll(item.label)}
                        >
                          Export Label
                        </button>
                        <button
                          type="button"
                          className={styles.btnDanger}
                          onClick={() => handleDeleteLabel(item.label, item.count)}
                        >
                          Delete Label
                        </button>
                      </div>
                    </td>
                  </tr>

                  {/* Expanded Sample Inspector */}
                  {expandedLabel === item.label && (
                    <tr>
                      <td colSpan={4} style={{ background: "rgba(5, 8, 15, 0.9)", padding: "16px" }}>
                        <div style={{ fontSize: "0.82rem", fontWeight: 700, marginBottom: "8px", color: "var(--text-secondary)" }}>
                          Individual Samples for '{item.label}':
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                          {sampleList.map((s, idx) => (
                            <div
                              key={s.id}
                              style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "space-between",
                                padding: "8px 12px",
                                background: "rgba(255,255,255,0.03)",
                                borderRadius: "4px",
                                fontSize: "0.8rem",
                              }}
                            >
                              <span>
                                Sample #{idx + 1} (ID: {s.id.slice(0, 8)}...) • Captured: {new Date(s.createdAt).toLocaleString()} • Performer: {s.metadata?.performerId || "N/A"}
                              </span>
                              <button
                                type="button"
                                className={styles.btnDanger}
                                onClick={() => handleDeleteSample(s.id, item.label)}
                              >
                                Delete
                              </button>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ textAlign: "center", padding: "32px", color: "var(--text-muted)", fontSize: "0.9rem" }}>
            No dataset samples collected yet. Use the <strong>Collector Tool</strong> to start recording sign sequences.
          </div>
        )}
      </div>

      {/* Import Validation Results Modal (§9) */}
      {importResultModal && (
        <div className={styles.modalOverlay} role="dialog" aria-label="Import Validation Results">
          <div className={styles.modalContent}>
            <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: importResultModal.success ? "var(--accent-green)" : "var(--accent-coral)" }}>
              {importResultModal.success ? "✅ Dataset Import Completed" : "⚠️ Dataset Import Failed"}
            </h3>

            <div style={{ fontSize: "0.9rem", lineHeight: 1.5 }}>
              <div>Imported Samples: <strong>{importResultModal.importedCount}</strong></div>
              <div>Skipped Bad Samples: <strong>{importResultModal.skippedCount}</strong></div>
            </div>

            {importResultModal.errors.length > 0 && (
              <>
                <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--accent-amber)" }}>
                  Validation Issues Encountered:
                </div>
                <div className={styles.errorList}>
                  {importResultModal.errors.map((err, i) => (
                    <div key={i} className={styles.errorItem}>
                      {err.sampleIndex !== undefined ? `Sample #${err.sampleIndex}: ` : ""}
                      {err.message}
                    </div>
                  ))}
                </div>
              </>
            )}

            <button
              type="button"
              className={styles.btn}
              onClick={() => setImportResultModal(null)}
              style={{ alignSelf: "flex-end" }}
            >
              Close Report
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
