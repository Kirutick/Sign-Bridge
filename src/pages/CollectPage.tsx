import React, { useState, useEffect, useCallback } from "react";
import { useWebcam } from "../hooks/useWebcam";
import { useHandLandmarker } from "../hooks/useHandLandmarker";
import { datasetStorage } from "../services/datasetStorage";
import { DatasetCollector } from "../components/DatasetCollector";
import { DatasetManager } from "../components/DatasetManager";
import { ControlBar } from "../components/ControlBar";
import { MediaUploadStudio } from "../components/MediaUploadStudio";
import { scanForDuplicates, DatasetDuplicateReport } from "../dataPipeline/datasetDuplicateScan";
import { detectOutliers, OutlierFlag } from "../dataPipeline/outlierDetection";
import type { RecordedSample } from "../dataCollection/validateSample";
import styles from "../styles/CollectPage.module.css";

export const CollectPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"record" | "manage" | "audit" | "upload">("record");
  const [existingLabels, setExistingLabels] = useState<string[]>([]);
  const [duplicateReport, setDuplicateReport] = useState<DatasetDuplicateReport | null>(null);
  const [outlierFlags, setOutlierFlags] = useState<OutlierFlag[]>([]);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [allSamples, setAllSamples] = useState<RecordedSample[]>([]);

  const { stream, status: webcamStatus, start, stop } = useWebcam();
  const { isReady: isLandmarkerReady } = useHandLandmarker();

  const loadLabels = useCallback(async () => {
    const lbls = await datasetStorage.listLabels();
    setExistingLabels(lbls.map((l) => l.label));
  }, []);

  useEffect(() => {
    loadLabels();
  }, [loadLabels]);

  const handleRunAudit = async () => {
    setIsScanning(true);
    try {
      const lbls = await datasetStorage.listLabels();
      const samples: RecordedSample[] = [];
      for (const l of lbls) {
        const sList = await datasetStorage.listSamples(l.label);
        for (const s of sList) {
          samples.push({
            id: s.id,
            label: s.label,
            sequence: s.sequence,
            sequenceLength: s.sequenceLength || 30,
            featureCount: s.featureCount || 63,
            timestamp: s.createdAt,
            handedness: (s.handedness?.[0] as any) || "Right",
            source: "webcam",
            datasetVersion: "v1",
            normalizationVersion: "v1",
          });
        }
      }
      setAllSamples(samples);

      const dupReport = scanForDuplicates(samples, 0.05);
      setDuplicateReport(dupReport);

      const flags = detectOutliers(samples);
      setOutlierFlags(flags);
    } catch (e) {
      console.warn("Audit error:", e);
    } finally {
      setIsScanning(false);
    }
  };

  return (
    <div className={styles.pageContainer}>
      {activeTab === "record" && (
        <ControlBar webcamStatus={webcamStatus} onStart={start} onStop={stop} />
      )}

      <div className={styles.subNavBar}>
        <button
          type="button"
          className={`${styles.subTabBtn} ${
            activeTab === "record" ? styles.subTabActive : ""
          }`}
          onClick={() => setActiveTab("record")}
        >
          📹 Sequence Recorder
        </button>

        <button
          type="button"
          className={`${styles.subTabBtn} ${
            activeTab === "manage" ? styles.subTabActive : ""
          }`}
          onClick={() => setActiveTab("manage")}
        >
          🗄️ Dataset Manager & Export/Import
        </button>

        <button
          type="button"
          className={`${styles.subTabBtn} ${
            activeTab === "audit" ? styles.subTabActive : ""
          }`}
          onClick={() => {
            setActiveTab("audit");
            handleRunAudit();
          }}
        >
          🔍 Quality &amp; Duplicate Audit
        </button>

        <button
          type="button"
          className={`${styles.subTabBtn} ${
            activeTab === "upload" ? styles.subTabActive : ""
          }`}
          onClick={() => setActiveTab("upload")}
        >
          📁 Upload Media
        </button>
      </div>

      {activeTab === "record" && (
        <DatasetCollector
          stream={stream}
          isLandmarkerReady={isLandmarkerReady}
          existingLabels={existingLabels}
          onSampleSaved={loadLabels}
        />
      )}

      {activeTab === "manage" && (
        <DatasetManager onRefreshTrigger={loadLabels} />
      )}

      {activeTab === "upload" && (
        <MediaUploadStudio onSampleSaved={loadLabels} />
      )}

      {activeTab === "audit" && (
        <div style={{ width: "100%", maxWidth: "900px", display: "flex", flexDirection: "column", gap: "16px" }}>
          <div style={{ padding: "20px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-cyan)" }}>
                Dataset Integrity & Duplicate Scanner
              </h3>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                Audits all samples in IndexedDB for exact duplicates, suspicious similarities, and geometric outliers.
              </p>
            </div>
            <button
              type="button"
              onClick={handleRunAudit}
              disabled={isScanning}
              style={{ padding: "8px 16px", background: "var(--accent-cyan)", border: "none", borderRadius: "6px", color: "#000", fontWeight: 700, cursor: "pointer" }}
            >
              {isScanning ? "Scanning..." : "Re-run Audit"}
            </button>
          </div>

          {/* Audit Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px" }}>
            <div style={{ padding: "14px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
              <div style={{ fontSize: "1.6rem", fontWeight: 800, color: "var(--text-primary)" }}>{allSamples.length}</div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Total Samples Scanned</div>
            </div>
            <div style={{ padding: "14px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
              <div style={{ fontSize: "1.6rem", fontWeight: 800, color: duplicateReport?.exactDuplicates.length ? "#ef4444" : "var(--accent-green)" }}>
                {duplicateReport?.exactDuplicates.length ?? 0}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Exact Duplicates</div>
            </div>
            <div style={{ padding: "14px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
              <div style={{ fontSize: "1.6rem", fontWeight: 800, color: duplicateReport?.potentialDuplicates.length ? "#f59e0b" : "var(--accent-green)" }}>
                {duplicateReport?.potentialDuplicates.length ?? 0}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Near-Duplicates Flagged</div>
            </div>
            <div style={{ padding: "14px", background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", textAlign: "center" }}>
              <div style={{ fontSize: "1.6rem", fontWeight: 800, color: outlierFlags.length ? "#f59e0b" : "var(--accent-green)" }}>
                {outlierFlags.length}
              </div>
              <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Outliers Flagged</div>
            </div>
          </div>

          {/* Duplicates Log */}
          {duplicateReport && duplicateReport.exactDuplicates.length > 0 && (
            <div style={{ padding: "16px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid #ef4444", borderRadius: "8px" }}>
              <strong style={{ color: "#fca5a5", display: "block", marginBottom: "8px" }}>
                ⚠️ Exact Duplicates Detected:
              </strong>
              {duplicateReport.exactDuplicates.map((dup, idx) => (
                <div key={idx} style={{ fontSize: "0.85rem", padding: "4px 0", color: "var(--text-primary)" }}>
                  Sample 1: {dup.sampleId1} ({dup.label1}) ↔ Sample 2: {dup.sampleId2} ({dup.label2})
                </div>
              ))}
            </div>
          )}

          {duplicateReport && duplicateReport.potentialDuplicates.length > 0 && (
            <div style={{ padding: "16px", background: "rgba(245, 158, 11, 0.1)", border: "1px solid #f59e0b", borderRadius: "8px" }}>
              <strong style={{ color: "#fcd34d", display: "block", marginBottom: "8px" }}>
                Near-Duplicates (High Similarity):
              </strong>
              {duplicateReport.potentialDuplicates.slice(0, 10).map((dup, idx) => (
                <div key={idx} style={{ fontSize: "0.85rem", padding: "4px 0", color: "var(--text-primary)" }}>
                  {dup.label1} vs {dup.label2} | Distance: {dup.similarityScore.toFixed(4)}
                </div>
              ))}
            </div>
          )}

          {(!duplicateReport || (duplicateReport.exactDuplicates.length === 0 && duplicateReport.potentialDuplicates.length === 0)) && (
            <div style={{ padding: "16px", background: "rgba(16, 185, 129, 0.1)", border: "1px solid var(--accent-green)", borderRadius: "8px", color: "#34d399", textAlign: "center" }}>
              ✓ No duplicate sequence anomalies detected in IndexedDB.
            </div>
          )}
        </div>
      )}
    </div>
  );
};
