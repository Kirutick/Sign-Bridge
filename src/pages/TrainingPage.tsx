import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import { datasetStorage } from "../services/datasetStorage";
import type { LabelInfo, DatasetSample } from "../types/dataset";
import styles from "../styles/TrainingPage.module.css";

// ─── Types ────────────────────────────────────────────────────────────────────
interface EpochLog {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss?: number;
  valAccuracy?: number;
}

interface PerClassMetrics {
  label: string;
  correct: number;
  total: number;
  accuracy: number;
}

type TrainingStatus =
  | "IDLE"
  | "TRAINING"
  | "EVALUATING"
  | "TARGET_REACHED"
  | "TARGET_NOT_REACHED"
  | "DATA_LIMITED";

// ─── Constants ────────────────────────────────────────────────────────────────
const TARGET_ACC = 0.90;
const SEQ_LEN = 30;
const FEAT_COUNT = 63;

// ─── Status badge colors ──────────────────────────────────────────────────────
const STATUS_COLOR: Record<TrainingStatus, string> = {
  IDLE: "#6b7280",
  TRAINING: "#3b82f6",
  EVALUATING: "#f59e0b",
  TARGET_REACHED: "#22c55e",
  TARGET_NOT_REACHED: "#ef4444",
  DATA_LIMITED: "#f59e0b",
};

// ─── Component ────────────────────────────────────────────────────────────────
export const TrainingPage: React.FC = () => {
  const [labels, setLabels] = useState<LabelInfo[]>([]);
  const [totalSamples, setTotalSamples] = useState<number>(0);

  // Config
  const [maxEpochs, setMaxEpochs] = useState(100);
  const [maxExperiments, setMaxExperiments] = useState(5);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.001);
  const [splitRatio, setSplitRatio] = useState(0.8);

  // Training state
  const [status, setStatus] = useState<TrainingStatus>("IDLE");
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [currentExperiment, setCurrentExperiment] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<EpochLog[]>([]);
  const [currentValAcc, setCurrentValAcc] = useState<number | null>(null);
  const [currentTestAcc, setCurrentTestAcc] = useState<number | null>(null);
  const [bestValAcc, setBestValAcc] = useState<number>(0);
  const [bestTestAcc, setBestTestAcc] = useState<number>(0);
  const [perClassMetrics, setPerClassMetrics] = useState<PerClassMetrics[]>([]);
  const [confusionMatrix, setConfusionMatrix] = useState<number[][] | null>(null);
  const [trainedModel, setTrainedModel] = useState<tf.LayersModel | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [stopReason, setStopReason] = useState<string>("");
  const abortRef = useRef(false);

  useEffect(() => {
    datasetStorage.listLabels().then((lbls) => {
      setLabels(lbls);
      setTotalSamples(lbls.reduce((s, l) => s + l.count, 0));
    }).catch(console.warn);
  }, []);

  // ─── Build model ──────────────────────────────────────────────────────────
  function buildModel(numClasses: number, units1 = 128, units2 = 64, lr = 0.001): tf.Sequential {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: units1, returnSequences: true, inputShape: [SEQ_LEN, FEAT_COUNT] }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.lstm({ units: units2 }));
    model.add(tf.layers.dropout({ rate: 0.25 }));
    model.add(tf.layers.dense({ units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: numClasses, activation: "softmax" }));
    model.compile({ optimizer: tf.train.adam(lr), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
    return model;
  }

  // ─── Confusion matrix + per-class metrics ─────────────────────────────────
  function computeMetrics(
    yTrue: number[], yPred: number[], numClasses: number, labelMap: string[]
  ): { perClass: PerClassMetrics[]; cm: number[][] } {
    const cm: number[][] = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
    yTrue.forEach((t, i) => { cm[t][yPred[i]]++; });

    const perClass: PerClassMetrics[] = labelMap.map((label, i) => {
      const total = cm[i].reduce((s, v) => s + v, 0);
      const correct = cm[i][i];
      return { label, correct, total, accuracy: total > 0 ? correct / total : 0 };
    });

    return { perClass, cm };
  }

  // ─── Training loop ────────────────────────────────────────────────────────
  const handleStartTraining = async () => {
    if (labels.length < 2) {
      setErrorMsg("Need at least 2 sign classes. Record or upload samples first.");
      return;
    }
    if (totalSamples < 6) {
      setErrorMsg("Too few samples. Record at least 3 per class.");
      return;
    }

    abortRef.current = false;
    setErrorMsg(null);
    setCurrentEpoch(0);
    setCurrentExperiment(0);
    setTrainingLogs([]);
    setCurrentValAcc(null);
    setCurrentTestAcc(null);
    setBestValAcc(0);
    setBestTestAcc(0);
    setPerClassMetrics([]);
    setConfusionMatrix(null);
    setStopReason("");
    setStatus("TRAINING");

    try {
      // Gather data from IndexedDB
      const allSamples: DatasetSample[] = [];
      const labelMap: Record<string, number> = {};
      labels.forEach((l, i) => { labelMap[l.label] = i; });
      const labelArray = labels.map((l) => l.label);
      const numClasses = labels.length;

      for (const l of labels) {
        const samples = await datasetStorage.listSamples(l.label);
        allSamples.push(...samples);
      }

      // Shuffle
      for (let i = allSamples.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allSamples[i], allSamples[j]] = [allSamples[j], allSamples[i]];
      }

      // Filter valid 30×63 sequences
      const valid = allSamples.filter(
        (s) => s.sequence?.length === SEQ_LEN && s.sequence.every((r) => Array.isArray(r) && r.length === FEAT_COUNT)
      );

      if (valid.length < 4) throw new Error("Not enough valid 30×63 sequences.");

      const Xall = valid.map((s) => s.sequence);
      const yall = valid.map((s) => {
        const oh = new Array(numClasses).fill(0);
        oh[labelMap[s.label]] = 1;
        return oh;
      });

      // Train/Val/Test split (80%/10%/10%)
      const n = Xall.length;
      const trainEnd = Math.floor(n * splitRatio);
      const valEnd = trainEnd + Math.floor(n * ((1 - splitRatio) / 2));

      const xTrain = tf.tensor3d(Xall.slice(0, trainEnd), [trainEnd, SEQ_LEN, FEAT_COUNT]);
      const yTrain = tf.tensor2d(yall.slice(0, trainEnd), [trainEnd, numClasses]);
      const xVal = tf.tensor3d(Xall.slice(trainEnd, valEnd), [valEnd - trainEnd, SEQ_LEN, FEAT_COUNT]);
      const yVal = tf.tensor2d(yall.slice(trainEnd, valEnd), [valEnd - trainEnd, numClasses]);
      const xTest = tf.tensor3d(Xall.slice(valEnd), [n - valEnd, SEQ_LEN, FEAT_COUNT]);
      const yTest = tf.tensor2d(yall.slice(valEnd), [n - valEnd, numClasses]);
      const yTestTrue = valid.slice(valEnd).map((s) => labelMap[s.label]);

      // Experiment configs
      const expConfigs = [
        { units1: 128, units2: 64, lr: learningRate },
        { units1: 128, units2: 64, lr: learningRate * 0.5 },
        { units1: 256, units2: 128, lr: learningRate },
        { units1: 64,  units2: 32,  lr: learningRate * 2 },
        { units1: 256, units2: 128, lr: learningRate * 0.3 },
      ].slice(0, maxExperiments);

      let bestModel: tf.LayersModel | null = null;
      let bestVal = 0;
      let bestTest = 0;
      let targetReached = false;

      for (let expIdx = 0; expIdx < expConfigs.length && !abortRef.current; expIdx++) {
        const exp = expConfigs[expIdx];
        setCurrentExperiment(expIdx + 1);
        setTrainingLogs([]);
        setStatus("TRAINING");

        const model = buildModel(numClasses, exp.units1, exp.units2, exp.lr);
        const logsAcc: EpochLog[] = [];
        let bestValThisRun = 0;

        await model.fit(xTrain, yTrain, {
          epochs: maxEpochs,
          batchSize,
          validationData: [xVal, yVal],
          shuffle: true,
          callbacks: {
            onEpochEnd: async (epoch, logs) => {
              if (abortRef.current) {
                model.stopTraining = true;
                return;
              }
              setCurrentEpoch(epoch + 1);
              const entry: EpochLog = {
                epoch: epoch + 1,
                loss: logs?.loss || 0,
                accuracy: logs?.acc || logs?.accuracy || 0,
                valLoss: logs?.val_loss,
                valAccuracy: logs?.val_acc || logs?.val_accuracy,
              };
              logsAcc.push(entry);
              setTrainingLogs([...logsAcc]);

              const va = entry.valAccuracy ?? 0;
              setCurrentValAcc(va);
              if (va > bestValThisRun) bestValThisRun = va;

              await tf.nextFrame();
            },
          },
        });

        if (abortRef.current) {
          model.dispose();
          break;
        }

        // Evaluate test set
        setStatus("EVALUATING");
        const testProbs = model.predict(xTest) as tf.Tensor;
        const yTestPred = Array.from(await tf.argMax(testProbs, -1).data());
        testProbs.dispose();

        const testCorrect = yTestPred.filter((p, i) => p === yTestTrue[i]).length;
        const testAcc = yTestTrue.length > 0 ? testCorrect / yTestTrue.length : 0;
        setCurrentTestAcc(testAcc);

        if (bestValThisRun > bestVal) {
          bestVal = bestValThisRun;
          if (bestModel) bestModel.dispose();
          bestModel = model;
        } else {
          model.dispose();
        }
        if (testAcc > bestTest) bestTest = testAcc;

        setBestValAcc(bestVal);
        setBestTestAcc(bestTest);

        // Per-class metrics
        const { perClass, cm } = computeMetrics(yTestTrue, yTestPred, numClasses, labelArray);
        setPerClassMetrics(perClass);
        setConfusionMatrix(cm);

        // 90% check
        if (bestValThisRun >= TARGET_ACC && testAcc >= TARGET_ACC) {
          targetReached = true;
          setStatus("TARGET_REACHED");
          setStopReason(`✅ 90% reached at experiment ${expIdx + 1}: val=${(bestValThisRun * 100).toFixed(1)}% test=${(testAcc * 100).toFixed(1)}%`);
          break;
        }
      }

      if (!targetReached && !abortRef.current) {
        if (valid.length < 50) {
          setStatus("DATA_LIMITED");
          setStopReason(`⚠️ Budget exhausted. ${valid.length} sequences is likely too few for 90%. Collect more ISL data (target: 50+ per class from multiple signers).`);
        } else {
          setStatus("TARGET_NOT_REACHED");
          setStopReason(`❌ 90% not reached after ${maxExperiments} experiments. Best val: ${(bestVal * 100).toFixed(1)}%. Collect more diverse ISL data.`);
        }
      }

      if (bestModel) setTrainedModel(bestModel);

      xTrain.dispose(); yTrain.dispose();
      xVal.dispose(); yVal.dispose();
      xTest.dispose(); yTest.dispose();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Training error");
      setStatus("IDLE");
    }
  };

  const handleAbort = () => { abortRef.current = true; };

  const handleDownloadModel = async () => {
    if (!trainedModel) return;
    await trainedModel.save("downloads://sign_bridge_isl_model");
    const metadata = {
      signLanguage: "Indian Sign Language", signLanguageCode: "ISL",
      outputLanguage: "English", outputLanguageCode: "en",
      featureCount: FEAT_COUNT, sequenceLength: SEQ_LEN,
      trainedAt: new Date().toISOString(),
      numClasses: labels.length, classes: labels.map((l) => l.label),
      validationAccuracy: bestValAcc, testAccuracy: bestTestAcc,
    };
    const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "model_metadata.json" });
    a.click();
    URL.revokeObjectURL(a.href);
  };

  // ─── Render helpers ───────────────────────────────────────────────────────
  const progressPct = maxEpochs > 0 ? Math.round((currentEpoch / maxEpochs) * 100) : 0;
  const isActive = status === "TRAINING" || status === "EVALUATING";
  const statusColor = STATUS_COLOR[status];
  const card: React.CSSProperties = { padding: "20px 24px", background: "var(--bg-card)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-md)" };
  const metricBox: React.CSSProperties = { background: "rgba(0,0,0,0.3)", border: "1px solid var(--border-color)", borderRadius: "var(--radius-sm)", padding: "16px", textAlign: "center" };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>In-Browser ISL Training Studio</h1>
        <p className={styles.subtitle}>Real deep LSTM training on your recorded/uploaded ISL landmark sequences. Target: ≥90% validation accuracy.</p>
      </div>

      {errorMsg && (
        <div style={{ padding: "12px 16px", background: "rgba(239,68,68,0.15)", border: "1px solid #ef4444", borderRadius: 8, color: "#fca5a5" }}>⚠️ {errorMsg}</div>
      )}

      {/* ─── 90% DASHBOARD ────────────────────────────────────── */}
      <div style={{ ...card, background: "linear-gradient(135deg, rgba(0,242,254,0.05) 0%, rgba(79,172,254,0.05) 100%)", border: "1px solid var(--accent-cyan)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--accent-cyan)" }}>🎯 90% Accuracy Dashboard</h2>
          <span style={{ padding: "4px 12px", borderRadius: 999, background: `${statusColor}22`, border: `1px solid ${statusColor}`, color: statusColor, fontWeight: 700, fontSize: "0.8rem", letterSpacing: "0.05em" }}>
            {status.replace("_", " ")}
          </span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
          <div style={metricBox}>
            <div style={{ fontSize: "2rem", fontWeight: 900, color: "var(--accent-cyan)" }}>90%</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>TARGET</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "2rem", fontWeight: 900, color: currentValAcc !== null && currentValAcc >= 0.90 ? "var(--accent-green)" : "#f59e0b" }}>
              {currentValAcc !== null ? `${(currentValAcc * 100).toFixed(1)}%` : "—"}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Current Validation</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "2rem", fontWeight: 900, color: currentTestAcc !== null && currentTestAcc >= 0.90 ? "var(--accent-green)" : "#f59e0b" }}>
              {currentTestAcc !== null ? `${(currentTestAcc * 100).toFixed(1)}%` : "—"}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Current Test</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "2rem", fontWeight: 900, color: bestValAcc >= 0.90 ? "var(--accent-green)" : "var(--text-primary)" }}>
              {bestValAcc > 0 ? `${(bestValAcc * 100).toFixed(1)}%` : "—"}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Best Validation</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "2rem", fontWeight: 900, color: bestTestAcc >= 0.90 ? "var(--accent-green)" : "var(--text-primary)" }}>
              {bestTestAcc > 0 ? `${(bestTestAcc * 100).toFixed(1)}%` : "—"}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Best Test</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "var(--text-primary)" }}>
              {isActive ? `${currentEpoch}/${maxEpochs}` : "—"}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Epoch</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "var(--text-primary)" }}>
              {currentExperiment > 0 ? `${currentExperiment}/${maxExperiments}` : "—"}
            </div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Experiment</div>
          </div>
          <div style={metricBox}>
            <div style={{ fontSize: "1.4rem", fontWeight: 900, color: "var(--text-primary)" }}>{labels.length}</div>
            <div style={{ fontSize: "0.7rem", color: "var(--text-secondary)", textTransform: "uppercase" }}>Classes</div>
          </div>
        </div>

        {stopReason && (
          <div style={{ marginTop: 14, padding: "10px 14px", background: status === "TARGET_REACHED" ? "rgba(34,197,94,0.1)" : "rgba(239,68,68,0.1)", borderRadius: 8, fontSize: "0.87rem", color: status === "TARGET_REACHED" ? "#34d399" : "#fca5a5" }}>
            {stopReason}
          </div>
        )}
      </div>

      {/* ─── Config ────────────────────────────────────────────── */}
      <div style={card}>
        <div style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: 16, color: "var(--accent-cyan)" }}>⚙️ Training Configuration</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14 }}>
          {[
            { label: `Max Epochs: ${maxEpochs}`, render: () => <input type="range" min={20} max={300} step={10} value={maxEpochs} disabled={isActive} onChange={(e) => setMaxEpochs(+e.target.value)} style={{ width: "100%" }} /> },
            { label: `Max Experiments: ${maxExperiments}`, render: () => <input type="range" min={1} max={20} step={1} value={maxExperiments} disabled={isActive} onChange={(e) => setMaxExperiments(+e.target.value)} style={{ width: "100%" }} /> },
          ].map((f) => (
            <div key={f.label}>
              <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: 6 }}>{f.label}</label>
              {f.render()}
            </div>
          ))}
          <div>
            <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: 6 }}>Batch Size</label>
            <select className={styles.input} value={batchSize} disabled={isActive} onChange={(e) => setBatchSize(+e.target.value)}>
              {[8, 16, 32, 64].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: 6 }}>Learning Rate</label>
            <select className={styles.input} value={learningRate} disabled={isActive} onChange={(e) => setLearningRate(+e.target.value)}>
              <option value={0.005}>0.005 (Fast)</option>
              <option value={0.001}>0.001 (Default)</option>
              <option value={0.0005}>0.0005 (Conservative)</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--text-secondary)", display: "block", marginBottom: 6 }}>Train Split</label>
            <select className={styles.input} value={splitRatio} disabled={isActive} onChange={(e) => setSplitRatio(+e.target.value)}>
              <option value={0.8}>80% train / 10% val / 10% test</option>
              <option value={0.7}>70% / 15% / 15%</option>
            </select>
          </div>
        </div>

        {/* Progress bar */}
        {isActive && (
          <div style={{ marginTop: 16 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", color: "var(--text-secondary)", marginBottom: 6 }}>
              <span>Experiment {currentExperiment}/{maxExperiments} — Epoch {currentEpoch}/{maxEpochs}</span>
              <span>{progressPct}%</span>
            </div>
            <div style={{ background: "rgba(255,255,255,0.08)", borderRadius: 999, height: 6, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${progressPct}%`, background: "linear-gradient(90deg, var(--accent-cyan), #4facfe)", transition: "width 0.3s ease", borderRadius: 999 }} />
            </div>
          </div>
        )}

        <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
          <button type="button" className={`${styles.btn} ${styles.btnPrimary}`} onClick={handleStartTraining} disabled={isActive || totalSamples < 4}>
            {isActive ? "⏳ Training…" : "🚀 Start Training Loop"}
          </button>
          {isActive && (
            <button type="button" className={styles.btn} onClick={handleAbort}>⏹️ Stop</button>
          )}
          {trainedModel && !isActive && (
            <button type="button" className={styles.btn} onClick={handleDownloadModel}>💾 Export Model</button>
          )}
        </div>
      </div>

      {/* ─── Epoch Log ─────────────────────────────────────────── */}
      {trainingLogs.length > 0 && (
        <div style={card}>
          <div style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: 12, color: "var(--accent-cyan)" }}>📊 Live Epoch Log (Experiment {currentExperiment})</div>
          <div className={styles.logBox}>
            {trainingLogs.slice(-20).map((log) => (
              <div key={log.epoch} className={styles.logEntry}>
                <span style={{ color: "var(--text-secondary)" }}>Ep {String(log.epoch).padStart(3)}</span>
                <span>Loss: {log.loss.toFixed(4)}</span>
                <span>Acc: {(log.accuracy * 100).toFixed(1)}%</span>
                {log.valAccuracy !== undefined && (
                  <span style={{ color: log.valAccuracy >= 0.90 ? "var(--accent-green)" : log.valAccuracy >= 0.70 ? "#f59e0b" : "#ef4444" }}>
                    Val: {(log.valAccuracy * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─── Per-class metrics ─────────────────────────────────── */}
      {perClassMetrics.length > 0 && (
        <div style={card}>
          <div style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: 12, color: "var(--accent-cyan)" }}>📋 Per-Class Test Accuracy</div>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border-color)", color: "var(--text-secondary)" }}>
                <th style={{ padding: "8px", textAlign: "left" }}>Sign</th>
                <th style={{ padding: "8px", textAlign: "right" }}>Correct</th>
                <th style={{ padding: "8px", textAlign: "right" }}>Total</th>
                <th style={{ padding: "8px", textAlign: "right" }}>Accuracy</th>
              </tr>
            </thead>
            <tbody>
              {perClassMetrics.map((m) => (
                <tr key={m.label} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                  <td style={{ padding: "8px", fontWeight: 700 }}>{m.label}</td>
                  <td style={{ padding: "8px", textAlign: "right" }}>{m.correct}</td>
                  <td style={{ padding: "8px", textAlign: "right" }}>{m.total}</td>
                  <td style={{ padding: "8px", textAlign: "right" }}>
                    <span style={{ fontWeight: 700, color: m.accuracy >= 0.90 ? "var(--accent-green)" : m.accuracy >= 0.70 ? "#f59e0b" : "#ef4444" }}>
                      {(m.accuracy * 100).toFixed(1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ─── Confusion matrix ──────────────────────────────────── */}
      {confusionMatrix && perClassMetrics.length > 0 && (
        <div style={card}>
          <div style={{ fontWeight: 700, fontSize: "0.95rem", marginBottom: 12, color: "var(--accent-cyan)" }}>🔢 Confusion Matrix (Test Set)</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ borderCollapse: "collapse", fontSize: "0.7rem" }}>
              <thead>
                <tr>
                  <th style={{ padding: 4 }} />
                  {perClassMetrics.map((m) => (
                    <th key={m.label} style={{ padding: "4px 2px", color: "var(--text-secondary)", writingMode: "vertical-rl", transform: "rotate(180deg)", maxHeight: 60 }}>{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, i) => {
                  const rowTotal = row.reduce((s, v) => s + v, 0);
                  return (
                    <tr key={i}>
                      <td style={{ padding: "2px 8px 2px 4px", fontWeight: 700, color: "var(--text-secondary)", fontSize: "0.72rem" }}>{perClassMetrics[i]?.label}</td>
                      {row.map((val, j) => {
                        const intensity = rowTotal > 0 ? val / rowTotal : 0;
                        const isCorrect = i === j;
                        return (
                          <td key={j} style={{
                            width: 32, height: 32, textAlign: "center", fontWeight: isCorrect && val > 0 ? 700 : 400,
                            background: isCorrect ? `rgba(34,197,94,${0.15 + intensity * 0.85})` : val > 0 ? `rgba(239,68,68,${0.1 + intensity * 0.6})` : "transparent",
                            color: isCorrect && val > 0 ? "#22c55e" : val > 0 ? "#fca5a5" : "var(--text-muted)",
                            fontSize: "0.72rem", border: "1px solid rgba(255,255,255,0.04)",
                          }}>{val > 0 ? val : ""}</td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
