import type {
  DatasetSample,
  DatasetExport,
  ImportOptions,
  ImportResult,
  ImportErrorDetail,
  LabelInfo,
  DatasetStorage,
} from "../types/dataset";

const DB_NAME = "SignBridgeDatasetDB";
const DB_VERSION = 1;
const STORE_NAME = "samples";

/** Helper to generate a simple UUID v4 string without external dependencies */
export function generateUUID(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

/** Open or upgrade IndexedDB connection */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    // Support fake-indexeddb in Vitest/Node environment
    const idbFactory =
      typeof window !== "undefined" && window.indexedDB
        ? window.indexedDB
        : globalThis.indexedDB;

    if (!idbFactory) {
      reject(new Error("IndexedDB is not supported in this environment."));
      return;
    }

    const request = idbFactory.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("label", "label", { unique: false });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

class IndexedDBDatasetStorage implements DatasetStorage {
  public async addSample(sample: DatasetSample): Promise<string> {
    if (!sample.id) {
      sample.id = generateUUID();
    }

    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.put(sample);

      req.onsuccess = () => resolve(sample.id);
      req.onerror = () => reject(req.error);
    });
  }

  /**
   * Convenience method that enriches a partial sample with ISL/English metadata
   * before storing. Used by MediaUploadStudio and DatasetCollector.
   */
  public async saveSample(partial: Partial<DatasetSample> & {
    label: string;
    sequence: number[][];
    source?: string;
    metadata?: Record<string, unknown>;
  }): Promise<string> {
    const sample: DatasetSample = {
      id: generateUUID(),
      label: partial.label.trim().toUpperCase().replace(/\s+/g, "_"),
      sequence: partial.sequence,
      sequenceLength: partial.sequenceLength ?? partial.sequence.length,
      featureCount: partial.featureCount ?? (partial.sequence[0]?.length ?? 63),
      handedness: partial.handedness ?? ["Unknown"],
      createdAt: new Date().toISOString(),
      datasetVersion: partial.datasetVersion ?? "v1",
      normalizationVersion: (partial as any).normalizationVersion ?? "v1",
      isSynthetic: (partial as any).isSynthetic ?? false,
      signLanguage: "Indian Sign Language",
      signLanguageCode: "ISL",
      outputLanguage: "English",
      outputLanguageCode: "en",
      sourceType: (partial.source ?? "webcam") as DatasetSample["sourceType"],
      sourceVideoId: (partial.metadata as any)?.sourceVideoHash ?? undefined,
      metadata: {
        ...(partial.metadata ?? {}),
        signerId: (partial.metadata as any)?.signerId ?? "signer_unknown",
        sourceType: partial.source ?? "webcam",
      },
    };
    return this.addSample(sample);
  }

  public async getSample(id: string): Promise<DatasetSample | null> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(id);

      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  }

  public async listSamples(label?: string): Promise<DatasetSample[]> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);

      let req: IDBRequest<DatasetSample[]>;
      if (label) {
        const index = store.index("label");
        req = index.getAll(label);
      } else {
        req = store.getAll();
      }

      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => reject(req.error);
    });
  }

  public async deleteSample(id: string): Promise<void> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.delete(id);

      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  public async deleteLabel(label: string): Promise<void> {
    const samples = await this.listSamples(label);
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);

      let completed = 0;
      if (samples.length === 0) {
        resolve();
        return;
      }

      for (const sample of samples) {
        const req = store.delete(sample.id);
        req.onsuccess = () => {
          completed++;
          if (completed === samples.length) {
            resolve();
          }
        };
        req.onerror = () => reject(req.error);
      }
    });
  }

  public async listLabels(): Promise<LabelInfo[]> {
    const allSamples = await this.listSamples();
    const labelMap = new Map<string, { count: number; performers: Set<string> }>();

    for (const sample of allSamples) {
      const lbl = sample.label;
      if (!labelMap.has(lbl)) {
        labelMap.set(lbl, { count: 0, performers: new Set() });
      }
      const entry = labelMap.get(lbl)!;
      entry.count++;
      if (sample.metadata?.performerId) {
        entry.performers.add(sample.metadata.performerId);
      }
    }

    const result: LabelInfo[] = [];
    labelMap.forEach((val, key) => {
      result.push({
        label: key,
        count: val.count,
        performersCount: val.performers.size || (val.count > 0 ? 1 : 0),
      });
    });

    return result.sort((a, b) => a.label.localeCompare(b.label));
  }

  public async clearAll(): Promise<void> {
    const db = await openDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      const req = store.clear();

      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
  }

  public async exportAll(targetLabel?: string): Promise<DatasetExport> {
    const samples = await this.listSamples(targetLabel);

    const featureCount = samples.length > 0 ? samples[0].featureCount : 63;
    const sequenceLength = samples.length > 0 ? samples[0].sequenceLength : 30;

    return {
      version: 1,
      featureCount,
      sequenceLength,
      exportedAt: new Date().toISOString(),
      signLanguage: "Indian Sign Language",
      signLanguageCode: "ISL",
      outputLanguage: "English",
      outputLanguageCode: "en",
      isSynthetic: false,
      samples,
    };
  }

  public async importAll(
    data: DatasetExport,
    opts: ImportOptions = {}
  ): Promise<ImportResult> {
    const skipInvalid = opts.skipInvalid !== false; // default true
    const errors: ImportErrorDetail[] = [];

    // Stage 1: Basic Root Format Validation (§9)
    if (!data || typeof data !== "object") {
      return {
        success: false,
        importedCount: 0,
        skippedCount: 0,
        errors: [{ message: "Import failed: File content is not a valid JSON object." }],
      };
    }

    if (data.version !== 1) {
      return {
        success: false,
        importedCount: 0,
        skippedCount: 0,
        errors: [
          {
            message: `Unsupported dataset export version: ${data.version}. Importer supports version 1.`,
          },
        ],
      };
    }

    if (!Array.isArray(data.samples)) {
      return {
        success: false,
        importedCount: 0,
        skippedCount: 0,
        errors: [{ message: "Import failed: Missing or invalid 'samples' array in export data." }],
      };
    }

    const declaredSeqLen = data.sequenceLength || 30;
    const declaredFeatCount = data.featureCount || 63;

    if (opts.mergeMode === "replace") {
      await this.clearAll();
    }

    let importedCount = 0;
    let skippedCount = 0;

    // Stage 2: Per-Sample Detailed Validation
    for (let i = 0; i < data.samples.length; i++) {
      const sample = data.samples[i];
      const sampleLabel = sample.label ? String(sample.label).trim() : `Sample #${i}`;

      let sampleError: string | null = null;

      if (!sample || typeof sample !== "object") {
        sampleError = `Sample #${i} is not a valid object.`;
      } else if (!sample.label || typeof sample.label !== "string" || !sample.label.trim()) {
        sampleError = `Sample #${i} has a missing or empty label.`;
      } else if (!Array.isArray(sample.sequence)) {
        sampleError = `Sample #${i} ('${sampleLabel}') sequence is missing or not an array.`;
      } else if (sample.sequence.length !== declaredSeqLen) {
        sampleError = `Sample #${i} ('${sampleLabel}') sequence length (${sample.sequence.length}) does not match dataset sequenceLength (${declaredSeqLen}).`;
      } else {
        // Validate frame feature vectors and finite numbers
        for (let f = 0; f < sample.sequence.length; f++) {
          const frame = sample.sequence[f];
          if (!Array.isArray(frame) || frame.length !== declaredFeatCount) {
            sampleError = `Sample #${i} ('${sampleLabel}') frame ${f} length (${frame?.length}) does not match declared featureCount (${declaredFeatCount}).`;
            break;
          }

          for (let v = 0; v < frame.length; v++) {
            const val = frame[v];
            if (typeof val !== "number" || !Number.isFinite(val)) {
              sampleError = `Sample #${i} ('${sampleLabel}') frame ${f} index ${v} contains a non-finite value: ${val}.`;
              break;
            }
          }
          if (sampleError) break;
        }
      }

      if (sampleError) {
        errors.push({
          sampleIndex: i,
          label: sample.label,
          message: sampleError,
        });

        if (!skipInvalid) {
          return {
            success: false,
            importedCount: 0,
            skippedCount: data.samples.length,
            errors,
          };
        }

        skippedCount++;
        continue;
      }

      // Valid Sample: Generate fresh ID to prevent collision (§9)
      const cleanSample: DatasetSample = {
        id: generateUUID(),
        label: sample.label.trim().toUpperCase().replace(/\s+/g, "_"),
        sequence: sample.sequence,
        sequenceLength: declaredSeqLen,
        featureCount: declaredFeatCount,
        createdAt: sample.createdAt || new Date().toISOString(),
        handedness: sample.handedness,
        isSynthetic: false,
        signLanguage: "Indian Sign Language",
        signLanguageCode: "ISL",
        outputLanguage: "English",
        outputLanguageCode: "en",
        sourceType: sample.sourceType ?? "webcam",
        sourceVideoId: sample.sourceVideoId ?? sample.metadata?.sourceVideoHash,
        metadata: sample.metadata || {},
      };

      await this.addSample(cleanSample);
      importedCount++;
    }

    return {
      success: importedCount > 0,
      importedCount,
      skippedCount,
      errors,
    };
  }
}

export const datasetStorage = new IndexedDBDatasetStorage();
