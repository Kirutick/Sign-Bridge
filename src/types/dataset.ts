export interface DatasetSample {
  id: string; // UUID v4 or unique generated identifier
  label: string; // Normalized label, e.g. "THANK_YOU"
  sequence: number[][]; // sequence.length === sequenceLength, sequence[i].length === featureCount
  sequenceLength: number; // e.g. 30
  featureCount: number; // e.g. 63
  createdAt: string; // ISO 8601 string
  handedness?: string[]; // Per-sample detected handedness array

  // ISL-locked language metadata
  signLanguage?: string; // Always "Indian Sign Language"
  signLanguageCode?: string; // Always "ISL"
  outputLanguage?: string; // Always "English"
  outputLanguageCode?: string; // Always "en"

  // Quality and provenance fields
  isSynthetic?: boolean; // Must be false for real ISL training data
  datasetVersion?: string; // e.g. "v1"
  normalizationVersion?: string; // e.g. "v1"
  sourceType?: "webcam" | "image_upload" | "video_upload" | "screen_recording";
  sourceVideoId?: string; // Hash of source filename for leakage guard grouping

  metadata?: {
    performerId?: string; // Performer ID for variation tracking
    signerId?: string; // Signer/performer ID from upload studio
    notes?: string;
    fps?: number;
    sourceFile?: string; // Original filename
    sourceType?: string;
    sourceVideoHash?: string;
    isStaticSign?: boolean;
    normalizationVersion?: string;
    sourceResolution?: { width: number; height: number };
    appVersion?: string;
    [key: string]: unknown; // allow extra fields
  };
}

export interface DatasetExport {
  version: 1;
  featureCount: number;
  sequenceLength: number;
  exportedAt: string;
  signLanguage?: string;
  signLanguageCode?: string;
  outputLanguage?: string;
  outputLanguageCode?: string;
  isSynthetic?: boolean;
  samples: DatasetSample[];
}

export interface ImportOptions {
  skipInvalid?: boolean; // Default true: skip bad samples rather than aborting whole import
  mergeMode?: "additive" | "replace"; // Default 'additive'
}

export interface ImportErrorDetail {
  sampleIndex?: number;
  label?: string;
  message: string;
}

export interface ImportResult {
  success: boolean;
  importedCount: number;
  skippedCount: number;
  errors: ImportErrorDetail[];
}

export interface LabelInfo {
  label: string;
  count: number;
  performersCount: number;
}

export interface DatasetStorage {
  addSample(sample: DatasetSample): Promise<string>;
  saveSample(partial: Partial<DatasetSample> & { label: string; sequence: number[][]; source?: string; metadata?: Record<string, unknown> }): Promise<string>;
  getSample(id: string): Promise<DatasetSample | null>;
  listSamples(label?: string): Promise<DatasetSample[]>;
  deleteSample(id: string): Promise<void>;
  deleteLabel(label: string): Promise<void>;
  listLabels(): Promise<LabelInfo[]>;
  clearAll(): Promise<void>;
  exportAll(targetLabel?: string): Promise<DatasetExport>;
  importAll(data: DatasetExport, opts?: ImportOptions): Promise<ImportResult>;
}
