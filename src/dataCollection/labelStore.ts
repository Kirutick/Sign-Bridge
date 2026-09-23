import { datasetStorage } from "../services/datasetStorage";

export interface SignLabel {
  id: string; // Normalized name as stable key (e.g. "HELLO")
  name: string; // Display name
  createdAt: string; // ISO 8601
  sampleCount: number; // Derived dynamically from datasetStorage
  targetCount: number; // Soft goal target (default 20)
}

export class LabelStore {
  private labels: Map<string, SignLabel> = new Map();
  private globalTarget: number = 20;

  constructor(globalTarget: number = 20) {
    this.globalTarget = globalTarget;
  }

  public setGlobalTarget(target: number): void {
    this.globalTarget = target;
  }

  public getGlobalTarget(): number {
    return this.globalTarget;
  }

  /**
   * Syncs labels with IndexedDB datasetStorage (§2.2).
   */
  public async syncWithStorage(): Promise<SignLabel[]> {
    const allSamples = await datasetStorage.listSamples();
    const labelCounts = new Map<string, number>();

    for (const sample of allSamples) {
      const normName = sample.label.trim().toUpperCase().replace(/\s+/g, "_");
      labelCounts.set(normName, (labelCounts.get(normName) || 0) + 1);
    }

    // Update existing or add newly discovered labels
    for (const [name, count] of labelCounts.entries()) {
      if (!this.labels.has(name)) {
        this.labels.set(name, {
          id: name,
          name,
          createdAt: new Date().toISOString(),
          sampleCount: count,
          targetCount: this.globalTarget,
        });
      } else {
        const existing = this.labels.get(name)!;
        existing.sampleCount = count;
      }
    }

    // Reset counts for labels with 0 samples
    for (const [id, label] of this.labels.entries()) {
      if (!labelCounts.has(id)) {
        label.sampleCount = 0;
      }
    }

    return this.getLabels();
  }

  public getLabels(sortBy: "name" | "sampleCountAsc" = "name"): SignLabel[] {
    const list = Array.from(this.labels.values());
    if (sortBy === "sampleCountAsc") {
      return list.sort((a, b) => a.sampleCount - b.sampleCount);
    }
    return list.sort((a, b) => a.name.localeCompare(b.name));
  }

  public createLabel(rawName: string, targetCount?: number): SignLabel {
    const normName = rawName.trim().toUpperCase().replace(/\s+/g, "_");

    if (!normName) {
      throw new Error("Label name cannot be empty.");
    }

    if (this.labels.has(normName)) {
      throw new Error(`Label "${normName}" already exists.`);
    }

    const newLabel: SignLabel = {
      id: normName,
      name: normName,
      createdAt: new Date().toISOString(),
      sampleCount: 0,
      targetCount: targetCount ?? this.globalTarget,
    };

    this.labels.set(normName, newLabel);
    return newLabel;
  }

  public setLabelTarget(labelId: string, targetCount: number): void {
    const label = this.labels.get(labelId);
    if (label) {
      label.targetCount = targetCount;
    }
  }

  public async deleteLabel(labelId: string): Promise<number> {
    const label = this.labels.get(labelId);
    if (!label) return 0;

    const count = label.sampleCount;
    await datasetStorage.deleteLabel(labelId);
    this.labels.delete(labelId);
    return count;
  }
}
