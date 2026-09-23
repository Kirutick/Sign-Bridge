export type ReviewActionType = "MARK_VALID" | "MARK_EXCLUDED";

export interface ReviewAction {
  sampleId: string;
  action: ReviewActionType;
  timestamp: string;
  reviewer?: string;
}

/**
 * Review Store (§5.2, §5.4)
 * Manages an audit log of manual human review decisions on flagged samples.
 * Does NOT mutate raw storage.
 */
export class ReviewStore {
  private actions = new Map<string, ReviewAction>();

  /**
   * Clears the current review session.
   */
  public clear(): void {
    this.actions.clear();
  }

  /**
   * Logs a review action for a specific sample.
   */
  public logAction(sampleId: string, action: ReviewActionType, reviewer: string = "local_reviewer"): void {
    this.actions.set(sampleId, {
      sampleId,
      action,
      timestamp: new Date().toISOString(),
      reviewer,
    });
  }

  /**
   * Retrieves the review status for a specific sample, if any.
   */
  public getStatus(sampleId: string): ReviewAction | undefined {
    return this.actions.get(sampleId);
  }

  /**
   * Returns all sample IDs that were explicitly marked as EXCLUDED.
   */
  public getExcludedSampleIds(): Set<string> {
    const excluded = new Set<string>();
    for (const action of this.actions.values()) {
      if (action.action === "MARK_EXCLUDED") {
        excluded.add(action.sampleId);
      }
    }
    return excluded;
  }

  /**
   * Returns all sample IDs that were explicitly marked as VALID.
   */
  public getValidSampleIds(): Set<string> {
    const valid = new Set<string>();
    for (const action of this.actions.values()) {
      if (action.action === "MARK_VALID") {
        valid.add(action.sampleId);
      }
    }
    return valid;
  }

  /**
   * Returns the complete audit log of all manual review actions.
   */
  public getAuditLog(): ReviewAction[] {
    return Array.from(this.actions.values());
  }
}

// Singleton instance for the pipeline session
export const reviewStore = new ReviewStore();
