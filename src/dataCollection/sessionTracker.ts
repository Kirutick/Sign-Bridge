export interface CollectionSession {
  id: string;
  startedAt: string;
  labelsRecorded: string[];
  samplesRecordedCount: number;
  device: {
    userAgent: string;
  };
}

export class SessionTracker {
  private session: CollectionSession;

  constructor() {
    this.session = {
      id: `session_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
      startedAt: new Date().toISOString(),
      labelsRecorded: [],
      samplesRecordedCount: 0,
      device: {
        userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "node_environment",
      },
    };
  }

  public getSession(): CollectionSession {
    return { ...this.session, labelsRecorded: [...this.session.labelsRecorded] };
  }

  public recordSampleSaved(labelId: string): void {
    this.session.samplesRecordedCount += 1;
    if (!this.session.labelsRecorded.includes(labelId)) {
      this.session.labelsRecorded.push(labelId);
    }
  }

  public recordSampleDeleted(_labelId: string): void {
    if (this.session.samplesRecordedCount > 0) {
      this.session.samplesRecordedCount -= 1;
    }
  }
}
