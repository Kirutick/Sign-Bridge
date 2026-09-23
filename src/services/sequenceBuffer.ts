/**
 * Fixed-size sliding window sequence buffer for real-time temporal feature accumulation.
 */
export class SequenceBuffer {
  private buffer: number[][] = [];
  private sequenceLength: number;
  private featureCount: number;
  private stalledCount: number = 0;

  constructor(sequenceLength: number = 30, featureCount: number = 63) {
    this.sequenceLength = sequenceLength;
    this.featureCount = featureCount;
  }

  /**
   * Pushes a valid feature vector into the ring buffer.
   * Enforces featureCount length matching.
   */
  public push(frame: number[]): void {
    if (!frame || frame.length !== this.featureCount) {
      throw new Error(
        `Invalid frame feature length (${frame?.length || 0}). Expected exactly ${this.featureCount}.`
      );
    }

    this.buffer.push([...frame]);
    if (this.buffer.length > this.sequenceLength) {
      this.buffer.shift(); // Drop oldest frame
    }
  }

  /**
   * Called when hand tracking is lost mid-stream (§4).
   * Clears the current buffer to preserve temporal sequence contiguity and tracks stalled count.
   */
  public handleHandLoss(): void {
    if (this.buffer.length > 0) {
      this.stalledCount++;
    }
    this.clear();
  }

  /**
   * Returns true once the buffer contains exactly sequenceLength frames.
   */
  public isFull(): boolean {
    return this.buffer.length === this.sequenceLength;
  }

  /**
   * Returns a deep COPY of the current sequence matrix [30, 63].
   */
  public getSequence(): number[][] {
    return this.buffer.map((row) => [...row]);
  }

  /**
   * Clears the internal frame buffer.
   */
  public clear(): void {
    this.buffer = [];
  }

  /**
   * Returns current buffer frame count.
   */
  public length(): number {
    return this.buffer.length;
  }

  /**
   * Returns total count of buffer clears caused by hand loss.
   */
  public getStalledCount(): number {
    return this.stalledCount;
  }

  /**
   * Resets stalled counter.
   */
  public resetStalledCount(): void {
    this.stalledCount = 0;
  }
}
