/**
 * Utility for calculating real-time Frames Per Second (FPS) and average latency.
 */
export class FpsCounter {
  private frameTimes: number[] = [];
  private lastTime: number = performance.now();
  private sampleSize: number;

  constructor(sampleSize: number = 30) {
    this.sampleSize = sampleSize;
  }

  /**
   * Called on each frame loop iteration. Returns current FPS.
   */
  public tick(): number {
    const now = performance.now();
    const delta = now - this.lastTime;
    this.lastTime = now;

    if (delta > 0) {
      this.frameTimes.push(1000 / delta);
      if (this.frameTimes.length > this.sampleSize) {
        this.frameTimes.shift();
      }
    }

    return this.getFps();
  }

  /**
   * Returns current average FPS over sample window.
   */
  public getFps(): number {
    if (this.frameTimes.length === 0) return 0;
    const sum = this.frameTimes.reduce((acc, val) => acc + val, 0);
    return Math.round(sum / this.frameTimes.length);
  }

  /**
   * Resets the counter.
   */
  public reset(): void {
    this.frameTimes = [];
    this.lastTime = performance.now();
  }
}
