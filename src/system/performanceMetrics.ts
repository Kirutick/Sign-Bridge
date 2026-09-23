export interface PerformanceSnapshot {
  cameraFps: number;
  mediapipeLatencyMs: number;
  lstmLatencyMs: number;
  totalLatencyMs: number;
}

export class PerformanceMetricsTracker {
  private bufferSize: number;
  private cameraFpsBuffer: number[] = [];
  private mediapipeBuffer: number[] = [];
  private lstmBuffer: number[] = [];
  private totalLatencyBuffer: number[] = [];

  constructor(bufferSize: number = 30) {
    this.bufferSize = bufferSize;
  }

  private pushToBuffer(buffer: number[], val: number) {
    buffer.push(val);
    if (buffer.length > this.bufferSize) {
      buffer.shift();
    }
  }

  private calculateAverage(buffer: number[]): number {
    if (buffer.length === 0) return 0;
    const sum = buffer.reduce((acc, curr) => acc + curr, 0);
    return Math.round((sum / buffer.length) * 10) / 10;
  }

  public recordCameraFps(fps: number): void {
    this.pushToBuffer(this.cameraFpsBuffer, fps);
  }

  public recordMediaPipeLatency(ms: number): void {
    this.pushToBuffer(this.mediapipeBuffer, ms);
  }

  public recordLstmLatency(ms: number): void {
    this.pushToBuffer(this.lstmBuffer, ms);
  }

  public recordTotalLatency(ms: number): void {
    this.pushToBuffer(this.totalLatencyBuffer, ms);
  }

  public getSnapshot(): PerformanceSnapshot {
    return {
      cameraFps: this.calculateAverage(this.cameraFpsBuffer),
      mediapipeLatencyMs: this.calculateAverage(this.mediapipeBuffer),
      lstmLatencyMs: this.calculateAverage(this.lstmBuffer),
      totalLatencyMs: this.calculateAverage(this.totalLatencyBuffer),
    };
  }

  public reset(): void {
    this.cameraFpsBuffer = [];
    this.mediapipeBuffer = [];
    this.lstmBuffer = [];
    this.totalLatencyBuffer = [];
  }
}
