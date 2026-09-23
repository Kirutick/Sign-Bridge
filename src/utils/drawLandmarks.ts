import { HandLandmarks } from "../types/landmarks";

/**
 * Official MediaPipe 21-point hand connection index topology.
 * Maps wrist (0) to thumb, index, middle, ring, pinky finger joint chains.
 */
export const HAND_CONNECTIONS: Array<[number, number]> = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Pinky finger
  [13, 17], [17, 18], [18, 19], [19, 20],
  // Palm connection
  [0, 17],
];

export interface DrawOptions {
  leftHandColor?: string;
  rightHandColor?: string;
  jointRadius?: number;
  lineWidth?: number;
  showLabels?: boolean;
}

const DEFAULT_OPTIONS: Required<DrawOptions> = {
  leftHandColor: "#00f2fe", // Vibrant Cyan
  rightHandColor: "#ff0844", // Neon Coral / Pink
  jointRadius: 5,
  lineWidth: 3,
  showLabels: true,
};

/**
 * Renders hand landmarks, connection lines, and handedness labels on 2D canvas context.
 */
export function drawHandLandmarks(
  ctx: CanvasRenderingContext2D,
  hands: HandLandmarks[],
  width: number,
  height: number,
  options: DrawOptions = {}
): void {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Clear previous frame canvas
  ctx.clearRect(0, 0, width, height);

  if (!hands || hands.length === 0) return;

  for (const hand of hands) {
    const { landmarks, handedness, handednessScore } = hand;
    if (landmarks.length !== 21) continue;

    // Pick distinct visual color scheme based on handedness
    const primaryColor =
      handedness === "Left" ? opts.leftHandColor : opts.rightHandColor;
    const secondaryColor =
      handedness === "Left" ? "rgba(0, 242, 254, 0.4)" : "rgba(255, 8, 68, 0.4)";

    // 1. Draw connection skeleton lines
    ctx.beginPath();
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = opts.lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
      const p1 = landmarks[startIdx];
      const p2 = landmarks[endIdx];

      const x1 = p1.x * width;
      const y1 = p1.y * height;
      const x2 = p2.x * width;
      const y2 = p2.y * height;

      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
    }
    ctx.stroke();

    // 2. Draw 21 landmark joint points with glow effect
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      const px = lm.x * width;
      const py = lm.y * height;

      // Outer glow circle
      ctx.beginPath();
      ctx.arc(px, py, opts.jointRadius + 2, 0, 2 * Math.PI);
      ctx.fillStyle = secondaryColor;
      ctx.fill();

      // Core joint dot
      ctx.beginPath();
      ctx.arc(px, py, opts.jointRadius, 0, 2 * Math.PI);
      // Fingertip points (4, 8, 12, 16, 20) get white core highlights
      const isFingertip = i % 4 === 0 && i > 0;
      ctx.fillStyle = isFingertip ? "#ffffff" : primaryColor;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }

    // 3. Draw Handedness Label near wrist landmark (index 0)
    if (opts.showLabels && landmarks[0]) {
      const wrist = landmarks[0];
      const lx = wrist.x * width;
      const ly = wrist.y * height + 25; // 25px below wrist

      const confidencePct = Math.round(handednessScore * 100);
      const labelText = `${handedness} (${confidencePct}%)`;

      ctx.save();
      // Unflip local matrix so text renders left-to-right when canvas is CSS mirrored (transform: scaleX(-1))
      ctx.translate(lx, ly);
      ctx.scale(-1, 1);

      ctx.font = "600 13px Inter, system-ui, sans-serif";
      const textWidth = ctx.measureText(labelText).width;

      // Label background badge
      ctx.fillStyle = "rgba(10, 13, 20, 0.85)";
      ctx.beginPath();
      ctx.roundRect(-textWidth / 2 - 8, -14, textWidth + 16, 22, 6);
      ctx.fill();
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label text
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "center";
      ctx.fillText(labelText, 0, 0);
      ctx.restore();
    }
  }
}
