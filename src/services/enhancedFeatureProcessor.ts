/**
 * Sign Bridge - Enhanced Hand Geometry & Motion Feature Processor (TypeScript Implementation)
 * 
 * Provides 100% mathematical parity with Python reference implementation (enhanced_features.py).
 * Calculates anatomical joint angles, segment lengths, pairwise fingertip distances, palm geometry,
 * hand orientation, landmark velocities, accelerations, global motion, and temporal summaries.
 */

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export type FeatureSchemaId =
  | "v1_single_hand_63"
  | "v2_single_hand_geometry"
  | "v3_single_hand_geom_vel"
  | "v4_single_hand_geom_vel_acc"
  | "v5_single_hand_all";

export const FEATURE_SCHEMA_DIMS: Record<FeatureSchemaId, number> = {
  v1_single_hand_63: 63,
  v2_single_hand_geometry: 134,
  v3_single_hand_geom_vel: 203,
  v4_single_hand_geom_vel_acc: 266,
  v5_single_hand_all: 271,
};

const WRIST = 0;
const THUMB_CMC = 1, THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_DIP = 7, INDEX_TIP = 8;
const MIDDLE_MCP = 9, MIDDLE_PIP = 10, MIDDLE_DIP = 11, MIDDLE_TIP = 12;
const RING_MCP = 13, RING_PIP = 14, RING_DIP = 15, RING_TIP = 16;
const PINKY_MCP = 17, PINKY_PIP = 18, PINKY_DIP = 19, PINKY_TIP = 20;

const EPSILON = 1e-7;

const ANGLE_TRIPLETS: [number, number, number][] = [
  // Thumb (3)
  [WRIST, THUMB_CMC, THUMB_MCP],
  [THUMB_CMC, THUMB_MCP, THUMB_IP],
  [THUMB_MCP, THUMB_IP, THUMB_TIP],
  // Index (3)
  [WRIST, INDEX_MCP, INDEX_PIP],
  [INDEX_MCP, INDEX_PIP, INDEX_DIP],
  [INDEX_PIP, INDEX_DIP, INDEX_TIP],
  // Middle (3)
  [WRIST, MIDDLE_MCP, MIDDLE_PIP],
  [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP],
  [MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
  // Ring (3)
  [WRIST, RING_MCP, RING_PIP],
  [RING_MCP, RING_PIP, RING_DIP],
  [RING_PIP, RING_DIP, RING_TIP],
  // Pinky (3)
  [WRIST, PINKY_MCP, PINKY_PIP],
  [PINKY_MCP, PINKY_PIP, PINKY_DIP],
  [PINKY_PIP, PINKY_DIP, PINKY_TIP],
];

const FINGER_SEGMENTS: [number, number][] = [
  // Thumb (3)
  [THUMB_CMC, THUMB_MCP], [THUMB_MCP, THUMB_IP], [THUMB_IP, THUMB_TIP],
  // Index (3)
  [INDEX_MCP, INDEX_PIP], [INDEX_PIP, INDEX_DIP], [INDEX_DIP, INDEX_TIP],
  // Middle (3)
  [MIDDLE_MCP, MIDDLE_PIP], [MIDDLE_PIP, MIDDLE_DIP], [MIDDLE_DIP, MIDDLE_TIP],
  // Ring (3)
  [RING_MCP, RING_PIP], [RING_PIP, RING_DIP], [RING_DIP, RING_TIP],
  // Pinky (3)
  [PINKY_MCP, PINKY_PIP], [PINKY_PIP, PINKY_DIP], [PINKY_DIP, PINKY_TIP],
];

const FINGERTIP_PAIRS: [number, number][] = [
  [THUMB_TIP, INDEX_TIP],
  [THUMB_TIP, MIDDLE_TIP],
  [THUMB_TIP, RING_TIP],
  [THUMB_TIP, PINKY_TIP],
  [INDEX_TIP, MIDDLE_TIP],
  [INDEX_TIP, RING_TIP],
  [INDEX_TIP, PINKY_TIP],
  [MIDDLE_TIP, RING_TIP],
  [MIDDLE_TIP, PINKY_TIP],
  [RING_TIP, PINKY_TIP],
];

const FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
const MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];

function euclideanDist(p1: LandmarkPoint, p2: LandmarkPoint): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = p1.z - p2.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function calculateJointAngle(pA: LandmarkPoint, pB: LandmarkPoint, pC: LandmarkPoint): number {
  const ux = pA.x - pB.x;
  const uy = pA.y - pB.y;
  const uz = pA.z - pB.z;

  const vx = pC.x - pB.x;
  const vy = pC.y - pB.y;
  const vz = pC.z - pB.z;

  const normU = Math.sqrt(ux * ux + uy * uy + uz * uz);
  const normV = Math.sqrt(vx * vx + vy * vy + vz * vz);

  if (normU < EPSILON || normV < EPSILON) {
    return 0.0;
  }

  const dotVal = ux * vx + uy * vy + uz * vz;
  let cosTheta = dotVal / (normU * normV);

  if (cosTheta > 1.0) cosTheta = 1.0;
  else if (cosTheta < -1.0) cosTheta = -1.0;

  return Math.acos(cosTheta);
}

/**
 * Extracts 134 static and geometric features (Groups A through G) for a single normalized hand frame.
 */
export function extractGeometricFeaturesFrame(landmarks: LandmarkPoint[]): number[] {
  if (landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, got ${landmarks.length}`);
  }

  const features: number[] = [];

  // Group A: Raw Normalized Coordinates (63 features)
  for (let i = 0; i < 21; i++) {
    features.push(landmarks[i].x, landmarks[i].y, landmarks[i].z);
  }

  // Group B: Finger Joint Angles (15 features)
  for (let i = 0; i < ANGLE_TRIPLETS.length; i++) {
    const [a, b, c] = ANGLE_TRIPLETS[i];
    features.push(calculateJointAngle(landmarks[a], landmarks[b], landmarks[c]));
  }

  // Group C: Finger Segment Lengths & Totals (20 features)
  const segLengths: number[] = [];
  for (let i = 0; i < FINGER_SEGMENTS.length; i++) {
    const [p1, p2] = FINGER_SEGMENTS[i];
    const len = euclideanDist(landmarks[p1], landmarks[p2]);
    segLengths.push(len);
    features.push(len);
  }

  // Total finger lengths (5 features)
  for (let f = 0; f < 5; f++) {
    const total = segLengths[f * 3] + segLengths[f * 3 + 1] + segLengths[f * 3 + 2];
    features.push(total);
  }

  // Group D: Fingertip Distances (15 features)
  for (let i = 0; i < FINGERTIP_PAIRS.length; i++) {
    const [t1, t2] = FINGERTIP_PAIRS[i];
    features.push(euclideanDist(landmarks[t1], landmarks[t2]));
  }

  const pWrist = landmarks[WRIST];
  for (let i = 0; i < FINGERTIPS.length; i++) {
    features.push(euclideanDist(pWrist, landmarks[FINGERTIPS[i]]));
  }

  // Group E: Finger Spread & MCP Angles (7 features)
  features.push(calculateJointAngle(landmarks[THUMB_MCP], pWrist, landmarks[INDEX_MCP]));
  features.push(calculateJointAngle(landmarks[INDEX_MCP], pWrist, landmarks[MIDDLE_MCP]));
  features.push(calculateJointAngle(landmarks[MIDDLE_MCP], pWrist, landmarks[RING_MCP]));
  features.push(calculateJointAngle(landmarks[RING_MCP], pWrist, landmarks[PINKY_MCP]));

  features.push(calculateJointAngle(landmarks[THUMB_TIP], pWrist, landmarks[INDEX_TIP]));
  features.push(calculateJointAngle(landmarks[INDEX_TIP], pWrist, landmarks[MIDDLE_TIP]));
  features.push(calculateJointAngle(landmarks[MIDDLE_TIP], pWrist, landmarks[RING_TIP]));

  // Group F: Palm Geometry (8 features)
  for (let i = 0; i < MCPS.length; i++) {
    features.push(euclideanDist(pWrist, landmarks[MCPS[i]]));
  }

  features.push(euclideanDist(landmarks[INDEX_MCP], landmarks[PINKY_MCP])); // Palm span
  features.push(euclideanDist(landmarks[INDEX_MCP], landmarks[MIDDLE_MCP]));
  features.push(euclideanDist(landmarks[MIDDLE_MCP], landmarks[RING_MCP]));
  features.push(euclideanDist(landmarks[RING_MCP], landmarks[PINKY_MCP]));

  // Group G: Hand Orientation & Palm Normal (6 features)
  const v1x = landmarks[INDEX_MCP].x - pWrist.x;
  const v1y = landmarks[INDEX_MCP].y - pWrist.y;
  const v1z = landmarks[INDEX_MCP].z - pWrist.z;

  const v2x = landmarks[PINKY_MCP].x - pWrist.x;
  const v2y = landmarks[PINKY_MCP].y - pWrist.y;
  const v2z = landmarks[PINKY_MCP].z - pWrist.z;

  const normX = v1y * v2z - v1z * v2y;
  const normY = v1z * v2x - v1x * v2z;
  const normZ = v1x * v2y - v1y * v2x;

  const nLen = Math.sqrt(normX * normX + normY * normY + normZ * normZ);
  if (nLen < EPSILON) {
    features.push(0.0, 0.0, 1.0);
  } else {
    features.push(normX / nLen, normY / nLen, normZ / nLen);
  }

  const vDirX = landmarks[MIDDLE_MCP].x - pWrist.x;
  const vDirY = landmarks[MIDDLE_MCP].y - pWrist.y;
  const vDirZ = landmarks[MIDDLE_MCP].z - pWrist.z;

  const dLen = Math.sqrt(vDirX * vDirX + vDirY * vDirY + vDirZ * vDirZ);
  if (dLen < EPSILON) {
    features.push(0.0, 1.0, 0.0);
  } else {
    features.push(vDirX / dLen, vDirY / dLen, vDirZ / dLen);
  }

  return features;
}

/**
 * Extracts full enhanced feature sequence [30, N] from raw sequence [30, 63] or [30, 21] landmarks.
 */
export function extractSequenceEnhancedFeatures(
  sequence: number[][] | LandmarkPoint[][],
  schema: FeatureSchemaId = "v5_single_hand_all"
): number[][] {
  const T = 30;
  if (!sequence || sequence.length !== T) {
    throw new Error(`Expected sequence length of 30, got ${sequence?.length || 0}`);
  }

  // Convert to LandmarkPoint[30][21]
  const parsedSeq: LandmarkPoint[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    const frame = sequence[t];
    if (Array.isArray(frame) && typeof frame[0] === "number") {
      const numFrame = frame as number[];
      if (numFrame.length !== 63) {
        throw new Error(`Frame ${t} length ${numFrame.length} does not match 63 floats.`);
      }
      const lms: LandmarkPoint[] = new Array(21);
      for (let i = 0; i < 21; i++) {
        lms[i] = {
          x: numFrame[i * 3],
          y: numFrame[i * 3 + 1],
          z: numFrame[i * 3 + 2],
        };
      }
      parsedSeq[t] = lms;
    } else {
      parsedSeq[t] = frame as LandmarkPoint[];
    }
  }

  // 1. Geometric features (134 per frame)
  const geomFeatures: number[][] = new Array(T);
  for (let t = 0; t < T; t++) {
    geomFeatures[t] = extractGeometricFeaturesFrame(parsedSeq[t]);
  }

  if (schema === "v1_single_hand_63") {
    return geomFeatures.map((row) => row.slice(0, 63));
  }

  if (schema === "v2_single_hand_geometry") {
    return geomFeatures;
  }

  // 2. Velocities (63 per frame)
  const velocities: LandmarkPoint[][] = new Array(T);
  velocities[0] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  for (let t = 1; t < T; t++) {
    const frameV: LandmarkPoint[] = new Array(21);
    for (let i = 0; i < 21; i++) {
      frameV[i] = {
        x: parsedSeq[t][i].x - parsedSeq[t - 1][i].x,
        y: parsedSeq[t][i].y - parsedSeq[t - 1][i].y,
        z: parsedSeq[t][i].z - parsedSeq[t - 1][i].z,
      };
    }
    velocities[t] = frameV;
  }

  // 3. Accelerations (63 per frame)
  const accelerations: LandmarkPoint[][] = new Array(T);
  accelerations[0] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  accelerations[1] = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  for (let t = 2; t < T; t++) {
    const frameA: LandmarkPoint[] = new Array(21);
    for (let i = 0; i < 21; i++) {
      frameA[i] = {
        x: velocities[t][i].x - velocities[t - 1][i].x,
        y: velocities[t][i].y - velocities[t - 1][i].y,
        z: velocities[t][i].z - velocities[t - 1][i].z,
      };
    }
    accelerations[t] = frameA;
  }

  // 4. Global motion (6) & Temporal summary (5)
  const globalMotion: number[][] = new Array(T);
  const temporalSummary: number[][] = new Array(T);
  let cumulativeDisplacement = 0.0;

  for (let t = 0; t < T; t++) {
    const vWrist = velocities[t][WRIST];
    const wristSpeed = Math.sqrt(vWrist.x * vWrist.x + vWrist.y * vWrist.y + vWrist.z * vWrist.z);

    let palmSpeed = 0.0;
    if (t > 0) {
      const palmCurrX = (parsedSeq[t][WRIST].x + parsedSeq[t][INDEX_MCP].x + parsedSeq[t][MIDDLE_MCP].x + parsedSeq[t][PINKY_MCP].x) * 0.25;
      const palmCurrY = (parsedSeq[t][WRIST].y + parsedSeq[t][INDEX_MCP].y + parsedSeq[t][MIDDLE_MCP].y + parsedSeq[t][PINKY_MCP].y) * 0.25;
      const palmCurrZ = (parsedSeq[t][WRIST].z + parsedSeq[t][INDEX_MCP].z + parsedSeq[t][MIDDLE_MCP].z + parsedSeq[t][PINKY_MCP].z) * 0.25;

      const palmPrevX = (parsedSeq[t - 1][WRIST].x + parsedSeq[t - 1][INDEX_MCP].x + parsedSeq[t - 1][MIDDLE_MCP].x + parsedSeq[t - 1][PINKY_MCP].x) * 0.25;
      const palmPrevY = (parsedSeq[t - 1][WRIST].y + parsedSeq[t - 1][INDEX_MCP].y + parsedSeq[t - 1][MIDDLE_MCP].y + parsedSeq[t - 1][PINKY_MCP].y) * 0.25;
      const palmPrevZ = (parsedSeq[t - 1][WRIST].z + parsedSeq[t - 1][INDEX_MCP].z + parsedSeq[t - 1][MIDDLE_MCP].z + parsedSeq[t - 1][PINKY_MCP].z) * 0.25;

      const dPx = palmCurrX - palmPrevX;
      const dPy = palmCurrY - palmPrevY;
      const dPz = palmCurrZ - palmPrevZ;
      palmSpeed = Math.sqrt(dPx * dPx + dPy * dPy + dPz * dPz);
    }

    let speedSum = 0.0;
    let maxSpeed = 0.0;
    for (let i = 0; i < 21; i++) {
      const sp = Math.sqrt(velocities[t][i].x * velocities[t][i].x + velocities[t][i].y * velocities[t][i].y + velocities[t][i].z * velocities[t][i].z);
      speedSum += sp;
      if (sp > maxSpeed) maxSpeed = sp;
    }
    const meanSpeed = speedSum / 21;

    globalMotion[t] = [vWrist.x, vWrist.y, vWrist.z, wristSpeed, palmSpeed, meanSpeed];

    cumulativeDisplacement += palmSpeed;

    let maxAccel = 0.0;
    let meanAccel = 0.0;
    if (t > 1) {
      let accelSum = 0.0;
      for (let i = 0; i < 21; i++) {
        const ac = Math.sqrt(accelerations[t][i].x * accelerations[t][i].x + accelerations[t][i].y * accelerations[t][i].y + accelerations[t][i].z * accelerations[t][i].z);
        accelSum += ac;
        if (ac > maxAccel) maxAccel = ac;
      }
      meanAccel = accelSum / 21;
    }

    temporalSummary[t] = [cumulativeDisplacement, maxSpeed, meanSpeed, maxAccel, meanAccel];
  }

  // Assemble full matrix
  const output: number[][] = new Array(T);
  const targetDim = FEATURE_SCHEMA_DIMS[schema];

  for (let t = 0; t < T; t++) {
    const row: number[] = [...geomFeatures[t]]; // 134

    if (schema === "v3_single_hand_geom_vel" || schema === "v4_single_hand_geom_vel_acc" || schema === "v5_single_hand_all") {
      // Group H: velocities (63)
      for (let i = 0; i < 21; i++) {
        row.push(velocities[t][i].x, velocities[t][i].y, velocities[t][i].z);
      }
      // Group J: global motion (6)
      row.push(...globalMotion[t]);
    }

    if (schema === "v4_single_hand_geom_vel_acc" || schema === "v5_single_hand_all") {
      // Group I: accelerations (63)
      for (let i = 0; i < 21; i++) {
        row.push(accelerations[t][i].x, accelerations[t][i].y, accelerations[t][i].z);
      }
    }

    if (schema === "v5_single_hand_all") {
      // Group K: temporal summary (5)
      row.push(...temporalSummary[t]);
    }

    if (row.length !== targetDim) {
      throw new Error(`Schema ${schema} expected ${targetDim} features, assembled ${row.length}`);
    }

    output[t] = row;
  }

  return output;
}
