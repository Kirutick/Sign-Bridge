import { normalizeHandLandmarks, extractFeatureVector } from "../../src/services/landmarkProcessor";
import * as fs from "fs";

const inputPath = process.argv[2];
const outputPath = process.argv[3];

const fixtures = JSON.parse(fs.readFileSync(inputPath, "utf-8"));
const results = [];

for (const fix of fixtures) {
  const normRes = normalizeHandLandmarks(fix.landmarks);
  if (!normRes.ok) {
    results.push({ name: fix.name, ok: false, reason: normRes.reason, features: [] });
  } else {
    const vec = extractFeatureVector(normRes.normalized.landmarks);
    results.push({ name: fix.name, ok: true, reason: "", features: vec });
  }
}

fs.writeFileSync(outputPath, JSON.stringify(results, null, 2), "utf-8");
