import * as fs from "fs";
import { extractSequenceEnhancedFeatures, FeatureSchemaId } from "../../src/services/enhancedFeatureProcessor";

function main() {
  const args = process.argv.slice(2);
  if (args.length < 3) {
    console.error("Usage: tsx run_ts_enhanced_features.ts <input_json> <output_json> <schema>");
    process.exit(1);
  }

  const inPath = args[0];
  const outPath = args[1];
  const schema = args[2] as FeatureSchemaId;

  const rawData = JSON.parse(fs.readFileSync(inPath, "utf-8"));
  // rawData is an array of test sequences: [{ name: string, sequence: number[30][63] }]
  const results = rawData.map((item: { name: string; sequence: number[][] }) => {
    try {
      const enhanced = extractSequenceEnhancedFeatures(item.sequence, schema);
      return {
        name: item.name,
        ok: true,
        features: enhanced,
      };
    } catch (err: any) {
      return {
        name: item.name,
        ok: false,
        error: err.message,
      };
    }
  });

  fs.writeFileSync(outPath, JSON.stringify(results, null, 2), "utf-8");
}

main();
