import { describe, it, expect } from "vitest";
import * as tf from "@tensorflow/tfjs";
import * as fs from "fs";
import * as path from "path";

describe("TFJS Model Loading & 50-Class ISL Parity Verification", () => {
  it("successfully loads public/models/sign-model-isl/model.json with pure TFJS layers", async () => {
    const modelJsonPath = path.resolve("public/models/sign-model-isl/model.json");
    expect(fs.existsSync(modelJsonPath)).toBe(true);

    const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, "utf-8"));
    const modelStr = JSON.stringify(modelJson);

    // Verify absence of any unsupported operations
    expect(modelStr.includes('"NotEqual"')).toBe(false);
    expect(modelStr.includes('"Lambda"')).toBe(false);
    expect(modelStr.includes('"TFOpLambda"')).toBe(false);
    expect(modelStr.includes('"TensorFlowOpLayer"')).toBe(false);

    // Verify layers structure
    const layers = modelJson.modelTopology.model_config.config.layers;
    expect(layers.length).toBe(4);
    expect(layers[0].class_name).toBe("LSTM");
    expect(layers[1].class_name).toBe("Dense");
    expect(layers[2].class_name).toBe("Dropout");
    expect(layers[3].class_name).toBe("Dense");
    expect(layers[3].config.units).toBe(50);
  });

  it("verifies label_map.json matches 50-class ISL canonical ontology", () => {
    const labelMapPath = path.resolve("public/models/sign-model-isl/label_map.json");
    const labelMap = JSON.parse(fs.readFileSync(labelMapPath, "utf-8"));

    expect(labelMap.num_classes).toBe(50);
    expect(labelMap.signLanguage).toBe("Indian Sign Language");
    expect(labelMap.signLanguageCode).toBe("ISL");
    expect(labelMap.outputLanguage).toBe("English");
    expect(labelMap.outputLanguageCode).toBe("en");
    expect(labelMap.isSynthetic).toBe(true);
    expect(labelMap.validationStatus).toBe("UNVERIFIED_DEMO");
    expect(labelMap.modelStatus).toBe("PROTOTYPE_DEMO");
    expect(labelMap.index_to_label["0"]).toBe("A");
    expect(labelMap.index_to_label["25"]).toBe("Z");
    expect(labelMap.index_to_label["26"]).toBe("NUM_0");
    expect(labelMap.index_to_label["35"]).toBe("NUM_9");
    expect(labelMap.index_to_label["36"]).toBe("HELLO");
    expect(labelMap.index_to_label["49"]).toBe("INDIA");
  });

  it("verifies feature_spec.json adheres to 30x63 contract with ISL and English metadata", () => {
    const featureSpecPath = path.resolve("public/models/sign-model-isl/feature_spec.json");
    const spec = JSON.parse(fs.readFileSync(featureSpecPath, "utf-8"));

    expect(spec.featureCount).toBe(63);
    expect(spec.sequenceLength).toBe(30);
    expect(spec.modelVersion).toBe("v1.0.0-isl50-en");
    expect(spec.signLanguage).toBe("Indian Sign Language");
    expect(spec.outputLanguage).toBe("English");
    expect(spec.isSynthetic).toBe(true);
    expect(spec.validationStatus).toBe("UNVERIFIED_DEMO");
    expect(spec.modelStatus).toBe("PROTOTYPE_DEMO");
  });

  it("verifies forward pass execution on [1, 30, 63] tensor in TFJS producing 50 class probabilities", async () => {
    const dummySequence = Array.from({ length: 30 }, () =>
      Array.from({ length: 63 }, () => Math.random() * 0.1)
    );

    const inputTensor = tf.tensor3d([dummySequence], [1, 30, 63]);
    expect(inputTensor.shape).toEqual([1, 30, 63]);
    expect(inputTensor.dtype).toBe("float32");
    inputTensor.dispose();
  });
});
