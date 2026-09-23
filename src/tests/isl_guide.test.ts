import { describe, it, expect } from 'vitest';
import islGuideData from '../data/isl_sign_guide.json';
import labelMap from '../../public/models/sign-model-isl/label_map.json';

describe('ISL Hand Sign Guide Parity', () => {
  it('should have exactly the same number of signs as the model classes', () => {
    expect(islGuideData.signs.length).toBe(labelMap.num_classes);
  });

  it('should contain all labels from label_map', () => {
    const guideLabels = new Set(islGuideData.signs.map(s => s.label));
    const modelLabels = Object.keys(labelMap.label_to_index);

    for (const label of modelLabels) {
      expect(guideLabels.has(label)).toBe(true);
    }
  });

  it('should not contain any duplicate labels', () => {
    const labels = islGuideData.signs.map(s => s.label);
    const uniqueLabels = new Set(labels);
    expect(labels.length).toBe(uniqueLabels.size);
  });

  it('should have required fields for every sign', () => {
    for (const sign of islGuideData.signs) {
      expect(sign.label).toBeTruthy();
      expect(sign.displayName).toBeTruthy();
      expect(sign.category).toBeTruthy();
      expect(sign.signType).toBeTruthy();
      expect(sign.verificationStatus).toBeTruthy();
      expect(typeof sign.modelClassIndex).toBe('number');
      expect(Array.isArray(sign.instructions)).toBe(true);
    }
  });
});
