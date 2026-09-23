import pytest
import numpy as np

from ml.evaluation.metrics import (
    compute_confidence_analysis,
    compute_threshold_analysis,
    analyze_confusion_matrix
)
from ml.evaluation.error_analysis import perform_error_analysis

def test_confidence_analysis():
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 1])
    
    # 0 is correct, 1 is incorrect (true=1, pred=0), 2 is correct, 3 is correct, 4 is incorrect
    # Wait, y_true=[0,1,0,1,2]
    # y_pred=[0,0,0,1,1]
    # Correct mask: True, False, True, True, False
    
    y_probs = np.array([
        [0.9, 0.1, 0.0], # correct, prob=0.9
        [0.6, 0.3, 0.1], # incorrect, max=0.6
        [0.8, 0.1, 0.1], # correct, prob=0.8
        [0.1, 0.7, 0.2], # correct, prob=0.7
        [0.2, 0.5, 0.3], # incorrect, max=0.5
    ])
    
    res = compute_confidence_analysis(y_true, y_pred, y_probs)
    
    assert res['correct_count'] == 3
    assert res['incorrect_count'] == 2
    
    # Correct max probs: 0.9, 0.8, 0.7 -> mean = 0.8, median = 0.8
    np.testing.assert_almost_equal(res['correct_mean'], 0.8)
    np.testing.assert_almost_equal(res['correct_median'], 0.8)
    
    # Incorrect max probs: 0.6, 0.5 -> mean = 0.55, median = 0.55
    np.testing.assert_almost_equal(res['incorrect_mean'], 0.55)
    np.testing.assert_almost_equal(res['incorrect_median'], 0.55)

def test_threshold_analysis():
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 1])
    
    y_probs = np.array([
        [0.9, 0.1, 0.0], # correct, 0.9
        [0.6, 0.3, 0.1], # incorrect, 0.6
        [0.8, 0.1, 0.1], # correct, 0.8
        [0.1, 0.7, 0.2], # correct, 0.7
        [0.2, 0.5, 0.3], # incorrect, 0.5
    ])
    
    res = compute_threshold_analysis(y_true, y_pred, y_probs, thresholds=[0.60, 0.75])
    
    # At 0.60: 
    # Max probs: [0.9, 0.6, 0.8, 0.7, 0.5]
    # Accepted: indices 0, 1, 2, 3 (count=4)
    # Correct in accepted: 0, 2, 3 (count=3)
    # Coverage: 4/5 = 0.8
    # Accuracy: 3/4 = 0.75
    
    assert res[0.60]['accepted_count'] == 4
    np.testing.assert_almost_equal(res[0.60]['coverage'], 0.8)
    np.testing.assert_almost_equal(res[0.60]['accuracy'], 0.75)
    
    # At 0.75:
    # Accepted: indices 0, 2 (count=2)
    # Correct in accepted: 0, 2 (count=2)
    # Coverage: 2/5 = 0.4
    # Accuracy: 2/2 = 1.0
    
    assert res[0.75]['accepted_count'] == 2
    np.testing.assert_almost_equal(res[0.75]['coverage'], 0.4)
    np.testing.assert_almost_equal(res[0.75]['accuracy'], 1.0)

def test_analyze_confusion_matrix():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 1, 1, 1, 0, 2])
    
    # Class 0: 4 samples (2 correct, 1 as 1, 1 as 2)
    # Class 1: 3 samples (3 correct)
    # Class 2: 2 samples (1 as 0, 1 correct)
    
    id_to_label = {0: "A", 1: "B", 2: "C"}
    
    # Support fraction 0.2
    # Class 0: total 4. predicted as 1: 1 (0.25 > 0.2). predicted as 2: 1 (0.25 > 0.2)
    # Class 1: total 3. all correct.
    # Class 2: total 2. predicted as 0: 1 (0.5 > 0.2)
    
    cm, pairs = analyze_confusion_matrix(y_true, y_pred, id_to_label, support_fraction=0.2)
    
    assert len(pairs) == 3
    # Sorted by count, then rate
    # C as A: count=1, rate=0.5
    # A as B: count=1, rate=0.25
    # A as C: count=1, rate=0.25
    
    assert pairs[0]["true_label"] == "C"
    assert pairs[0]["pred_label"] == "A"
    assert pairs[0]["rate"] == 0.5
    
    assert pairs[1]["true_label"] == "A"
    assert pairs[1]["rate"] == 0.25

def test_error_analysis():
    # 4 classes
    y_train = np.array([0]*100 + [1]*50 + [2]*40 + [3]*60)
    # medians: [40, 50, 60, 100] -> median is 55.0
    
    f1_scores = [0.9, 0.6, 0.5, 0.4]
    support = [10, 10, 10, 10]
    id_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    # Freq pair: D confused with A
    frequent_pairs = [{"true_label": "D", "pred_label": "A", "count": 4, "rate": 0.4, "true_support": 10}]
    
    res = perform_error_analysis(y_train, f1_scores, support, id_to_label, frequent_pairs, f1_threshold=0.70)
    
    # A > 0.70 so not analyzed
    assert "A" not in res
    
    # B: count=50. median=55. No freq pair. 50 > 55*0.5. So no clear cause.
    assert res["B"] == "no clear cause identified from available data"
    
    # C: count=40. < 50 threshold. 
    assert "insufficient examples" in res["C"]
    
    # D: count=60. Freq pair.
    assert "similar hand configurations: frequently confused with 'A'" in res["D"]
