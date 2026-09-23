# EVALUATION REPORT: Sign Bridge LSTM (exp_mock)
Generated on: 2026-08-26T00:12:49.465536
Dataset version: 4f441224858b338e11e9c06e405e7900

## 1. Overall Metrics
- **Test Samples**: 15
- **Loss**: 7.1923
- **Accuracy**: 0.3333
- **Macro F1** (Primary): 0.2265 *(Note: Macro is primary to prevent high-support classes from hiding poor performance on rare ones)*
- **Weighted F1**: 0.2120

## 2. Confusion Matrix & Confused Pairs
*(Full confusion matrix array saved in evaluation artifacts)*

### Frequently Confused Pairs (>10% of true class support)
1. **THANK_YOU** predicted as **NO** (3/4 cases, 75.0%)
2. **HELLO** predicted as **NO** (2/4 cases, 50.0%)
3. **HELLO** predicted as **YES** (2/4 cases, 50.0%)
4. **YES** predicted as **NO** (1/3 cases, 33.3%)
5. **NO** predicted as **YES** (1/4 cases, 25.0%)
6. **THANK_YOU** predicted as **YES** (1/4 cases, 25.0%)

## 3. Per-Class Report

### Sorted by F1 (Ascending - Worst First)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| HELLO | 0.0000 | 0.0000 | 0.0000 | 4 |
| THANK_YOU | 0.0000 | 0.0000 | 0.0000 | 4 |
| YES | 0.3333 | 0.6667 | 0.4444 | 3 |
| NO | 0.3333 | 0.7500 | 0.4615 | 4 |

### Sorted Alphabetically
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| HELLO | 0.0000 | 0.0000 | 0.0000 | 4 |
| NO | 0.3333 | 0.7500 | 0.4615 | 4 |
| THANK_YOU | 0.0000 | 0.0000 | 0.0000 | 4 |
| YES | 0.3333 | 0.6667 | 0.4444 | 3 |

## 4. Error Analysis (Evidence-Cited)
*(Investigating classes with F1 < 0.70)*

- **HELLO**: insufficient examples: 17 training samples (threshold: 50); similar hand configurations: frequently confused with 'NO' (2/4 cases, 50.0%)
- **NO**: insufficient examples: 18 training samples (threshold: 50); similar hand configurations: frequently confused with 'YES' (1/4 cases, 25.0%)
- **THANK_YOU**: insufficient examples: 17 training samples (threshold: 50); similar hand configurations: frequently confused with 'NO' (3/4 cases, 75.0%)
- **YES**: insufficient examples: 18 training samples (threshold: 50); similar hand configurations: frequently confused with 'NO' (1/3 cases, 33.3%)

## 5. Cross-Performer Testing
- **Split Strategy**: `stratified-random` (Fallback).
- **WARNING**: A genuine unseen-performer evaluation is **not possible** from this test set, because performers were not held out during splitting. Test accuracy may be optimistic due to near-duplicate sequences across splits.

## 6. Real-World Testing
NO REAL-WORLD SAMPLES FOUND — SECTION SKIPPED

## 7. Confidence Analysis
- Mean confidence for CORRECT predictions: 0.9337
- Mean confidence for INCORRECT predictions: 0.9953
- **Finding**: The model is highly overconfident when wrong.

## 8. Threshold Analysis
| Threshold | Coverage | Accepted Accuracy |
|---|---|---|
| 0.60 | 100.0% | 33.3% |
| 0.70 | 93.3% | 28.6% |
| 0.80 | 93.3% | 28.6% |
| 0.90 | 93.3% | 28.6% |

**Recommended Operating Point**: Threshold **0.90**. This retains 93.3% coverage at 28.6% accuracy, balancing responsiveness with reliability.

## 9. Limitations
- This report is a snapshot of the held-out test set and is not subject to further model-selection tuning.
- **No real-world data yet**: Generalization to diverse environments and out-of-distribution recording conditions is entirely unknown.
- **Unseen-performer testing unavailable**: The training split did not hold out performers due to missing metadata.
- **Low-support classes**: The following classes have < 10 test samples, making their metrics statistically unstable: HELLO, NO, THANK_YOU, YES.
