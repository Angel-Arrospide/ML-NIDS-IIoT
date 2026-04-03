# ML-Based NIDS for Industrial IoT — Documentation

## Introduction

This project develops a Machine Learning-based Network Intrusion Detection System (NIDS) for Industrial IoT (IIoT) environments using the Edge IIoTset dataset. The dataset contains network packet captures from a controlled IIoT lab, labeled across 15 classes (Normal traffic and 14 attack types). Experiments are organized in two tracks: binary and multiclass classification.

For dataset exploration and per-column cleaning decisions, see [preliminary.md](preliminary.md).

---

## Binary Classification

Binary experiments use `Attack_label` (0 = Normal, 1 = Attack) as the target.

### 1. All Numeric Features (Baseline)

Uses all 50 numeric features produced by the cleaning phase, including environment-specific columns. Establishes the upper-bound performance ceiling with no feature selection applied.

→ [exp1_Binary_ML_Numeric_All.md](exp1_Binary_ML_Numeric_All.md)

### 2. Environment-Agnostic Feature Selection

Removes 10 features that encode environment-specific information (IP category encodings, frame timing) to improve generalizability across IIoT deployments. Reduces the feature set from 50 to 40. Performance delta vs. Experiment 1 quantifies how much the baseline relied on deployment shortcuts.

→ [exp2_Binary_ML_Numeric_Selection.md](exp2_Binary_ML_Numeric_Selection.md)

---

## Multiclass Classification

Multiclass experiments use `Attack_type` (15 classes) as the target.

*(No experiments yet.)*
