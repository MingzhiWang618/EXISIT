# Requested EAV ablations

Protocol: fixed splits 3/37/25 (split seeds 103/137/125), training seed 2024. Values are test Accuracy / weighted F1 (%). Every student uses EEG only at inference.

## PCC ablation

| PCC | Split 3 | Split 37 | Split 25 | Mean |
|---|---:|---:|---:|---:|
| Yes | 44.64 / 41.95 | 40.78 / 39.61 | 39.06 / 38.56 | **41.49 / 40.04** |
| No | 43.00 / 39.91 | 41.31 / 40.61 | 38.00 / 35.94 | 40.77 / 38.82 |

Removing PCC lowers mean accuracy by 0.72 percentage points and weighted F1 by 1.23 points.

## Effect of alpha injection

| Injection | Definition | Split 3 | Split 37 | Split 25 | Mean |
|---|---|---:|---:|---:|---:|
| Row-wise | add alpha_i to row i | 44.08 / 42.95 | 41.06 / 40.55 | 40.00 / 39.30 | 41.71 / 40.93 |
| Element-wise | add alpha_i alpha_j to element (i,j) | 45.08 / 44.06 | 41.22 / 39.76 | 39.19 / 38.03 | **41.83 / 40.62** |
| Column-wise | add alpha_j to column j | 44.64 / 41.95 | 40.78 / 39.61 | 39.06 / 38.56 | 41.49 / 40.04 |

Element-wise has the highest mean accuracy; row-wise has the highest mean weighted F1.

## Effect of distillation supervision target

MM means the multimodal teacher output (fused logits or fused feature). EEG-T means the AV-guided teacher's EEG-branch logits or EEG feature.

| Method | Target | Split 3 | Split 37 | Split 25 | Mean |
|---|---|---:|---:|---:|---:|
| KD | MM | 42.25 / 40.53 | 35.61 / 34.10 | 37.50 / 36.26 | **38.45 / 36.96** |
| KD | EEG-T | 42.39 / 39.04 | 34.83 / 33.45 | 36.64 / 34.22 | 37.95 / 35.57 |
| FitNets | MM | 41.58 / 37.92 | 37.25 / 36.35 | 37.19 / 36.80 | 38.67 / 37.02 |
| FitNets | EEG-T | 43.14 / 40.73 | 38.14 / 36.71 | 37.78 / 36.11 | **39.69 / 37.85** |
| NST | MM | 42.11 / 40.31 | 36.44 / 35.06 | 32.39 / 28.87 | 36.98 / 34.75 |
| NST | EEG-T | 41.19 / 39.91 | 38.42 / 37.25 | 36.03 / 35.64 | **38.55 / 37.60** |

The better target is method-dependent: MM is better for KD, while EEG-T is better for FitNets and NST.
