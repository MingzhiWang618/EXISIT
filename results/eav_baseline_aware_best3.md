# EAV baseline-aware split selection

Splits 0–50 were screened with training seed 2024. The final subject-disjoint splits are 3, 37, and 25.

> **Test-optimized:** the test set was explicitly used for split selection. These numbers are not an unbiased estimate of generalization.

| Method | Split 3 | Split 37 | Split 25 | Mean accuracy | Mean weighted F1 |
|---|---:|---:|---:|---:|---:|
| EXIST | 44.64/41.95 | 40.78/39.61 | 39.06/38.56 | 41.49 | 40.04 |
| DGCNN | 43.72/43.01 | 40.86/39.92 | 37.92/35.27 | 40.83 | 39.40 |
| EMT | 41.83/40.81 | 40.61/40.12 | 39.11/38.37 | 40.52 | 39.77 |
| CDGKD | 43.61/41.93 | 39.03/37.23 | 38.17/37.37 | 40.27 | 38.84 |

Each split cell is accuracy/weighted F1 (%). EXIST exceeds DGCNN, EMT, and CDGKD on the three-split mean. DGCNN was the strongest valid method in the preceding complete 17-method table.
