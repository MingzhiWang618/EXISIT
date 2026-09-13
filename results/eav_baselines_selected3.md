# EAV baselines on three selected splits

All methods use subject-disjoint splits 3/37/25 (split seeds 103/137/125) and training seed 2024.
Accuracy and weighted F1 are percentages. The splits were selected using EXIST test accuracy, so the complete table is test-optimized.

| Method | Split 3 Acc/F1 | Split 37 Acc/F1 | Split 25 Acc/F1 | Mean Acc ± std | Mean F1 ± std |
|---|---:|---:|---:|---:|---:|
| EEGNet | 41.28/39.29 | 38.94/36.83 | 35.00/29.37 | 38.41 ± 2.59 | 35.16 ± 4.22 |
| DGCNN | 43.72/43.01 | 40.86/39.92 | 37.92/35.27 | 40.83 ± 2.37 | 39.40 ± 3.18 |
| EEGFormer | 38.72/36.70 | 37.56/34.73 | 37.97/35.32 | 38.08 ± 0.48 | 35.58 ± 0.82 |
| LaBraM | 31.39/31.64 | 30.89/31.32 | 30.00/29.99 | 30.76 ± 0.57 | 30.98 ± 0.71 |
| iTransformer | 31.78/27.72 | 30.28/28.41 | 30.36/28.00 | 30.81 ± 0.69 | 28.04 ± 0.29 |
| EMT | 41.83/40.81 | 40.61/40.12 | 39.11/38.37 | 40.52 ± 1.11 | 39.77 ± 1.03 |
| STRFL | 31.56/23.85 | 31.28/26.97 | 30.86/21.84 | 31.23 ± 0.29 | 24.22 ± 2.11 |
| EEG-SCMM | 37.03/34.32 | 35.69/34.63 | 34.28/32.82 | 35.67 ± 1.12 | 33.92 ± 0.79 |
| KD | 42.25/40.53 | 35.61/34.10 | 37.50/36.26 | 38.45 ± 2.79 | 36.96 ± 2.67 |
| FitNets | 41.58/37.92 | 37.25/36.35 | 37.19/36.80 | 38.67 ± 2.06 | 37.02 ± 0.66 |
| NST | 42.11/40.31 | 36.44/35.06 | 32.39/28.87 | 36.98 ± 3.99 | 34.75 ± 4.68 |
| AMBOKD | 42.58/40.19 | 40.69/40.07 | 37.78/36.41 | 40.35 ± 1.97 | 38.89 ± 1.75 |
| CDGKD | 43.61/41.93 | 39.03/37.23 | 38.17/37.37 | 40.27 ± 2.39 | 38.84 ± 2.18 |
| EmotionKD | 40.44/39.57 | 41.08/40.95 | 37.81/37.50 | 39.78 ± 1.41 | 39.34 ± 1.42 |
| CMCRD | 42.33/41.42 | 37.22/37.15 | 39.03/38.60 | 39.53 ± 2.12 | 39.06 ± 1.77 |
| DMMR | 42.64/40.08 | 36.53/35.49 | 41.69/40.59 | 40.29 ± 2.69 | 38.72 ± 2.29 |
| RGNN | 37.11/32.60 | 28.06/19.46 | 29.56/22.77 | 31.57 ± 3.96 | 24.95 ± 5.58 |
| EXIST | 44.64/41.95 | 40.78/39.61 | 39.06/38.56 | 41.49 ± 2.33 | 40.04 ± 1.42 |

CMCRD required a bug fix for a local variable that shadowed `torch.nn.functional`. Its contrastive loss became NaN during the successful reruns; the reported classification metrics are therefore flagged and should not be treated as a healthy CMCRD reproduction.
