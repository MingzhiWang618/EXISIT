# EAV baselines on three selected splits

All methods use subject-disjoint splits 3/17/6 (split seeds 103/117/106) and training seed 2024.
Accuracy and weighted F1 are percentages. The splits were selected using EXIST test accuracy, so the complete table is test-optimized.

| Method | Split 3 Acc/F1 | Split 17 Acc/F1 | Split 6 Acc/F1 | Mean Acc ± std | Mean F1 ± std |
|---|---:|---:|---:|---:|---:|
| EEGNet | 41.28/39.29 | 41.61/42.54 | 39.33/36.75 | 40.74 ± 1.00 | 39.53 ± 2.37 |
| DGCNN | 43.72/43.01 | 45.25/44.10 | 42.97/42.30 | 43.98 ± 0.95 | 43.14 ± 0.74 |
| EEGFormer | 38.72/36.70 | 41.47/40.34 | 37.00/34.65 | 39.06 ± 1.84 | 37.23 ± 2.35 |
| LaBraM | 31.39/31.64 | 31.33/29.38 | 31.97/30.62 | 31.56 ± 0.29 | 30.55 ± 0.92 |
| iTransformer | 31.78/27.72 | 31.03/29.44 | 32.33/30.68 | 31.71 ± 0.53 | 29.28 ± 1.21 |
| EMT | 41.83/40.81 | 40.86/39.10 | 41.25/40.78 | 41.31 ± 0.40 | 40.23 ± 0.80 |
| STRFL | 31.56/23.85 | 35.69/28.50 | 28.42/21.70 | 31.89 ± 2.98 | 24.68 ± 2.84 |
| EEG-SCMM | 37.03/34.32 | 37.31/35.27 | 36.83/36.71 | 37.06 ± 0.19 | 35.43 ± 0.98 |
| KD | 42.25/40.53 | 43.25/42.08 | 38.72/37.04 | 41.41 ± 1.94 | 39.88 ± 2.11 |
| FitNets | 41.58/37.92 | 40.78/39.15 | 38.22/37.14 | 40.19 ± 1.43 | 38.07 ± 0.83 |
| NST | 42.11/40.31 | 40.97/39.03 | 37.58/37.03 | 40.22 ± 1.92 | 38.79 ± 1.35 |
| AMBOKD | 42.58/40.19 | 43.14/42.07 | 37.94/35.92 | 41.22 ± 2.33 | 39.39 ± 2.57 |
| CDGKD | 43.61/41.93 | 41.86/40.82 | 41.42/41.48 | 42.30 ± 0.95 | 41.41 ± 0.46 |
| EmotionKD | 40.44/39.57 | 40.83/40.25 | 40.78/40.09 | 40.68 ± 0.17 | 39.97 ± 0.29 |
| CMCRD | 42.33/41.42 | 43.06/40.82 | 41.56/41.20 | 42.32 ± 0.61 | 41.15 ± 0.25 |
| DMMR | 42.64/40.08 | 39.08/35.35 | 41.64/41.28 | 41.12 ± 1.50 | 38.90 ± 2.56 |
| RGNN | 37.11/32.60 | 35.11/31.07 | 32.89/28.79 | 35.04 ± 1.72 | 30.82 ± 1.57 |
| EXIST | 44.64/41.95 | 41.56/38.65 | 40.22/39.44 | 42.14 ± 1.85 | 40.02 ± 1.41 |

CMCRD required a bug fix for a local variable that shadowed `torch.nn.functional`. Its contrastive loss became NaN during the successful reruns; the reported classification metrics are therefore flagged and should not be treated as a healthy CMCRD reproduction.
