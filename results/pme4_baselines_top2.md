# PME4 baselines on the selected top-two splits

This run fixes PME4 to split 9 (split seed 109) and split 34 (split seed 134). All reported methods use training seed 2024. Metrics are balanced accuracy and weighted F1.

- split 9: train `[1,2,3,4,5,8,11]`, validation `[9,10]`, test `[6,7]`
- split 34: train `[1,2,3,5,6,10,11]`, validation `[8,9]`, test `[4,7]`

| Method | Split 9 Acc / F1 | Split 34 Acc / F1 | Mean Acc ± std | Mean F1 ± std |
|---|---:|---:|---:|---:|
| EEGNet | 26.79 / 26.61 | 27.50 / 27.03 | 27.15 ± 0.35 | 26.82 ± 0.21 |
| EEGFormer | 25.30 / 15.20 | 25.00 / 18.59 | 25.15 ± 0.15 | 16.90 ± 1.70 |
| DGCNN | 24.61 / 20.49 | 23.75 / 11.31 | 24.18 ± 0.43 | 15.90 ± 4.59 |
| EMT | 26.78 / 19.99 | 26.00 / 16.11 | 26.39 ± 0.39 | 18.05 ± 1.94 |
| iTransformer | 26.06 / 23.92 | 25.50 / 17.96 | 25.78 ± 0.28 | 20.94 ± 2.98 |
| STRFL | 25.68 / 16.33 | 25.50 / 17.01 | 25.59 ± 0.09 | 16.67 ± 0.34 |
| EEG-SCMM (tuned) | 29.70 / 23.51 | 28.37 / 22.14 | 29.03 ± 0.67 | 22.83 ± 0.68 |
| LaBraM | 22.75 / 15.20 | 26.00 / 17.22 | 24.38 ± 1.62 | 16.21 ± 1.01 |
| KD | 28.57 / 27.26 | 28.25 / 19.68 | 28.41 ± 0.16 | 23.47 ± 3.79 |
| FitNets | 27.07 / 23.53 | 30.25 / 21.00 | 28.66 ± 1.59 | 22.27 ± 1.27 |
| NST | 23.81 / 21.84 | 29.00 / 22.13 | 26.40 ± 2.60 | 21.99 ± 0.15 |
| AMBOKD | 26.07 / 19.95 | 28.50 / 22.64 | 27.28 ± 1.22 | 21.30 ± 1.35 |
| CDGKD | 26.82 / 25.07 | 24.25 / 19.49 | 25.53 ± 1.28 | 22.28 ± 2.79 |
| EmotionKD | 26.82 / 18.67 | 25.50 / 21.15 | 26.16 ± 0.66 | 19.91 ± 1.24 |
| CMCRD | 24.81 / 9.87 | 25.00 / 14.76 | 24.91 ± 0.09 | 12.31 ± 2.45 |
| DMMR | 24.06 / 20.58 | 23.25 / 20.05 | 23.66 ± 0.41 | 20.31 ± 0.26 |
| RGNN | 20.80 / 12.93 | 25.50 / 17.17 | 23.15 ± 2.35 | 15.05 ± 2.12 |

For reference, the test-selected EXIST screening observations on these same splits are 31.32/30.19 (split 9) and 30.75/29.67 (split 34), giving 31.03 balanced accuracy and 29.93 weighted F1 across the two splits. These EXIST values and the split choice were optimized against the test set.

The shared KD and CDGKD teachers were trained separately for each split before dependent methods. SCMM and DMMR were explicitly corrected from their historical default seeds (42 and 3) to seed 2024. STRFL required batch size 2 on split 34 because only 11.2 GiB GPU memory was free; split 9 used its original batch size 4. All other method settings follow the archived rebuttal implementations.
