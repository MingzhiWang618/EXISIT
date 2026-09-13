# EAV Tables A.3 and A.4 under the current protocol

Splits 3/37/25 (split seeds 103/137/125), training seed 2024. Values are mean ± population standard deviation across splits.

## Table A.3: Effect of AV guidance on EEG teacher on EAV

| EEG teacher configuration | Acc (%) | Weighted F1 (%) |
|---|---:|---:|
| w/o guidance | 40.47 ± 1.31 | 39.92 ± 1.41 |
| w/o spatial (temporal only) | 45.01 ± 1.96 | 43.61 ± 2.81 |
| w/o temporal (spatial only) | 46.06 ± 1.27 | 45.37 ± 1.49 |
| w/ both | 47.33 ± 2.65 | 45.91 ± 4.27 |

## Table A.4: Effect of temporal window size on EAV

| Window (s) | T | EEG Student Acc (%) | EEG Student F1 (%) | EEG Teacher Acc (%) | EEG Teacher F1 (%) |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 50 | 39.67 ± 1.32 | 38.12 ± 1.41 | 43.90 ± 5.12 | 40.42 ± 7.53 |
| 0.5 | 10 | 41.50 ± 2.18 | 39.85 ± 2.31 | 46.04 ± 4.02 | 44.00 ± 5.11 |
| 1.0 | 5 | 41.49 ± 2.33 | 40.04 ± 1.42 | 47.33 ± 2.65 | 45.91 ± 4.27 |
| 2.5 | 2 | 39.60 ± 2.30 | 37.92 ± 2.45 | 44.93 ± 2.53 | 44.42 ± 2.70 |
