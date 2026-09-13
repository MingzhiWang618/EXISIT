# EXIST without distillation

The student architecture, fixed subject splits, optimizer settings, and training seed 2024 are unchanged. Only the total distillation weight is set to `alpha=0`, so training uses cross-entropy alone.

| Dataset | Splits | No-distill Acc ± std | No-distill F1 ± std | Distilled Acc | Distilled F1 | Δ Acc | Δ F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| EAV | 3/37/25 | 41.72 ± 2.58 | 40.04 ± 2.55 | 41.49 | 40.04 | +0.23 | -0.01 |
| PME4 | 9/34 | 27.76 ± 0.24 | 22.81 ± 0.33 | 31.03 | 29.93 | -3.28 | -7.12 |

## EAV

| Split | Accuracy | Weighted F1 | Best validation accuracy |
|---:|---:|---:|---:|
| 3 | 45.28 | 43.45 | 38.09 |
| 37 | 40.64 | 39.33 | 34.53 |
| 25 | 39.25 | 37.33 | 42.56 |

## PME4

| Split | Accuracy | Weighted F1 | Best validation accuracy |
|---:|---:|---:|---:|
| 9 | 27.51 | 22.49 | 27.75 |
| 34 | 28.00 | 23.14 | 25.60 |

> The splits were selected using distilled EXIST test performance, so both the reference and this same-split ablation remain test-optimized.
