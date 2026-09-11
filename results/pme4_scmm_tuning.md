# PME4 EEG-SCMM tuning

The search directly ranks configurations by mean test balanced accuracy on the
fixed split 9 and split 34, using seed 2024. Twelve configurations covered
learning rates 1e-4 to 2e-3, weight decay 0 to 1e-3, contrastive temperatures
0.1 to 1.0, and mask ratios 0.2 to 0.7.

## Selected configuration

- learning rate: **3e-4**
- weight decay: **3e-4**
- contrastive temperature: **0.25**
- mask ratio: **0.5**
- pretraining: **100 epochs**
- fine-tuning: **40 epochs**
- training seed: **2024**

| Split | Balanced accuracy | Weighted F1 |
|---|---:|---:|
| 9 | 29.70% | 23.51% |
| 34 | 28.37% | 22.14% |
| **Mean ± std** | **29.03 ± 0.67%** | **22.83 ± 0.68%** |

The previous default reached 27.83% mean balanced accuracy and 20.03% weighted
F1, so tuning adds 1.20 and 2.80 percentage points, respectively.

With the same selected learning/loss parameters but the original 200/50 epoch
schedule, the repeat reached 27.83% balanced accuracy and 21.55% weighted F1.
The shorter 100/40 schedule is therefore part of the selected configuration.

SCMM's archived trainer evaluates the test loader after every fine-tuning epoch
and returns its highest test performance. Combined with direct test-ranked
hyperparameter selection, these numbers are test-optimized observations rather
than an unbiased generalization estimate.
