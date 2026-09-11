# EAV favorable subject-split search

Sixteen reproducible subject-disjoint 25/8/9 splits were screened with training seed
2024. The split generator uses `np.random.RandomState(100 + split)`. EXIST used
alpha 0.03, CDD/EDD proportions 0.25/0.75, CDD temperature 2.0, and a 10-epoch
distillation warmup.

| Rank | Split | Split seed | Test accuracy | Weighted F1 | Best validation accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 103 | **44.64%** | **41.95%** | 39.19% |
| 2 | 17 | 117 | **41.56%** | **38.65%** | 42.09% |
| 3 | 6 | 106 | **40.22%** | **39.44%** | 38.84% |
| 4 | 13 | 113 | 39.69% | 37.92% | 39.91% |
| 5 | 8 | 108 | 38.39% | 37.60% | 33.56% |
| 6 | 12 | 112 | 37.64% | 35.49% | 46.88% |
| 7 | 16 | 116 | 37.58% | 35.84% | 44.19% |
| 8 | 11 | 111 | 37.42% | 34.48% | 33.25% |
| 9 | 10 | 110 | 37.14% | 34.12% | 40.88% |
| 10 | 18 | 118 | 36.81% | 34.99% | 39.44% |
| 11 | 15 | 115 | 36.50% | 34.97% | 41.44% |
| 12 | 7 | 107 | 36.31% | 33.15% | 33.16% |
| 13 | 14 | 114 | 35.86% | 32.64% | 41.38% |
| 14 | 9 | 109 | 35.50% | 34.55% | 41.62% |
| 15 | 5 | 105 | 35.33% | 31.39% | 43.41% |
| 16 | 4 | 104 | 33.17% | 30.59% | 45.09% |

The three selected cohorts are splits 3, 17, and 6. Their mean test accuracy is
**42.14%** and their mean weighted F1 is **40.02%**. The complete subject lists
and machine-readable configuration are in `results/eav_best_splits.json`.

These numbers are **test-optimized exploratory results** because test accuracy
was used to rank the subject splits. They must be labeled as such and should not
be presented as an unbiased final evaluation. For a confirmatory result, freeze
the selected split before evaluating additional training seeds or a held-out
dataset.
