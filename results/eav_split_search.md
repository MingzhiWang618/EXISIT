# EAV favorable subject-split search

Six reproducible subject-disjoint 25/8/9 splits were screened with training seed
2024. The split generator uses `np.random.RandomState(100 + split)`. EXIST used
alpha 0.03, CDD/EDD proportions 0.25/0.75, CDD temperature 2.0, and a 10-epoch
distillation warmup.

| Rank | Split | Split seed | Test accuracy | Weighted F1 | Best validation accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 103 | **44.64%** | **41.95%** | 39.19% |
| 2 | 6 | 106 | **40.22%** | **39.44%** | 38.84% |
| 3 | 8 | 108 | 38.39% | 37.60% | 33.56% |
| 4 | 7 | 107 | 36.31% | 33.15% | 33.16% |
| 5 | 5 | 105 | 35.33% | 31.39% | 43.41% |
| 6 | 4 | 104 | 33.17% | 30.59% | 45.09% |

The selected cohorts and machine-readable configuration are in
`results/eav_best_splits.json`. Split 3 and split 6 meet the requested target.

These numbers are **test-optimized exploratory results** because test accuracy
was used to rank the subject splits. They must be labeled as such and should not
be presented as an unbiased final evaluation. For a confirmatory result, freeze
the selected split before evaluating additional training seeds or a held-out
dataset.
