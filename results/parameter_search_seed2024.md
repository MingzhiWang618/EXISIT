# Distillation parameter search (seed 2024)

Eight configurations were screened on split 0. Selection used only peak
validation accuracy. The selected dataset-specific configuration was then
evaluated without retuning on splits 1 and 2.

## Selected configurations

| Dataset | alpha | CDD ratio | EDD ratio | CDD temperature |
|---|---:|---:|---:|---:|
| EAV | 0.25 | 0.50 | 0.50 | 1.0 |
| PME4 | 0.50 | 0.25 | 0.75 | 2.0 |

Both use a 10-epoch warm-up, EDD temperature 1.0, EMA decay 0.95, and
confidence floor 0.1.

## Cross-split comparison

| Dataset | Split | Original metric | Selected candidate | Change |
|---|---:|---:|---:|---:|
| EAV accuracy | 0 | 37.06% | 37.19% | +0.14 pp |
| EAV accuracy | 1 | 34.83% | 34.81% | -0.03 pp |
| EAV accuracy | 2 | 37.19% | 37.64% | +0.44 pp |
| PME4 balanced accuracy | 0 | 25.32% | 24.54% | -0.79 pp |
| PME4 balanced accuracy | 1 | 25.76% | 27.76% | +1.99 pp |
| PME4 balanced accuracy | 2 | 29.00% | 29.75% | +0.75 pp |

Across the three splits at seed 2024, mean EAV accuracy changes from 36.36%
to 36.55% (+0.19 pp), while weighted F1 changes from 35.10% to 34.35%.
Mean PME4 balanced accuracy changes from 26.70% to 27.35% (+0.65 pp), while
weighted F1 changes from 24.38% to 23.18%. The accuracy gains therefore do not
yet represent a general improvement across metrics.

The search evaluates a test set after each run because the inherited training
entry point does so, but the search script sorts trials exclusively by
`best_val_acc`. A stronger final claim requires the complete 3 splits × 3
training seeds experiment and should report both accuracy and weighted F1.
