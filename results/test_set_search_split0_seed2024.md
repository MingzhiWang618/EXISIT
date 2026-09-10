# Test-set-tuned distillation parameters

This experiment directly selects hyperparameters on the split 0 test set, as
explicitly requested. These numbers are optimization results and are not an
unbiased estimate of generalization.

Ten fine-search configurations covered alpha 0.00–0.15 and CDD ratios
0.10–0.50. All used seed 2024, a 10-epoch warm-up, CDD temperature 2.0, EDD
temperature 1.0, EMA decay 0.95, and confidence floor 0.1. The selected
configurations were rerun with the standard patience of 30.

| Dataset | alpha | CDD ratio | EDD ratio | Original | Tuned | Change |
|---|---:|---:|---:|---:|---:|---:|
| EAV accuracy | 0.03 | 0.25 | 0.75 | 37.06% | **38.31%** | **+1.25 pp** |
| EAV weighted F1 | 0.03 | 0.25 | 0.75 | 35.99% | **36.91%** | **+0.92 pp** |
| PME4 balanced accuracy | 0.10 | 0.40 | 0.60 | 25.32% | **26.88%** | **+1.55 pp** |
| PME4 weighted F1 | 0.10 | 0.40 | 0.60 | 22.52% | **22.70%** | **+0.18 pp** |

The EAV optimum uses very weak distillation, indicating that its previous
distillation gradient was too strong. PME4 benefits from a larger but still
moderate weight. Dataset-specific alpha is therefore necessary under this
design.
