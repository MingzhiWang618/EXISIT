# Stable CDD/EDD candidate: split 0, seed 2024

This is a controlled smoke experiment on the same rebuttal split 0 and
training seed 2024. The teacher checkpoint is frozen; only the maintained
student attention and distillation objective are changed.

| Dataset | Original EXIST | Candidate | Change |
|---|---:|---:|---:|
| EAV accuracy | 37.06% | 37.56% | +0.50 pp |
| EAV weighted F1 | 35.99% | 35.94% | -0.05 pp |
| PME4 balanced accuracy | 25.32% | 24.52% | -0.80 pp |
| PME4 weighted F1 | 22.52% | 18.32% | -4.20 pp |

The objectives are trainable and finite. EAV training CDD fell from 3.912 to
3.072 and EDD from 1.364 to 1.193. PME4 training CDD fell from 1.960 to 1.915
and EDD from 1.432 to 0.795 by its final training epoch.

This candidate improves EAV accuracy but does not improve PME4. It should be
treated as an experimental branch until its weights are tuned on validation
data and the complete 3 splits × 3 seeds matrix is rerun.
