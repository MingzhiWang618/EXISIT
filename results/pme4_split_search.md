# PME4 test-set split search

The search directly uses the test set, as explicitly requested. Every split is
subject-disjoint and keeps the 7/2/2 train/validation/test subject counts.
Random partitions are reproduced by `numpy.RandomState(100 + split_index)`.

We trained fresh split-specific multimodal teachers and EXIST students for
split indices 3 through 34. Together with the three earlier splits, 35 random
partitions were examined. Screening used alpha 0.10, CDD/EDD ratio 0.40/0.60,
CDD temperature 2.0, seed 2024, and shortened early stopping.

## Recommended partition

- split index: **9**
- split seed: **109**
- train subjects: **[1, 2, 3, 4, 5, 8, 11]**
- validation subjects: **[9, 10]**
- test subjects: **[6, 7]**
- balanced accuracy: **29.54%**
- weighted F1: **27.91%**

Compared with the original split 0 result, balanced accuracy rises from
25.32% to 29.54% (+4.22 percentage points) and weighted F1 rises from 22.52%
to 27.91% (+5.39 points).

The next best newly searched partitions by balanced accuracy were split 13
(28.32%), split 26 (27.83%), split 34 (27.75%), and split 15 (27.75%).

An additional 12-trial learning-rate/dropout/weight-decay search on split 9
did not exceed the screening result. Training seeds 42 and 3407 reached only
23.82% and 23.76%, respectively, so the recommended training seed is **2024**.

Because both the partition and training seed were chosen using test results,
29.54% is a test-optimized score rather than an unbiased estimate.

## Wide distillation-weight search

We evaluated 71 test-ranked configurations on split 9 with training seed 2024.
The total CDD/EDD weight covered 0.001 through 10.0; the CDD share covered
0%, 25%, 50%, 75%, and 100%, and a fine search varied the CDD temperature over
0.5, 1.0, 2.0, and 4.0.

The highest observed screening result used total weight **0.001**, pure CDD
(CDD/EDD = **1.0/0.0**), CDD temperature **1.0**, and a 10-epoch linear
warmup. It reached **31.32%** balanced accuracy and **30.19%** weighted F1.
The previously best 0.005/pure-CDD configuration reached 30.28%. Weights 5.0,
7.5, and 10.0 did not improve the result; the high-weight runs stayed below
27.61% balanced accuracy. This teacher therefore benefits, at most, from a
very weak feature-distillation constraint on PME4.

An independent full-schedule repeat of the selected configuration reached
26.75% balanced accuracy and 16.76% weighted F1. The 31.32% figure is the
highest test-selected observation, not a stable repeated estimate. PME4's
CUDA training and validation-based checkpoint selection currently show high
run-to-run variance even with the same nominal seed.

## Top three partitions with the selected loss

We re-evaluated all 35 partitions with the same selected configuration:
training seed 2024, total distillation weight 0.001, pure CDD, CDD temperature
1.0, 10-epoch warmup, and the stable student. Screening used 80 epochs and
patience 15. Ranking directly by test balanced accuracy selected:

1. **split 9 / split seed 109**: train `[1,2,3,4,5,8,11]`, validation
   `[9,10]`, test `[6,7]`; balanced accuracy **31.32%**, weighted F1 **30.19%**.
2. **split 34 / split seed 134**: train `[1,2,3,5,6,10,11]`, validation
   `[8,9]`, test `[4,7]`; balanced accuracy **30.75%**, weighted F1 **29.67%**.
3. **split 13 / split seed 113**: train `[1,2,4,5,8,9,11]`, validation
   `[7,10]`, test `[3,6]`; balanced accuracy **28.31%**, weighted F1 **22.09%**.

Their single-seed mean is **30.13%** balanced accuracy and **27.32%** weighted
F1. These values select both the partitions and loss parameters on the test
set. The next experiment should run all three training seeds on these fixed
partitions without replacing a partition based on those additional outcomes.

## Original network and logit-distillation check

On split 9, the original student attention implementation reached 29.19%
balanced accuracy without logit distillation, while the stable student reached
29.54%. Adding teacher `eeg_logits` KL at weights 0.10 or 0.30 did not improve
either architecture; the best logit-distilled run was 28.27%. The teacher's
EEG predictions are too unstable across held-out subjects to serve as a strong
class-probability target. The maintained default therefore keeps the stable
attention structure and leaves `logit_weight=0`.

The runner supports `--student-architecture original` for controlled
ablations and optional `--logit-weight` and `--logit-temperature` arguments.
