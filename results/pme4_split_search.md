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
