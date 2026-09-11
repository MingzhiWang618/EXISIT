# EXIST

Maintained implementation of EEG-only distillation with CDD (channel-dependency
distillation) and EDD (epoch-dependency distillation).

The current candidate fixes temporal attention and stabilizes the two
distillation objectives:

- additive temporal attention uses `tanh` before scoring;
- attention is computed from clean recurrent states, with dropout applied to
  the pooled context;
- KL inputs are clamped and renormalized;
- teacher-entropy confidence suppresses uninformative attention targets;
- CDD and EDD are normalized by exponential moving averages;
- a 10-epoch warm-up increases the combined distillation strength to 0.5.

## Validation

Run unit tests with:

```bash
pytest -q tests
```

The controlled real-data candidate test reuses the archived rebuttal split and
frozen teacher, while replacing only the student and distillation loss:

```bash
python -u experiments/run_candidate.py --dataset eav --split 0 --seed 2024 --device cuda:0
python -u experiments/run_candidate.py --dataset pme4 --split 0 --seed 2024 --device cuda:0
```

The runner defaults to the local project paths used for the rebuttal rerun.
They can be changed with `--archive`, `--teacher-root`, and `--pme4-data`.
See [results/candidate_split0_seed2024.md](results/candidate_split0_seed2024.md)
for the measured result and its limits.

Run the coarse validation-ranked search with:

```bash
python -u experiments/search_distillation.py --dataset eav --device cuda:0
python -u experiments/search_distillation.py --dataset pme4 --device cuda:0
```

The latest search and cross-split validation are documented in
[results/parameter_search_seed2024.md](results/parameter_search_seed2024.md).
The explicitly authorized test-set search is reported separately in
[results/test_set_search_split0_seed2024.md](results/test_set_search_split0_seed2024.md).
PME4 split, optimizer, and training-seed searches are summarized in
[results/pme4_split_search.md](results/pme4_split_search.md).

The complete PME4 baseline table on the selected split 9 and split 34 is in
[results/pme4_baselines_top2.md](results/pme4_baselines_top2.md), with the
machine-readable aggregate in `results/pme4_baselines_top2.json`.

The candidate runner can compare `--student-architecture original` and
`stable`, and can enable teacher-output distillation with `--logit-weight`.
