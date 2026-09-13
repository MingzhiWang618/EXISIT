#!/usr/bin/env python3
"""Aggregate the fixed-split EXIST alpha=0 ablation."""
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / 'ablations/no_distillation'


def load(dataset, split):
    value = json.loads((ABLATION / dataset / f'split{split}/seed2024/result.json').read_text())
    assert value['loss_config']['alpha'] == 0.0
    result = value['result']
    return {'split': split, 'accuracy': result['acc'], 'weighted_f1': result['f1'],
            'best_validation_accuracy': result['best_val_acc']}


def summarize(rows):
    return {
        'by_split': rows,
        'mean_accuracy': statistics.mean(x['accuracy'] for x in rows),
        'std_accuracy': statistics.pstdev(x['accuracy'] for x in rows),
        'mean_weighted_f1': statistics.mean(x['weighted_f1'] for x in rows),
        'std_weighted_f1': statistics.pstdev(x['weighted_f1'] for x in rows),
    }


def main():
    no_distill = {
        'eav': summarize([load('eav', split) for split in (3, 37, 25)]),
        'pme4': summarize([load('pme4', split) for split in (9, 34)]),
    }
    eav_reference = json.loads((ROOT / 'results/eav_best_splits.json').read_text())['selected_mean']
    with_distill = {
        'eav': {'mean_accuracy': eav_reference['accuracy'],
                'mean_weighted_f1': eav_reference['weighted_f1']},
        'pme4': {'mean_accuracy': (0.3131818181818182 + 0.3075) / 2,
                 'mean_weighted_f1': (0.3019386338801082 + 0.2966706962392659) / 2},
    }
    payload = {'training_seed': 2024, 'ablation': 'alpha=0 (cross-entropy only)',
               'student_architecture': 'stable', 'no_distillation': no_distill,
               'with_distillation_reference': with_distill}
    for dataset in ('eav', 'pme4'):
        payload.setdefault('delta_no_distillation_minus_distillation', {})[dataset] = {
            'accuracy': no_distill[dataset]['mean_accuracy'] - with_distill[dataset]['mean_accuracy'],
            'weighted_f1': no_distill[dataset]['mean_weighted_f1'] - with_distill[dataset]['mean_weighted_f1'],
        }
    (ROOT / 'results/no_distillation_ablation.json').write_text(json.dumps(payload, indent=2) + '\n')

    lines = ['# EXIST without distillation', '',
             'The student architecture, fixed subject splits, optimizer settings, and training seed 2024 are unchanged. Only the total distillation weight is set to `alpha=0`, so training uses cross-entropy alone.', '',
             '| Dataset | Splits | No-distill Acc ± std | No-distill F1 ± std | Distilled Acc | Distilled F1 | Δ Acc | Δ F1 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for dataset, splits in [('eav', '3/37/25'), ('pme4', '9/34')]:
        n, d = no_distill[dataset], with_distill[dataset]
        da = n['mean_accuracy'] - d['mean_accuracy']; df = n['mean_weighted_f1'] - d['mean_weighted_f1']
        lines.append(f"| {dataset.upper()} | {splits} | {100*n['mean_accuracy']:.2f} ± {100*n['std_accuracy']:.2f} | {100*n['mean_weighted_f1']:.2f} ± {100*n['std_weighted_f1']:.2f} | {100*d['mean_accuracy']:.2f} | {100*d['mean_weighted_f1']:.2f} | {100*da:+.2f} | {100*df:+.2f} |")
    for dataset, title in [('eav', 'EAV'), ('pme4', 'PME4')]:
        lines += ['', f'## {title}', '', '| Split | Accuracy | Weighted F1 | Best validation accuracy |', '|---:|---:|---:|---:|']
        for row in no_distill[dataset]['by_split']:
            lines.append(f"| {row['split']} | {100*row['accuracy']:.2f} | {100*row['weighted_f1']:.2f} | {100*row['best_validation_accuracy']:.2f} |")
    lines += ['', '> The splits were selected using distilled EXIST test performance, so both the reference and this same-split ablation remain test-optimized.', '']
    (ROOT / 'results/no_distillation_ablation.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
