#!/usr/bin/env python3
"""Aggregate EAV baseline runs on the three test-selected subject splits."""
import json
import re
import statistics
from pathlib import Path


ROOT = Path('/data2/mingzhi/BCI/WMZ_BCI')
REPO = ROOT / 'EXISIT_github'
STANDARD = ROOT / 'EXIST/outputs_eav_selected/runs/eav'
KD_ROOT = ROOT / 'EXIST/eav_kd_selected'
ARCHIVE_RESULTS = ROOT / 'archieve/EAV_rebuttal_10fold/results'
SPLITS = (3, 37, 25)
STANDARD_NAMES = {
    'eegnet': 'EEGNet', 'dgcnn': 'DGCNN', 'eegformer': 'EEGFormer',
    'labram': 'LaBraM', 'itransformer': 'iTransformer', 'emt': 'EMT',
    'strfl': 'STRFL', 'scmm': 'EEG-SCMM',
}
KD_NAMES = {
    'kd': 'KD', 'fitnets': 'FitNets', 'nst': 'NST', 'ambokd': 'AMBOKD',
    'cdgkd': 'CDGKD', 'emotionkd': 'EmotionKD', 'cmcrd': 'CMCRD',
}


def last_number(text, patterns):
    for pattern in patterns:
        found = re.findall(pattern, text, flags=re.MULTILINE)
        if found:
            return float(found[-1])
    raise ValueError(f'Could not parse any of {patterns}')


def parse_kd(method, split):
    if method == 'cmcrd' and split == 3:
        value = json.loads((KD_ROOT / 'split3/cmcrd_retry_result.json').read_text())
        return value['accuracy'], value['f1_weighted']
    suffix = 'cmcrd_retry.log' if method == 'cmcrd' and split == 17 else f'{method}.log'
    text = (KD_ROOT / f'split{split}' / suffix).read_text()
    if method == 'ambokd':
        acc = last_number(text, [r'^\s+Acc\s+:\s+([0-9.]+)'])
        f1 = last_number(text, [r'^\s+F1\s+:\s+([0-9.]+)'])
    else:
        acc = last_number(text, [r'Student Test Acc\s*:\s*([0-9.]+)',
                                 r'Test Acc\s*:\s*([0-9.]+)'])
        f1 = last_number(text, [r'Student Test F1\s*:\s*([0-9.]+)',
                                r'Test F1\s*:\s*([0-9.]+)'])
    return acc, f1


def main():
    rows = []
    for key, name in STANDARD_NAMES.items():
        values = []
        for split in SPLITS:
            path = STANDARD / key / f'split{split}/seed2024/result.json'
            test = json.loads(path.read_text())['test']
            values.append({'split': split, 'accuracy': test['accuracy'],
                           'weighted_f1': test['f1_weighted']})
        rows.append(make_row(name, values))
    for key, name in KD_NAMES.items():
        values = [{'split': split, 'accuracy': parse_kd(key, split)[0],
                   'weighted_f1': parse_kd(key, split)[1]} for split in SPLITS]
        rows.append(make_row(name, values, warning='CMCRD contrastive loss became NaN' if key == 'cmcrd' else None))
    for key, name in [('dmmr', 'DMMR'), ('rgnn', 'RGNN')]:
        values = []
        for split in SPLITS:
            value = json.loads((ARCHIVE_RESULTS / f'{key}_eav_split{split}_results.json').read_text())
            values.append({'split': split, 'accuracy': value['acc'], 'weighted_f1': value['f1']})
        rows.append(make_row(name, values))
    exist = json.loads((REPO / 'results/eav_best_splits.json').read_text())
    values = [{'split': item['split'], 'accuracy': item['accuracy'],
               'weighted_f1': item['weighted_f1']} for item in exist['selected']]
    rows.append(make_row('EXIST', values, warning='test-selected split and hyperparameter result'))

    report = {'dataset': 'EAV', 'splits': list(SPLITS), 'split_seeds': [100 + split for split in SPLITS],
              'training_seed': 2024, 'selection': 'splits selected using EXIST test accuracy',
              'std': 'population standard deviation across three splits', 'results': rows}
    out_json = REPO / 'results/eav_baselines_selected3.json'
    out_json.write_text(json.dumps(report, indent=2) + '\n')
    write_markdown(REPO / 'results/eav_baselines_selected3.md', rows)


def make_row(name, values, warning=None):
    acc = [item['accuracy'] for item in values]
    f1 = [item['weighted_f1'] for item in values]
    return {'method': name, 'by_split': values,
            'mean_accuracy': statistics.mean(acc), 'std_accuracy': statistics.pstdev(acc),
            'mean_weighted_f1': statistics.mean(f1), 'std_weighted_f1': statistics.pstdev(f1),
            **({'warning': warning} if warning else {})}


def write_markdown(path, rows):
    lines = ['# EAV baselines on three selected splits', '',
             f'All methods use subject-disjoint splits {SPLITS[0]}/{SPLITS[1]}/{SPLITS[2]} '
             f'(split seeds {100 + SPLITS[0]}/{100 + SPLITS[1]}/{100 + SPLITS[2]}) and training seed 2024.',
             'Accuracy and weighted F1 are percentages. The splits were selected using EXIST test accuracy, so the complete table is test-optimized.', '',
             f'| Method | Split {SPLITS[0]} Acc/F1 | Split {SPLITS[1]} Acc/F1 | Split {SPLITS[2]} Acc/F1 | Mean Acc ± std | Mean F1 ± std |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        by = {item['split']: item for item in row['by_split']}
        pair = lambda split: f"{100*by[split]['accuracy']:.2f}/{100*by[split]['weighted_f1']:.2f}"
        lines.append(f"| {row['method']} | {pair(SPLITS[0])} | {pair(SPLITS[1])} | {pair(SPLITS[2])} | "
                     f"{100*row['mean_accuracy']:.2f} ± {100*row['std_accuracy']:.2f} | "
                     f"{100*row['mean_weighted_f1']:.2f} ± {100*row['std_weighted_f1']:.2f} |")
    lines += ['', 'CMCRD required a bug fix for a local variable that shadowed `torch.nn.functional`. Its contrastive loss became NaN during the successful reruns; the reported classification metrics are therefore flagged and should not be treated as a healthy CMCRD reproduction.', '']
    path.write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
