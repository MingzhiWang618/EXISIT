#!/usr/bin/env python3
"""Run archived EAV distillation baselines on one reproducible subject split."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ARCHIVE = Path('/data2/mingzhi/BCI/WMZ_BCI/archieve')
SOURCE = ARCHIVE / 'KDbaseline'
STEPS = [
    ('teacher', 'train_teacher.py'),
    ('kd', 'KD.py'),
    ('fitnets', 'FitNets.py'),
    ('nst', 'NST.py'),
    ('ambokd', 'AMBOKD.py'),
    ('cdgkd_teacher', 'train_CDGKD_teacher.py'),
    ('cdgkd', 'CDGKD.py'),
    ('emotionkd', 'EmotionKD.py'),
    ('cmcrd', 'CMCRD.py'),
]


def main(args):
    run_dir = Path(args.output).resolve() / f'split{args.split}'
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['EAV_SPLIT_IDX'] = str(args.split)
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PYTHONHASHSEED'] = str(args.seed)
    env['PYTHONPATH'] = str(ARCHIVE) + os.pathsep + env.get('PYTHONPATH', '')
    status = {'split': args.split, 'split_seed': 100 + args.split,
              'training_seed': args.seed, 'completed': [], 'failed': []}
    for name, script in STEPS:
        log_path = run_dir / f'{name}.log'
        print(f'RUN split={args.split} method={name} log={log_path}', flush=True)
        with log_path.open('w') as log:
            proc = subprocess.run([sys.executable, '-u', str(SOURCE / script)],
                                  cwd=run_dir, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
        (status['completed'] if proc.returncode == 0 else status['failed']).append(name)
        (run_dir / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
        if proc.returncode and not args.keep_going:
            raise SystemExit(proc.returncode)
    if status['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--output', default='eav_kd_baselines')
    parser.add_argument('--keep-going', action='store_true')
    main(parser.parse_args())
