#!/usr/bin/env python3
"""Train a matched EAV teacher/student pair for PCC or alpha-injection ablations."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path('/data2/mingzhi/BCI/WMZ_BCI/archieve')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    sys.path[:0] = [str(REPO), str(ARCHIVE), str(ARCHIVE / 'OurMethod_rebuttal'),
                    str(ARCHIVE / 'EAV_rebuttal_10fold')]
    from exist_method.architecture_ablation import configure_teacher

    trainer = load('ablation_teacher_trainer', ARCHIVE / 'OurMethod_rebuttal/run_eav_teacher.py')
    teacher_module = load('ablation_teacher_model', ARCHIVE / 'multimodal/model/Teacher.py')
    trainer.TeacherModel = configure_teacher(teacher_module, args.alpha_injection, args.use_pcc)
    root = Path(args.output).resolve()
    checkpoint_dir = root / 'runs/eav' / f'split{args.split}/seed2024/checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.CKPT_DIR = str(checkpoint_dir)
    config = trainer.Config()
    config.seed, config.device = 2024, args.device
    config.epochs, config.patience = 100, 15
    teacher_result = trainer.train(config, args.split)

    command = [sys.executable, '-u', str(REPO / 'experiments/run_candidate.py'),
               '--dataset', 'eav', '--split', str(args.split), '--seed', '2024',
               '--device', args.device, '--teacher-root', str(root), '--output', str(root / 'students'),
               '--alpha', '0.03', '--cdd-ratio', '0.25', '--cdd-temperature', '2',
               '--warmup-epochs', '10', '--epochs', '80', '--patience', '15',
               '--alpha-injection', args.alpha_injection,
               '--use-pcc' if args.use_pcc else '--no-use-pcc']
    log = root / f'student_split{args.split}.log'
    with log.open('w') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    student_path = root / f'students/eav/split{args.split}/seed2024/result.json'
    payload = {'split': args.split, 'seed': 2024, 'alpha_injection': args.alpha_injection,
               'use_pcc': args.use_pcc, 'teacher': teacher_result,
               'student': json.loads(student_path.read_text())['result']}
    (root / f'result_split{args.split}.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--alpha-injection', choices=('row', 'element', 'column'), default='column')
    parser.add_argument('--use-pcc', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', required=True)
    main(parser.parse_args())
