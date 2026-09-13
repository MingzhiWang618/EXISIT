#!/usr/bin/env python3
"""Run current-protocol EAV guidance or temporal-window appendix ablations."""
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO.parent / "archieve"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main(args):
    sys.path[:0] = [str(REPO), str(ARCHIVE), str(ARCHIVE/'OurMethod_rebuttal'),
                    str(ARCHIVE/'EAV_rebuttal_10fold')]
    from exist_method.architecture_ablation import configure_teacher_guidance
    from exist_method.window_ablation import configure_eav_window

    trainer = load('appendix_teacher_trainer', ARCHIVE/'OurMethod_rebuttal/run_eav_teacher.py')
    teacher_module = load('appendix_teacher_model', ARCHIVE/'multimodal/model/Teacher.py')
    trainer.TeacherModel = configure_teacher_guidance(
        teacher_module, args.spatial_guidance, args.temporal_guidance,
        injection='column', use_pcc=True)
    if args.window_seconds is not None:
        import dataset.dataset as dataset_module
        configure_eav_window(dataset_module.CrossSubjectMultiModalDataset, args.window_seconds)

    root = Path(args.output).resolve()
    checkpoint_dir = root/'runs/eav'/f'split{args.split}/seed2024/checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True); trainer.CKPT_DIR = str(checkpoint_dir)
    cfg=trainer.Config(); cfg.seed=2024; cfg.device=args.device; cfg.epochs=100; cfg.patience=15
    teacher_result=trainer.train(cfg,args.split)
    payload={'split':args.split,'seed':2024,'window_seconds':args.window_seconds,
             'spatial_guidance':args.spatial_guidance,'temporal_guidance':args.temporal_guidance,
             'teacher':teacher_result}
    if args.train_student:
        command=[sys.executable,'-u',str(REPO/'experiments/run_candidate.py'),
                 '--dataset','eav','--split',str(args.split),'--seed','2024','--device',args.device,
                 '--teacher-root',str(root),'--output',str(root/'students'),'--alpha','0.03',
                 '--cdd-ratio','0.25','--cdd-temperature','2','--warmup-epochs','10',
                 '--epochs','80','--patience','15']
        if args.window_seconds is not None: command += ['--window-seconds',str(args.window_seconds)]
        log=root/f'student_split{args.split}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream: subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
        result_path=root/'students/eav'/f'split{args.split}/seed2024/result.json'
        payload['student']=json.loads(result_path.read_text())['result']
    out=root/f'result_split{args.split}.json'; out.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--split',type=int,required=True); p.add_argument('--device',required=True)
    p.add_argument('--window-seconds',type=float); p.add_argument('--train-student',action='store_true')
    p.add_argument('--spatial-guidance',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--temporal-guidance',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output',required=True); main(p.parse_args())
