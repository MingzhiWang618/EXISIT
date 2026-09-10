#!/usr/bin/env python3
"""Evaluate the maintained student/loss against a frozen rebuttal teacher."""
import argparse, importlib.util, json, shutil, statistics, sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from exist_method.distillation import DistillationConfig, StableDistillationLoss

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None; spec.loader.exec_module(mod); return mod

def main(a):
    archive=Path(a.archive).resolve(); source=archive/"OurMethod_rebuttal"
    # Resolve the archived teacher and its dependencies before loading the
    # maintained student.  The teacher checkpoint must keep its exact training
    # architecture for a controlled student/loss comparison.
    sys.path.insert(0, str(archive))
    sys.path.insert(0, str(source))
    original=load(f"candidate_{a.dataset}",source/f"run_{a.dataset}_distill.py")
    teacher_file=archive/("multimodal" if a.dataset=="eav" else "PME4")/"model"/"Teacher.py"
    teacher_mod=load(f"frozen_teacher_{a.dataset}",teacher_file)
    original.TeacherModel=teacher_mod.TeacherModel
    student_mod=load(f"maintained_student_{a.dataset}",REPO/("multimodal" if a.dataset=="eav" else "PME4")/"model"/"Student.py")
    original.ST_GCLSTM=student_mod.ST_GCLSTM
    if a.dataset=="pme4": original.DATA_ROOT=a.pme4_data
    output=Path(a.output).resolve()/a.dataset/f"split{a.split}"/f"seed{a.seed}"
    ckpt=output/"checkpoints"; ckpt.mkdir(parents=True,exist_ok=True); original.CKPT_DIR=str(ckpt)
    teacher_dir=Path(a.teacher_root)/"runs"/a.dataset/f"split{a.split}"/f"seed{a.seed}"/"checkpoints"
    teacher=teacher_dir/f"teacher_{a.dataset}_split{a.split}_best.pt"
    if a.dataset=="eav" and a.seed!=2024: teacher=teacher_dir/f"teacher_eav_split{a.split}_seed{a.seed}_best.pt"
    shutil.copy2(teacher,ckpt/f"teacher_{a.dataset}_split{a.split}_best.pt")
    criterion=StableDistillationLoss(DistillationConfig()).to(a.device); epoch={"value":0}; batches=[]; history=[]
    def loss_fn(cfg,s,t,y):
        values=criterion(s,t,y,epoch["value"]); batches.append({k:float(v.detach()) for k,v in values.items()}); return values["loss"]
    base_epoch=original.run_epoch
    def run_epoch(cfg,student,teacher,loader,optimizer,device,is_train):
        if is_train: epoch["value"]+=1
        criterion.train(is_train); start=len(batches)
        result=base_epoch(cfg,student,teacher,loader,optimizer,device,is_train)
        rows=batches[start:]
        if rows:
            summary={k:statistics.mean(r[k] for r in rows) for k in rows[0]}
            summary.update(epoch=epoch["value"],phase="train" if is_train else "val",accuracy=result["acc"])
            history.append(summary); print("CANDIDATE_LOSS "+json.dumps(summary),flush=True)
        return result
    original.distill_loss=loss_fn; original.run_epoch=run_epoch
    cfg=original.Config(); cfg.seed=a.seed; cfg.device=a.device
    result=original.train(cfg,a.split)
    payload={"dataset":a.dataset,"split":a.split,"split_seed":100+a.split,"seed":a.seed,
             "candidate":"stable_attention_ema_warmup_confidence","loss_config":vars(criterion.config),
             "result":result,"history":history}
    (output/"result.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(f"Saved {output/'result.json'}")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("eav","pme4"),required=True)
    p.add_argument("--split",type=int,default=0); p.add_argument("--seed",type=int,default=2024); p.add_argument("--device",required=True)
    p.add_argument("--archive",default="/data2/mingzhi/BCI/WMZ_BCI/archieve")
    p.add_argument("--teacher-root",default="/data2/mingzhi/BCI/WMZ_BCI/EXIST/rebuttal_rerun/outputs_3x3")
    p.add_argument("--pme4-data",default="/data2/zhiwen/bci/dataset/PME4")
    p.add_argument("--output",default="candidate_outputs"); main(p.parse_args())
