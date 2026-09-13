#!/usr/bin/env python3
"""Run KD-family baselines with an AV-guided EEG teacher target on fixed EAV splits."""
import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
ARCHIVE = ROOT / "archieve"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    os.environ["EAV_SPLIT_IDX"] = str(args.split)
    sys.path[:0] = [str(ARCHIVE), str(REPO)]
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    filenames = {"kd": "KD.py", "fitnets": "FitNets.py", "nst": "NST.py"}
    script = load(f"eav_{args.method}_eegt", ARCHIVE / "KDbaseline" / filenames[args.method])
    teacher_module = load("eav_eeg_target_teacher", ARCHIVE / "multimodal/model/Teacher.py")

    class EEGTargetTeacher(teacher_module.TeacherModel):
        """Expose the EEG branch through the keys expected by legacy KD runners."""
        def __init__(self, *values, **kwargs):
            kwargs["dk"] = 32
            super().__init__(*values, **kwargs)

        def forward(self, *values, **kwargs):
            output = super().forward(*values, **kwargs)
            output["logits"] = output["eeg_logits"]
            output["fused"] = output["eeg_feat"]
            return output

    output = Path(args.output).resolve() / args.method / f"split{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = REPO / "eav_split_search/runs/eav" / f"split{args.split}/seed2024/checkpoints" / f"teacher_eav_split{args.split}_best.pt"
    adapted_checkpoint = output / "eeg_teacher.pt"
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    checkpoint_data["best_val_acc"] = float(checkpoint_data.get("best_eeg_acc", 0.0))
    torch.save(checkpoint_data, adapted_checkpoint)
    script.TeacherModel = EEGTargetTeacher
    if args.method == "kd":
        script.Config.device = args.device
        script.Config.save_dir = str(output)
        script.Config.teacher_ckpt = str(adapted_checkpoint)
        script.Config.student_ckpt = str(output / "best_student_kd.pth")
    else:
        script.DEVICE = args.device
        script.SAVE_DIR = str(output)
        script.TEACHER_CKPT = str(adapted_checkpoint)
        script.TEACHER_DIM = script.STUDENT_DIM
    script.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("kd", "fitnets", "nst"), required=True)
    parser.add_argument("--split", type=int, choices=(3, 25, 37), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", default="ablations/eav_supervision_target/eeg_t")
    main(parser.parse_args())
