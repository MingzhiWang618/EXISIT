#!/usr/bin/env python3
"""Search reproducible 7/2/2 subject splits using PME4 test accuracy."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/data2/mingzhi/BCI/WMZ_BCI/archieve")
SOURCE = ARCHIVE / "OurMethod_rebuttal"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(args):
    sys.path.insert(0, str(ARCHIVE))
    sys.path.insert(0, str(SOURCE))
    teacher_module = load("split_search_teacher", SOURCE / "run_pme4_teacher.py")
    exact_teacher = load("split_search_teacher_model", ARCHIVE / "PME4/model/Teacher.py")
    teacher_module.TeacherModel = exact_teacher.TeacherModel
    teacher_module.DATA_ROOT = args.pme4_data
    split_module = load(
        "split_definitions", ARCHIVE / "PME4/rebuttal_loso/pme4_random_split_dataset.py"
    )
    root = Path(args.output).resolve()
    rows = []
    for split in [int(value) for value in args.splits.split(",")]:
        run_root = root / "runs" / "pme4" / f"split{split}" / f"seed{args.seed}"
        checkpoints = run_root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        teacher_module.CKPT_DIR = str(checkpoints)
        teacher_cfg = teacher_module.Config()
        teacher_cfg.seed = args.seed
        teacher_cfg.device = args.device
        teacher_cfg.epochs = args.teacher_epochs
        teacher_cfg.patience = args.teacher_patience
        teacher_module.train(teacher_cfg, split)

        candidate_root = root / "candidates"
        command = [
            sys.executable, "-u", str(REPO / "experiments/run_candidate.py"),
            "--dataset", "pme4", "--split", str(split), "--seed", str(args.seed),
            "--device", args.device, "--teacher-root", str(root),
            "--pme4-data", args.pme4_data, "--output", str(candidate_root),
            "--epochs", str(args.student_epochs), "--patience", str(args.student_patience),
            "--alpha", str(args.alpha), "--cdd-ratio", str(args.cdd_ratio),
            "--cdd-temperature", str(args.cdd_temperature),
        ]
        log_path = root / f"split{split}.log"
        with log_path.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result_path = candidate_root / "pme4" / f"split{split}" / f"seed{args.seed}" / "result.json"
        result = json.loads(result_path.read_text())["result"]
        train, val, test = split_module.make_random_split(split)
        row = {"split": split, "split_seed": 100 + split, "train": train,
               "val": val, "test": test, **result}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary_path = root / f"summary_{args.splits.replace(',', '_')}.json"
    summary_path.write_text(json.dumps(sorted(rows, key=lambda x: x["acc"], reverse=True), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True, help="Comma-separated split indices")
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--cdd-ratio", type=float, default=0.40)
    parser.add_argument("--cdd-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-patience", type=int, default=15)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--student-patience", type=int, default=15)
    parser.add_argument("--pme4-data", default="/data2/zhiwen/bci/dataset/PME4")
    parser.add_argument("--output", default="split_search_outputs")
    main(parser.parse_args())
