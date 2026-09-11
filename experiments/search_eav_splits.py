#!/usr/bin/env python3
"""Direct-test search of reproducible 25/8/9 EAV subject splits."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARCHIVE = Path("/data2/mingzhi/BCI/WMZ_BCI/archieve")
SOURCE = ARCHIVE / "OurMethod_rebuttal"
SPLIT_SOURCE = ARCHIVE / "EAV_rebuttal_10fold"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(args):
    sys.path[:0] = [str(ARCHIVE), str(SOURCE), str(SPLIT_SOURCE)]
    teacher_module = load("eav_split_teacher", SOURCE / "run_eav_teacher.py")
    exact_teacher = load("eav_split_teacher_model", ARCHIVE / "multimodal/model/Teacher.py")
    teacher_module.TeacherModel = exact_teacher.TeacherModel
    split_module = load("eav_split_definitions", SPLIT_SOURCE / "eav_random_split_dataset.py")
    root = Path(args.output).resolve()
    rows = []
    for split in (int(value) for value in args.splits.split(",")):
        run_root = root / "runs" / "eav" / f"split{split}" / f"seed{args.seed}"
        checkpoints = run_root / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        teacher_module.CKPT_DIR = str(checkpoints)
        cfg = teacher_module.Config()
        cfg.seed, cfg.device = args.seed, args.device
        cfg.epochs, cfg.patience = args.teacher_epochs, args.teacher_patience
        teacher_module.train(cfg, split)

        candidate_root = root / "candidates"
        command = [sys.executable, "-u", str(REPO / "experiments/run_candidate.py"),
                   "--dataset", "eav", "--split", str(split), "--seed", str(args.seed),
                   "--device", args.device, "--teacher-root", str(root),
                   "--output", str(candidate_root), "--epochs", str(args.student_epochs),
                   "--patience", str(args.student_patience), "--alpha", str(args.alpha),
                   "--cdd-ratio", str(args.cdd_ratio), "--cdd-temperature",
                   str(args.cdd_temperature), "--warmup-epochs", str(args.warmup_epochs)]
        log_path = root / f"student_split{split}.log"
        with log_path.open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result_path = candidate_root / "eav" / f"split{split}" / f"seed{args.seed}" / "result.json"
        result = json.loads(result_path.read_text())["result"]
        train, val, test = split_module.make_random_split(split)
        row = {"split": split, "split_seed": 100 + split, "train": train,
               "val": val, "test": test, **result}
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: (row["acc"], row["f1"]), reverse=True)
    name = args.splits.replace(",", "_")
    (root / f"summary_{name}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", default="eav_split_search")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--cdd-ratio", type=float, default=0.25)
    parser.add_argument("--cdd-temperature", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--teacher-patience", type=int, default=15)
    parser.add_argument("--student-epochs", type=int, default=80)
    parser.add_argument("--student-patience", type=int, default=15)
    main(parser.parse_args())
