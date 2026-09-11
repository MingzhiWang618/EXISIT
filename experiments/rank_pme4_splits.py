#!/usr/bin/env python3
"""Rank existing PME4 subject splits with one fixed EXIST configuration."""
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPLIT_FILE = Path(
    "/data2/mingzhi/BCI/WMZ_BCI/archieve/PME4/rebuttal_loso/"
    "pme4_random_split_dataset.py"
)


def load_splits():
    spec = importlib.util.spec_from_file_location("pme4_splits", SPLIT_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(args):
    split_module = load_splits()
    root = Path(args.output).resolve()
    rows = []
    for split in (int(value) for value in args.splits.split(",")):
        out = root / f"split{split}"
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-u", str(REPO / "experiments/run_candidate.py"),
            "--dataset", "pme4", "--split", str(split), "--seed", str(args.seed),
            "--device", args.device, "--teacher-root", args.teacher_root,
            "--output", str(out), "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--alpha", str(args.alpha),
            "--cdd-ratio", str(args.cdd_ratio), "--cdd-temperature",
            str(args.cdd_temperature), "--warmup-epochs", str(args.warmup_epochs),
            "--student-architecture", "stable",
        ]
        with (out / "train.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result_file = out / "pme4" / f"split{split}" / f"seed{args.seed}" / "result.json"
        result = json.loads(result_file.read_text())["result"]
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
    parser.add_argument("--teacher-root", default="split_search_outputs")
    parser.add_argument("--output", default="pme4_split_ranking")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--cdd-ratio", type=float, default=1.0)
    parser.add_argument("--cdd-temperature", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    main(parser.parse_args())
