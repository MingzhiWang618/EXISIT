#!/usr/bin/env python3
"""Validation-ranked coarse search for stable CDD/EDD hyperparameters."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


SEARCH_SPACE = [
    {"alpha": 0.10, "cdd_ratio": 0.25, "cdd_temperature": 2.0},
    {"alpha": 0.10, "cdd_ratio": 0.50, "cdd_temperature": 2.0},
    {"alpha": 0.25, "cdd_ratio": 0.25, "cdd_temperature": 2.0},
    {"alpha": 0.25, "cdd_ratio": 0.50, "cdd_temperature": 2.0},
    {"alpha": 0.25, "cdd_ratio": 0.75, "cdd_temperature": 2.0},
    {"alpha": 0.50, "cdd_ratio": 0.25, "cdd_temperature": 2.0},
    {"alpha": 0.25, "cdd_ratio": 0.50, "cdd_temperature": 1.0},
    {"alpha": 0.25, "cdd_ratio": 0.50, "cdd_temperature": 4.0},
]


def main(args):
    root = Path(args.output).resolve() / args.dataset
    rows = []
    for index, config in enumerate(SEARCH_SPACE):
        run_dir = root / f"trial{index:02d}"
        result_file = run_dir / args.dataset / f"split{args.split}" / f"seed{args.seed}" / "result.json"
        command = [
            sys.executable, "-u", str(Path(__file__).with_name("run_candidate.py")),
            "--dataset", args.dataset, "--split", str(args.split), "--seed", str(args.seed),
            "--device", args.device, "--output", str(run_dir), "--epochs", str(args.epochs),
            "--patience", str(args.patience), "--alpha", str(config["alpha"]),
            "--cdd-ratio", str(config["cdd_ratio"]),
            "--cdd-temperature", str(config["cdd_temperature"]),
        ]
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "train.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        payload = json.loads(result_file.read_text())
        rows.append({"trial": index, **config, **payload["result"]})
        print(json.dumps(rows[-1]), flush=True)
    rows.sort(key=lambda row: row["best_val_acc"], reverse=True)
    summary = {"selection_metric": "best_val_acc", "dataset": args.dataset, "rows": rows}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("BEST " + json.dumps(rows[0]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("eav", "pme4"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output", default="search_outputs")
    main(parser.parse_args())
