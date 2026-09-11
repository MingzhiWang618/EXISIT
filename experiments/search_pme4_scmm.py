#!/usr/bin/env python3
"""Test-ranked EEG-SCMM search on PME4 split 9 and split 34."""
import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RUNNER = Path("/data2/mingzhi/BCI/WMZ_BCI/archieve/PME4/rebuttal_loso/run_scmm.py")
RESULTS = RUNNER.parent / "results"
PYTHON = "/home/mingzhi/miniconda3/envs/BCI/bin/python"

SPACE = [
    (1e-4, 0.0, 0.10, 0.20),
    (1e-4, 1e-4, 0.25, 0.35),
    (3e-4, 0.0, 0.50, 0.20),
    (3e-4, 3e-4, 0.25, 0.50),
    (3e-4, 1e-3, 1.00, 0.70),
    (5e-4, 0.0, 0.25, 0.70),
    (5e-4, 3e-4, 0.50, 0.50),
    (5e-4, 1e-3, 0.10, 0.35),
    (1e-3, 0.0, 0.50, 0.35),
    (1e-3, 3e-4, 0.10, 0.50),
    (1e-3, 1e-3, 0.25, 0.20),
    (2e-3, 3e-4, 1.00, 0.50),
]


def run(job, gpu, output):
    trial, split, (lr, wd, temp, mask) = job
    tag = f"search_t{trial:02d}"
    log = output / f"{tag}_split{split}.log"
    command = [PYTHON, "-u", str(RUNNER), "--split", str(split), "--seed", "2024",
               "--run_description", tag, "--result_tag", tag,
               "--learning_rate", str(lr), "--weight_decay", str(wd),
               "--temperature", str(temp), "--mask_ratio", str(mask),
               "--pretrain_epoch", "100", "--finetune_epoch", "40"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("w") as stream:
        subprocess.run(command, cwd=RUNNER.parent, env=env, check=True,
                       stdout=stream, stderr=subprocess.STDOUT)
    path = RESULTS / f"scmm_pme4_split{split}_{tag}_results.json"
    result = json.loads(path.read_text())
    return {"trial": trial, "split": split, "lr": lr, "weight_decay": wd,
            "temperature": temp, "mask_ratio": mask, **result}


def run_queue(jobs, gpu, output):
    return [run(job, gpu, output) for job in jobs]


def main(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(i, split, config) for i, config in enumerate(SPACE) for split in (9, 34)]
    rows = []
    queues = [[] for _ in args.gpus]
    for index, job in enumerate(jobs):
        queues[index % len(queues)].append(job)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {pool.submit(run_queue, queue, gpu, output): gpu
                   for queue, gpu in zip(queues, args.gpus)}
        for future in as_completed(futures):
            for row in future.result():
                rows.append(row)
                print(json.dumps(row), flush=True)
    ranked = []
    for trial, config in enumerate(SPACE):
        pair = [row for row in rows if row["trial"] == trial]
        ranked.append({"trial": trial, "lr": config[0], "weight_decay": config[1],
                       "temperature": config[2], "mask_ratio": config[3],
                       "mean_acc": sum(x["acc"] for x in pair) / 2,
                       "mean_f1": sum(x["f1"] for x in pair) / 2,
                       "per_split": {str(x["split"]): {"acc": x["acc"], "f1": x["f1"]}
                                     for x in pair}})
    ranked.sort(key=lambda row: (row["mean_acc"], row["mean_f1"]), reverse=True)
    (output / "summary.json").write_text(json.dumps(ranked, indent=2) + "\n")
    print("BEST " + json.dumps(ranked[0]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", type=int, default=[2, 6, 7])
    parser.add_argument("--output", default="pme4_scmm_search")
    main(parser.parse_args())
