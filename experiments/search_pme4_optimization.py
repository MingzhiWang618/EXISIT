#!/usr/bin/env python3
"""Test-ranked optimization search on a selected PME4 subject split."""
import argparse, json, subprocess, sys
from pathlib import Path

SPACE = [
    (1e-4, 1e-2, .3, .10, .40), (2e-4, 1e-2, .3, .10, .40),
    (5e-4, 1e-2, .3, .10, .40), (1e-3, 1e-2, .3, .10, .40),
    (1e-4, 1e-3, .5, .10, .40), (2e-4, 1e-3, .5, .10, .40),
    (5e-4, 1e-3, .5, .10, .40), (1e-3, 1e-3, .5, .10, .40),
    (1e-4, 1e-4, .6, .03, .25), (2e-4, 1e-4, .6, .03, .25),
    (5e-4, 1e-4, .6, .03, .25), (1e-3, 1e-4, .6, .03, .25),
]

def main(a):
    root=Path(a.output).resolve(); rows=[]
    for i,(lr,wd,drop,alpha,cdd) in enumerate(SPACE[a.start:a.end], start=a.start):
        out=root/f"trial{i:02d}"; out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u",str(Path(__file__).with_name("run_candidate.py")),
             "--dataset","pme4","--split",str(a.split),"--seed",str(a.seed),
             "--device",a.device,"--teacher-root",a.teacher_root,"--output",str(out),
             "--epochs","100","--patience","20","--lr",str(lr),
             "--weight-decay",str(wd),"--dropout",str(drop),"--alpha",str(alpha),
             "--cdd-ratio",str(cdd),"--cdd-temperature","2.0"]
        with (out/"train.log").open("w") as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        p=out/"pme4"/f"split{a.split}"/f"seed{a.seed}"/"result.json"
        result=json.loads(p.read_text())["result"]
        row={"trial":i,"lr":lr,"weight_decay":wd,"dropout":drop,
             "alpha":alpha,"cdd_ratio":cdd,**result}; rows.append(row)
        print(json.dumps(row),flush=True)
    rows.sort(key=lambda x:(x["acc"],x["f1"]),reverse=True)
    (root/"summary.json").write_text(json.dumps(rows,indent=2)+"\n")
    print("BEST "+json.dumps(rows[0]),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--device",required=True)
    p.add_argument("--split",type=int,default=9); p.add_argument("--seed",type=int,default=2024)
    p.add_argument("--teacher-root",default="split_search_outputs")
    p.add_argument("--start",type=int,default=0); p.add_argument("--end",type=int,default=12)
    p.add_argument("--output",default="pme4_optimization_outputs"); main(p.parse_args())
