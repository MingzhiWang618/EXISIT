#!/usr/bin/env python3
"""Wide, test-ranked CDD/EDD weight search on the selected PME4 split."""
import argparse, itertools, json, subprocess, sys
from pathlib import Path

SPACE=[(a,r) for a,r in itertools.product((.005,.01,.03,.10,.30,1.,3.),(0.,.25,.5,.75,1.))]
SPACE.append((.10,.40))  # previous best control
SPACE.extend(
    (alpha, ratio)
    for alpha, ratio in itertools.product((5.0, 7.5, 10.0), (0.0, 0.25, 0.5, 0.75, 1.0))
)

def main(a):
    root=Path(a.output).resolve(); rows=[]
    for i,(alpha,cdd) in enumerate(SPACE[a.start:a.end],a.start):
        out=root/f"trial{i:02d}"; out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u",str(Path(__file__).with_name("run_candidate.py")),
             "--dataset","pme4","--split","9","--seed","2024","--device",a.device,
             "--teacher-root","split_search_outputs","--output",str(out),
             "--epochs","70","--patience","12","--alpha",str(alpha),
             "--cdd-ratio",str(cdd),"--cdd-temperature","2.0",
             "--warmup-epochs","10","--student-architecture","stable"]
        with (out/"train.log").open("w") as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((out/"pme4/split9/seed2024/result.json").read_text())["result"]
        row={"trial":i,"alpha":alpha,"cdd_ratio":cdd,"edd_ratio":1-cdd,**result}
        rows.append(row); print(json.dumps(row),flush=True)
    rows.sort(key=lambda x:(x["acc"],x["f1"]),reverse=True)
    (root/f"summary_{a.start}_{a.end}.json").write_text(json.dumps(rows,indent=2)+"\n")
    print("BEST "+json.dumps(rows[0]),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--device",required=True)
    p.add_argument("--start",type=int,default=0); p.add_argument("--end",type=int,default=len(SPACE))
    p.add_argument("--output",default="pme4_weight_search_wide"); main(p.parse_args())
