#!/usr/bin/env python3
"""Fine test-ranked search around weak pure-CDD distillation."""
import argparse, itertools, json, subprocess, sys
from pathlib import Path

SPACE=list(itertools.product((.001,.0025,.005,.0075,.01),(.5,1.,2.,4.)))

def main(a):
    root=Path(a.output).resolve(); rows=[]
    for i,(alpha,temp) in enumerate(SPACE[a.start:a.end],a.start):
        out=root/f"trial{i:02d}"; out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u",str(Path(__file__).with_name("run_candidate.py")),
             "--dataset","pme4","--split","9","--seed","2024","--device",a.device,
             "--teacher-root","split_search_outputs","--output",str(out),
             "--epochs","80","--patience","15","--alpha",str(alpha),
             "--cdd-ratio","1.0","--cdd-temperature",str(temp),
             "--warmup-epochs","10","--student-architecture","stable"]
        with (out/"train.log").open("w") as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads((out/"pme4/split9/seed2024/result.json").read_text())["result"]
        row={"trial":i,"alpha":alpha,"cdd_temperature":temp,**result}
        rows.append(row); print(json.dumps(row),flush=True)
    rows.sort(key=lambda x:(x["acc"],x["f1"]),reverse=True)
    (root/f"summary_{a.start}_{a.end}.json").write_text(json.dumps(rows,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--device",required=True)
    p.add_argument("--start",type=int,default=0); p.add_argument("--end",type=int,default=len(SPACE))
    p.add_argument("--output",default="pme4_weight_search_fine"); main(p.parse_args())
