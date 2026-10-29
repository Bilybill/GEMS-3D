"""Aggregate repeated OpenFWI benchmark JSON files by model."""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__); p.add_argument("inputs",type=Path,nargs="+"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
rows=[json.loads(x.read_text()) for x in a.inputs]
grouped=defaultdict(list)
for row in rows:
    key=row.get("model_key",row.get("model","fno").split()[-1].lower())
    grouped[key].append(row)

models={}
for key,model_rows in sorted(grouped.items()):
    model_rows=sorted(model_rows,key=lambda item:item["seed"])
    pr=np.array([x["heldout_test"]["pearson_r"]["mean"] for x in model_rows])
    rl=np.array([x["heldout_test"]["relative_l2"]["mean"] for x in model_rows])
    models[key]={"model":model_rows[0]["model"],"parameter_count":model_rows[0].get("parameter_count"),"num_seeds":len(model_rows),"seeds":[x["seed"] for x in model_rows],"pearson_r":{"mean":float(pr.mean()),"std_across_seeds":float(pr.std(ddof=1)) if len(pr)>1 else 0.0},"relative_l2":{"mean":float(rl.mean()),"std_across_seeds":float(rl.std(ddof=1)) if len(rl)>1 else 0.0},"runs":[{"seed":x["seed"],"pearson_r":float(pr[i]),"relative_l2":float(rl[i])} for i,x in enumerate(model_rows)]}

out={"dataset":"OpenFWI CurveFault-B","setting":rows[0].get("setting"),"protocol":rows[0]["split"],"models":models}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2),encoding="utf-8")
s=out["setting"]
lines=["| Model | Parameters | Pearson | Relative L2 |","|---|---:|---:|---:|"]
for key,result in models.items():
    lines.append(f"| {key.upper()} | {result['parameter_count'] or 'n/a'} | {result['pearson_r']['mean']:.3f} +/- {result['pearson_r']['std_across_seeds']:.3f} | {result['relative_l2']['mean']:.3f} +/- {result['relative_l2']['std_across_seeds']:.3f} |")
summary="\n".join(lines)
print(summary)
a.output.with_suffix(".md").write_text(summary + "\n",encoding="utf-8")
