"""Compare released and refined seismic responses sample by sample."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def load(path, key=None):
    if path.suffix == ".npy": return np.asarray(np.load(path))
    with np.load(path, allow_pickle=False) as data:
        name = key or (data.files[0] if len(data.files)==1 else None)
        if name is None: raise ValueError(f"Pass --key for {path}; keys={data.files}")
        return np.asarray(data[name])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--released",type=Path,required=True); p.add_argument("--refined",type=Path,required=True); p.add_argument("--released-key"); p.add_argument("--refined-key"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    x,y=load(a.released,a.released_key),load(a.refined,a.refined_key)
    if x.shape!=y.shape: raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    rows=[]
    for i,(u,v) in enumerate(zip(x.reshape(x.shape[0],-1),y.reshape(y.shape[0],-1))):
        uc,vc=u-u.mean(),v-v.mean(); r=float(np.dot(uc,vc)/max(np.linalg.norm(uc)*np.linalg.norm(vc),1e-12)); rl2=float(np.linalg.norm(u-v)/max(np.linalg.norm(v),1e-12)); rows.append({"sample":i,"pearson_r":r,"relative_l2_difference":rl2})
    keys=("pearson_r","relative_l2_difference"); result={"shape":list(x.shape),"num_samples":len(rows),"mean":{k:float(np.mean([z[k] for z in rows])) for k in keys},"std":{k:float(np.std([z[k] for z in rows],ddof=1)) if len(rows)>1 else 0.0 for k in keys},"samples":rows}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result["mean"],indent=2))
if __name__ == "__main__": main()
