import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from recurrence import build,METRICS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from analysis_core import read_trial,digest
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--data-root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
qc=pd.read_csv(ROOT/'inputs/disjoint_signed/input_qc.csv').sort_values(['subject','video'])
rows=list(qc.itertuples())[a.shard::a.shards]
if a.limit:rows=rows[:a.limit]
for row in rows:
    target=a.output/f'{row.subject:03d}_{row.video:02d}.npz'
    path=Path(row.full_path)
    if a.data_root:
        marker='EEG-data-raw-multiscale'
        parts=path.parts
        path=a.data_root/Path(*parts[parts.index(marker):])
    if digest(path)!=row.full_sha256:raise ValueError('Raw source hash mismatch')
    x,meta=read_trial(path,7500)
    assert (meta['subject'],meta['video'])==(row.subject,row.video)
    gs=[];fs=[]
    for block in [x[:3500],x[4000:7500]]:
        g,f=build(block);gs.append(g);fs.append(f)
    np.savez_compressed(target,key=[row.subject,row.video],graphs=gs,metrics=fs,source_sha256=row.full_sha256,code_sha256=digest(Path(__file__).with_name('recurrence.py')))
    print(target.name,flush=True)
(a.output/f'manifest_{a.shard}.json').write_text(json.dumps({'count':len(rows),'metrics':METRICS,'windows':[[500,3000],[4500,7000]],'layers':['abs_spearman_4_45','alpha_wpli','recurrence_jaccard'],'target_eligible_recurrence_rate':.05,'embedding_m':3,'delay_ms':16,'anchor_step_ms':40,'theiler_exclude_ms':96},indent=2))
