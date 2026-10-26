"""Run a bounded, resumable local queue; each task refits all five participant folds."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
MODELS=['hybrid_gnn','sfgnn','lda_full']
SEED=20261024


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage',choices=['observed','sensitivity','null'],required=True)
    ap.add_argument('--out-dir',type=Path,default=HERE/'results')
    ap.add_argument('--cache',type=Path,default=HERE.parent/'seed7_replication/cache')
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--permutations',type=int,default=99)
    args=ap.parse_args()
    if args.workers<1 or args.threads<1:ap.error('workers and threads must be positive')
    if args.stage=='observed':tasks=[(m,SEED,0,True) for m in MODELS]
    elif args.stage=='sensitivity':tasks=[(m,s,0,True) for s in [11,29] for m in MODELS[:2]]
    else:tasks=[(m,SEED,p,False) for p in range(1,args.permutations+1) for m in MODELS]
    logdir=args.out_dir/'logs';logdir.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),MPLCONFIGDIR='/tmp/eeg_participant_mpl')
    def one(task):
        model,seed,perm,importance=task;stem=f'{model}_seed{seed}_perm{perm:03d}'
        cmd=[sys.executable,str(HERE/'run.py'),'--model',model,'--seed',str(seed),'--perm',str(perm),
             '--out-dir',str(args.out_dir),'--cache',str(args.cache),'--threads',str(args.threads)]
        if importance:cmd.append('--importance')
        t=time.time()
        with (logdir/(stem+'.log')).open('a') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        status={'task':stem,'returncode':result.returncode,'seconds':round(time.time()-t,1)}
        print(json.dumps(status),flush=True)
        return status
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,task) for task in tasks]
        for future in as_completed(futures):results.append(future.result())
    (args.out_dir/(args.stage+'_batch.json')).write_text(json.dumps(results,indent=2)+'\n')
    return 1 if any(r['returncode'] for r in results) else 0


if __name__=='__main__':
    sys.exit(main())
