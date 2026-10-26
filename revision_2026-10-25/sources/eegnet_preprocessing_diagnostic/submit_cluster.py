"""Submit the reviewed isolated nine-fit audit once, with bounded concurrency."""
import json
from pathlib import Path
import subprocess
from protocol import HERE, save_json, sources


def submit_pipeline():
    record=HERE/'CLUSTER_JOBS.json'
    # Exclusive creation also blocks concurrent submitters and partial resubmission.
    state={'project':str(HERE),'gpu_tasks':9,'max_gpu_concurrent':2,
           'source_sha256':sources(),'jobs':{},'submission_complete':False}
    with record.open('x') as handle:
        json.dump(state,handle,indent=2,sort_keys=True); handle.write('\n')
    (HERE/'logs').mkdir(exist_ok=True)
    previous=None
    for stage,gpu,memory,hours,array in [('prepare',False,'8G',1,None),
                                        ('smoke',True,'12G',1,None),
                                        ('train',True,'16G',2,'0-8%2'),
                                        ('report',False,'8G',1,None)]:
        command=['sbatch','--parsable','--job-name=eegnet_audit_'+stage,
                 '--output=logs/'+stage+'_%A_%a.out',
                 '--partition='+('shared-gpu' if gpu else 'shared-cpu'),
                 '--cpus-per-task=2','--mem='+memory,f'--time={hours:02d}:00:00']
        if previous: command.append('--dependency=afterok:'+previous)
        if gpu: command.append('--gres=gpu:1')
        if array: command.append('--array='+array)
        command+=['cluster.slurm',stage]
        # Persist the intended command before sbatch; a partial record requires review.
        state['pending_submission']={'stage':stage,'command':command}; save_json(record,state)
        job=subprocess.check_output(command,cwd=HERE,text=True).strip().split(';')[0]
        if not job.isdigit(): raise ValueError('Unexpected sbatch response: '+job)
        state['jobs'][stage]=job; state.pop('pending_submission',None); save_json(record,state)
        print(json.dumps(dict(stage=stage,job=job)),flush=True)
        previous=job
    state['submission_complete']=True; save_json(record,state)
    return state


if __name__=='__main__':
    submit_pipeline()
