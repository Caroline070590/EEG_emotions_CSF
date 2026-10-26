"""Submit bounded feature/GPU arrays with validation dependencies, exactly once."""
import argparse
import json
from pathlib import Path
import subprocess
from common import HERE, atomic_json


def main(setup_job, probe_job=None):
    record = HERE / 'CLUSTER_JOBS.json'
    if record.exists():
        raise SystemExit('Submission record exists; inspect it before submitting more jobs.')
    (HERE / 'logs').mkdir(exist_ok=True)
    state = dict(project=str(HERE), feature_tasks=256, gpu_fold_tasks=300,
                 max_feature_concurrent=32, max_gpu_concurrent=8,
                 jobs={'gpu_environment': str(setup_job)})
    atomic_json(record,state)

    def submit(stage, dependency=None, array=None, gpu=False, hours=1, memory='8G'):
        cmd = ['sbatch', '--parsable', '--job-name=eegnet_' + stage,
               '--output=logs/' + stage + '_%A_%a.out',
               '--partition=' + ('shared-gpu' if gpu else 'shared-cpu'),
               '--cpus-per-task=2', '--mem=' + memory, f'--time={hours:02d}:00:00']
        if dependency: cmd += ['--dependency=afterok:' + dependency]
        if array: cmd += ['--array=' + array]
        if gpu: cmd += ['--gres=gpu:1']
        cmd += ['cluster.slurm', stage]
        job = subprocess.check_output(cmd, cwd=HERE, text=True).strip().split(';')[0]
        if not job.isdigit(): raise ValueError('Unexpected sbatch result: ' + job)
        state['jobs'][stage] = job; atomic_json(record,state)
        print(json.dumps(dict(stage=stage, job=job)),flush=True)
        return job

    preflight = submit('preflight')
    if probe_job:
        probe = str(probe_job)
        state['jobs']['probe'] = probe; atomic_json(record,state)
    else:
        probe = submit('probe', str(setup_job), gpu=True, memory='12G')
    features = submit('feature', preflight, '0-255%32', hours=2)
    merge = submit('merge', features, '0-3%4', memory='12G')
    configured = submit('configure', merge)
    smoke = submit('smoke', configured + ':' + probe, gpu=True, memory='16G')
    trained = submit('train', smoke, '0-299%8', gpu=True, hours=8, memory='16G')
    submit('report', trained)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-job', type=int, required=True)
    parser.add_argument('--probe-job', type=int)
    args = parser.parse_args()
    main(args.setup_job, args.probe_job)
