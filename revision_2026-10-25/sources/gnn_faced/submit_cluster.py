"""Submit a preflight, bounded analysis array, and dependent validated report once."""
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
state_path = HERE / 'CLUSTER_JOBS.json'
if state_path.exists():
    raise SystemExit('Submission record already exists; inspect it before submitting again.')
(HERE / 'cluster_logs').mkdir(exist_ok=True)
state = {'project': str(HERE), 'tasks': 304, 'max_concurrent': 24, 'jobs': {}}


def submit(name, options, stage):
    cmd = ['sbatch', '--parsable', '--job-name=eeg_part_' + name,
           '--output=cluster_logs/' + name + '_%A_%a.log', *options, 'cluster.slurm', stage]
    job = subprocess.check_output(cmd, cwd=HERE, text=True).strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected sbatch response: ' + job)
    state['jobs'][name] = job
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({'stage': name, 'job': job}), flush=True)
    return job


preflight = submit('preflight', ['--time=00:15:00'], 'preflight')
array = submit('analysis', ['--dependency=afterok:' + preflight,
                           '--array=0-303%24'], 'task')
submit('report', ['--dependency=afterok:' + array, '--time=00:20:00'], 'report')
