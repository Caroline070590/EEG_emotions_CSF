"""One prespecified fit: training/validation only, then one final test evaluation."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from model_snapshot import EEGNet
from protocol import digest, fixed_splits, read_config, save_json, SIGNAL_NAMES, metrics


def fit_scaler(signal, rows, recipe):
    if recipe == 'trial_fourier_div100':
        return {'mean':np.zeros(30, np.float32), 'std':np.full(30,100.,np.float32),
                'constant':np.zeros(30,bool), 'fit_rows':np.array([],dtype=np.int64)}
    if recipe not in ('window_filtered_zscore','trial_fourier_zscore') or not len(rows):
        raise ValueError('Unknown recipe or empty training set')
    count = 0; mean = np.zeros(30); m2 = np.zeros(30)
    for start in range(0,len(rows),16):
        values = np.asarray(signal[rows[start:start+16]],dtype=np.float64)
        n = len(values) * values.shape[2]
        batch_mean = values.mean((0,2)); delta = batch_mean - mean
        m2 += values.var((0,2)) * n + delta * delta * count * n / (count+n)
        mean += delta * n / (count+n); count += n
    std = np.sqrt(np.maximum(m2/count,0)); constant = std <= 1e-12; std[constant] = 1
    return dict(mean=mean.astype(np.float32),std=std.astype(np.float32),constant=constant,fit_rows=np.asarray(rows))


def batch(signal, rows, scaler, device):
    x = np.array(signal[rows],dtype=np.float32,copy=True)
    x = (x-scaler['mean'][None,:,None])/scaler['std'][None,:,None]
    x[:,scaler['constant'],:] = 0
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite model input')
    return torch.from_numpy(x).to(device)


def trial_metrics(probabilities, meta, rows):
    trials, first, inverse = np.unique(meta['trial_idx'][rows],return_index=True,return_inverse=True)
    if not np.array_equal(meta['y'][rows], meta['y'][rows][first][inverse]):
        raise ValueError('Inconsistent trial labels')
    if not np.all(np.bincount(inverse)==3):
        raise ValueError('Expected exactly three windows per trial')
    sums = np.zeros((len(trials),9)); np.add.at(sums,inverse,probabilities)
    return metrics(sums/3,meta['y'][rows][first])


@torch.no_grad()
def predict(model, signal, rows, scaler, device, batch_size):
    model.eval(); probabilities=[]
    for start in range(0,len(rows),batch_size):
        probabilities.append(model(batch(signal,rows[start:start+batch_size],scaler,device)).softmax(1).cpu().numpy())
    return np.concatenate(probabilities)


def improves(current, best):
    return (current['balanced_accuracy'] > best['balanced_accuracy'] + 1e-12 or
            (abs(current['balanced_accuracy']-best['balanced_accuracy']) <= 1e-12 and
             current['cross_entropy'] < best['cross_entropy']-1e-12))


def train_and_evaluate(signal, meta, split, recipe, seed, settings, out, device):
    """No call to predict(test) occurs until the best validation state is frozen."""
    out = Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists() or (out/'test_started.json').exists():
        raise FileExistsError('An existing test evaluation must not be repeated; review incomplete outputs explicitly')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    scaler=fit_scaler(signal,split['train'],recipe)
    np.savez_compressed(out/'scaler.npz',**scaler)
    model=EEGNet(9,2000,n_channels=30).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    counts=np.bincount(meta['y'][split['train']],minlength=9)
    if np.any(counts==0): raise ValueError('Training set is missing a class')
    weights=torch.as_tensor(len(split['train'])/(9*counts),dtype=torch.float32,device=device)
    best={'balanced_accuracy':-1.,'cross_entropy':float('inf')}; selected=0; stale=0; curves=[]
    checkpoint=out/'selected.pt'
    for epoch in range(1,settings['max_epochs']+1):
        model.train(); order=np.random.default_rng(seed+epoch).permutation(split['train'])
        numerator=0.; denominator=0.
        for start in range(0,len(order),settings['batch_size']):
            rows=order[start:start+settings['batch_size']]
            y=torch.as_tensor(meta['y'][rows],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True)
            losses=F.cross_entropy(model(batch(signal,rows,scaler,device)),y,weight=weights,reduction='none')
            weight_sum=weights[y].sum(); loss=losses.sum()/weight_sum
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); optimizer.step(); model.apply_constraints()
            numerator+=float(losses.detach().sum().cpu()); denominator+=float(weight_sum.cpu())
        validation=predict(model,signal,split['validation'],scaler,device,settings['batch_size'])
        score=metrics(validation,meta['y'][split['validation']])
        if improves(score,best):
            best=score; selected=epoch; stale=0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'epoch':epoch},checkpoint)
            np.savez_compressed(out/'selected_validation.npz',probabilities=validation,
                                **{k:v[split['validation']] for k,v in meta.items()})
        else: stale+=1
        row=dict(epoch=epoch,train_loss=numerator/denominator,validation=score)
        curves.append(row); save_json(out/'curves.json',curves)
        print(json.dumps(row),flush=True)
        if epoch>=settings['min_epochs'] and stale>=settings['patience']: break
    chosen=torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(chosen['state_dict'])
    # Test is accessed for the first and only prediction pass here.
    save_json(out/'test_started.json',dict(selected_epoch=selected,recipe=recipe,seed=seed))
    probability=predict(model,signal,split['test'],scaler,device,settings['batch_size'])
    np.savez_compressed(out/'test_predictions.npz',probabilities=probability,
                        **{k:v[split['test']] for k,v in meta.items()})
    return dict(selected_epoch=selected,selection_epochs=len(curves),validation=best,
                window_test=metrics(probability,meta['y'][split['test']]),
                trial_test=trial_metrics(probability,meta,split['test']),
                roles={k:np.unique(meta['subject'][v]).tolist() for k,v in split.items()},
                test_prediction_passes=1, refit=False)


def run(config_path, task_index, device, smoke=False):
    config_path=Path(config_path); config=read_config(config_path); base=config_path.parent
    if task_index not in range(9): raise ValueError('Task index must be 0 through 8')
    task=config['tasks'][task_index]
    manifest=json.loads((base/'prepared.json').read_text())
    if manifest['config_sha256']!=digest(config_path): raise ValueError('Prepared config hash mismatch')
    signal_path=base/SIGNAL_NAMES[task['recipe']]
    if digest(signal_path)!=manifest['artifact_sha256'][signal_path.name]: raise ValueError('Signal hash mismatch')
    if digest(base/'metadata.npz')!=config['input_sha256']['metadata']: raise ValueError('Metadata hash mismatch')
    signal=np.load(signal_path,mmap_mode='r',allow_pickle=False)
    if signal.shape!=(10332,30,2000) or signal.dtype!=np.float32: raise ValueError('Prepared shape mismatch')
    with np.load(base/'metadata.npz',allow_pickle=False) as stored: meta={k:stored[k] for k in stored.files}
    split=fixed_splits(meta['subject']); settings=config['training'].copy()
    if smoke:
        # Keep the first video from each class and all three windows, one participant/role.
        split={k:np.concatenate([v[(meta['subject'][v]==meta['subject'][v][0]) &
                                  (meta['video'][v]==np.min(meta['video'][v][meta['y'][v]==c]))]
                                 for c in range(9)]) for k,v in split.items()}
        settings.update(max_epochs=1,min_epochs=1,patience=1)
    elif not str(device).startswith('cuda'):
        raise ValueError('Production fits require a GPU; use --smoke for CPU checks')
    label=f"{task['recipe']}_seed{task['seed']}"
    out=base/('smoke' if smoke else 'runs')/label
    if (out/'result.json').exists():
        result=json.loads((out/'result.json').read_text())
        if result['config_sha256']!=digest(config_path): raise ValueError('Completed result config mismatch')
        for rel,expected in result['artifact_sha256'].items():
            if digest(out/rel)!=expected: raise ValueError('Completed result artifact mismatch')
        return result
    started=time.monotonic()
    result=train_and_evaluate(signal,meta,split,task['recipe'],task['seed'],settings,out,torch.device(device))
    result.update(task,config_sha256=digest(config_path),complete=not smoke,smoke=smoke,
                  elapsed_seconds=time.monotonic()-started,
                  torch_version=torch.__version__,numpy_version=np.__version__,
                  artifact_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    save_json(out/'result.json',result)
    return result


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--task',required=True,type=int)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    run(args.config,args.task,args.device,args.smoke)
