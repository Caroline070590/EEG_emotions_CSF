#!/usr/bin/env python3
"""Matched baseline/GNN experiments with participant, stimulus, and joint holdout."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from model import LAYERS,matrices,GraphTransform,MultiLayerGNN,FeatureMLP,shapley_from_coalitions

ROOT=Path(__file__).resolve().parents[1]
EMOTIONS=['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']
LABELS=np.repeat(np.arange(9),[3,3,3,3,4,3,3,3,3])
MODELS=['logistic_spearman','logistic_mtf','logistic_qg','logistic_fused','mlp_fused',
        'gnn_spearman','gnn_mtf','gnn_qg','gnn_no_interaction','gnn_multilayer']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path,subjects=None):
    d=np.load(path);keys=d['keys'].astype(int)
    if keys.shape!=(3444,2) or len(np.unique(keys,axis=0))!=3444:raise ValueError('Expected unique 3444 subject/video keys')
    for s in np.unique(keys[:,0]):
        if set(keys[keys[:,0]==s,1])!=set(range(28)):raise ValueError('Incomplete video coverage')
    V=np.stack([d[k] for k in LAYERS],1).astype(np.float32)
    if V.shape!=(3444,3,435) or not np.isfinite(V).all() or V.min()<0 or V.max()>1.00001:raise ValueError('Invalid weighted matrices')
    if subjects is not None:
        keep=keys[:,0]<subjects;V=V[keep];keys=keys[keep]
    return V,keys


def make_folds(keys,protocol,video_labels=LABELS):
    vf=np.empty(28,int)
    for category in range(9):
        ids=np.flatnonzero(video_labels==category);vf[ids]=np.arange(len(ids))%3
    subjects=np.unique(keys[:,0]);shuffled=np.random.default_rng(731).permutation(subjects)
    mapping={s:i%3 for i,s in enumerate(shuffled)}
    sf=np.array([mapping[s] for s in keys[:,0]])
    folds=[]
    combinations=[(i,None) for i in range(3)] if protocol=='participant' else [(None,j) for j in range(3)] if protocol=='video' else [(i,j) for i in range(3) for j in range(3)]
    coverage=np.zeros(len(keys),int)
    for sid,vid in combinations:
        train=np.ones(len(keys),bool);test=np.ones(len(keys),bool)
        if sid is not None:train &= sf!=sid;test &= sf==sid
        if vid is not None:train &= vf[keys[:,1]]!=vid;test &= vf[keys[:,1]]==vid
        if not train.any() or not test.any():raise ValueError('Too few participants for chosen folds')
        if sid is not None and set(keys[train,0]) & set(keys[test,0]):raise AssertionError('Participant leakage')
        if vid is not None and set(keys[train,1]) & set(keys[test,1]):raise AssertionError('Video leakage')
        if set(video_labels[keys[train,1]])!=set(range(9)):raise ValueError('Missing training category')
        coverage[test]+=1;folds.append((np.flatnonzero(train),np.flatnonzero(test)))
    if not np.all(coverage==1):raise AssertionError('Each observation must be tested exactly once')
    return folds


def predict(model,inputs,batch=512):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(inputs[0]),batch):
            out.append(model(*(x[start:start+batch] for x in inputs)).softmax(-1).numpy())
    return np.concatenate(out)


def fit_predict(name,V,keys,train,test,labels,seed,args,checkpoint):
    y=labels[keys[:,1]];torch.manual_seed(seed);rng=np.random.default_rng(seed)
    scaler=None;graph_transform=None
    if name.startswith('logistic') or name=='mlp_fused':
        layer=name.split('_')[-1]
        flat=V.reshape(len(V),-1) if layer=='fused' else V[:,LAYERS.index(layer)]
        scaler=StandardScaler().fit(flat[train]);X=scaler.transform(flat).astype(np.float32)
        if name.startswith('logistic'):
            model=LogisticRegression(C=1.,class_weight='balanced',max_iter=3000,solver='lbfgs',random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);model.fit(X[train],y[train])
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):raise RuntimeError('Logistic regression did not converge')
            probs=model.predict_proba(X[test])
            if args.save_models:joblib.dump(dict(model=model,scaler=scaler),checkpoint.with_suffix('.joblib'))
            return probs,None,dict(parameters=int(model.coef_.size+model.intercept_.size),train_loss=np.nan)
        model=FeatureMLP();inputs=(torch.from_numpy(X),)
    else:
        A=matrices(V);graph_transform=GraphTransform().fit(A[train]);inputs=graph_transform.transform(A)
        layers=(LAYERS.index(name[4:]),) if name[4:] in LAYERS else (0,1,2)
        model=MultiLayerGNN(layers=layers,hidden=args.hidden,interaction=name=='gnn_multilayer')
    weights=len(train)/(9*np.bincount(y[train],minlength=9))
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    target=torch.tensor(y,dtype=torch.long)
    # Fixed training duration; neither test outcomes nor test losses select epochs.
    for epoch in range(args.epochs):
        model.train();total=0.
        shuffled=rng.permutation(train)
        for start in range(0,len(train),args.batch):
            idx=shuffled[start:start+args.batch]
            optimizer.zero_grad();logits=model(*(x[idx] for x in inputs))
            loss=criterion(logits,target[idx]);loss.backward();optimizer.step()
            total+=float(loss.detach())*len(idx)
        total/=len(train)
        if not np.isfinite(total):raise RuntimeError('Nonfinite training loss')
    probs=predict(model,tuple(x[test] for x in inputs));attribution=None
    if name=='gnn_multilayer' and args.attributions:
        values=[]
        for mask in range(8):
            replaced=A[test].copy()
            for layer in range(3):
                if not mask & (1<<layer):replaced[:,layer]=graph_transform.mean[layer]
            prob=predict(model,graph_transform.transform(replaced))
            values.append(prob[np.arange(len(test)),y[test]])
        coalitions=np.stack(values,axis=1);phi=shapley_from_coalitions(coalitions)
        if not np.allclose(phi.sum(1),coalitions[:,-1]-coalitions[:,0],atol=1e-6):raise AssertionError('Shapley efficiency failed')
        strength=A[test].mean(axis=(2,3))
        training_strength=A[train].mean(axis=(2,3))
        color=(strength-training_strength.mean(0))/np.maximum(training_strength.std(0),1e-8)
        attribution=dict(phi=phi,coalitions=coalitions,strength_z=color)
    if args.save_models:
        torch.save(dict(state_dict=model.state_dict(),model=name,hidden=args.hidden,
                        graph_transform=graph_transform,scaler=scaler,seed=seed,
                        train_keys=keys[train],test_keys=keys[test]),checkpoint.with_suffix('.pt'))
    return probs,attribution,dict(parameters=sum(p.numel() for p in model.parameters()),train_loss=total)


def run_one(name,protocol,seed,V,keys,labels,args,tag='observed'):
    directory=args.out/'checkpoints'/tag;directory.mkdir(parents=True,exist_ok=True)
    all_probs=np.zeros((len(V),9));fold_id=np.empty(len(V),int);attributes=[];logs=[]
    for fold,(train,test) in enumerate(make_folds(keys,protocol,labels)):
        path=directory/f'{protocol}_{name}_seed{seed}_fold{fold}.npz'
        if path.exists():
            saved=np.load(path);probs=saved['probabilities']
            if not np.array_equal(saved['test'],test):raise ValueError('Checkpoint fold mismatch')
            attr={k:saved[k] for k in ['phi','coalitions','strength_z']} if 'phi' in saved else None
            meta=json.loads(str(saved['metadata']))
        else:
            before=time.time();probs,attr,meta=fit_predict(name,V,keys,train,test,labels,seed,args,path)
            meta.update(seconds=time.time()-before,n_train=len(train),n_test=len(test),
                        train_subjects=len(np.unique(keys[train,0])),test_subjects=len(np.unique(keys[test,0])),
                        train_videos=len(np.unique(keys[train,1])),test_videos=len(np.unique(keys[test,1])))
            payload=dict(test=test,probabilities=probs,metadata=json.dumps(meta))
            if attr is not None:payload.update(attr)
            temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**payload);os.replace(temp,path)
            print(f'{tag} {protocol} {name} seed={seed} fold={fold} {meta["seconds"]:.1f}s',flush=True)
        all_probs[test]=probs;fold_id[test]=fold
        logs.append(dict(protocol=protocol,model=name,seed=seed,fold=fold,tag=tag,**meta))
        if attr is not None:attributes.append((test,attr))
    return all_probs,fold_id,attributes,logs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vectors',type=Path,default=ROOT/'rerun_fulltrial_q30_2026-09-02/outputs/trial_matrix_vectors_q30_fulltrial.npz')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    p.add_argument('--protocols',nargs='+',choices=['participant','video','joint'],default=['participant','video','joint'])
    p.add_argument('--seeds',type=int,nargs='+',default=[11,29,47])
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--hidden',type=int,default=16)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--subjects',type=int,help='Subset for debugging only')
    p.add_argument('--permutations',type=int,default=0,help='Video-protocol nulls for logistic_fused and gnn_multilayer; train whole pipeline again')
    p.add_argument('--attributions',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--save-models',action=argparse.BooleanOptionalAction,default=True)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    if a.hidden%2:p.error('Hidden size must be even')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='threads'}
    config.update(vectors_sha256=sha(a.vectors),code_hashes={f:sha(Path(__file__).parent/f) for f in ['experiment.py','model.py']},
                  lineage='audited_full30_q30_weighted_graphs',scope='full_dataset' if a.subjects is None else 'smoke_not_paper_results')
    path=a.out/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Configuration/code changed: use a new output path')
    path.write_text(json.dumps(config,indent=2))
    V,keys=load_data(a.vectors,a.subjects);logs=[]
    (a.out/'predictions').mkdir(exist_ok=True)
    for protocol in a.protocols:
        for name in a.models:
            for seed in a.seeds:
                probs,folds,attrs,diagnostics=run_one(name,protocol,seed,V,keys,LABELS,a)
                logs.extend(diagnostics)
                payload=dict(keys=keys,y=LABELS[keys[:,1]],probabilities=probs,fold=folds)
                if attrs:
                    for field,width in [('phi',3),('coalitions',8),('strength_z',3)]:
                        result=np.empty((len(V),width))
                        for idx,attr in attrs:result[idx]=attr[field]
                        payload[field]=result
                np.savez_compressed(a.out/'predictions'/f'{protocol}_{name}_seed{seed}.npz',**payload)
                pd.DataFrame(logs).to_csv(a.out/'fold_audit.csv',index=False)
    if a.permutations:
        from sklearn.metrics import balanced_accuracy_score
        rng=np.random.default_rng(20261012);null=[]
        for b in range(a.permutations):
            shuffled=rng.permutation(LABELS)
            for name in ['logistic_fused','gnn_multilayer']:
                predictions=[]
                # Same seed averaging as the observed statistic, same fixed training schedule.
                for seed in a.seeds:
                    old_attr=a.attributions;old_save=a.save_models;a.attributions=False;a.save_models=False
                    probs,_,_,_=run_one(name,'video',seed,V,keys,shuffled,a,tag=f'perm{b:04d}')
                    a.attributions=old_attr;a.save_models=old_save;predictions.append(probs)
                accuracy=balanced_accuracy_score(shuffled[keys[:,1]],np.mean(predictions,axis=0).argmax(1))
                null.append(dict(permutation=b,model=name,balanced_accuracy=accuracy))
                pd.DataFrame(null).to_csv(a.out/'permutation_null.csv',index=False)
    (a.out/'COMPLETE.json').write_text(json.dumps(dict(observed_complete=True,permutations=a.permutations,scope=config['scope']),indent=2))
    print('Experiment complete',flush=True)


if __name__=='__main__':main()
