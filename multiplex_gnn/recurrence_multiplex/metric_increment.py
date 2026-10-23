"""Exploratory RQA increment over synchrony; fixed C=1, whole-video permutations."""
from pathlib import Path
import argparse,json,sys,warnings,hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel,delayed
from statsmodels.stats.multitest import multipletests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'multiplex_prediction'))
from experiment import make_folds,LABELS
NAMES=['synchrony','rqa','synchrony_rqa','synchrony_controls','synchrony_controls_rqa']
def design(sync,controls,rqa,model,direction,cross):
    b='A' if direction==0 else 'B';r=rqa[:,1-direction if cross else direction].reshape(len(rqa),-1)
    parts={'synchrony':[sync['sync_'+b]],'rqa':[r], 'synchrony_rqa':[sync['sync_'+b],r], 'synchrony_controls':[sync['sync_'+b],controls], 'synchrony_controls_rqa':[sync['sync_'+b],controls,r]}
    return np.column_stack(parts[model])
def fit(index,keys,designs,out):
    labels=LABELS.copy() if index==0 else np.random.default_rng(np.random.SeedSequence([20261014,index])).permutation(LABELS)
    y=labels[keys[:,1]];pred=np.zeros((len(NAMES),2,len(keys),9))
    for train,test in make_folds(keys,'video',labels):
        for m,name in enumerate(NAMES):
            for direction in range(2):
                x=designs[name][direction];scaler=StandardScaler().fit(x[train])
                model=LogisticRegression(C=1,class_weight='balanced',max_iter=5000)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);model.fit(scaler.transform(x[train]),y[train])
                pred[m,direction,test]=model.predict_proba(scaler.transform(x[test]))
    assert np.isfinite(pred).all() and np.allclose(pred.sum(-1),1)
    dest=out/f'prediction_{index:04d}.npz';np.savez_compressed(dest,keys=keys,y=y,probabilities=pred,video_labels=labels)
    scores={name:float(balanced_accuracy_score(y,p.argmax(1))) for name,p in zip(NAMES,pred.mean(1))}
    scores['rqa_increment']=scores['synchrony_rqa']-scores['synchrony']
    scores['controlled_increment']=scores['synchrony_controls_rqa']-scores['synchrony_controls']
    return scores

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--synchrony',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--permutations',type=int,default=99);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    with np.load(a.synchrony) as z:sync={k:z[k] for k in ['keys','sync_A','sync_B','controls']}
    keys=sync['keys'];assert keys.shape==(3444,2) and len(np.unique(keys,axis=0))==3444
    rqa=[];expected=hashlib.sha256(Path(__file__).with_name('recurrence.py').read_bytes()).hexdigest()
    for subject,video in keys:
        with np.load(a.features/f'{subject:03d}_{video:02d}.npz') as z:
            assert np.array_equal(z['key'],[subject,video]) and str(z['code_sha256'])==expected
            rqa.append(z['metrics'])
    rqa=np.asarray(rqa);assert rqa.shape==(3444,2,30,6)
    assert all(np.isfinite(v).all() for v in sync.values()) and np.isfinite(rqa).all()
    a.output.mkdir(parents=True,exist_ok=True)
    config={'models':NAMES,'metrics':['RR','DET','LAM','mean_diagonal_length','max_diagonal_length','diagonal_entropy'],'recurrence_shape':list(rqa.shape),'permutations':a.permutations,'sync_sha256':hashlib.sha256(a.synchrony.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'primary':'same-window controlled increment','secondary':'same-window unadjusted increment and both opposite-window increments','correction':'Holm over four incremental tests','status':'exploratory; model specifications fixed before this run'}
    (a.output/'CONFIG.json').write_text(json.dumps(config,indent=2))
    tests=[]
    for cross in [False,True]:
        name='opposite_window' if cross else 'same_window';out=a.output/name;out.mkdir(exist_ok=True)
        designs={m:[design(sync,sync['controls'],rqa,m,d,cross) for d in range(2)] for m in NAMES}
        observed=fit(0,keys,designs,out);print(name,observed,flush=True)
        null=Parallel(n_jobs=a.jobs)(delayed(fit)(i,keys,designs,out) for i in range(1,a.permutations+1))
        (out/'scores.json').write_text(json.dumps([observed]+null,indent=2))
        for contrast in ['rqa_increment','controlled_increment']:
            values=np.array([r[contrast] for r in null]);tests.append(dict(window=name,contrast=contrast,gain=observed[contrast],null_mean=float(values.mean()),p=float((1+np.sum(values>=observed[contrast]))/(1+len(values)))))
    for t,q in zip(tests,multipletests([t['p'] for t in tests],method='holm')[1]):t['p_holm']=float(q)
    (a.output/'RESULTS.json').write_text(json.dumps(tests,indent=2));print(tests,flush=True)
if __name__=='__main__':main()
