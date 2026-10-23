"""Fixed-setting ablations; whole-video label permutations, full refitting."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'multiplex_prediction'))
from experiment import make_folds,LABELS
class Model(nn.Module):
    def __init__(self,layers,metrics):
        super().__init__();self.layers=layers;self.metrics=metrics
        self.first=nn.ModuleList([nn.Linear(36 if metrics else 30,16) for _ in layers]);self.second=nn.ModuleList([nn.Linear(16,16) for _ in layers])
        self.head=nn.Sequential(nn.Linear(16*len(layers),32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,9))
    def forward(self,g,x):
        # Average node states across corresponding layers: explicit interlayer sensor coupling.
        hs=[torch.relu(f(g[:,k]@x)) for k,f in zip(self.layers,self.first)]
        shared=torch.stack(hs).mean(0)
        hs=[torch.relu(f(g[:,k]@((h+shared)/2))).mean(1) for k,f,h in zip(self.layers,self.second,hs)]
        return self.head(torch.cat(hs,1))
def scramble_recurrence(graphs,keys,seed=20261015):
    """Permute recurrence node identities only; preserve weighted graph structure."""
    result=graphs.copy()
    for i,(subject,video) in enumerate(keys):
        for window in range(2):
            order=np.random.default_rng(np.random.SeedSequence([seed,int(subject),int(video),window])).permutation(30)
            result[i,window,2]=graphs[i,window,2][np.ix_(order,order)]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--permutation',type=int,default=0);p.add_argument('--epochs',type=int,default=40);p.add_argument('--seeds',type=int,nargs='+',default=[11,29,47]);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    files=sorted(a.features.glob('*.npz'));data=[]
    for f in files:
        with np.load(f) as item:
            data.append({key:item[key] for key in item.files})
    print(f'Loaded {len(data)} trial feature files; device={device}',flush=True)
    keys=np.array([d['key'] for d in data]);g=np.array([d['graphs'] for d in data]);r=np.array([d['metrics'] for d in data])
    expected=hashlib.sha256(Path(__file__).with_name('recurrence.py').read_bytes()).hexdigest()
    assert all(str(d['code_sha256'])==expected for d in data), 'Mixed or stale extraction code'
    assert len(np.unique(keys,axis=0))==len(keys)
    if not a.smoke:assert keys.shape==(3444,2)
    assert np.isfinite(g).all() and np.isfinite(r).all()
    labels=LABELS.copy()
    if a.permutation:labels=np.random.default_rng(np.random.SeedSequence([20261013,a.permutation])).permutation(labels)
    y=labels[keys[:,1]];folds=make_folds(keys,'video',labels)
    g=scramble_recurrence(g,keys)
    variants={'multiplex_scrambled':((0,1,2),True)}
    out={};scores={}
    for name,(layers,metrics) in variants.items():
        ensemble=[]
        for seed in a.seeds:
            for window in range(2):
                pred=np.zeros((len(keys),9))
                for train,test in folds:
                    torch.manual_seed(seed);rng=np.random.default_rng(seed)
                    mu=r[train,window].mean(0);sd=np.maximum(r[train,window].std(0),1e-6)
                    identity=np.broadcast_to(np.eye(30),(len(keys),30,30))
                    x=np.concatenate([identity,(r[:,window]-mu)/sd],axis=-1) if metrics else identity.copy()
                    w=g[:,window]+np.eye(30)[None,None];degree=w.sum(-1);w=w/np.sqrt(degree[:,:,:,None]*degree[:,:,None,:])
                    X=torch.tensor(x,dtype=torch.float32,device=device);G=torch.tensor(w,dtype=torch.float32,device=device);Y=torch.tensor(y,dtype=torch.long,device=device)
                    model=Model(layers,metrics).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                    weights=len(train)/(9*np.bincount(y[train],minlength=9));lossfun=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
                    for epoch in range(a.epochs):
                        model.train()
                        for ids in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/128)))):
                            opt.zero_grad();loss=lossfun(model(G[ids],X[ids]),Y[ids]);loss.backward();opt.step()
                            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    model.eval()
                    with torch.no_grad():pred[test]=model(G[test],X[test]).softmax(-1).cpu().numpy()
                ensemble.append(pred)
        out[name]=np.mean(ensemble,axis=0);scores[name]=float(balanced_accuracy_score(y,out[name].argmax(1)))
        print(name,scores[name],flush=True)
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/f'prediction_{a.permutation:04d}.npz',keys=keys,y=y,**out)
    (a.output/f'scores_{a.permutation:04d}.json').write_text(json.dumps({'scores':scores,'control':'recurrence graph node identities permuted independently per trial/window; metrics and other layers unchanged','config':vars(a)|{'features':str(a.features),'output':str(a.output)},'device':device},indent=2))
if __name__=='__main__':main()
