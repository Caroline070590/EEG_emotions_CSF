"""Multiplex-inspired graph-level fusion; no sensor/state node correspondence."""
import itertools
import math
import numpy as np
import torch
from torch import nn

LAYERS=['spearman','mtf','qg']


def matrices(vectors):
    n=len(vectors);A=np.zeros((n,3,30,30),np.float32);tri=np.triu_indices(30,1)
    for k in range(3):
        A[:,k,tri[0],tri[1]]=vectors[:,k]
        A[:,k]=A[:,k]+A[:,k].transpose(0,2,1)
    return A


class GraphTransform:
    """Only training data determine scales and replacement graph baselines."""
    def fit(self,A):
        self.mean=A.mean(0)
        self.scale=np.maximum(A.mean(axis=(0,2,3)),1e-8)
        strength=A.mean(-1)
        self.strength_mean=strength.mean(0)
        self.strength_sd=np.maximum(strength.std(0),1e-6)
        return self

    def transform(self,A):
        W=A/self.scale[None,:,None,None]
        W=W+np.eye(30,dtype=np.float32)[None,None]
        inv=1/np.sqrt(W.sum(-1))
        W=W*inv[:,:,:,None]*inv[:,:,None,:]
        strength=(A.mean(-1)-self.strength_mean)/self.strength_sd
        identity=np.broadcast_to(np.eye(30,dtype=np.float32),(len(A),3,30,30))
        X=np.concatenate([identity,strength[:,:,:,None]],axis=-1)
        return torch.from_numpy(W.astype(np.float32)),torch.from_numpy(X.astype(np.float32))


class Encoder(nn.Module):
    def __init__(self,hidden=16):
        super().__init__()
        self.one=nn.Linear(31,hidden);self.two=nn.Linear(hidden,hidden)
    def forward(self,A,X):
        h=torch.relu(self.one(torch.bmm(A,X)))
        h=torch.relu(self.two(torch.bmm(A,h)))
        return h.mean(1)


class MultiLayerGNN(nn.Module):
    def __init__(self,layers=(0,1,2),hidden=16,interaction=True):
        super().__init__();self.layers=layers;self.interaction=interaction and len(layers)>1
        self.encoders=nn.ModuleList([Encoder(hidden) for _ in layers])
        if self.interaction:
            self.attention=nn.MultiheadAttention(hidden,num_heads=2,batch_first=True)
            self.norm=nn.LayerNorm(hidden)
        self.readout=nn.Sequential(nn.Linear(hidden*len(layers),32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,9))
    def forward(self,A,X):
        tokens=torch.stack([encoder(A[:,layer],X[:,layer]) for encoder,layer in zip(self.encoders,self.layers)],1)
        if self.interaction:
            mixed,_=self.attention(tokens,tokens,tokens,need_weights=False)
            tokens=self.norm(tokens+mixed)
        return self.readout(tokens.flatten(1))


class FeatureMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1305,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,9))
    def forward(self,X):return self.net(X)


def shapley_from_coalitions(values):
    """values: n x 8, subset encoded by three-bit integer. Exact three-player SHAP."""
    phi=np.zeros((len(values),3))
    for i in range(3):
        for mask in range(8):
            if mask & (1<<i):continue
            size=mask.bit_count() if hasattr(int,'bit_count') else bin(mask).count('1')
            weight=math.factorial(size)*math.factorial(2-size)/math.factorial(3)
            phi[:,i]+=weight*(values[:,mask|(1<<i)]-values[:,mask])
    return phi
