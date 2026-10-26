"""One-way repeated-measures ANOVA on a participants x conditions table (complete cases), with Greenhouse-Geisser correction."""
import numpy as np
from scipy import stats

def rm_anova(cell):
    Y=cell.dropna().values;n,k=Y.shape;gm=Y.mean()
    ss_cond=n*((Y.mean(0)-gm)**2).sum();ss_subj=k*((Y.mean(1)-gm)**2).sum()
    ss_err=((Y-Y.mean(1,keepdims=True)-Y.mean(0,keepdims=True)+gm)**2).sum()
    df1,df2=k-1,(k-1)*(n-1);F=(ss_cond/df1)/(ss_err/df2)
    S=np.cov(Y,rowvar=False);P=np.eye(k)-1/k;Sc=P@S@P            # double-centred covariance
    eps=float(np.trace(Sc)**2/((k-1)*np.sum(Sc*Sc)));eps=min(max(eps,1/(k-1)),1.0)
    return dict(F=float(F),df1=df1,df2=df2,n=n,p_unc=float(stats.f.sf(F,df1,df2)),eps_gg=eps,
                p_gg=float(stats.f.sf(F,df1*eps,df2*eps)),ges=float(ss_cond/(ss_cond+ss_subj+ss_err)))
