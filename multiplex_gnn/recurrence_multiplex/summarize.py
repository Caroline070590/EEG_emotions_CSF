import argparse,json
from pathlib import Path
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import balanced_accuracy_score
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--permutations',type=int,default=99);a=p.parse_args()
rows=[json.loads((a.directory/f'scores_{i:04d}.json').read_text()) for i in range(a.permutations+1)]
if any(r['config']['smoke'] for r in rows):raise ValueError('Smoke data are not inferential results')
for i,row in enumerate(rows):
    d=np.load(a.directory/f'prediction_{i:04d}.npz')
    assert d['keys'].shape==(3444,2) and len(np.unique(d['keys'],axis=0))==3444
    for name,score in row['scores'].items():
        prob=d[name];assert prob.shape==(3444,9) and np.isfinite(prob).all()
        np.testing.assert_allclose(prob.sum(1),1,atol=1e-5)
        np.testing.assert_allclose(balanced_accuracy_score(d['y'],prob.argmax(1)),score)
    for key in ['epochs','seeds','smoke']:
        assert row['config'][key]==rows[0]['config'][key]
metrics=['connectivity','recurrence','multiplex','connectivity_rqa','increment'];result=[]
for name in metrics:
    v=np.array([r['increment'] if name=='increment' else r['scores'][name] for r in rows]);pvalue=(1+np.sum(v[1:]>=v[0]))/len(v)
    result.append({'model_or_contrast':name,'observed':v[0],'mean_null':v[1:].mean(),'p':pvalue})
for row,q in zip(result,multipletests([r['p'] for r in result],method='holm')[1]):row['p_holm']=q
(a.directory/'SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
