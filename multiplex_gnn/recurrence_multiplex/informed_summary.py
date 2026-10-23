from pathlib import Path
import json,sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'multiplex_prediction'))
from experiment import LABELS
rows=[]
for i in range(100):
    with np.load(HERE/f'informed99/prediction_{i:04d}.npz') as d:
        keys=d['keys'];assert keys.shape==(3444,2) and len(np.unique(keys,axis=0))==3444
        labels=LABELS.copy() if i==0 else np.random.default_rng(np.random.SeedSequence([20261013,i])).permutation(LABELS)
        np.testing.assert_array_equal(d['y'],labels[keys[:,1]])
        scores={}
        for name in ['no_graph','distance_graph','recurrence_graph']:
            prob=d[name];assert prob.shape==(3444,9) and np.isfinite(prob).all();np.testing.assert_allclose(prob.sum(1),1,atol=1e-5)
            scores[name]=float(balanced_accuracy_score(d['y'],prob.argmax(1)))
        rows.append(scores)
tests=[]
for a,b in [('distance_graph','no_graph'),('recurrence_graph','no_graph'),('distance_graph','recurrence_graph')]:
    values=np.array([r[a]-r[b] for r in rows]);tests.append({'contrast':a+' minus '+b,'gain':float(values[0]),'null_mean':float(values[1:].mean()),'p':float((1+sum(values[1:]>=values[0]))/100)})
for t,q in zip(tests,multipletests([t['p'] for t in tests],method='holm')[1]):t['p_holm']=float(q)
result={'observed':rows[0],'tests':tests,'scope':'Exploratory matched adjacency comparison; 99 whole-video label permutations; one-sided gain tests; three-test Holm correction','all_scores':rows}
(HERE/'informed99/RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='all_scores'},indent=2))
