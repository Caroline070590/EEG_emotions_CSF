from pathlib import Path
import json
import numpy as np
from sklearn.metrics import balanced_accuracy_score
HERE=Path(__file__).resolve().parent;rows=[]
for i in range(100):
    with np.load(HERE/f'pilot99/prediction_{i:04d}.npz') as a, np.load(HERE/f'graph_control99/prediction_{i:04d}.npz') as b:
        np.testing.assert_array_equal(a['keys'],b['keys']);np.testing.assert_array_equal(a['y'],b['y'])
        intact=balanced_accuracy_score(a['y'],a['multiplex'].argmax(1));control=balanced_accuracy_score(b['y'],b['multiplex_scrambled'].argmax(1));rows.append({'permutation':i,'intact':intact,'control':control,'gain':intact-control})
null=np.array([r['gain'] for r in rows[1:]]);result={'observed':rows[0],'null_mean_gain':float(null.mean()),'p':float((1+sum(null>=rows[0]['gain']))/100),'scope':'exploratory additional test of sensor alignment; one fixed graph randomization, not a distribution over graph randomizations','rows':rows}
(HERE/'graph_control99/RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
