from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import hilbert

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'corrected_pipeline/src'))
from lib import synchrony


class StandardWPLITest(unittest.TestCase):
    def test_analytic_amplitudes_are_retained(self):
        rng=np.random.default_rng(19)
        time=np.arange(250)/250
        amplitude=0.2+np.exp(1.8*np.sin(2*np.pi*2*time))
        x=np.c_[amplitude*np.sin(2*np.pi*11*time)+rng.normal(0,.2,250),
                np.sin(2*np.pi*11*time+.7*np.sin(2*np.pi*3*time))+rng.normal(0,.2,250),
                rng.normal(size=250)]
        filtered=synchrony._bandpass(x);z=hilbert(filtered,axis=0)
        expected=[];phase_only=[]
        for i in range(x.shape[1]):
            for j in range(i+1,x.shape[1]):
                imag=np.imag(z[:,i]*np.conj(z[:,j]))
                expected.append(abs(np.sum(imag))/np.sum(abs(imag)))
                phase=np.sin(np.angle(z[:,i])-np.angle(z[:,j]))
                phase_only.append(abs(np.sum(phase))/np.sum(abs(phase)))
        with patch.object(synchrony,'load_raw_trial_csv',return_value=x):
            got=synchrony.synchrony_one(Path('Patient001/trial_001.csv'))
        self.assertAlmostEqual(got['wpli'],np.mean(expected),places=13)
        self.assertGreater(abs(got['wpli']-np.mean(phase_only)),.01)

    def test_untagged_and_tampered_cache_are_recomputed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            cache=Path(td)
            calls=[]
            def fake_one(path,emotion):
                calls.append(path)
                return dict(subject=1,emotion=emotion,trial='trial_001',plv=.2,pli=.1,
                            wpli=.4,amp_env_corr=.2,max_cross_corr=.5,lag_ms=4.)
            (cache/'trial_synchrony_metrics.csv').write_text('subject,emotion,trial,wpli\n1,anger,trial_001,0.01\n')
            with patch.object(synchrony,'CACHE',cache),patch.object(synchrony,'_list_trials',return_value=[('trial','Patient001','anger')]),patch.object(synchrony,'_one_with_emo',side_effect=fake_one):
                first=synchrony.compute_synchrony(n_jobs=1)
                self.assertEqual(first.wpli.iloc[0],.4)
                self.assertEqual(len(calls),1)
                second=synchrony.compute_synchrony(n_jobs=1)
                self.assertEqual(len(calls),1)
                self.assertEqual(second.wpli.iloc[0],.4)
                csv=cache/'trial_synchrony_metrics.csv'
                csv.write_text(csv.read_text().replace('0.4','0.01'))
                synchrony.compute_synchrony(n_jobs=1)
                self.assertEqual(len(calls),2)
                info=json.loads((cache/'trial_synchrony_metrics.metadata.json').read_text())
                self.assertEqual(info['estimator_version'],'analytic_cross_spectrum_wpli_v1')


if __name__=='__main__':unittest.main()
