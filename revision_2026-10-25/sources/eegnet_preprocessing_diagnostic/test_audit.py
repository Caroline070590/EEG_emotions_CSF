import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample
import protocol
from prepare import process_trial
import run
import report


class AuditTests(unittest.TestCase):
    def test_fixed_participants_and_windows(self):
        trials=dict(idx=np.arange(3444),subject=np.repeat(np.arange(123),28),
                    video=np.tile(np.arange(28),123),y=np.tile(np.repeat(np.arange(9),[3,3,3,3,4,3,3,3,3]),123))
        meta=protocol.window_metadata(trials); splits=protocol.fixed_splits(meta['subject'])
        self.assertEqual([len(splits[k]) for k in ['train','validation','test']],[6720,1680,1932])
        self.assertEqual(len(set(splits['train']) & set(splits['test'])),0)
        self.assertTrue(np.all(np.bincount(meta['trial_idx'])==3))

    def test_metadata_rejects_reordered_source(self):
        class Raw:
            shape=(3444,30,7500)
            dtype=np.dtype('float32')
        y=np.tile(np.repeat(np.arange(9),[3,3,3,3,4,3,3,3,3]),123)
        frame=pd.DataFrame(dict(idx=np.arange(3444),subject=np.repeat(np.arange(123),28),
                                video=np.tile(np.arange(28),123),emotion=np.asarray(protocol.CLASSES)[y]))
        protocol.validate_trials(Raw(),frame)
        with self.assertRaises(ValueError): protocol.validate_trials(Raw(),frame.iloc[::-1])

    def test_reference_resamples_before_segmentation(self):
        raw=np.random.default_rng(9).normal(size=(30,7500)).astype(np.float32)
        expected=resample(raw.astype(np.float64),6000,axis=1).reshape(30,3,2000).transpose(1,0,2).astype(np.float32)
        np.testing.assert_array_equal(process_trial(raw,'trial_fourier_div100'),expected)
        np.testing.assert_array_equal(process_trial(raw,'trial_fourier_zscore'),expected)
        self.assertEqual(process_trial(raw,'window_filtered_zscore').shape,(3,30,2000))

    def test_scaler_uses_only_training_values(self):
        raw=np.random.default_rng(2).normal(size=(6,30,100)).astype(np.float32); raw[:,0]=4
        a=run.fit_scaler(raw,np.array([0,1]),'window_filtered_zscore')
        raw[2:]+=100000
        b=run.fit_scaler(raw,np.array([0,1]),'trial_fourier_zscore')
        for key in ['mean','std','constant']: np.testing.assert_array_equal(a[key],b[key])
        self.assertTrue(a['constant'][0])
        fixed=run.fit_scaler(raw,np.array([0,1]),'trial_fourier_div100')
        self.assertEqual(len(fixed['fit_rows']),0)
        np.testing.assert_array_equal(fixed['std'],np.full(30,100))

    def test_three_window_trial_aggregation(self):
        y=np.repeat(np.arange(9),3); probability=np.eye(9)[y]
        meta=dict(y=y,trial_idx=np.repeat(np.arange(9),3))
        self.assertEqual(run.trial_metrics(probability,meta,np.arange(27))['balanced_accuracy'],1)
        with self.assertRaises(ValueError): run.trial_metrics(probability[:-1],meta,np.arange(26))

    def test_canonical_model_accepts_audited_shape(self):
        model=run.EEGNet(9,2000,n_channels=30)
        with torch.no_grad(): output=model(torch.zeros(2,30,2000))
        self.assertEqual(tuple(output.shape),(2,9))

    def test_training_never_predicts_test_until_selection_finishes(self):
        class Tiny(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__(); self.layer=torch.nn.Linear(30,9)
            def forward(self,x): return self.layer(x.mean(-1))
            def apply_constraints(self): pass
        raw=np.random.default_rng(3).normal(size=(81,30,40)).astype(np.float32)
        meta=dict(y=np.tile(np.repeat(np.arange(9),3),3),trial_idx=np.repeat(np.arange(27),3),
                  subject=np.repeat([0,80,100],27),window_id=np.tile(np.arange(3),27))
        split=dict(train=np.arange(27),validation=np.arange(27,54),test=np.arange(54,81))
        settings=dict(max_epochs=2,min_epochs=2,patience=2,batch_size=27,learning_rate=.001)
        calls=[]; original=run.predict
        def record(model,signal,rows,*args):
            calls.append(rows.copy()); return original(model,signal,rows,*args)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run,'EEGNet',Tiny),patch.object(run,'predict',record):
                result=run.train_and_evaluate(raw,meta,split,'window_filtered_zscore',4,settings,Path(directory),torch.device('cpu'))
            self.assertEqual(len(calls),3)
            for rows in calls[:-1]: np.testing.assert_array_equal(rows,split['validation'])
            np.testing.assert_array_equal(calls[-1],split['test'])
            self.assertEqual(result['test_prediction_passes'],1)
            with self.assertRaises(FileExistsError):
                run.train_and_evaluate(raw,meta,split,'window_filtered_zscore',4,settings,Path(directory),torch.device('cpu'))

    def test_report_marks_pending_and_rejects_changed_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory); config_path=base/'config.json'; config_path.write_text('{}')
            task=dict(recipe=protocol.RECIPES[0],seed=protocol.SEEDS[0])
            other=dict(recipe=protocol.RECIPES[1],seed=protocol.SEEDS[0])
            folder=base/'runs'/f"{task['recipe']}_seed{task['seed']}"; folder.mkdir(parents=True)
            trials=dict(idx=np.arange(3444),subject=np.repeat(np.arange(123),28),
                        video=np.tile(np.arange(28),123),y=np.tile(np.repeat(np.arange(9),[3,3,3,3,4,3,3,3,3]),123))
            meta=protocol.window_metadata(trials); np.savez_compressed(base/'metadata.npz',**meta)
            expected={k:v[8400:] for k,v in meta.items()}; y=expected['y']
            probability=np.eye(9)[y]
            predictions=folder/'test_predictions.npz'
            np.savez_compressed(predictions,probabilities=probability,**expected)
            result=dict(task,complete=True,smoke=False,config_sha256=protocol.digest(config_path),
                        artifact_sha256={'test_predictions.npz':protocol.digest(predictions)},
                        window_test=dict(balanced_accuracy=1),trial_test=dict(balanced_accuracy=1),
                        validation=dict(balanced_accuracy=.5),selected_epoch=2,selection_epochs=22)
            protocol.save_json(folder/'result.json',result)
            fixture={'tasks':[task,other],'input_sha256':{'metadata':protocol.digest(base/'metadata.npz')}}
            with patch.object(report,'read_config',return_value=fixture):
                summary=report.report(config_path)
                self.assertFalse(summary['complete']); self.assertEqual(len(summary['pending']),1)
                self.assertEqual(summary['paired_differences'],[])
                changed={k:v.copy() for k,v in expected.items()}; changed['video'][0]=27
                np.savez_compressed(predictions,probabilities=probability,**changed)
                result['artifact_sha256']['test_predictions.npz']=protocol.digest(predictions)
                protocol.save_json(folder/'result.json',result)
                with self.assertRaisesRegex(ValueError,'metadata differs'): report.report(config_path)
                np.savez_compressed(predictions,probabilities=probability,**expected)
                result['artifact_sha256']['test_predictions.npz']=protocol.digest(predictions)
                protocol.save_json(folder/'result.json',result)
                with predictions.open('ab') as handle: handle.write(b'changed')
                with self.assertRaises(ValueError): report.report(config_path)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
