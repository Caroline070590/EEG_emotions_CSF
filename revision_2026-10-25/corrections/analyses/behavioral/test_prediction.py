"""Regression checks for prediction leakage and invertible fold transformations."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from correct_prediction import METRICS,MODELS,VIDEO_EMOTION,fit_fold,validate_frame,make_design


def example():
    rng=np.random.default_rng(293)
    subject=np.repeat(np.arange(5),28);video=np.tile(np.arange(1,29),5)
    d=pd.DataFrame(dict(subject_id=subject,video_id=video,emotion=[VIDEO_EMOTION[v] for v in video],
                        Valence=rng.uniform(0,7,len(video)),Arousal=rng.uniform(0,7,len(video))))
    for k,m in enumerate(METRICS):
        d[m]=(k+1)*(subject+.2*d.Valence+.1*d.Arousal+rng.normal(size=len(d)))
    return d


def test_heldout_targets_cannot_change_fitted_scales_or_predictions():
    d=example();a=fit_fold(d,8)
    changed=d.copy();changed.loc[changed.video_id==8,METRICS]=1e8+changed.loc[changed.video_id==8,METRICS]*-273
    b=fit_fold(changed,8)
    for key in ["means","stds","scales","prediction_original","prediction_training_standardized"]:
        np.testing.assert_array_equal(a[key],b[key])
    for x,y in zip(a["coef"],b["coef"]):np.testing.assert_array_equal(x,y)
    assert not np.array_equal(a["y_raw"],b["y_raw"])


def test_subject_reference_matches_training_mean_in_original_units():
    d=example();f=fit_fold(d,12)
    expected=d[d.video_id!=12].groupby("subject_id")[METRICS].mean().to_numpy()
    np.testing.assert_allclose(f["prediction_original"][:,0,:],expected,atol=1e-12)
    np.testing.assert_allclose(f["prediction_training_standardized"][:,0,:],0,atol=1e-13)
    train=d[d.video_id!=12].groupby("subject_id")[METRICS]
    np.testing.assert_allclose(f["stds"],train.std(ddof=0).to_numpy(),rtol=1e-14)


def test_fold_fit_matches_independent_explicit_ols_and_inverse_transform():
    d=example();f=fit_fold(d,19)
    train=d.video_id.to_numpy()!=19;test=~train
    mu=d.loc[train].groupby("subject_id")[METRICS].mean()
    sd=d.loc[train].groupby("subject_id")[METRICS].std(ddof=0)
    z=np.stack([(d.loc[i,METRICS].to_numpy(float)-mu.loc[d.loc[i,"subject_id"]].to_numpy()) /
                sd.loc[d.loc[i,"subject_id"]].to_numpy() for i in d.index])
    for j,model in enumerate(MODELS):
        xtr,xte=make_design(d,train,test,model)
        coef=np.linalg.lstsq(xtr,z[train],rcond=None)[0]
        pred=xte@coef
        np.testing.assert_allclose(f["prediction_training_standardized"][:,j],pred,atol=2e-13)
        np.testing.assert_allclose(f["prediction_original"][:,j],pred*sd.to_numpy()+mu.to_numpy(),atol=3e-12)


def test_constant_training_outcome_rule_does_not_inspect_heldout_value():
    d=example();d.loc[d.subject_id==2,"Lag"]=10
    d.loc[(d.subject_id==2)&(d.video_id==7),"Lag"]=900
    f=fit_fold(d,7)
    assert f["zero_scale"][2,5] and f["scales"][2,5]==1
    assert f["y_standardized"][2,5]==890
    assert abs(f["prediction_original"][2,0,5]-10)<1e-12


def test_heldout_rating_cannot_affect_training_design():
    d=example();train=d.video_id.to_numpy()!=3;test=~train
    for model in ["rating","combined"]:
        a,_=make_design(d,train,test,model)
        changed=d.copy();changed.loc[test,["Valence","Arousal"]]=1e8
        b,_=make_design(changed,train,test,model)
        np.testing.assert_array_equal(a,b)


def test_global_target_standardization_exhibits_the_original_dependence():
    d=example();subject=d[d.subject_id==0]
    y=subject.PLV.to_numpy();z=(y-y.mean())/y.std(ddof=0)
    assert np.isclose(z[1:].mean(),-z[0]/27)
    changed=y.copy();changed[0]+=100
    z2=(changed-changed.mean())/changed.std(ddof=0)
    assert not np.allclose(z[1:],z2[1:])


def test_metadata_validation_rejects_incomplete_or_wrong_labels():
    d=example();validate_frame(d,full=False)
    with pytest.raises(ValueError,match="Incomplete"):validate_frame(d.iloc[1:],full=False)
    bad=d.copy();bad.loc[0,"emotion"]="fear"
    with pytest.raises(ValueError,match="mapping"):validate_frame(bad,full=False)
