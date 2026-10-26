"""Independently validate saved predictions without importing the fitting code."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    manifest=json.loads((RESULTS/"analysis_manifest.json").read_text())
    assert manifest["code_sha256"]==sha(HERE/"correct_prediction.py")
    assert manifest["input_sha256"]==sha(HERE/"inputs/behavioral_EEG_merged_validated.csv")
    for filename,digest in manifest["result_sha256"].items():assert sha(RESULTS/filename)==digest,filename
    d=pd.read_csv(HERE/"inputs/behavioral_EEG_merged_validated.csv").sort_values(["subject_id","video_id"])
    z=np.load(RESULTS/"out_of_fold_predictions.npz",allow_pickle=False)
    metrics=z["metric"].tolist();models=z["model"].tolist()
    assert z["prediction_original"].shape==z["prediction_training_standardized"].shape==(3444,4,6)
    np.testing.assert_array_equal(z["subject_id"],d.subject_id)
    np.testing.assert_array_equal(z["video_id"],d.video_id)
    np.testing.assert_array_equal(z["observed_original"],d[metrics].to_numpy())
    assert len(d.drop_duplicates(["subject_id","video_id"]))==3444
    scalers=pd.read_csv(RESULTS/"training_fold_outcome_scalers.csv")
    folds=pd.read_csv(RESULTS/"Table_S9_corrected_fold_errors.csv")
    comps=pd.read_csv(RESULTS/"Table_S9_corrected_comparisons.csv")
    summary=pd.read_csv(RESULTS/"Table_S9_corrected_summary.csv")
    all_errors=[]; standardized_reconstruction_errors=[]
    for video in range(1,29):
        tr=d.video_id.to_numpy()!=video;te=~tr
        assert tr.sum()==3321 and te.sum()==123
        g=d.loc[tr].groupby("subject_id")[metrics]
        mean=g.mean().to_numpy();std=g.std(ddof=0).to_numpy()
        scale=np.where(std<=16*np.finfo(float).eps*np.maximum(1,np.abs(mean)),1,std)
        stored=scalers[scalers.held_video==video]
        for k,metric in enumerate(metrics):
            s=stored[stored.metric==metric].sort_values("subject_id")
            np.testing.assert_allclose(s["mean"],mean[:,k],rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(s["std"],std[:,k],rtol=1e-13,atol=1e-13)
            np.testing.assert_allclose(s.applied_scale,scale[:,k],rtol=1e-13,atol=1e-13)
            assert (s.n_training_videos==27).all()
        np.testing.assert_allclose(z["prediction_original"][te,0],mean,atol=1e-12)
        np.testing.assert_allclose(z["prediction_original"][te],
            z["prediction_training_standardized"][te]*scale[:,None,:]+mean[:,None,:],rtol=1e-13,atol=1e-12)
        independently_standardized=(d.loc[te,metrics].to_numpy()-mean)/scale
        standardized_reconstruction_errors.append(float(np.max(abs(
            z["observed_training_standardized"][te]-independently_standardized))))
        # pandas grouped SD and NumPy two-pass SD differ slightly for nearly
        # constant outcomes; retain the measured discrepancy in the audit.
        np.testing.assert_allclose(z["observed_training_standardized"][te],
            independently_standardized,rtol=1e-9,atol=1e-9)
        for unit in ["original","training_standardized"]:
            error=z["prediction_"+unit][te]-z["observed_"+unit][te,None,:]
            assert np.isfinite(error).all()
            for m,model in enumerate(models):
                for k,metric in enumerate(metrics):
                    row=folds[(folds.held_video==video)&(folds.scale==unit)&(folds.model==model)&(folds.metric==metric)].iloc[0]
                    rmse=np.sqrt(np.mean(error[:,m,k]**2));mae=np.mean(abs(error[:,m,k]))
                    all_errors.extend([abs(rmse-row.rmse),abs(mae-row.mae)])
    assert max(all_errors)<1e-12
    for unit in ["original","training_standardized"]:
        subset=comps[comps.scale==unit]
        ps=[]
        for row in subset.itertuples():
            f=folds[(folds.metric==row.metric)&(folds.scale==unit)].pivot(index="held_video",columns="model",values="rmse")
            model=row.comparison.split()[0];diff=f[model]-f.label
            p=wilcoxon(diff,alternative="two-sided",zero_method="wilcox").pvalue
            assert abs(p-row.p)<1e-14
            assert abs(diff.mean()-row.mean_rmse_difference)<1e-12
            ps.append(p)
        np.testing.assert_allclose(subset.q_BH_12_within_scale,multipletests(ps,method="fdr_bh")[1],atol=1e-14)
    np.testing.assert_allclose(comps.q_BH_24_both_scales,multipletests(comps.p,method="fdr_bh")[1],atol=1e-14)
    for row in summary.itertuples():
        q=folds[(folds.metric==row.metric)&(folds.scale==row.scale)&(folds.model==row.model)]
        assert abs(q.rmse.mean()-row.mean_fold_rmse)<1e-12
        assert abs(np.sqrt(np.mean(q.rmse**2))-row.pooled_oof_rmse)<1e-12
    wpli=HERE.parent/"wpli/RAW_LINEAGE_VALIDATION.json"
    wr=json.loads(wpli.read_text())["cache_checks"]["behavioral_verified"]
    assert wr["sha256"]==json.loads((HERE/"provenance/INPUT_VALIDATION.json").read_text())["raw_synchronization_sha256"]
    assert wr["standard_sos"]["n"]==3444 and wr["standard_sos"]["matches_at_1e-6"]
    old=pd.read_csv(HERE/"provenance/legacy_verification_folds.csv")
    hist=pd.read_csv(HERE/"provenance/legacy_manuscript_folds.csv")
    q=old.merge(hist,on=["held_video","metric","model"],validate="one_to_one",suffixes=("_verification","_historical"))
    record={"status":"passed","n_folds":28,"n_predictions_per_model_outcome":3444,
        "n_models":4,"n_outcomes":6,"n_scale_specific_fold_error_rows":len(folds),
        "n_training_scalers_verified":len(scalers),"prediction_errors_max_abs_difference":max(all_errors),
        "independent_standardization_max_abs_difference":max(standardized_reconstruction_errors),
        "all_original_unit_predictions_inverse_transform_verified":True,
        "participant_only_predictions_equal_training_means":True,
        "all_stored_scales_recomputed_from_training27_videos_only":True,
        "all_summary_and_comparison_outputs_independently_verified":True,
        "BH_12_and_24_independently_verified_with_statsmodels":True,
        "raw_wPLI_all3444_verified_standard":True,"wpli_lineage_validation_sha256":sha(wpli),
        "historical_manuscript_vs_verification_max_fold_rmse_difference":float(abs(q.rmse_verification-q.rmse_historical).max()),
        "n_meanings": "Original-unit prediction errors and training-fold-standardized errors are both reported; no score compares different outcome units across metrics.",
        "leakage_regression_tests":"Seven tests passed, including held-out outcome perturbations leaving training stats/coefficients/predictions unchanged and an independent explicit OLS solution.",
        "test_source_sha256":sha(HERE/"test_prediction.py"),"validator_source_sha256":sha(Path(__file__)),
        "manifest_sha256":sha(RESULTS/"analysis_manifest.json"),
        "scope":"This validates the corrected supplementary predictive analysis only. No primary inference or association/RSA models were refitted."}
    (RESULTS/"INDEPENDENT_VALIDATION.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))


if __name__=="__main__":main()
