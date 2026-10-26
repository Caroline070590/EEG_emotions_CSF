"""Correct Appendix J.4 outcome leakage, leaving its source data untouched.

Each of 28 folds fits participant outcome means/SDs on the other 27 videos.
Four original OLS designs are retained. Original-unit RMSE is the principal
comparison; training-standardized RMSE is a sensitivity analysis. The 12
original paired-video Wilcoxon comparisons are retained per scale, with both
within-scale BH-12 and across-scale BH-24 corrections explicitly supplied.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
METRICS = ["PLV", "PLI", "wPLI", "AEC", "MaxXC", "Lag"]
MODELS = ["subject_only", "label", "rating", "combined"]
EMOTIONS = ["anger", "disgust", "fear", "sadness", "neutral",
            "amusement", "inspiration", "joy", "tenderness"]
VIDEO_EMOTION = dict(zip(range(1, 29), np.repeat(EMOTIONS, [3,3,3,3,4,3,3,3,3])))
SCALES = ["original", "training_standardized"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    out[order] = np.minimum(1, np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    return out


def validate_frame(d, full=True):
    required = ["subject_id", "video_id", "emotion", "Valence", "Arousal"] + METRICS
    if set(required) - set(d.columns):
        raise ValueError("Missing required columns")
    if d.duplicated(["subject_id", "video_id"]).any():
        raise ValueError("Duplicate participant-video rows")
    if not np.isfinite(d[["Valence", "Arousal"] + METRICS].to_numpy(float)).all():
        raise ValueError("Nonfinite ratings or raw outcomes")
    if not d.emotion.eq(d.video_id.map(VIDEO_EMOTION)).all():
        raise ValueError("Video-condition mapping mismatch")
    cross = pd.crosstab(d.subject_id, d.video_id)
    if not (cross.to_numpy() == 1).all():
        raise ValueError("Incomplete participant-video design")
    if full and (set(d.subject_id) != set(range(123)) or set(d.video_id) != set(range(1,29))):
        raise ValueError("Expected participants 0–122 and videos 1–28")


def make_design(d, train, test, model):
    # Indicator levels are the known experimental design. Every level occurs
    # in each training fold; no target values define these columns.
    subject = pd.get_dummies(d.subject_id.astype(str), drop_first=True, dtype=float)
    emotion = pd.get_dummies(pd.Categorical(d.emotion, categories=EMOTIONS),
                             drop_first=True, dtype=float)
    a = [np.ones((train.sum(),1)), subject.loc[train].to_numpy()]
    b = [np.ones((test.sum(),1)), subject.loc[test].to_numpy()]
    if model in ("label", "combined"):
        a.append(emotion.loc[train].to_numpy())
        b.append(emotion.loc[test].to_numpy())
    if model in ("rating", "combined"):
        r = d[["Valence", "Arousal"]].to_numpy(float)
        mean, sd = r[train].mean(0), r[train].std(0, ddof=0)
        sd = np.where(sd > 0, sd, 1.)
        a.append((r[train] - mean) / sd)
        b.append((r[test] - mean) / sd)
    return np.column_stack(a), np.column_stack(b)


def fit_fold(d, held_video, legacy_global_z=False):
    """Return predictions/stats using training outcomes only, except explicit legacy audit.

    A zero training SD is replaced by one original outcome unit. This keeps
    the affine transform invertible and does not use the held-out outcome to
    decide its scale. Such participant-outcome folds are separately reported.
    """
    test = d.video_id.to_numpy() == held_video
    train = ~test
    if not test.any() or not train.any():
        raise ValueError("Empty training/test partition")
    subjects = np.sort(d.subject_id.unique())
    subject_index = np.searchsorted(subjects, d.subject_id.to_numpy())
    y = d[METRICS].to_numpy(float)
    means, stds = [], []
    for subject in subjects:
        rows = (d.subject_id.to_numpy() == subject) & train
        if not rows.any():
            raise ValueError("Held-out participant lacks training videos")
        means.append(y[rows].mean(0)); stds.append(y[rows].std(0, ddof=0))
    means, stds = np.stack(means), np.stack(stds)
    # Values smaller than round-off at the original scale count as constant.
    zero = stds <= 16 * np.finfo(float).eps * np.maximum(1, np.abs(means))
    scales = np.where(zero, 1., stds)
    ys = (y - means[subject_index]) / scales[subject_index]
    if legacy_global_z:
        ys = d[[m + "_z" for m in METRICS]].to_numpy(float)
    pred, coefs = [], []
    for model in MODELS:
        xtrain, xtest = make_design(d, train, test, model)
        fit = LinearRegression(fit_intercept=False).fit(xtrain, ys[train])
        pred.append(fit.predict(xtest)); coefs.append(fit.coef_)
    pred = np.stack(pred, axis=1)  # heldout rows × models × outcomes
    pred_raw = pred * scales[subject_index[test],None,:] + means[subject_index[test],None,:]
    assert np.isfinite(pred).all() and np.isfinite(pred_raw).all()
    return {"test_indices":np.flatnonzero(test), "train_indices":np.flatnonzero(train),
            "subject_ids":subjects, "means":means, "stds":stds,
            "scales":scales, "zero_scale":zero, "coef":coefs,
            "y_raw":y[test], "y_standardized":ys[test],
            "prediction_original":pred_raw, "prediction_training_standardized":pred}


def run_folds(d, legacy=False):
    n = len(d)
    prediction_raw = np.full((n,len(MODELS),len(METRICS)),np.nan)
    prediction_std = np.full_like(prediction_raw,np.nan)
    observed_std = np.full((n,len(METRICS)),np.nan)
    rows, scaler_rows = [], []
    covered = np.zeros(n,int)
    for held in sorted(d.video_id.unique()):
        f = fit_fold(d,held,legacy_global_z=legacy)
        ix = f["test_indices"]; covered[ix] += 1
        prediction_raw[ix] = f["prediction_original"]
        prediction_std[ix] = f["prediction_training_standardized"]
        observed_std[ix] = f["y_standardized"]
        for scale in SCALES:
            target = f["y_raw"] if scale == "original" else f["y_standardized"]
            err = f["prediction_" + scale] - target[:,None,:]
            for im, model in enumerate(MODELS):
                for iy, metric in enumerate(METRICS):
                    e = err[:,im,iy]
                    rows.append(dict(held_video=int(held),held_emotion=VIDEO_EMOTION[held],
                                     metric=metric,model=model,scale=scale,
                                     rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(np.abs(e))),
                                     n_train=len(f["train_indices"]),n_test=len(ix)))
        for j, subject in enumerate(f["subject_ids"]):
            for k, metric in enumerate(METRICS):
                scaler_rows.append(dict(held_video=int(held),subject_id=int(subject),metric=metric,
                                        n_training_videos=int(((d.subject_id==subject)&(d.video_id!=held)).sum()),
                                        mean=f["means"][j,k],std=f["stds"][j,k],
                                        applied_scale=f["scales"][j,k],zero_training_sd=bool(f["zero_scale"][j,k])))
    assert np.all(covered == 1)
    arrays = dict(subject_id=d.subject_id.to_numpy(),video_id=d.video_id.to_numpy(),
                  metric=np.array(METRICS),model=np.array(MODELS),
                  observed_original=d[METRICS].to_numpy(float),observed_training_standardized=observed_std,
                  prediction_original=prediction_raw,prediction_training_standardized=prediction_std)
    return pd.DataFrame(rows),pd.DataFrame(scaler_rows),arrays


def compare(folds):
    rows = []
    for scale in SCALES:
        for metric in METRICS:
            wide = folds.loc[(folds.scale==scale)&(folds.metric==metric)].pivot(
                index="held_video",columns="model",values="rmse")
            for model in ["rating","combined"]:
                diff = (wide[model]-wide.label).to_numpy()
                w = stats.wilcoxon(diff,zero_method="wilcox",alternative="two-sided")
                rows.append(dict(scale=scale,metric=metric,comparison=model+" minus label",
                                 mean_rmse_difference=diff.mean(),median_rmse_difference=np.median(diff),
                                 wilcoxon_statistic=w.statistic,p=w.pvalue,n_held_videos=len(diff)))
    result = pd.DataFrame(rows)
    result["q_BH_12_within_scale"] = result.groupby("scale")["p"].transform(lambda s:bh(s.to_numpy()))
    result["q_BH_24_both_scales"] = bh(result.p.to_numpy())
    return result


def plot_predictions(summary,out):
    colors = ["#777777","#E2A330","#4477AA","#228833"]
    names = ["Participant only","Assigned label","Ratings","Combined"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(8,4.8),constrained_layout=True)
    for metric,ax in zip(METRICS,axes.ravel()):
        q=summary[(summary.scale=="original")&(summary.metric==metric)].set_index("model").loc[MODELS]
        ax.bar(np.arange(4),q.mean_fold_rmse,color=colors,width=.65)
        ax.set_xticks(np.arange(4));ax.set_xticklabels(names,rotation=25,ha="right",fontsize=7)
        ax.set_title(metric);ax.set_ylabel("Mean held-video RMSE"+(" (ms)" if metric=="Lag" else ""))
        ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"behavioral_prediction_original_units.pdf",bbox_inches="tight")
    fig.savefig(out/"behavioral_prediction_original_units.png",dpi=220,bbox_inches="tight");plt.close(fig)


def plot_replacement_s23(folds,out):
    """Reproduce unchanged association panels and replace only prediction results.

    This new generator explicitly reindexes each metric; the preserved old
    generator did not align sorted groupby rows to its displayed labels.
    """
    spec=importlib.util.spec_from_file_location("preserved_behavioral",HERE/"provenance/original_behavioral_grounding_appendix.py")
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    primary=pd.read_csv(HERE/"inputs/association_primary_unchanged.csv")
    sensitivity=pd.read_csv(HERE/"inputs/association_sensitivity_unchanged.csv")
    old.set_style()
    fig,axes=plt.subplots(1,3,figsize=(9.2,3.4),constrained_layout=True)
    palette={"Valence":"#4C78A8","Arousal":"#E45756","Liking":"#72B7B2","Familiarity":"#B279A2"}
    for ax,table,title,letter,predictors in [(axes[0],primary,"Within-video associations","a",["Valence","Arousal"]),
            (axes[1],sensitivity,"With liking and familiarity","b",["Valence","Arousal","Liking","Familiarity"])]:
        for y,metric in enumerate(METRICS):
            q=table[table.metric==metric].set_index("predictor")
            for j,predictor in enumerate(predictors):
                r=q.loc[predictor];yy=y+(j-(len(predictors)-1)/2)*.16
                ax.errorbar(r.beta,yy,xerr=[[r.beta-r.ci_low],[r.ci_high-r.beta]],fmt="o",color=palette[predictor],ms=3,lw=.8,capsize=1.5)
        ax.axvline(0,color="black",lw=.6,ls="--");ax.set_yticks(range(6));ax.set_yticklabels(METRICS);ax.invert_yaxis()
        ax.set_xlabel("Standardized association (β ± 95% CI)");ax.set_title(title);old.panel(ax,letter)
        handles=[plt.Line2D([0],[0],marker="o",linestyle="",color=palette[x],label=x,markersize=4) for x in predictors]
        ax.legend(handles=handles,frameon=False,fontsize=6,loc="best")
    means=folds[folds.scale=="training_standardized"].groupby(["metric","model"]).rmse.mean().unstack().reindex(METRICS)
    ax=axes[2]; x=np.arange(6);width=.19
    for j,(model,color) in enumerate(zip(MODELS,["#BDBDBD","#F2A541","#4C78A8","#59A14F"])):
        ax.bar(x+(j-1.5)*width,means[model].to_numpy(),width,label=model.replace("_"," "),color=color)
    ax.set_xticks(x);ax.set_xticklabels(METRICS,rotation=45,ha="right")
    ax.set_ylabel("Mean held-video RMSE (training-SD units)");ax.set_title("Training-only outcome scaling")
    ax.legend(frameon=False,fontsize=5.5,loc="lower center",ncol=2);old.panel(ax,"c")
    fig.savefig(out/"Supplementary_Figure_S23_behavioral_EEG_models_corrected.pdf",bbox_inches="tight")
    fig.savefig(out/"Supplementary_Figure_S23_behavioral_EEG_models_corrected.png",dpi=220,bbox_inches="tight");plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,default=HERE/"inputs/behavioral_EEG_merged_validated.csv")
    ap.add_argument("--synchronization",type=Path,help="Optional verified raw synchronization table, joined by subject_id/video_id")
    ap.add_argument("--out",type=Path,default=HERE/"results")
    args=ap.parse_args();start=time.time()
    if args.out.exists():raise SystemExit("Refusing to overwrite results")
    d=pd.read_csv(args.input).sort_values(["subject_id","video_id"]).reset_index(drop=True)
    validate_frame(d)
    if args.synchronization:
        s=pd.read_csv(args.synchronization)
        s=s[["subject_id","video_id"]+METRICS]
        d=d.drop(columns=METRICS).merge(s,on=["subject_id","video_id"],validate="one_to_one",how="left")
        validate_frame(d)
    with threadpool_limits(limits=1):
        folds,scalers,arrays=run_folds(d)
        legacy,_,_=run_folds(d,legacy=True)
    ref=pd.read_csv(HERE/"provenance/legacy_verification_folds.csv")
    old=legacy[legacy.scale=="training_standardized"].merge(ref,on=["held_video","metric","model"],suffixes=("_new","_old"),validate="one_to_one")
    legacy_delta=float(np.max(np.abs(old.rmse_new-old.rmse_old)))
    if not legacy_delta<1e-10:raise AssertionError(("Legacy reproduction mismatch",legacy_delta))
    comparisons=compare(folds)
    summary=folds.groupby(["scale","metric","model"]).agg(mean_fold_rmse=("rmse","mean"),
        pooled_oof_rmse=("rmse",lambda x:np.sqrt(np.mean(x**2))),mean_fold_mae=("mae","mean"),n_folds=("rmse","size")).reset_index()
    args.out.mkdir(parents=True)
    folds.to_csv(args.out/"Table_S9_corrected_fold_errors.csv",index=False)
    summary.to_csv(args.out/"Table_S9_corrected_summary.csv",index=False)
    comparisons.to_csv(args.out/"Table_S9_corrected_comparisons.csv",index=False)
    scalers.to_csv(args.out/"training_fold_outcome_scalers.csv",index=False)
    np.savez_compressed(args.out/"out_of_fold_predictions.npz",**arrays)
    # Training-only correction deltas in the old standardized reporting scale.
    delta=folds[folds.scale=="training_standardized"].merge(ref,on=["held_video","metric","model"],suffixes=("_corrected","_legacy"),validate="one_to_one")
    delta["rmse_delta"]=delta.rmse_corrected-delta.rmse_legacy
    delta.to_csv(args.out/"outcome_scaling_correction_delta.csv",index=False)
    plot_predictions(summary,args.out);plot_replacement_s23(folds,args.out)
    manifest={"status":"completed","analysis":"Appendix J.4 training-only outcome scaling correction",
        "input":str(args.input.resolve()),"input_sha256":sha(args.input),
        "synchronization_override":str(args.synchronization.resolve()) if args.synchronization else None,
        "synchronization_override_sha256":sha(args.synchronization) if args.synchronization else None,
        "code_sha256":sha(Path(__file__)),"original_code_sha256":sha(HERE/"provenance/original_behavioral_grounding_appendix.py"),
        "n_rows":len(d),"n_participants":123,"n_videos":28,"n_folds":28,"models":MODELS,"metrics":METRICS,
        "training_rows_per_fold":3321,"test_rows_per_fold":123,"outcome_standardization":"Per participant and outcome, mean and population SD from 27 training videos only; held-out target not used during fitting",
        "constant_scale_rule":"Training SD <=16*machine_epsilon*max(1,abs(training_mean)) replaced by one original outcome unit; no observations dropped",
        "n_zero_training_sd_participant_outcome_folds":int(scalers.zero_training_sd.sum()),
        "zero_scale_subject_metrics":scalers[scalers.zero_training_sd][["subject_id","metric"]].drop_duplicates().to_dict("records"),
        "comparison_scope":"Original-unit errors are the principal comparison. Training-standardized errors are sensitivity results. Same 12 paired-video comparisons per scale; BH-12 within scale and BH-24 across both supplied. These exploratory fold comparisons condition on the fixed familiar participants and overlapping training sets.",
        "legacy_global_scaling_reproduction_max_abs_rmse_error":legacy_delta,
        "legacy_global_scaling_outputs_are_validation_only":True,
        "association_panels":"Reused unchanged historical Table S7/S7b outputs; prediction panel is newly generated with explicit metric order. No within-video association model was refitted.",
        "runtime_seconds":time.time()-start,"python":platform.python_version(),
        "public_release":"Local correction only; nothing uploaded or published",
        "result_sha256":{p.name:sha(p) for p in sorted(args.out.iterdir()) if p.is_file()}}
    (args.out/"analysis_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(summary.to_string(index=False));print(comparisons.to_string(index=False))
    print(json.dumps({"runtime_seconds":manifest["runtime_seconds"],"legacy_max_abs_error":legacy_delta,
                      "zero_training_sd_folds":manifest["n_zero_training_sd_participant_outcome_folds"]}))


if __name__=="__main__":main()
