"""Write reviewable manuscript snippets from verified outputs; never edit a manuscript."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
RESULTS=HERE/"results"
METRICS=["PLV","PLI","wPLI","AEC","MaxXC","Lag"]
MODELS=["subject_only","label","rating","combined"]


def main():
    validation=json.loads((RESULTS/"INDEPENDENT_VALIDATION.json").read_text())
    assert validation["status"]=="passed"
    s=pd.read_csv(RESULTS/"Table_S9_corrected_summary.csv")
    c=pd.read_csv(RESULTS/"Table_S9_corrected_comparisons.csv")
    lines=[r"\begin{table}[!htbp]",r"\centering",
        r"\caption{\textbf{Leave-one-video-out prediction with training-only outcome scaling.} Panels A and B show mean RMSE across 28 held-video folds in original outcome units and training-fold standard-deviation units, respectively. Panel C compares rating-only and combined models against the assigned-label model using original-unit fold errors. Tests are paired two-sided Wilcoxon comparisons across held videos, with BH correction across the 12 comparisons; negative differences favor the first model named.}",
        r"\label{tab:s9-loov}",r"\scriptsize"]
    for unit,title in [("original","A. Mean held-video RMSE in original units"),
                       ("training_standardized","B. Mean held-video RMSE in training-SD units")]:
        lines += [r"\textbf{Panel "+title+r"}\\[3pt]",r"\begin{tabular}{lrrrr}",r"\toprule",
                  r"Metric & Participant only & Assigned label & Ratings & Combined \\",r"\midrule"]
        for metric in METRICS:
            q=s[(s.scale==unit)&(s.metric==metric)].set_index("model").loc[MODELS]
            digits=3 if metric=="Lag" or unit=="training_standardized" else 4
            vals=[f"{v:.{digits}f}" for v in q.mean_fold_rmse]
            lines.append((metric+(" (ms)" if metric=="Lag" and unit=="original" else ""))+" & "+" & ".join(vals)+r" \\")
        lines += [r"\bottomrule",r"\end{tabular}\par",r"\vspace{7pt}"]
    lines += [r"\textbf{Panel C. Paired comparisons in original units}\\[3pt]",
              r"\begin{tabular}{llrrrr}",r"\toprule",
              r"Metric & Comparison & Mean difference & Median difference & $p$ & BH--FDR $q$ \\",r"\midrule"]
    for metric in METRICS:
        for model in ["rating","combined"]:
            row=c[(c.scale=="original")&(c.metric==metric)&(c.comparison==model+" minus label")].iloc[0]
            digits=4 if metric=="Lag" else 6
            lines.append(f"{metric} & {model} $-$ label & {row.mean_rmse_difference:.{digits}f} & {row.median_rmse_difference:.{digits}f} & {row.p:.3f} & {row.q_BH_12_within_scale:.3f}"+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}\par",r"\vspace{3pt}",r"\begin{minipage}{0.96\linewidth}",
        r"\footnotesize Predictions were transformed back to original units for Panels A and C; Lag is in milliseconds and the other outcomes are dimensionless. Outcome means and population SDs were fitted separately for each participant using 27 training videos. One participant had constant Lag; a scale of one original unit was used in its 28 folds, without dropping observations. No comparison survived BH correction using training-SD errors either (all $q\geq0.179$). Correcting jointly across both scales (24 tests) also yielded no significant comparison. These exploratory fold comparisons use overlapping training sets.",
        r"\end{minipage}",r"\end{table}"]
    (RESULTS/"Table_S9_corrected.tex").write_text("\n".join(lines)+"\n")
    paragraph=r"""To compare assigned labels with reported experience across videos, we used 28 leave-one-video-out folds, retaining the same participants in training and testing. All four models included participant indicators: the reference model used these alone; the label model added assigned condition; the rating model added valence and arousal; and the combined model included both. Within each fold, each participant's outcome mean and population standard deviation were estimated from the 27 training videos only and applied to training and held-out observations. Rating predictors were likewise standardized using training observations only. A constant training outcome was assigned a scale of one original unit; this occurred for one participant's Lag measure. Predictions were transformed back to original units before the principal RMSE comparison; training-standardized errors were retained as a sensitivity analysis (Supplementary Table~\ref{tab:s9-loov}). Neither rating-only nor combined models differed from the assigned-label model after correction across the 12 paired comparisons (all $q\geq0.611$). The same conclusion held for training-standardized errors (all $q\geq0.179$). These exploratory comparisons relate final-second EEG descriptors to assigned categories and subsequent whole-video ratings; the training sets overlap across held-video folds.
"""
    (RESULTS/"Appendix_J4_prediction_replacement.tex").write_text(paragraph)
    caption=r"""\caption{\textbf{Behaviorally grounded models of EEG synchronization.} \textbf{a,} Participant- and video-fixed-effect estimates for within-video-centered valence and arousal. Outcomes are within-participant standardized synchronization measures; predictors are scaled within-video rating deviations. Intervals are pointwise participant- and video-clustered 95\% confidence intervals. \textbf{b,} Sensitivity model additionally controlling for liking and familiarity. The association estimates in a and b are unchanged. \textbf{c,} Mean RMSE across 28 leave-one-video-out folds for participant-only, assigned-label, rating-only, and combined models. Outcome standardization is fitted separately for each participant using the 27 training videos only; rating scaling also uses training data only. Errors are shown in training-SD units; lower values indicate better prediction. Original-unit errors and multiplicity-corrected comparisons are reported in Supplementary Table~\ref{tab:s9-loov}. \textit{Abbreviations:} AEC, amplitude-envelope correlation; CI, confidence interval; Lag, lag at maximum cross-correlation; MaxXC, maximum cross-correlation; PLI, phase-lag index; PLV, phase-locking value; RMSE, root-mean-square error; wPLI, weighted phase-lag index.}
"""
    (RESULTS/"Behavioral_models_corrected_caption.tex").write_text(caption)
    response="""We corrected the outcome preprocessing in the supplementary leave-one-video-out prediction analysis. Each participant's outcome mean and standard deviation are now estimated from the 27 training videos and applied unchanged to the held-out video; all four models, including the participant-only reference, were refitted. Held-out outcome values do not enter scaling or model fitting. We report RMSE after returning predictions to original outcome units and provide training-standardized errors as a sensitivity analysis. No rating-only or combined versus assigned-label comparison survives correction in either scale (minimum BH-adjusted q=0.611 and 0.179, respectively). The updated table and prediction panel therefore preserve the conclusion without relying on held-out outcome information. These changes concern the supplementary prediction analysis; the separate within-video association and representational analyses are unchanged.
"""
    (RESULTS/"Response_behavioral_correction.txt").write_text(response)
    print("Created Table S9 and three integration snippets from independently verified results.")


if __name__=="__main__":main()
