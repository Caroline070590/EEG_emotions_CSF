SEED-VII GNN evaluation — October 2026

This is an isolated exploratory extension of the FACED participant evaluation.
Read ANALYSIS_PLAN.txt for the fixed protocol and its interpretation.
The existing manuscript and existing FACED jobs are not modified by these scripts.

Three models receive the same 5,520 inputs: hybrid GNN, summary-first GNN and LDA.
Five folds hold out four of the twenty participants each. Models are trained within
SEED-VII to predict its seven conditions, with familiar clips and sessions across
training and test participants. This is not zero-shot FACED-to-SEED weight transfer.

Cluster execution:
  preflight: verify all source/data hashes, run the tests, initialize configuration;
  analysis array: 3 observed models + 297 full null refits + 4 neural seed checks;
  report: recording-session-only diagnostic, complete-result validation, tables
          and figures, including paired comparisons and feature-group rankings.

run.py and batch.py can also run locally with the existing seed7_replication/cache.
The production cluster results use a separate configuration because software
versions and input paths differ. Never mix local and cluster runs within a report.

The smoke/ directory is SOFTWARE VALIDATION ONLY: one epoch and two feature
shuffles, not the planned 40-epoch/30-shuffle analysis. Do not cite its figures or
model scores in the manuscript. cluster_results/ is the production result folder.

The source-block null preserves labels within each source-by-original-condition
block; it does not preserve session-condition counts or the relationship between
the two condition blocks of a mixed-condition source. The session-only comparison
is a diagnostic for recording-design information, not a conditional permutation
test. Above-null accuracy alone does not establish emotion-specific physiology.

Feature rankings quantify the fitted model's reliance on predefined input groups.
Intervals are descriptive participant-bootstrap intervals conditional on the fitted
models and this stimulus/session set; correlated groups are not additive.
