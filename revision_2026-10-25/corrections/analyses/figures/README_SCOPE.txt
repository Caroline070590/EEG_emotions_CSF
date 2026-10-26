Current main Figure 3 reconstruction

Run part4_regional/run_figure5_corrected.py from this folder. Historical filenames
contain Figure5; the current manuscript numbers this composite Figure3 across
two continued pages. The cache directory contains derived aggregate graph
matrices and metadata, not EEG time series. The regional coefficient table is
verified against raw-derived standard wPLI in the adjacent wpli audit.

The copied library snapshot is retained for the figure's historical imports.
This renderer does not import or execute lib/synchrony.py. For recomputation of
global synchronization use the corrected public source at
faced_1s_exploratory/src/lib/synchrony.py in the repository root, not an unused
historical helper in this figure-only snapshot. Numerical estimator corrections
and historical source identities remain explicitly separated.
