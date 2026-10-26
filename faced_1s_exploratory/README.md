# Exploratory analyses of the final 1-s segments (FACED)

These analyses use the final second of each released FACED epoch (one 250-sample segment per participant × video; 3,444 segments). They preserve the original main-paper network description and remain exploratory. The complete 30-s epochs are analyzed separately as supplementary validation in `../faced/`.

Main Figure 2 uses one-to-one trial matching and within-participant standardization, followed by participant-clustered OLS for pooled and emotion-resolved synchronization–topology associations. It does not use the later 1,107-row MixedLM scaffold. Regional coupling uses trial-level within-participant standardized nodal strength on synchronization. See `../revision_2026-10-25/corrections/analyses/wpli/PROVENANCE_AND_SCOPE.md` for estimator, input, figure and sensitivity provenance.

The global wPLI source now retains amplitudes in the analytic cross-spectrum and requires matching estimator/source/output cache metadata. All retained global wPLI values were already standard wPLI; all 3,444 rows were independently verified from the raw input. Existing raw or signal caches are not distributed. The cache guard does not verify changed raw input files, so use `force=True` after replacing raw inputs.

## Layout

```
src/lib/                    data loading, network representations, graph metrics, synchronization, statistics
src/part1_topology/         scalar topology of Spearman, MTF and QG networks (q = 30, 10% density)
src/part2_synchrony/        broadband synchronization and synchronization-topology coupling
src/part3_embedding_topology/  embeddings and representational geometry
src/part4_regional/         regional analyses and the composite network figure
src/part5_enrichment/       network and embedding enrichment
src/part6_advanced/         NBS, centrality, simplices, rich club, split-half and sensitivity analyses
src/stimulus_aware/         stimulus deletion, bootstrap, retrieval and held-out decoding
src/behavioral/             behavioral validation and brain-behavior analyses (Figs. S27-S29)
derived/                    per-trial scalar synchronization used by the behavioral and split-half analyses
figures/, tables/           outputs, including the full 1-s topology (144) and synchronization (48) contrast tables
```

## Running

Place the 1-s export at `src/EEG-data-raw/<emotion>/Patient###/trial_*.csv`, then from `src/`:

```bash
python validate_inputs.py --data-root EEG-data-raw        # checks the 3,444 participant-video segments
python run_all.py --n_jobs -1 --n_rand 20 --n_boot 2000    # base pipeline
python part6_advanced/run_density_sweep_q30.py EEG-data-raw     # Fig. S9
python part6_advanced/run_region_coupling_splithalf.py          # Fig. S19
python part6_advanced/run_contrast_splithalf.py ../derived/trial_scalar_synchronization.csv   # Fig. S25
```

The behavioral analyses need the FACED behavioral ratings; `src/behavioral/inputs/NEMAR_SOURCE.md` describes
the source files and their checksums.

Figure S24 (leave-one-participant-out stability) has no recovered generator. Figures S9, S19 and S25 were
regenerated with documented procedures.
