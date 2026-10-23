# Sourced by slurm/*.sh. Point PATH to the Python environment that has requirements.txt installed.
export PATH="${EEG_ENV:-$HOME/conda_envs/eeg_revision}/bin:$PATH"
export PYTHONNOUSERSITE=1   # ignore ~/.local packages
