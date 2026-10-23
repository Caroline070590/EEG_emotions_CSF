#!/usr/bin/env python3
"""NBS threshold sensitivity for the historical 1-s Spearman lineage.

This is a transparent implementation of the analysis described in the
manuscript: participant-level emotion-minus-neutral edge differences,
two-sided supra-threshold components, 5,000 sign-flip permutations, and
BH-FDR across the eight emotion contrasts separately at each threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


EMOTIONS = ["anger", "disgust", "fear", "sadness", "amusement",
            "inspiration", "joy", "tenderness"]
THRESHOLDS = (2.5, 3.0, 3.5)
N_PERMUTATIONS = 5000
N_CHANNELS = 30
UPPER = np.triu_indices(N_CHANNELS, 1)


def participant_differences(matrices: np.ndarray, metadata: pd.DataFrame,
                            emotion: str) -> np.ndarray:
    rows = []
    for subject in sorted(metadata["subject"].unique()):
        target = np.flatnonzero((metadata["subject"] == subject)
                                & (metadata["emotion"] == emotion))
        neutral = np.flatnonzero((metadata["subject"] == subject)
                                 & (metadata["emotion"] == "neutral"))
        if len(target) == 1 and len(neutral) == 1:
            rows.append(matrices[target[0]] - matrices[neutral[0]])
    if not rows:
        raise ValueError(f"No matched participant rows for {emotion}")
    return np.vstack(rows)


def edge_t(differences: np.ndarray) -> np.ndarray:
    standard_error = differences.std(axis=0, ddof=1) / np.sqrt(len(differences))
    return np.divide(differences.mean(axis=0), standard_error,
                     out=np.zeros(differences.shape[1]), where=standard_error > 0)


def largest_component_edges(t_values: np.ndarray, threshold: float) -> int:
    graph = nx.Graph()
    for index in np.flatnonzero(np.abs(t_values) > threshold):
        graph.add_edge(int(UPPER[0][index]), int(UPPER[1][index]))
    if graph.number_of_edges() == 0:
        return 0
    return max(graph.subgraph(nodes).number_of_edges()
               for nodes in nx.connected_components(graph))


def run(matrices: np.ndarray, metadata: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for threshold in THRESHOLDS:
        threshold_rows = []
        for emotion in EMOTIONS:
            differences = participant_differences(matrices, metadata, emotion)
            observed = largest_component_edges(edge_t(differences), threshold)
            null = np.empty(N_PERMUTATIONS, dtype=int)
            for permutation in range(N_PERMUTATIONS):
                signs = rng.choice((-1, 1), size=len(differences))[:, None]
                null[permutation] = largest_component_edges(
                    edge_t(differences * signs), threshold)
            p_value = (1 + np.count_nonzero(null >= observed)) / (N_PERMUTATIONS + 1)
            threshold_rows.append({
                "threshold": threshold,
                "emotion": emotion,
                "component_edges": observed,
                "p_cluster_fwe": p_value,
                "n_permutations": N_PERMUTATIONS,
            })
        q_values = multipletests(
            [row["p_cluster_fwe"] for row in threshold_rows], method="fdr_bh")[1]
        for row, q_value in zip(threshold_rows, q_values):
            row["q_bh_within_threshold"] = q_value
        rows.extend(threshold_rows)
    return pd.DataFrame(rows)


def main() -> None:
    package_root = Path(__file__).resolve().parents[3]
    default_reference = package_root / "results_reference" / "supplementary_reported"
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", type=Path,
                        default=default_reference / "agg_spearman.npy")
    parser.add_argument("--metadata", type=Path,
                        default=default_reference / "agg_meta_spearman.csv")
    parser.add_argument("--output", type=Path,
                        default=default_reference / "TableS_NBS_threshold_sensitivity_regenerated.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matrices = np.load(args.matrices)
    metadata = pd.read_csv(args.metadata)
    required = {"subject", "emotion"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata lacks columns: {sorted(required - set(metadata.columns))}")
    if len(matrices) != len(metadata) or matrices.shape[1] != len(UPPER[0]):
        raise ValueError(f"Unexpected matrix/metadata shapes: {matrices.shape}, {metadata.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(matrices, metadata, args.seed)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
