Fresh independent check performed2026-10-25. This is not a recovered historical40-trial record.

Sample40 distinct participant/trial pairs with seed20261025 from the3444-record full-trial manifest sorted by numeric participant/trial id. Compare all30scalp columns and250rows of each supplied1-s CSV against the final250rows of the corresponding7500-row30-s CSV. Exact Decimal comparison retains every exported digit; no floating-point downcast or numerical tolerance is used.

All40 pairs, comprising38participants, matched on300000 EEG values. Original EEG value strings were identical too. Identity metadata and indexing checks all passed. One-second time_index0–249 is local to that file; matching full-epoch indices are7250–7499(samples7251–7500), corresponding to29–30s at250Hz. sample_index differs across export segmentation schemes and is not used to pair files.

The trial-level CSV contains only identifiers, paths, hashes, counts, timing metadata and match booleans; it contains no raw EEG. See check_terminal_segments.py --help for the portable command.
