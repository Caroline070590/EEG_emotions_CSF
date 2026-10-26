#!/usr/bin/env python3
"""Audit the supplied 1-s FACED CSV export without changing source files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

CHANNELS = [
    "FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "CZ", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "PZ", "P3", "P4", "P7", "P8", "PO3", "PO4", "OZ", "O1", "O2",
]
EMOTIONS = {
    "anger", "disgust", "fear", "sadness", "neutral", "amusement",
    "inspiration", "joy", "tenderness",
}


def _number(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def inspect_file(path: Path, root: Path) -> dict:
    frame = pd.read_csv(path)
    rel = path.relative_to(root)
    folder_emotion = rel.parts[0].lower() if rel.parts else ""
    patient_1based = _number(path.parent.name, r"Patient[_-]?(\d+)")
    video_1based = _number(path.stem.replace(".csv", ""), r"trial[_-]?(\d+)")
    missing_channels = [c for c in CHANNELS if c not in frame.columns]

    def unique_value(column):
        if column not in frame:
            return None
        values = frame[column].dropna().unique()
        return values[0] if len(values) == 1 else None

    subject_0based = unique_value("subject_id")
    video_0based = unique_value("trial_id")
    file_emotion = unique_value("emotion_name")
    time_ok = False
    if "time_index" in frame:
        values = pd.to_numeric(frame["time_index"], errors="coerce")
        time_ok = values.notna().all() and values.tolist() == list(range(len(frame)))

    return {
        "path": str(rel),
        "rows": len(frame),
        "columns": len(frame.columns),
        "folder_emotion": folder_emotion,
        "file_emotion": None if file_emotion is None else str(file_emotion).lower(),
        "patient_folder_1based": patient_1based,
        "subject_id_0based": subject_0based,
        "video_filename_1based": video_1based,
        "trial_id_0based": video_0based,
        "missing_channel_count": len(missing_channels),
        "metadata_constant": all(
            c not in frame or frame[c].nunique(dropna=False) == 1
            for c in ("subject_id", "trial_id", "emotion_label", "emotion_name")
        ),
        "time_index_sequential": bool(time_ok),
        "emotion_matches_folder": (
            file_emotion is not None and str(file_emotion).lower() == folder_emotion
        ),
        "subject_offset_matches": (
            patient_1based is not None
            and subject_0based is not None
            and patient_1based == int(subject_0based) + 1
        ),
        "video_offset_matches": (
            video_1based is not None
            and video_0based is not None
            and video_1based == int(video_0based) + 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("input_audit"))
    args = parser.parse_args()

    root = args.data_root.resolve()
    files = sorted(root.glob("*/*/trial_*.csv"))
    compressed = sorted(root.glob("*/*/trial_*.csv.gz"))
    rows = [inspect_file(path, root) for path in files]
    audit = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "input_file_qc.csv", index=False)

    duplicate_keys = 0
    if not audit.empty:
        duplicate_keys = int(
            audit.duplicated(["subject_id_0based", "trial_id_0based"], keep=False).sum()
        )
    checks = {
        "data_root": str(root),
        "csv_files": len(files),
        "csv_gz_files": len(compressed),
        "expected_csv_files": 3444,
        "participants": int(audit["subject_id_0based"].nunique()) if not audit.empty else 0,
        "videos": int(audit["trial_id_0based"].nunique()) if not audit.empty else 0,
        "emotions": sorted(audit["folder_emotion"].dropna().unique().tolist()) if not audit.empty else [],
        "duplicate_participant_video_rows": duplicate_keys,
        "all_250_samples": bool((audit["rows"] == 250).all()) if not audit.empty else False,
        "all_30_channels_present": bool((audit["missing_channel_count"] == 0).all()) if not audit.empty else False,
        "all_metadata_constant": bool(audit["metadata_constant"].all()) if not audit.empty else False,
        "all_time_indices_sequential": bool(audit["time_index_sequential"].all()) if not audit.empty else False,
        "all_emotions_match_folders": bool(audit["emotion_matches_folder"].all()) if not audit.empty else False,
        "all_subject_offsets_match": bool(audit["subject_offset_matches"].all()) if not audit.empty else False,
        "all_video_offsets_match": bool(audit["video_offset_matches"].all()) if not audit.empty else False,
    }
    checks["passes_expected_export_contract"] = bool(
        checks["csv_files"] == 3444
        and checks["csv_gz_files"] == 0
        and checks["participants"] == 123
        and checks["videos"] == 28
        and set(checks["emotions"]) == EMOTIONS
        and checks["duplicate_participant_video_rows"] == 0
        and all(
            checks[name]
            for name in (
                "all_250_samples", "all_30_channels_present", "all_metadata_constant",
                "all_time_indices_sequential", "all_emotions_match_folders",
                "all_subject_offsets_match", "all_video_offsets_match",
            )
        )
    )
    with (args.output_dir / "input_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)
    print(json.dumps(checks, indent=2))
    if not checks["passes_expected_export_contract"]:
        raise SystemExit("Input audit failed; inspect input_file_qc.csv before analysis.")


if __name__ == "__main__":
    main()
