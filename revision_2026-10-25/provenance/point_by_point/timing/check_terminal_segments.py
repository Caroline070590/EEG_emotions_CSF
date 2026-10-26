#!/usr/bin/env python3
"""Fresh independent terminal-segment check; no recovery of the historical sample.

python check_terminal_segments.py --workspace-root /path/to/Emotions-eeg --out /tmp/timing
Uses Python standard library only. Reads raw local CSVs; exports only ids, file hashes,
counts, timing metadata and exact-match booleans, never EEG sample values.
"""
import argparse
import collections
import csv
import datetime
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import random

CHANNELS = ['FP1','FP2','FZ','F3','F4','F7','F8','FC1','FC2','FC5','FC6','CZ','C3','C4','T7','T8','CP1','CP2','CP5','CP6','PZ','P3','P4','P7','P8','PO3','PO4','OZ','O1','O2']

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        data = list(reader)
    return header, data

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--n-trials', type=int, default=40)
    args = ap.parse_args()
    root = args.workspace_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_rel = 'EEG-data-raw-multiscale/metadata/manifest_full_trial.csv'
    _, manifest = rows(root / manifest_rel)
    ordered = sorted(manifest, key=lambda x: (int(x['subject_id']), int(x['trial_id']), x['csv_file']))
    if len({(x['subject_id'],x['trial_id']) for x in ordered}) != len(ordered):
        raise ValueError('Duplicate participant/trial identities in manifest')
    indices = sorted(random.Random(args.seed).sample(range(len(ordered)), args.n_trials))
    results = []
    for sample_order, idx in enumerate(indices):
        entry = ordered[idx]
        one_rel = str(Path('EEG-data-raw')/entry['emotion_name']/entry['patient_folder']/f"trial_{int(entry['trial_id'])+1:03d}.csv")
        full_rel = entry['csv_file']
        one_path, full_path = root/one_rel, root/full_rel
        one_header, one = rows(one_path)
        full_header, full = rows(full_path)
        one_channels = [x for x in one_header if x in CHANNELS]
        full_channels = [x for x in full_header if x in CHANNELS]
        if one_channels != CHANNELS or full_channels != CHANNELS:
            raise ValueError(f'Unexpected scalp-channel set/order in {one_rel} or {full_rel}')
        if len(one) != 250 or len(full) != 7500:
            raise ValueError(f'Unexpected sample count in {one_rel} or {full_rel}')
        tail = full[-250:]
        unequal_numeric = 0
        unequal_strings = 0
        nonfinite = 0
        # Decimal reads all digits as exported, without float32 or float64 rounding.
        for a,b in zip(one,tail):
            for ch in CHANNELS:
                av,bv = Decimal(a[ch]),Decimal(b[ch])
                if not av.is_finite() or not bv.is_finite():
                    nonfinite += 1
                if av != bv:
                    unequal_numeric += 1
                if a[ch] != b[ch]:
                    unequal_strings += 1
        identity_ok = all(str(row[k]) == entry[k] for rows_ in (one,full) for row in rows_ for k in ('subject_id','trial_id','emotion_name','emotion_label'))
        times_ok = [int(x['time_index']) for x in one] == list(range(250)) and [int(x['time_index']) for x in full] == list(range(7500))
        results.append(dict(sample_order=sample_order,sorted_manifest_index=idx,subject_id=entry['subject_id'],patient_folder=entry['patient_folder'],trial_id=entry['trial_id'],emotion_name=entry['emotion_name'],one_second_path=one_rel,thirty_second_path=full_rel,one_second_sha256=sha(one_path),thirty_second_sha256=sha(full_path),one_second_rows=len(one),thirty_second_rows=len(full),scalp_channels=len(CHANNELS),compared_values=250*len(CHANNELS),unequal_decimal_values=unequal_numeric,unequal_text_values=unequal_strings,nonfinite_values=nonfinite,all_values_exact_match=unequal_numeric==0 and nonfinite==0,identity_metadata_match=identity_ok,time_index_match_expected=times_ok,one_second_time_index_start=one[0]['time_index'],one_second_time_index_end=one[-1]['time_index'],full_terminal_time_index_start=tail[0]['time_index'],full_terminal_time_index_end=tail[-1]['time_index'],one_second_sample_index=one[0]['sample_index'],thirty_second_sample_index=full[0]['sample_index']))
    csv_path=args.out/'TRIAL_LEVEL_TERMINAL_MATCH.csv'
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    ok=all(x['all_values_exact_match'] and x['identity_metadata_match'] and x['time_index_match_expected'] for x in results)
    report=dict(status='pass' if ok else 'mismatch',check_type='Fresh independent verification on 2026-10-25; does not claim recovery of the historical40-trial sample.',generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=args.seed,sampling='random.Random(seed).sample of indices into manifest sorted by numeric subject_id, numeric trial_id, csv_file; sampled indices sorted for reporting',manifest_path=manifest_rel,manifest_sha256=sha(root/manifest_rel),manifest_trials=len(ordered),sampled_trials=len(results),sampled_participants=len(set(x['subject_id']for x in results)),sampled_conditions=dict(collections.Counter(x['emotion_name']for x in results)),matched_trials=sum(x['all_values_exact_match']for x in results),total_eeg_values_compared=sum(x['compared_values']for x in results),unequal_decimal_values=sum(x['unequal_decimal_values']for x in results),unequal_text_values=sum(x['unequal_text_values']for x in results),comparison_precision='Exact Decimal numeric equality on every exported CSV digit; no float32/float64 conversion, no tolerance.',channels=CHANNELS,match_location=dict(zero_based_full_epoch_indices=[7250,7499],one_based_samples=[7251,7500],fs_hz=250,interval_seconds=[29,30],one_second_local_indices=[0,249]),metadata_notes=['One-second files have local time_index0–249; full files have time_index0–7499. Position is established by sample values, not by assuming equal index columns.','sample_index describes different exported segmentation units in the two trees and is recorded, not used as a pairing key.','Pairs are resolved by participant, trial and emotion identifiers; the30 scalp columns are checked in explicit fixed order.','Full files contain extra clip/window/valence/time_sec metadata columns. These are not EEG channels.','All result files omit raw EEG values.'],trial_results_csv=csv_path.name,trial_results_sha256=sha(csv_path),script_sha256=sha(Path(__file__).resolve()))
    (args.out/'TERMINAL_MATCH_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ['status','sampled_trials','sampled_participants','matched_trials','total_eeg_values_compared','unequal_decimal_values','unequal_text_values']},indent=2))
    if not ok:raise SystemExit(1)

if __name__=='__main__':main()
