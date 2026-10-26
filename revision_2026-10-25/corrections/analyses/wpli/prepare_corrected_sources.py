"""Prepare isolated corrected copies; original release remains untouched."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'EEG_emotions_CSF_release/faced_1s_exploratory/src'
DEST=HERE/'corrected_pipeline/src'
DEST.mkdir(parents=True,exist_ok=True)
records=[]
for p in SOURCE.rglob('*.py'):
    relative=p.relative_to(SOURCE); target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
    records.append({'source':str(p.relative_to(ROOT)), 'copy':str(target.relative_to(HERE)),
                    'original_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
p=DEST/'lib/synchrony.py';s=p.read_text()
s=s.replace('  PLV, PLI, wPLI            (phase synchrony, vectorised via complex phases)',
            '  PLV, PLI                  (phase-difference descriptors)\n  wPLI                      (amplitude-weighted analytic cross-spectrum)')
s=s.replace('import time\n','import time\nimport hashlib\nimport json\n')
s=s.replace('FS = 250\n','FS = 250\nESTIMATOR_VERSION = "analytic_cross_spectrum_wpli_v1"\n')
old='Im = np.imag(z[:, :, None] * np.conj(z)[:, None, :])   # T x n x n'
new='Im = np.imag(A[:, :, None] * np.conj(A)[:, None, :])   # T x n x n; amplitude weights retained'
assert s.count(old)==1;s=s.replace(old,new)
old='''    if f.exists() and not force:
        print("[synchrony] cached"); return pd.read_csv(f)'''
new='''    metadata = CACHE / "trial_synchrony_metrics.metadata.json"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if f.exists() and not force:
        try:
            info = json.loads(metadata.read_text())
            valid = (info.get("estimator_version") == ESTIMATOR_VERSION
                     and info.get("source_sha256") == source_sha
                     and info.get("csv_sha256") == hashlib.sha256(f.read_bytes()).hexdigest())
        except (OSError, ValueError):
            valid = False
        if valid:
            print("[synchrony] verified amplitude-weighted cache"); return pd.read_csv(f)
        print("[synchrony] recomputing cache without matching estimator provenance")'''
assert s.count(old)==1;s=s.replace(old,new)
old='''    df.to_csv(f, index=False)
    print(f"[synchrony]'''
new='''    df.to_csv(f, index=False)
    metadata.write_text(json.dumps({"estimator_version": ESTIMATOR_VERSION,
                                   "source_sha256": source_sha,
                                   "csv_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                   "n_trials": len(df)}, indent=2) + "\\n")
    print(f"[synchrony]'''
assert s.count(old)==1;s=s.replace(old,new)
p.write_text(s)
# Reconstruct the model which actually generated all 240 displayed coupling
# cells.  The released scaffold's later 1,107-cell MixedLM was not that model.
historical=ROOT/'emotion_eeg_paper_updated-FINAL/emotion_eeg_paper_fixed'
p=DEST/'part2_synchrony/run_part2.py'
s=(historical/'part2_synchrony/run_part2.py').read_text()
s=s.replace('interaction-LMM per-emotion slopes', 'participant-clustered OLS interaction slopes')
s=s.replace('emotion_slope_lmm', 'emotion_slope_clustered_ols')
p.write_text(s)
# Keep unrelated released statistics and the explicitly guarded zero-variance
# contrast calculation. Replace only the erroneous coupling implementation.
p=DEST/'lib/stats.py'; s=p.read_text()
old_start=s.index('def emotion_slope_lmm(')
old_end=s.index('\ndef boot_vs_neutral(', old_start)
hs=(historical/'lib/stats.py').read_text()
start=hs.index('def emotion_slope_lmm('); end=hs.index('\ndef boot_vs_neutral(',start)
replacement=hs[start:end].replace('def emotion_slope_lmm(', 'def emotion_slope_clustered_ols(')
replacement+='\n# Historical name retained for importing modules; this estimator is OLS.\nemotion_slope_lmm = emotion_slope_clustered_ols\n\n'
s=s[:old_start]+replacement+s[old_end:];p.write_text(s)
for rec in records:
    rec['corrected_sha256']=hashlib.sha256((HERE/rec['copy']).read_bytes()).hexdigest()
    rec['changed']=rec['original_sha256']!=rec['corrected_sha256']
(HERE/'SOURCE_COPY_MANIFEST.json').write_text(json.dumps({'files':records,
    'changes':['Standard wPLI retains analytic amplitudes; PLV remains unit-phase based.',
               'Global synchronization cache reuse now requires matching estimator/source/CSV metadata; historical untagged caches are recomputed.',
               'Restore verified historical Figure 2 trial-level participant-clustered OLS, with an explicit estimator name; unrelated released statistics remain unchanged.'],
    'historical_coupling_sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [historical/'part2_synchrony/run_part2.py',historical/'lib/stats.py']},
    'originals_unchanged':all(hashlib.sha256((ROOT/r['source']).read_bytes()).hexdigest()==r['original_sha256'] for r in records)},indent=2)+'\n')
print('Prepared',len(records),'source copies; changed global wPLI/cache and verified coupling generator/statistic')
