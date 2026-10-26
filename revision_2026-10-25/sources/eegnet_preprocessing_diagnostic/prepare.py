"""Prepare two signal arrays shared by three equal-channel 10-s recipes."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy.signal import butter, resample, resample_poly, sosfiltfilt
from protocol import digest, read_config, save_json, SIGNAL_NAMES


def process_trial(raw, recipe):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (30, 7500) or not np.isfinite(raw).all():
        raise ValueError('Expected a finite 30-channel, 30-second trial at 250 Hz')
    if recipe in ('trial_fourier_div100','trial_fourier_zscore'):
        # Official FACED script resamples the complete trial before segmentation.
        full = resample(raw, 6000, axis=1)
        windows = full.reshape(30, 3, 2000).transpose(1,0,2)
    elif recipe == 'window_filtered_zscore':
        sos = butter(4, [1.,45.], fs=200, btype='bandpass', output='sos')
        windows = np.stack([sosfiltfilt(sos, resample_poly(raw[:, j*2500:(j+1)*2500], 4, 5, axis=1), axis=1)
                            for j in range(3)])
    else:
        raise ValueError('Unknown recipe')
    result = np.ascontiguousarray(windows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite processed EEG')
    return result


def prepare(config_path):
    config_path = Path(config_path); config = read_config(config_path)
    out = config_path.parent
    if digest(config['raw_path']) != config['input_sha256']['raw']:
        raise ValueError('Raw EEG hash mismatch')
    if digest(out / 'metadata.npz') != config['input_sha256']['metadata']:
        raise ValueError('Metadata hash mismatch')
    if (out / 'prepared.json').exists():
        manifest = json.loads((out / 'prepared.json').read_text())
        if manifest['config_sha256'] != digest(config_path):
            raise ValueError('Prepared configuration mismatch')
        for path, expected in manifest['artifact_sha256'].items():
            if digest(out / path) != expected:
                raise ValueError('Prepared artifact hash mismatch')
        return manifest
    raw = np.load(config['raw_path'], mmap_mode='r', allow_pickle=False)
    artifacts = {}
    for recipe in ['window_filtered_zscore', 'trial_fourier_div100']:
        path = out / SIGNAL_NAMES[recipe]
        temporary = path.with_suffix('.partial.npy')
        signal = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(10332,30,2000))
        for i in range(len(raw)):
            signal[i*3:(i+1)*3] = process_trial(raw[i], recipe)
            if i % 280 == 0:
                print(f'{recipe}: {i}/{len(raw)} trials', flush=True)
        signal.flush(); del signal
        os.replace(temporary, path)
        artifacts[path.name] = digest(path)
    manifest = {'config_sha256':digest(config_path), 'artifact_sha256':artifacts,
                'shape':[10332,30,2000], 'dtype':'float32'}
    save_json(out / 'prepared.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    prepare(parser.parse_args().config)
