"""Portable input preparation around the unchanged original feature functions.

No data are downloaded or processed on import. Raw files are obtained only by
an explicit call, at the pinned revision, and checked against the source manifest.
The released score grids are used directly, not recreated from another grid.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema_guard import enforce
from src.evaluation.background import non_action_mask
from src.evaluation.replay import load_settings
from . import assembly
from .physical import m4_window_features
from .splits import campaign_of_window, pre_action_mask
from .streaming import stream_m3_multi, stream_prov_multi
from .workspace import TrainingWorkspace, sha256_file

PHASES = ('attack22h', 'benign48h')
SOURCE_NAMES = {
    'M1': {'attack22h':'spade-provenance.json', 'benign48h':'spade-provenance.json'},
    'M2': {'attack22h':'rpi-provenance_rpi.json', 'benign48h':'rpi-provenance.json'},
    'M3': {'attack22h':'modbus.jsonl', 'benign48h':'modbus.jsonl'},
    'M4': {'attack22h':'physical_state.csv', 'benign48h':'physical_state.csv'},
}


def _window(window_s):
    if isinstance(window_s, bool) or window_s not in (5, 10, 30, 60):
        raise ValueError('Use one recorded window duration: 5, 10, 30 or 60 seconds')
    return int(window_s)


def input_inventory(workspace: TrainingWorkspace) -> pd.DataFrame:
    """Eight required files, not the packet captures or the whole dataset repo."""
    manifest = pd.read_csv(workspace.repository / 'data_manifest/provics_download_manifest.csv')
    names = {f'{phase}/{files[phase]}' for files in SOURCE_NAMES.values() for phase in PHASES}
    rows = manifest[manifest['path'].isin(names)].copy()
    if len(rows) != 8 or set(rows['path']) != names:
        raise ValueError('The released source manifest does not contain all eight required inputs')
    return rows.sort_values('path').reset_index(drop=True)


def verify_source(path: Path, record) -> None:
    """Manifest sha is SHA-256 for LFS files, Git-blob SHA-1 for ordinary files."""
    path = Path(path)
    expected_size = int(record['size'])
    expected = str(record['sha'])
    if not path.is_file() or path.stat().st_size != expected_size:
        raise ValueError(f'Source missing or wrong size: {record["path"]}')
    if re.fullmatch(r'[0-9a-f]{64}', expected):
        actual = sha256_file(path)
    elif re.fullmatch(r'[0-9a-f]{40}', expected):
        digest = hashlib.sha1(f'blob {expected_size}\0'.encode())
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        actual = digest.hexdigest()
    else:
        raise ValueError('Unknown source-hash format')
    if actual != expected:
        raise ValueError(f'Source checksum mismatch: {record["path"]}')


def download_inputs(workspace: TrainingWorkspace, log=print) -> pd.DataFrame:
    """Explicit, potentially large download; no PCAPs, no embedded credentials."""
    from huggingface_hub import hf_hub_download
    settings = load_settings()
    for _, record in input_inventory(workspace).iterrows():
        relative = Path('raw') / record['path']
        destination = workspace.output_path(relative)
        if not destination.exists():
            log(f'Downloading {record["path"]} ({int(record["size"]):,} bytes)')
            hf_hub_download(repo_id=settings['dataset']['huggingface_repo'],
                            repo_type='dataset', revision=settings['dataset']['pinned_revision'],
                            filename=record['path'], local_dir=str(workspace.output_path('raw')))
        verify_source(destination, record)
        log(f'Verified {record["path"]}')
    return input_inventory(workspace)


def _source(workspace, modality, phase):
    name = f'{phase}/{SOURCE_NAMES[modality][phase]}'
    record = input_inventory(workspace).set_index('path', drop=False).loc[name]
    path = workspace.output_path(Path('raw') / name)
    verify_source(path, record)
    return path


def load_metadata(workspace: TrainingWorkspace, window_s: int):
    """Read the same score grids and labels used by the saved primary arrays."""
    window_s = _window(window_s)
    settings = load_settings()
    grids = []
    for key in ('attack', 'benign'):
        relative = settings['score_metadata'][f'{key}_pattern'].format(window_s=window_s)
        meta = pd.read_parquet(workspace.repository / relative)
        if not meta['window_start'].is_monotonic_increasing:
            raise ValueError('Metadata must remain in saved chronological row order')
        grids.append(meta)
    events = pd.read_csv(workspace.repository / settings['dataset']['ground_truth_events'])
    for column in ('start', 'end'):
        events[column] = pd.to_datetime(events[column], utc=True)
    return *grids, events


def _verify_prepared(workspace, relative):
    """Reuse only a complete, checksummed prepared block; partial blocks fail closed."""
    directory = workspace.output_path(relative)
    if not directory.exists():
        return False
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'Incomplete prepared block {relative}; choose a fresh workspace')
    records = json.loads(manifest_path.read_text())['files']
    for name, digest in records.items():
        path = workspace.output_path(relative / name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Prepared-input checksum mismatch: {name}')
    return True


def prepare_cyber(workspace: TrainingWorkspace, modality: str, window_s: int, log=print):
    """Extract one modality, in the foreground, using corrected repeated-window merges."""
    window_s = _window(window_s)
    if modality not in ('M1','M2','M3'):
        raise ValueError('Cyber modality must be M1, M2 or M3')
    relative = Path('prepared') / f'w{window_s}' / modality
    if _verify_prepared(workspace, relative):
        log(f'Reusing verified {modality} counters for {window_s}-second windows')
        return
    paths = {phase:_source(workspace,modality,phase) for phase in PHASES}
    workspace.output_path(relative).mkdir(parents=True, exist_ok=False)
    files, counts = {}, {}
    for phase, source in paths.items():
        log(f'Extracting {modality} {phase}; progress stays in this cell')
        streamer = stream_m3_multi if modality == 'M3' else stream_prov_multi
        records, census = streamer(str(source), windows=(window_s,), log=log)
        name = f'{modality}_{phase}_w{window_s}.json'
        files[name] = workspace.write_json(relative / name, records[window_s])
        counts[phase] = census
    workspace.write_json(relative/'manifest.json', {'modality':modality,
        'window_s':window_s, 'files':files, 'census':counts,
        'source_sha256':{phase:sha256_file(path) for phase,path in paths.items()}})
    log(f'Prepared {modality}; repeated fragments were merged, not overwritten')


def prepare_physical(workspace: TrainingWorkspace, window_s: int, log=print):
    """Original 22 x 9 summaries + count; no interpolation or new feature search."""
    window_s = _window(window_s)
    relative = Path('prepared') / f'w{window_s}' / 'M4'
    if _verify_prepared(workspace, relative):
        log(f'Reusing verified M4 features for {window_s}-second windows')
        return
    paths = {phase:_source(workspace,'M4',phase) for phase in PHASES}
    grids = dict(zip(PHASES,load_metadata(workspace,window_s)[:2]))
    variables = json.loads((workspace.repository/'configs/frozen_specs/m4_variables.json').read_text())['process_variables']
    workspace.output_path(relative).mkdir(parents=True, exist_ok=False)
    files = {}
    for phase, source in paths.items():
        data = pd.read_csv(source)
        data = data.drop(columns=[c for c in data.columns if c.startswith('Unnamed') or c in ('result','table')])
        if set(data.columns) != {'_time',*variables}:
            raise ValueError('Physical source schema differs from the recorded 22-variable schema')
        data['_time'] = pd.to_datetime(data['_time'],utc=True,format='ISO8601')
        data = data.sort_values('_time').reset_index(drop=True)
        for column in variables:
            data[column] = pd.to_numeric(data[column],errors='raise')
        features = m4_window_features(data,grids[phase],variables,window_s)
        name = f'm4_{phase}_w{window_s}.parquet'
        files[name] = workspace.write_parquet(relative/name,features)
        log(f'M4 {phase}: {len(features):,} windows, {len(features.columns)} features')
    workspace.write_json(relative/'manifest.json',{'modality':'M4','window_s':window_s,
        'files':files,'source_sha256':{phase:sha256_file(path) for phase,path in paths.items()}})


def load_prepared_views(workspace: TrainingWorkspace, window_s: int):
    """Assemble train-vocabulary features; keep availability outside the model inputs."""
    window_s = _window(window_s)
    meta_a,meta_b,events = load_metadata(workspace,window_s)
    training = np.arange(int(round(len(meta_b)*0.70)))
    evaluation = np.arange(len(meta_b))[len(training):]
    frozen = json.loads((workspace.repository/'configs/frozen_specs/feature_lists_post_screening.json').read_text())
    views = {}
    for modality in ('M1','M2','M3','M4'):
        relative = Path('prepared') / f'w{window_s}' / modality
        if not _verify_prepared(workspace,relative):
            raise FileNotFoundError(f'Prepare {modality} first; its feature block is missing')
        directory = workspace.output_path(relative)
        if modality == 'M4':
            frames = [pd.read_parquet(directory/f'm4_{phase}_w{window_s}.parquet') for phase in PHASES]
            availability = [frame['n_samples_m4'].to_numpy()>0 for frame in frames]
            for frame in frames:
                frame.columns = ['m4_'+column for column in frame.columns]
        else:
            records = [assembly.load_records(str(directory),modality,phase,window_s) for phase in PHASES]
            train_keys = [str(i) for i in assembly.epoch_index(meta_b,window_s)[training]]
            vocabulary = assembly.fit_vocabs(records[1],modality,train_keys)
            frames = [assembly.build_view(record,meta,window_s,modality,vocabulary)
                      for record,meta in zip(records,(meta_a,meta_b))]
            availability = [assembly.availability(record,meta,window_s)>0
                            for record,meta in zip(records,(meta_a,meta_b))]
            frames = [frame.drop(columns=[c for c in frame if c.endswith('__present')]) for frame in frames]
        expected = frozen[f'{modality}_w{window_s}']['features']
        for frame in frames:
            enforce(frame.columns,where=f'{modality} training inputs')
            if list(frame.columns) != expected:
                raise ValueError(f'{modality} feature names/order differ from the frozen artifact; stop, do not retune')
        views[modality] = (*frames,*availability)
    return {'window_s':window_s,'meta_attack':meta_a,'meta_benign':meta_b,'events':events,
        'views':views,'benign_fit':training,'benign_eval':evaluation,
        'action':meta_a['label'].to_numpy()==1,
        'background':non_action_mask(meta_a,events,300),
        'pre_action':pre_action_mask(meta_a,events),'campaign':campaign_of_window(meta_a,events)}
