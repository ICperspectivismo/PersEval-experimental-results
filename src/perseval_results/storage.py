"""Relative managed paths, bounded reads, and immutable run registry descriptors."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from .bundle import MAX_FILE, MAX_TOTAL, validate_files
from .protocol import Contract, ProtocolError, canonical_json, canonical_path, digest, read_json
from .lfs import pointer, stored_reference

MANAGED = ('runs', 'registry')


def safe_read(path: Path, limit=MAX_FILE):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ProtocolError(f'not a bounded regular file: {path}')
            data = stream.read(limit + 1)
            if len(data) > limit:
                raise ProtocolError(f'file exceeds size limit: {path}')
            return data
    except OSError as exc:
        raise ProtocolError(f'cannot safely read {path}: {exc}') from exc


def tree_files(directory: Path, *, bundle=False):
    """Reject symlinks/special files at every depth, including the root."""
    if directory.is_symlink():
        raise ProtocolError(f'symlink source directory: {directory}')
    if not directory.exists():
        return {}
    if not directory.is_dir():
        raise ProtocolError(f'source must be a directory: {directory}')
    files = {}
    size = 0
    for path in sorted(directory.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ProtocolError(f'symlink or special source file: {path}')
        name = canonical_path(path.relative_to(directory).as_posix())
        data = safe_read(path)
        size += len(data)
        if bundle and size > MAX_TOTAL:
            raise ProtocolError('stored run exceeds expanded payload size limit')
        files[name] = data
    return files


def managed_snapshot(root: Path):
    return {f'{kind}/{name}': data for kind in MANAGED for name, data in tree_files(root / kind).items()}


def scientific_digest(files, docs, *, allow_lfs=False):
    """Copy-detection signature, not an assertion that executions were independent.

    Excludes execution UUID/seed/timestamps/environment/notes and selected artifacts.
    Equal predictions/scores under an equal configuration need explicit acknowledgement.
    """
    return digest(canonical_json({
        'profile':'perseval-scientific-content-v1',
        'configuration_id':docs['run']['configuration_id'],
        'scoring_rule':docs['metrics']['scoring_rule'],
        'metrics':docs['metrics'],
        'predictions_digest':payload_digest(files['predictions.csv'],allow_lfs),
    }))


def payload_digest(data, allow_lfs):
    ref = pointer(data) if allow_lfs else None
    return ref[0] if ref is not None else digest(data)


def run_index(files, docs, *, allow_lfs=False):
    run = docs['run']
    def size(name):
        ref = stored_reference(name,files[name],files['checksums.sha256']) if allow_lfs and (name=='predictions.csv' or name.startswith('artifacts/')) else None
        return ref[1] if ref is not None else len(files[name])
    def file_digest(name):
        ref = stored_reference(name,files[name],files['checksums.sha256']) if allow_lfs and (name=='predictions.csv' or name.startswith('artifacts/')) else None
        return ref[0] if ref else digest(files[name])
    return {'schema_version':'1.0', 'run_id':run['run_id'],
            'configuration_id':run['configuration_id'],
            'payload_digest':docs['manifest']['payload_digest'],
            'scientific_digest':scientific_digest(files, docs,allow_lfs=allow_lfs),
            'predictions_digest':payload_digest(files['predictions.csv'],allow_lfs),
            'path':f'runs/{run["run_id"]}',
            'payload_bytes':sum(size(key) for key in files),
            'artifact_bytes':sum(size(key) for key in files if key.startswith('artifacts/')),
            'files':{name:{'sha256':file_digest(name),
                           'size':size(name)} for name,data in sorted(files.items())}}


def revision_path(record):
    return f'registry/submissions/{record["submission_id"]}/revisions/{record["revision"]:06d}.json'


def current_path(sid):
    return f'registry/submissions/{sid}/current.json'


def current_pointer(record):
    return {'schema_version':'1.0','submission_id':record['submission_id'],
            'revision':record['revision'],'revision_path':revision_path(record),
            'revision_digest':digest(canonical_json(record))}


def duplicate_groups(indices):
    groups = {}
    for rid,index in sorted(indices.items()):
        groups.setdefault(index['scientific_digest'],[]).append(rid)
    return {key:ids for key,ids in groups.items() if len(ids)>1}


def duplicate_notice(signature, run_ids):
    return f'Identical scientific payload {signature} under run IDs {", ".join(sorted(run_ids))}; explicitly included, independent executions not verified.'


def validate_run_consistency(runs, indices):
    """Equal known test/configuration, predictions and rule cannot yield different metrics."""
    seen = {}
    for rid,docs in sorted(runs.items()):
        if docs['run']['configuration']['split']['test_fingerprint'] is None:
            continue
        key = digest(canonical_json({'configuration_id':docs['run']['configuration_id'],
                                     'predictions_digest':indices[rid]['predictions_digest'],
                                     'scoring_rule':docs['metrics']['scoring_rule']}))
        if key in seen:
            other = seen[key]
            if docs['metrics'] != runs[other]['metrics']:
                raise ProtocolError(f'inconsistent metrics for identical predictions/configuration/scoring: {other}, {rid}')
        else:
            seen[key] = rid
