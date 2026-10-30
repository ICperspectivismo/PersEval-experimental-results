"""Bounded, non-extracting bundle reader. No datasets, model code, or network."""
from __future__ import annotations

import csv
import gzip
import io
import os
import stat
import tarfile
import zlib
from pathlib import Path

from .protocol import (Contract, ProtocolError, canonical_json, canonical_path,
                       checksum_bytes, configuration_id, digest, read_json, timestamp)
from .lfs import stored_reference

CORE = {'manifest.json', 'run.json', 'metrics.json', 'predictions.csv', 'environment.json', 'checksums.sha256'}
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_FILE = 32 * 1024 * 1024
MAX_FILES = 256


def read_archive(path: Path) -> dict[str, bytes]:
    path = Path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ProtocolError('bundle input must be a regular file')
    if path.stat().st_size > MAX_ARCHIVE:
        raise ProtocolError('compressed archive exceeds 64 MiB')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ProtocolError('bundle input must be a regular file')
        if info.st_size > MAX_ARCHIVE:
            raise ProtocolError('compressed archive exceeds 64 MiB')
        compressed = source.read(MAX_ARCHIVE + 1)
        if len(compressed) > MAX_ARCHIVE:
            raise ProtocolError('compressed archive exceeds 64 MiB')
    files = {}
    total = 0
    raw_total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as stream:
            def take(size):
                nonlocal raw_total
                data = stream.read(size)
                raw_total += len(data)
                if raw_total > MAX_TOTAL + (MAX_FILES * 1024) + 10240:
                    raise ProtocolError('uncompressed archive exceeds size limits')
                if len(data) != size:
                    raise ProtocolError('truncated tar archive')
                return data

            while True:
                header = take(512)
                if header == bytes(512):
                    if take(512) != bytes(512):
                        raise ProtocolError('tar requires two zero end blocks')
                    # Reject appended archives/data; bound even zero-padding decompression.
                    while True:
                        tail = stream.read(8192)
                        raw_total += len(tail)
                        if raw_total > MAX_TOTAL + (MAX_FILES * 1024) + 10240:
                            raise ProtocolError('uncompressed archive exceeds size limits')
                        if not tail:
                            break
                        if any(tail):
                            raise ProtocolError('nonzero trailing tar data')
                    break
                if header[257:263] != b'ustar\0':
                    raise ProtocolError('only USTAR regular file headers are allowed')
                member = tarfile.TarInfo.frombuf(header, 'ascii', 'strict')
                name = canonical_path(member.name)
                if len(files) >= MAX_FILES or name in files:
                    raise ProtocolError('too many members or duplicate archive member')
                if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    raise ProtocolError('only regular, non-sparse, non-PAX files are allowed')
                if (member.mode, member.uid, member.gid, member.mtime, member.uname, member.gname) != (0o644, 0, 0, 0, '', ''):
                    raise ProtocolError('noncanonical tar metadata')
                total += member.size
                if member.size < 0 or member.size > MAX_FILE or total > MAX_TOTAL:
                    raise ProtocolError('uncompressed archive exceeds size limits')
                files[name] = take(member.size)
                padding = (-member.size) % 512
                if any(take(padding)):
                    raise ProtocolError('nonzero tar file padding')
    except (tarfile.TarError, tarfile.HeaderError, EOFError, OSError, UnicodeError, zlib.error) as exc:
        raise ProtocolError(f'invalid tar.gz: {exc}') from exc
    return files


def pack_files(files: dict[str, bytes]) -> bytes:
    """Reference deterministic tar.gz encoder for conformance, not a run recorder."""
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode='w', format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(files):
                canonical_path(name)
                info = tarfile.TarInfo(name)
                info.size = len(files[name])
                info.mode = 0o644
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ''
                archive.addfile(info, io.BytesIO(files[name]))
    return raw.getvalue()


def seal_payload(payload: dict[str, bytes], contract: Contract, artifacts=None) -> dict[str, bytes]:
    """Seal an already frozen payload; execution snapshots must have been persisted."""
    run = read_json(payload['run.json'])
    files = dict(payload)
    files['manifest.json'] = canonical_json({
        'schema_version': '1.0', 'schema_digest': contract.schema_digest,
        'run_id': run['run_id'], 'configuration_id': run['configuration_id'],
        'payload_digest': digest(checksum_bytes(payload)), 'artifacts': artifacts or [],
    })
    files['checksums.sha256'] = checksum_bytes(files)
    return files


def validate_files(files: dict[str, bytes], contract: Contract, *, allow_lfs=False):
    if not CORE <= files.keys():
        raise ProtocolError(f'missing required files: {sorted(CORE - files.keys())}')
    pointers = {}
    if allow_lfs:
        for name,data in files.items():
            if name=='predictions.csv' or name.startswith('artifacts/'):
                ref = stored_reference(name,data,files['checksums.sha256'])
                if ref is not None:
                    pointers[name] = ref
    sizes = {name:pointers[name][1] if name in pointers else len(data) for name,data in files.items()}
    if len(files) > MAX_FILES or sum(sizes.values()) > MAX_TOTAL:
        raise ProtocolError('bundle exceeds size limits')
    for name, data in files.items():
        canonical_path(name)
        if sizes[name] > MAX_FILE:
            raise ProtocolError('file exceeds size limit')
    sums = {name:pointers[name][0] if name in pointers else digest(data)
            for name,data in files.items() if name!='checksums.sha256'}
    def inventory(names):
        return ''.join(f'{sums[name][7:]}  {name}\n' for name in sorted(names)).encode('ascii')
    if files['checksums.sha256'] != inventory(sums):
        raise ProtocolError('checksums must list every payload file exactly once, in canonical order')
    docs = {name: read_json(files[name + '.json']) for name in ('manifest', 'run', 'metrics', 'environment')}
    for name, document in docs.items():
        contract.validate(name, document)
    manifest, run, metrics = (docs[key] for key in ('manifest', 'run', 'metrics'))
    if manifest['schema_digest'] != contract.schema_digest:
        raise ProtocolError('unsupported schema digest (version alone is insufficient)')
    if run['configuration_id'] != configuration_id(run['configuration']):
        raise ProtocolError('configuration_id does not match configuration')
    if any(manifest[key] != run[key] for key in ('run_id', 'configuration_id')):
        raise ProtocolError('manifest and run identity disagree')
    payload = {k: v for k, v in files.items() if k not in ('manifest.json', 'checksums.sha256')}
    if manifest['payload_digest'] != digest(inventory(payload)):
        raise ProtocolError('immutable payload digest mismatch')
    if any(len(name) > 100 for name in files):
        raise ProtocolError('v1 bundle paths must fit in 100 ASCII bytes')
    if any(other.startswith(name + '/') for name in files for other in files if name != other):
        raise ProtocolError('file path also used as a directory')
    artifact_paths = [item['path'] for item in manifest['artifacts']]
    if len(set(artifact_paths)) != len(artifact_paths) or set(files) != CORE | set(artifact_paths):
        raise ProtocolError('only explicitly declared artifacts may accompany the core files')
    start, finish = (timestamp(run['execution'][key]) for key in ('started_at', 'finished_at'))
    if finish < start:
        raise ProtocolError('execution finished before it started')
    if 'predictions.csv' not in pointers:
        validate_predictions(files['predictions.csv'],metrics,run)
    # Full import always requires real payloads. A metadata-only source build checks
    # pointer OIDs against ordinary-Git checksums without claiming CSV verification.
    validate_evaluation_report({name:data for name,data in files.items() if name not in pointers}, docs)
    return docs


def validate_predictions(predictions, metrics, run):
    try:
        text = predictions.decode('utf-8')
        rows = list(csv.reader(io.StringIO(text, newline=''), strict=True))
        if not rows or rows[0] != ['user_id', 'text_id', 'predictions']:
            raise ProtocolError('prediction columns must be exactly user_id,text_id,predictions')
        data = rows[1:]
        if not data or any(len(row) != 3 or not all(value.strip() for value in row) for row in data):
            raise ProtocolError('predictions must contain three nonempty string cells per row')
        pairs = [(row[0], row[1]) for row in data]
        if pairs != sorted(set(pairs)):
            raise ProtocolError('prediction pairs must be unique and sorted lexicographically')
        out = io.StringIO(newline='')
        csv.writer(out, lineterminator='\n').writerows(rows)
        if out.getvalue() != text:
            raise ProtocolError('CSV must use canonical UTF-8, minimal quoting, and LF bytes')
        if len(data) != metrics['prediction_count']:
            raise ProtocolError('prediction count differs from metrics')
        mapping = run['provenance']['label_mapping']
        if mapping is not None and any(row[2] not in mapping.values() for row in data):
            raise ProtocolError('prediction label absent from label_mapping')
    except (csv.Error, UnicodeError) as exc:
        raise ProtocolError(f'invalid predictions CSV: {exc}') from exc


def validate_bundle(path: Path, contract: Contract):
    files = read_archive(path)
    return files, validate_files(files, contract)


def validate_evaluation_report(files, docs):
    """Cross-check PersEval's optional generated report; never pretend to have gold data."""
    path = 'artifacts/perseval-evaluation.json'
    if path not in files:
        return
    report = read_json(files[path])
    if not isinstance(report,dict) or report.get('format') != 'perseval-evaluation-v1':
        raise ProtocolError('unsupported generated evaluation report')
    try:
        if report['scoring_rule'] != docs['metrics']['scoring_rule']:
            raise ProtocolError('evaluation report scoring rule differs from metrics')
        for compact, full in [('global_f1','global'),('user_f1','user_average'),('text_f1','text_average')]:
            if report[full]['macro avg']['f1-score'] != docs['metrics']['compact'][compact]:
                raise ProtocolError(f'inconsistent metrics and evaluation report: {compact}')
        if report['global']['macro avg']['support'] != docs['metrics']['prediction_count']:
            raise ProtocolError('inconsistent global support and prediction count')
        task = report['task_score']
        if task['kind'] == 'macro-f1':
            expected = docs['metrics']['compact']['global_f1']
        elif task['kind'] == 'positive-class-f1':
            label = task['positive_label']
            expected = report['global'].get(label,{}).get('f1-score',0.0) if label is not None else None
        else:
            raise ProtocolError('unsupported generated task score')
        if task['value'] != expected:
            raise ProtocolError('inconsistent task score in evaluation report')
    except (KeyError,TypeError) as exc:
        raise ProtocolError(f'invalid generated evaluation report: {exc}') from exc
