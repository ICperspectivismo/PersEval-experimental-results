"""Immutable source records, explicit aggregation, and backend-only ranking."""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from .bundle import validate_files
from .storage import (tree_files, run_index, current_pointer, duplicate_groups, duplicate_notice, validate_run_consistency)
from .protocol import Contract, ProtocolError, canonical_json, digest, read_json, timestamp

METRICS = ('global_f1', 'user_f1', 'text_f1')
GROUP_FIELDS = ('benchmark_version', 'dataset', 'label', 'task', 'adaptation', 'extended', 'split_protocol', 'scoring_rule')


def limitations(run):
    c, p = run['configuration'], run['provenance']
    missing = []
    for name, value in {
        'dataset revision': c['dataset']['revision'], 'dataset fingerprint': c['dataset']['fingerprint'],
        'test fingerprint': c['split']['test_fingerprint'], 'split seed': c['split']['seed'],
        'model revision': c['model']['revision'], 'execution seed': run['seed'],
        'PersEval Git revision': p['perseval_git_revision'], 'label mapping': p['label_mapping'],
        'split provenance': p['split_provenance'],
    }.items():
        if value is None:
            missing.append(f'Unavailable: {name}')
    if p['perseval_git_dirty'] is not False:
        missing.append('PersEval working tree was dirty or its state is unknown')
    return missing


def compatible(run_ids, runs):
    if len(set(run_ids)) != len(run_ids) or not run_ids:
        raise ProtocolError('run references must be nonempty and unique')
    if any(rid not in runs for rid in run_ids):
        raise ProtocolError('submission references an unknown run')
    docs = [runs[rid] for rid in run_ids]
    if len({doc['run']['configuration_id'] for doc in docs}) != 1 or len({doc['metrics']['scoring_rule'] for doc in docs}) != 1:
        raise ProtocolError('all included runs must share configuration_id and scoring rule')
    if len(docs) > 1 and docs[0]['run']['configuration']['split']['test_fingerprint'] is None:
        raise ProtocolError('cannot establish multi-run compatibility without a test fingerprint')
    if len({canonical_json(doc['run']['provenance']['label_mapping']) for doc in docs}) != 1:
        raise ProtocolError('included runs must share their recorded label mapping')
    for key in METRICS:
        present = [doc['metrics']['compact'][key] is not None for doc in docs]
        if any(present) and not all(present):
            raise ProtocolError(f'inconsistent metric availability: {key}')
    return docs


def aggregate(docs):
    result = {}
    for key in METRICS:
        values = [d['metrics']['compact'][key] for d in docs]
        result[key] = {'mean': statistics.mean(values) if values[0] is not None else None,
                       'stddev': statistics.stdev(values) if len(values) > 1 and values[0] is not None else None}
    return result


def _files(directory):
    return tree_files(directory, bundle=True)


def validate_sources(root: Path, contract: Contract, *, metadata_only=False):
    # The bootstrap layout never held real results. Fail instead of silently omitting legacy data.
    for kind in ('runs', 'submissions', 'reported'):
        legacy = root / 'records' / kind
        if legacy.is_symlink() or (legacy.exists() and any(legacy.iterdir())):
            raise ProtocolError('legacy records/ sources require an explicit migration; no data was discarded')
    for kind in ('runs', 'registry'):
        tree_files(root / kind)  # Validate all ancestors/members before using any registry paths.
    registry = root / 'registry'
    if registry.exists() and any(p.name not in ('runs','submissions') for p in registry.iterdir()):
        raise ProtocolError('unexpected registry file or directory')
    runs, indices = {}, {}
    for directory in sorted((root / 'runs').glob('*')):
        if directory.is_symlink() or not directory.is_dir():
            raise ProtocolError(f'run source must be a directory: {directory}')
        files = _files(directory)
        index_path = registry / 'runs' / f'{directory.name}.json'
        if not index_path.is_file():
            raise ProtocolError(f'missing immutable run registry: {directory.name}')
        actual = read_json(index_path.read_bytes())
        contract.validate('run-index',actual,record=True)
        # Ordinary-Git inventory permits an offline metadata build with absent optional
        # payloads. Synthetic references never leave this validation function and do
        # not claim that predictions or artifact bytes have been inspected.
        if metadata_only:
            from .protocol import canonical_path
            for name, item in actual.get('files', {}).items():
                canonical_path(name)
                if name not in files and (name == 'predictions.csv' or name.startswith('artifacts/')):
                    files[name] = (f'version https://git-lfs.github.com/spec/v1\noid sha256:{item["sha256"][7:]}\nsize {item["size"]}\n').encode('ascii')
        docs = validate_files(files, contract,allow_lfs=metadata_only)
        rid = docs['run']['run_id']
        if directory.name != rid:
            raise ProtocolError('run directory and run_id disagree')
        expected = run_index(files, docs,allow_lfs=metadata_only)
        index_path = registry / 'runs' / f'{rid}.json'
        if not index_path.is_file():
            raise ProtocolError(f'missing immutable run registry: {rid}')
        actual = read_json(index_path.read_bytes())
        contract.validate('run-index',actual,record=True)
        if 'files' not in actual:  # Backward-compatible descriptors require pointer files.
            expected.pop('files')
        if actual != expected:
            raise ProtocolError(f'run registry differs from immutable run payload: {rid}')
        runs[rid], indices[rid] = docs, expected
    found = {p.name for p in (registry / 'runs').glob('*')}
    if found != {rid+'.json' for rid in runs}:
        raise ProtocolError('run registry contains unreferenced or unexpected files')
    validate_run_consistency(runs,indices)
    histories = {}
    for directory in sorted((registry / 'submissions').glob('*')):
        if not directory.is_dir():
            raise ProtocolError('submission source must be a directory')
        if {p.name for p in directory.iterdir()} != {'revisions','current.json'}:
            raise ProtocolError('submission must contain only revisions/ and current.json')
        revisions = directory / 'revisions'
        if not revisions.is_dir():
            raise ProtocolError('revisions must be a directory')
        history = []
        for index,path in enumerate(sorted(revisions.iterdir()),1):
            if not path.is_file() or path.name != f'{index:06d}.json':
                raise ProtocolError('revision paths must be consecutive canonical JSON filenames')
            record = read_json(path.read_bytes())
            kind = record.get('source_type')
            schema = 'submission' if kind == 'measured' else 'reported' if kind == 'reported' else None
            if schema is None:
                raise ProtocolError('unknown submission source type')
            contract.validate(schema,record,record=True)
            timestamp(record['published_at'])
            if record['submission_id'] != directory.name or record['revision'] != index:
                raise ProtocolError('revision identity and path disagree')
            expected = digest(canonical_json(history[-1])) if history else None
            if record['previous_revision_digest'] != expected:
                raise ProtocolError('broken revision digest chain')
            if record['superseded_by'] == record['submission_id'] or (record['status'] != 'superseded' and record['superseded_by'] is not None):
                raise ProtocolError('invalid supersession reference')
            if history:
                previous = history[-1]
                if previous['status'] != 'published':
                    raise ProtocolError('withdrawn and superseded submissions are terminal')
                if record['source_type'] != previous['source_type']:
                    raise ProtocolError('submission source type is immutable')
                if timestamp(record['published_at']) < timestamp(previous['published_at']):
                    raise ProtocolError('revision time moved backwards')
                if kind == 'measured' and not set(previous['run_ids']) <= set(record['run_ids']):
                    raise ProtocolError('run additions must preserve previously included runs')
                for key in ('configuration_id','scoring_rule') if kind == 'measured' else ('model','group','paper_variant','values','citation','experiment'):
                    if record.get(key) != previous.get(key):
                        raise ProtocolError(f'revision changed immutable field: {key}')
            if kind == 'measured':
                docs = compatible(record['run_ids'],runs)
                if record['configuration_id'] != docs[0]['run']['configuration_id'] or record['scoring_rule'] != docs[0]['metrics']['scoring_rule']:
                    raise ProtocolError('submission configuration/scoring differs from its runs')
                required = {item for doc in docs for item in limitations(doc['run'])}
                for signature,ids in duplicate_groups({rid:indices[rid] for rid in record['run_ids']}).items():
                    required.add(duplicate_notice(signature,ids))
                if not required <= set(record['provenance']['limitations']):
                    raise ProtocolError('submission must disclose all provenance and duplicate-content limitations')
            elif not record['provenance']['limitations']:
                raise ProtocolError('reported results must disclose unavailable run fingerprints')
            history.append(record)
        if not history:
            raise ProtocolError('empty submission history')
        pointer = read_json((directory/'current.json').read_bytes())
        contract.validate('current',pointer,record=True)
        if pointer != current_pointer(history[-1]):
            raise ProtocolError('current pointer differs from latest immutable revision')
        histories[directory.name] = history
    for history in histories.values():
        target = history[-1]['superseded_by']
        if target is not None and target not in histories:
            raise ProtocolError('supersession target does not exist')
    from .catalog import validate_catalog
    validate_catalog(root,contract,histories)
    return runs,histories


def leaderboard(runs, histories):
    entries = []
    for sid, history in sorted(histories.items()):
        record = history[-1]
        if record['status'] != 'published':
            continue
        entry = dict(record)
        if record['source_type'] == 'measured':
            docs = compatible(record['run_ids'], runs)
            run = docs[0]['run']
            c = run['configuration']
            entry['model'] = c['model']['name']
            group = {key: c[key] for key in ('benchmark_version', 'label', 'task', 'adaptation', 'extended')}
            group.update(dataset=c['dataset']['name'], split_protocol=c['split']['protocol'], scoring_rule=record['scoring_rule'])
            entry['metrics'] = aggregate(docs)
            entry['run_count'] = len(docs)
            entry['seeds'] = [doc['run']['seed'] for doc in docs]
            entry['configuration'] = c
            entry['test_fingerprint'] = c['split']['test_fingerprint']
            # Dataset identity and exact split parameters also guard against misleading grouping.
            partition = {'kind': 'recorded-test-fingerprint', 'test_fingerprint': c['split']['test_fingerprint'],
                         'dataset_revision': c['dataset']['revision'], 'dataset_fingerprint': c['dataset']['fingerprint'],
                         'split': c['split'], 'label_mapping': run['provenance']['label_mapping']}
            if any(any(value is None for value in (doc['run']['configuration']['split']['test_fingerprint'],
                    doc['run']['configuration']['dataset']['fingerprint'], doc['run']['configuration']['dataset']['revision'],
                    doc['run']['provenance']['label_mapping'], doc['run']['provenance']['split_provenance'])) for doc in docs):
                partition = {'kind': 'limited-provenance', 'submission_id': sid}
            entry['comparability'] = partition['kind']
            entry['verification'] = 'Bundle integrity and recorded provenance validated; metrics and dataset fingerprints are not independently recomputed.'
        else:
            group = record['group']
            partition = {'kind': 'documented-paper-variant', 'paper_variant_id': record['paper_variant']['id']}
            entry['comparability'] = partition['kind']
            entry['metrics'] = {key: {'mean': record['values'][key], 'stddev': None} for key in METRICS}
            entry['verification'] = 'Published values transcribed with citation; run-level comparability is unavailable.'
        entry['comparison_group'] = group
        entry['ranking_partition'] = partition
        entry['ranking_group_id'] = digest(canonical_json({'group': group, 'partition': partition}))
        entry['rank'] = None
        entries.append(entry)
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['ranking_group_id']].append(entry)
    ordered = []
    for gid in sorted(groups):
        group = sorted(groups[gid], key=lambda e: (-e['metrics']['global_f1']['mean'], e['submission_id']))
        previous, rank = None, None
        for index, entry in enumerate(group, 1):
            value = entry['metrics']['global_f1']['mean']
            if value != previous:
                rank = index
            entry['rank'] = rank
            previous = value
            ordered.append(entry)
    return {'ranking_rule': 'competition; descending full-precision mean global_f1; within ranking_group_id',
            'entries': ordered,
            'history': {sid: records for sid, records in sorted(histories.items())},
            'runs': {rid: {'run': docs['run'], 'manifest': docs['manifest'], 'metrics': docs['metrics'], 'environment': docs['environment']} for rid, docs in sorted(runs.items())}}
