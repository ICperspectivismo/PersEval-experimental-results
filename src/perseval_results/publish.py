"""One explicit publication decision, fully planned before transactional persistence."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .build import build
from .bundle import validate_bundle
from .gitops import GitRepository, DurableCommitError, git
from .lfs import matches as blob_matches
from .protocol import Contract, ProtocolError, canonical_json, digest
from .records import aggregate, compatible, leaderboard, limitations, validate_sources
from .storage import (MANAGED, managed_snapshot, tree_files, run_index, revision_path,
                      current_path, current_pointer, duplicate_groups, duplicate_notice, validate_run_consistency)


@contextmanager
def publisher_lock(root):
    path = root/'.perseval-results.lock'
    try:
        fd = os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
    except FileExistsError as exc:
        raise ProtocolError('another publisher is active, or a stale .perseval-results.lock needs inspection') from exc
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(str(os.getpid())+'\n')
        yield
    finally:
        path.unlink(missing_ok=True)


def validate_git_history(root):
    for line in git(root,'rev-list','--reverse','--parents','HEAD').splitlines():
        commit,*parents = line.split()
        for parent in parents or [None]:
            args = ['diff-tree','--no-commit-id','--name-status','-r','--no-renames']
            args += [parent,commit] if parent else ['--root',commit]
            for change in git(root,*args,'--','runs','registry').splitlines():
                status,name = change.split('\t',1)
                pointer = name.startswith('registry/submissions/') and name.endswith('/current.json')
                if status != 'A' and not (pointer and status == 'M'):
                    raise ProtocolError(f'immutable source modified in {commit}: {name}')
                if parent and status == 'A' and name.startswith('runs/'):
                    run_path = '/'.join(name.split('/')[:2])
                    if git(root,'ls-tree',parent,'--',run_path):
                        raise ProtocolError(f'artifact added to an already frozen run in {commit}: {name}')


def environment_summary(docs):
    environment = docs['environment']
    return {**{key:environment[key] for key in ('python','platform','machine','accelerator')},
            'package_count':len(environment['packages']) if environment['packages'] is not None else None,
            'unavailable_optional_fields':[key for key in ('python','platform','machine','packages') if environment[key] is None]}


def bundle_summary(files, docs):
    run = docs['run']; c = run['configuration']
    return {'run_id':run['run_id'],'configuration_id':run['configuration_id'],
            'dataset':c['dataset'],'label':c['label'],'model':c['model'],
            'task':c['task'],'adaptation':c['adaptation'],'extended':c['extended'],
            'seed':run['seed'],'split_seed':c['split']['seed'],
            'compact_metrics':docs['metrics']['compact'],'payload_digest':docs['manifest']['payload_digest'],
            'scientific_digest':run_index(files,docs)['scientific_digest'],
            'payload_bytes':sum(map(len,files.values())),
            'artifacts':[{'path':item['path'],'bytes':len(files[item['path']])} for item in docs['manifest']['artifacts']],
            'limitations':limitations(run),
            'environment':environment_summary(docs),
            'verification':'Checksums, schemas, identities and internal consistency validated; no gold dataset is present to independently verify metrics or claimed test fingerprints.'}


@dataclass
class Plan:
    record: dict
    writes: dict[str,bytes]
    before: dict[str,bytes]
    commit_files: dict[str,bytes]
    message: str
    summary: dict
    candidate: Path
    base_head: str | None
    git_state: object = None
    extra_before: dict = field(default_factory=dict)


@dataclass
class PublicationResult:
    record: dict
    status: str
    plan: dict
    commit_sha: str | None = None
    warnings: list[str] = field(default_factory=list)


def _same_metadata(record, metadata):
    return all(record[key] == value for key,value in metadata.items())


def _write_file(root, name, data):
    path = root/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def prepare(root, candidate, bundles, repo, *, name, submitter, notes, citation, submission,
            commit, reuse_existing_runs, allow_duplicate_content, git_state=None):
    contract = Contract(root)
    before = managed_snapshot(root)
    runs,histories = validate_sources(root,contract,metadata_only=True)
    repo.ensure_immutable(before)
    base_head = repo.head()
    original_ids = set(runs)
    selected,imports,summaries,indices = set(),{},[],{}
    for rid in runs:
        indices[rid] = read_json_index(root,rid)
    for bundle in bundles:
        files,docs = validate_bundle(Path(bundle),contract)
        rid = docs['run']['run_id']
        existing = {name.removeprefix(f'runs/{rid}/'):data for name,data in before.items() if name.startswith(f'runs/{rid}/')}
        if rid in original_ids and (set(existing)!=set(files) or any(
                not blob_matches(data,existing.get(name),f'runs/{rid}/{name}') for name,data in files.items())):
            raise ProtocolError(f'existing run_id has different immutable content: {rid}')
        if rid in imports and imports[rid] != files:
            raise ProtocolError(f'bundles reuse a run_id with different content: {rid}')
        if rid not in original_ids:
            imports[rid] = files
        if rid not in selected:
            summary = bundle_summary(files,docs)
            summary['import_status'] = 'already imported' if rid in original_ids else 'new'
            summaries.append(summary)
        selected.add(rid)
        runs[rid],indices[rid] = docs,run_index(files,docs)
    validate_run_consistency(runs,indices)
    if not selected:
        raise ProtocolError('at least one bundle path is required')
    previous = None
    if submission:
        if submission not in histories:
            raise ProtocolError('--submission must reference an existing submission ID')
        previous = histories[submission][-1]
        if previous['source_type'] != 'measured' or previous['status'] != 'published':
            raise ProtocolError('only an active measured submission can receive runs')
        if reuse_existing_runs:
            raise ProtocolError('--reuse-existing-runs is for a new independent submission, not an update')
    metadata = {key:(value if value is not None else previous[key] if previous else None)
                for key,value in [('name',name),('submitter',submitter),('notes',notes),('citation',citation)]}
    for key in ('name','submitter'):
        if not isinstance(metadata[key],str) or not metadata[key].strip() or any(ord(ch)<32 for ch in metadata[key]):
            raise ProtocolError(f'--{key} is required for a new submission and must be a nonblank single line')
    for key in ('notes','citation'):
        if metadata[key] == '': metadata[key] = None
    if metadata['citation'] is not None:
        try:
            uri = urlsplit(metadata['citation'])
        except (ValueError,TypeError) as exc:
            raise ProtocolError('--citation must be a valid HTTP(S) URL') from exc
        if uri.scheme not in ('http','https') or not uri.netloc or uri.username or any(ch.isspace() for ch in metadata['citation']):
            raise ProtocolError('--citation must be an HTTP(S) URL without credentials or whitespace')
    existing_selected = sorted(selected & set(before_run_ids(before)))
    noop = False
    # Only exact whole-entry retries resolve idempotently; never infer an update target.
    if not submission and not reuse_existing_runs:
        matches = [h[-1] for h in histories.values() if h[-1]['source_type']=='measured'
                   and h[-1]['status']=='published' and set(h[-1]['run_ids'])==selected
                   and _same_metadata(h[-1],metadata)]
        if len(matches)>1:
            raise ProtocolError('ambiguous exact retry; specify --submission explicitly')
        if matches:
            previous = matches[0]
            submission = previous['submission_id']
            noop = True
        elif existing_selected:
            raise ProtocolError('new submission would reuse already imported runs: '+', '.join(existing_selected)+
                                '. Use --submission to update an entry, or --reuse-existing-runs for an intentional new independent entry.')
    included = sorted(selected | (set(previous['run_ids']) if previous else set()))
    docs = compatible(included,runs)
    if previous and previous['configuration_id'] != docs[0]['run']['configuration_id']:
        raise ProtocolError('cannot change a published submission configuration')
    related = set(included)
    duplicates = {signature:ids for signature,ids in duplicate_groups(indices).items() if related & set(ids)}
    disclosures = {item for doc in docs for item in limitations(doc['run'])}
    warnings = []
    for signature,ids in duplicates.items():
        notice = duplicate_notice(signature,ids)
        # Previous explicit acknowledgements survive idempotent retries and revisions.
        acknowledged = previous and notice in previous['provenance']['limitations']
        if not allow_duplicate_content and not acknowledged:
            raise ProtocolError(notice+' Use --allow-duplicate-content only when these are genuine executions; no run was imported.')
        warnings.append(notice)
        disclosures.add(notice)
        inside = sorted(set(ids) & set(included))
        if len(inside)>1:
            disclosures.add(duplicate_notice(signature,inside))
    if previous:
        disclosures.update(previous['provenance']['limitations'])
    if reuse_existing_runs and existing_selected:
        notice = 'Existing measured runs explicitly reused in a separate submission; reuse does not create another execution.'
        warnings.append(notice);disclosures.add(notice)
    sid = submission or 'sub_'+str(uuid.uuid4())
    history = histories.get(sid,[])
    record = {'schema_version':'1.0','submission_id':sid,'revision':len(history)+1,
              'previous_revision_digest':digest(canonical_json(previous)) if previous else None,
              'source_type':'measured','status':'published','superseded_by':None,**metadata,
              'published_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
              'provenance':{'source':'explicit local bundle publication','limitations':sorted(disclosures)},
              'configuration_id':docs[0]['run']['configuration_id'],
              'scoring_rule':docs[0]['metrics']['scoring_rule'],'run_ids':included}
    contract.validate('submission',record,record=True)  # Validate IDs before any path construction.
    if previous and set(previous['run_ids'])==set(included) and _same_metadata(previous,metadata):
        # Explicitly acknowledging new information warrants a revision even with equal runs.
        noop = noop or set(previous['provenance']['limitations'])==set(disclosures)
    if noop:
        record = previous
    writes = {}
    if not noop:
        for rid,files in sorted(imports.items()):
            writes.update({f'runs/{rid}/{name}':data for name,data in files.items()})
            writes[f'registry/runs/{rid}.json'] = canonical_json(indices[rid])
        writes[revision_path(record)] = canonical_json(record)
        writes[current_path(sid)] = canonical_json(current_pointer(record))
    # Attribute rules are themselves an exact, reviewable part of the source plan.
    attribute_writes,lfs_files = repo.plan_lfs(writes)
    extra_before = {name:(root/name).read_bytes() if (root/name).exists() else None
                    for name in attribute_writes}
    writes.update(attribute_writes)
    # Validate a complete isolated source tree and site through the shared build service.
    for directory in ('schemas','website','catalog','papers'):
        if not (root/directory).exists():
            continue
        tree_files(root/directory)
        shutil.copytree(root/directory,candidate/directory)
    # Historical payloads are never needed for a publication/site build, only their ledger.
    from .historical import HISTORY
    for name in ('inventory.json','manifest.json','source-commit.txt'):
        path = root/HISTORY/name
        if path.exists():
            from .storage import safe_read
            _write_file(candidate,HISTORY+'/'+name,safe_read(path))
    for path,data in before.items():
        _write_file(candidate,path,data)
    for path,data in writes.items():
        target = candidate/path
        if target.exists():
            if path != current_path(sid):
                raise ProtocolError(f'refusing to overwrite immutable source: {path}')
            target.unlink()
        _write_file(candidate,path,data)
    if (root/'site-publication.json').exists() or (root/'site-publication.json').is_symlink():
        from .storage import safe_read
        _write_file(candidate,'site-publication.json',safe_read(root/'site-publication.json'))
    data = build(candidate,candidate/'site-build')
    # Clean-checkout preflight excludes pending work: commit exactly this plan's writes.
    commit_files = dict(writes) if commit else {}
    metrics = aggregate(docs)
    message = (f'Publish {record["name"]} ({sid} revision {record["revision"]})\n\n'
               f'Submission: {sid}\nRevision: {record["revision"]}\n'
               f'Included runs ({len(included)}):\n'+''.join(f'- {rid}\n' for rid in included)+
               f'\nGlobal F1 mean: {metrics["global_f1"]["mean"]!r}\n'
               f'Submitter: {record["submitter"]}')
    entry = next(e for e in data['entries'] if e['submission_id']==sid)
    summary = {'operation':'unchanged' if not writes else 'update' if history else 'create',
               'bundles':summaries,'new_run_ids':sorted(imports),'already_imported_run_ids':existing_selected,
               'submission':record,
               'included_runs':[{'run_id':rid,'seed':runs[rid]['run']['seed'],
                                 'configuration_id':runs[rid]['run']['configuration_id'],
                                 'compact_metrics':runs[rid]['metrics']['compact'],
                                 'artifact_bytes':indices[rid]['artifact_bytes'],
                                 'environment':environment_summary(runs[rid]),
                                 'limitations':limitations(runs[rid]['run']),
                                 'preserved_from_previous_revision':rid not in selected}
                                for rid in included],
               'aggregation':{'count':len(included),'run_ids':included,
                                                 'seeds':[runs[rid]['run']['seed'] for rid in included],
                                                 'metrics':aggregate(docs)},
               'rank':entry['rank'],'ranking_group_id':entry['ranking_group_id'],
               'comparability':entry['comparability'],'comparison_group':entry['comparison_group'],
               'limitations':record['provenance']['limitations'], 'duplicate_scientific_content':duplicates,
               'warnings':warnings,'planned_files':[{'path':path,'action':'replace' if path in before or extra_before.get(path) is not None else 'create','bytes':len(data)} for path,data in sorted(writes.items())],
               'commit':{'requested':commit,'message':message,'files':sorted(commit_files)},
               'git_preflight':{'clean_checkout_required':commit,
                                'head':git_state.head if git_state else None,
                                'branch':git_state.branch if git_state else None,
                                'upstream':git_state.upstream if git_state else None},
               'lfs':{'files':lfs_files,'attribute_files':sorted(attribute_writes)},
               'preview':'site-build/ (ignored derived output)', 'no_gold_verification':True}
    if managed_snapshot(root)!=before or repo.head()!=base_head:
        raise ProtocolError('repository changed during planning; retry to review a fresh plan')
    if git_state:
        repo.assert_state(git_state)
    return Plan(record,writes,before,commit_files,message,summary,candidate,base_head,git_state,extra_before)


def before_run_ids(snapshot):
    return {path.split('/')[1] for path in snapshot if path.startswith('runs/')}


def read_json_index(root,rid):
    from .protocol import read_json
    return read_json((root/f'registry/runs/{rid}.json').read_bytes())


def confirm_plan(yes, interactive, input_func):
    if yes:
        return True
    if not interactive:
        raise ProtocolError('noninteractive publication requires --yes; use --dry-run to validate and review without writes')
    try:
        answer = input_func('Publish this plan locally? [y/N] ')
    except EOFError:
        return False
    return answer.strip().casefold() in ('y','yes')


def _install_file(path, data):
    """Same-filesystem atomic replacement, with no final-looking partial file."""
    fd,name = tempfile.mkstemp(prefix='.publish-',dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def _refresh_preview(root, candidate):
    target = root/'site-build'
    if target.is_symlink():
        raise ProtocolError('preview must not be a symlink')
    temporary = Path(tempfile.mkdtemp(prefix='.perseval-preview-',dir=root))
    backup = None
    complete = False
    cleanup = True
    warnings = []
    try:
        shutil.copytree(candidate/'site-build',temporary/'site-build')
        if target.exists():
            backup = temporary/'previous'
            os.replace(target,backup)
        try:
            os.replace(temporary/'site-build',target)
            complete = True
        except BaseException:
            if backup is not None:
                try:
                    os.replace(backup,target)
                except OSError as exc:
                    cleanup = False
                    raise ProtocolError(f'preview replacement and rollback failed; previous preview is preserved at {backup}; restore it manually') from exc
            raise
    finally:
        try:
            if cleanup: shutil.rmtree(temporary)
        except OSError as exc:
            if complete:
                warnings.append(f'Validated preview installed; temporary backup cleanup failed at {temporary}: {exc}')
            else:
                raise
    return warnings


def apply_plan(root, repo, plan):
    created, parents, replaced = [], set(), {}
    durable = None
    warnings = []
    try:
        if managed_snapshot(root)!=plan.before or repo.head()!=plan.base_head:
            raise ProtocolError('repository changed after planning; no publication was applied')
        if plan.git_state:
            repo.assert_state(plan.git_state)
        for name,data in plan.extra_before.items():
            target = root/name
            if target.is_symlink() or (target.read_bytes() if target.exists() else None)!=data:
                raise ProtocolError(f'{name} changed concurrently after planning; no publication was applied')
        for name,data in sorted(plan.writes.items()):
            target = root/name
            parent = target.parent
            while parent != root and not parent.exists():
                parents.add(parent);parent=parent.parent
            target.parent.mkdir(parents=True,exist_ok=True)
            if name in plan.before or plan.extra_before.get(name) is not None:
                if not name.endswith('/current.json') and name not in plan.extra_before:
                    raise ProtocolError('only current pointers may be replaced')
                replaced[target] = plan.before[name] if name in plan.before else plan.extra_before[name]
            else:
                if target.exists() or target.is_symlink():
                    raise ProtocolError(f'planned new path already exists: {name}')
                created.append(target)
            _install_file(target,data)
        # Required validation/build already succeeded in isolation; validate installed registries again.
        validate_sources(root,Contract(root),metadata_only=True)
        if plan.commit_files:
            if managed_snapshot(root)!={key:value for key,value in {**plan.before,**plan.writes}.items()
                                        if key.startswith(('runs/','registry/'))}:
                raise ProtocolError('publication sources changed concurrently after installation; inspect the checkout')
            outcome = repo.commit(plan.commit_files,plan.message,plan.git_state)
            durable = outcome.sha
            warnings.extend(outcome.warnings)
        if durable is None:
            # A preview-copy failure before a durable commit is part of the transaction.
            warnings.extend(_refresh_preview(root,plan.candidate) or [])
        else:
            try:
                warnings.extend(_refresh_preview(root,plan.candidate) or [])
            except BaseException as exc:
                warnings.append(f'Commit {durable} is durable; preview refresh failed: {exc}. Run perseval-results site build locally.')
    except DurableCommitError as exc:
        durable = exc.sha
        warnings.append(f'Commit {durable} is durable; follow-up inspection failed: {exc}. No history or persistent files were rolled back.')
    except BaseException as exc:
        if durable is None:
            preserved = []
            for path,data in replaced.items():
                name = path.relative_to(root).as_posix()
                if path.is_file() and not path.is_symlink() and path.read_bytes()==data:
                    continue  # Installation failed before replacing the original bytes.
                if path.is_symlink() or not path.is_file() or path.read_bytes()!=plan.writes[name]:
                    preserved.append(name)
                else:
                    _install_file(path,data)
            for path in reversed(created):
                name = path.relative_to(root).as_posix()
                if path.exists() and (path.is_symlink() or not path.is_file() or path.read_bytes()!=plan.writes[name]):
                    preserved.append(name)
                else:
                    path.unlink(missing_ok=True)
            for directory in sorted(parents,key=lambda p:len(p.parts),reverse=True):
                try: directory.rmdir()
                except OSError: pass
            if preserved:
                raise ProtocolError(f'{exc}; concurrent edits preserved at {preserved!r}; review them manually') from exc
        raise
    return PublicationResult(plan.record,'committed' if durable else 'published',plan.summary,durable,warnings)


def publish(root, bundles, *, name=None, submitter=None, commit=False, submission=None,
            submission_id=None, notes=None, citation=None, yes=False, dry_run=False,
            reuse_existing_runs=False, allow_duplicate_content=False,
            interactive=None, input_func=None, output=print):
    root = Path(root).resolve()
    Contract(root)  # Validate the repository identity before any lock/write.
    repo = GitRepository(root)
    git_state = repo.preflight() if commit else None
    if submission and submission_id and submission!=submission_id:
        raise ProtocolError('conflicting submission IDs')
    submission = submission or submission_id
    if not bundles:
        raise ProtocolError('at least one explicitly selected bundle is required')
    interactive = sys.stdin.isatty() if interactive is None else interactive
    input_func = input if input_func is None else input_func
    lock = root/'.perseval-results.lock'
    if lock.exists() or lock.is_symlink():
        raise ProtocolError('publisher lock exists; inspect the active publisher or stale lock before retrying')

    def execute():
        if (root/'site-build').is_symlink():
            raise ProtocolError('preview must not be a symlink')
        outcome = None
        try:
            with tempfile.TemporaryDirectory(prefix='perseval-publication-') as temporary:
                plan = prepare(root,Path(temporary),bundles,repo,name=name,submitter=submitter,notes=notes,
                               citation=citation,submission=submission,commit=commit,
                               reuse_existing_runs=reuse_existing_runs,allow_duplicate_content=allow_duplicate_content,
                               git_state=git_state)
                output(json.dumps(plan.summary,indent=2,ensure_ascii=False,allow_nan=False))
                if dry_run:
                    outcome = PublicationResult(plan.record,'dry-run',plan.summary)
                elif not plan.writes and not plan.commit_files:
                    outcome = PublicationResult(plan.record,'unchanged',plan.summary)
                elif not confirm_plan(yes,interactive,input_func):
                    outcome = PublicationResult(plan.record,'cancelled',plan.summary)
                else:
                    try:
                        # Even the lock is created only after preflight and plan validation.
                        with publisher_lock(root):
                            outcome = apply_plan(root,repo,plan)
                    except BaseException as exc:
                        if outcome is not None and outcome.status in ('committed','published'):
                            outcome.warnings.append(f'Publication persisted; commit {outcome.commit_sha or "not requested"}; lock cleanup failed: {exc}')
                        else:
                            raise
            return outcome
        except BaseException as exc:
            if outcome is not None and outcome.status in ('committed','published'):
                outcome.warnings.append(f'Publication persisted; commit {outcome.commit_sha or "not requested"}; staging cleanup failed: {exc}')
                return outcome
            raise

    return execute()  # Dry-run/no-op never acquires a lock or writes into the checkout.
