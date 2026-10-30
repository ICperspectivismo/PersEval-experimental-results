"""Offline, deterministic, atomic leaderboard builds for publication and CI."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

from .protocol import Contract, ProtocolError, canonical_json, canonical_path, digest, read_json
from .records import METRICS, leaderboard, validate_sources
from .storage import safe_read, tree_files
from .lfs import stored_reference

OUTPUTS = ('site-build', 'preview')  # preview is the pre-v1 CLI compatibility alias.
POLICY = 'site-publication.json'
PAGE_DATA = 'data/leaderboard.json'


def _source_files(root):
    # Payload hydration is not a build input: checksums and ordinary-Git indices
    # bind those bytes. Neither the historical snapshot nor arbitrary artifacts
    # nor git timestamps are read into the build identity.
    files = {}
    for directory in ('schemas', 'website', 'registry', 'catalog', 'papers'):
        files.update({f'{directory}/{name}': data for name,data in tree_files(root/directory).items()})
    for path in sorted((root/'runs').glob('*')):
        for name in ('run.json','manifest.json','metrics.json','environment.json','checksums.sha256'):
            files[f'runs/{path.name}/{name}'] = safe_read(path/name)
    from .historical import HISTORY
    for name in ('inventory.json','manifest.json','source-commit.txt'):
        path = root/HISTORY/name
        if path.exists() or path.is_symlink(): files[HISTORY+'/'+name] = safe_read(path)
    if (root/POLICY).exists() or (root/POLICY).is_symlink():
        files[POLICY] = safe_read(root/POLICY)
    return files


def _artifact_metadata(root, rid, docs, policy, files):
    index = read_json(safe_read(root/'registry/runs'/f'{rid}.json'))
    checksums = safe_read(root/'runs'/rid/'checksums.sha256')
    sums = {line[66:]: 'sha256:'+line[:64] for line in checksums.decode('ascii').splitlines()}
    artifacts, complete = [], None
    for item in docs['manifest']['artifacts']:
        path = item['path']
        known = index.get('files',{}).get(path,{})
        local = root/'runs'/rid/path
        # Metadata-only site builds need not read or hydrate private payloads.
        size = known.get('size')
        if size is None and local.exists():
            raw = safe_read(local)
            ref = stored_reference(path,raw,checksums)
            size = ref[1] if ref else len(raw)
        public = policy.pop((rid,path),None)
        artifact = {**item, 'sha256':sums[path], 'size':size, 'public':public is not None,
                    'href':None, 'license':None, 'attribution':None}
        if public is not None:
            if not local.exists():
                raise ProtocolError(f'explicit public artifact is unavailable: {rid}/{path}; supply its verified bytes locally or remove the public selection')
            raw = safe_read(local)
            if digest(raw) != sums[path]:
                raise ProtocolError(f'public artifact is absent, unhydrated or corrupt: {rid}/{path}; no LFS fetch is automatic')
            # Serve downloads as plain-text downloads, never as same-origin HTML/JS/SVG.
            target = f'artifacts/{rid}/{path.removeprefix("artifacts/")}.download.txt'
            files[target] = raw
            artifact.update(href=target,license=public['license'],attribution=public['attribution'])
            if path == 'artifacts/perseval-evaluation.json':
                complete = read_json(raw)  # validate_sources already checked its compact metrics.
        artifacts.append(artifact)
    return sorted(artifacts,key=lambda a:a['path']), complete


def _policy(root, contract):
    if not (root/POLICY).exists() and not (root/POLICY).is_symlink():
        return {}
    document = read_json(safe_read(root/POLICY))
    contract.validate('public-artifacts',document,kind='site')
    result = {}
    for item in document['artifacts']:
        canonical_path(item['path'])
        key = item['run_id'], item['path']
        if key in result:
            raise ProtocolError('duplicate public artifact selection')
        result[key] = item
    return result


def _csv(data):
    columns = ['submission_id','revision','name','submitter','source_type','model','configuration_id',
               'ranking_group_id','comparability','rank','benchmark_version','dataset','label','task',
               'adaptation','extended','split_protocol','scoring_rule','n','run_ids','seeds']
    columns += [f'{metric}_{suffix}' for metric in METRICS for suffix in ('mean','stddev')]
    columns += ['citation','notes']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
    writer.writeheader()
    for entry in data['entries']:
        row = {name:entry.get(name,'') for name in columns}
        row.update(entry['comparison_group'])
        row['n'] = entry.get('run_count','')
        row['run_ids'] = json.dumps(entry['run_ids'],separators=(',',':')) if 'run_ids' in entry else ''
        row['seeds'] = json.dumps(entry['seeds'],separators=(',',':')) if 'seeds' in entry else ''
        for metric in METRICS:
            for suffix in ('mean','stddev'):
                value = entry['metrics'][metric][suffix]
                row[f'{metric}_{suffix}'] = '' if value is None else repr(value)
        # Spreadsheet formula injection: JSON retains exact authoritative strings.
        for name,value in row.items():
            if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@','\t','\r')):
                row[name] = "'"+value
        writer.writerow(row)
    return stream.getvalue().encode('utf-8')


def generate(root: Path, *, source_revision=None):
    """Pure build plan. No files are written and no Git/network command is run."""
    root = root.resolve()
    if source_revision is not None and (not isinstance(source_revision,str) or not source_revision.strip() or len(source_revision)>200 or any(ord(c)<32 for c in source_revision)):
        raise ProtocolError('source revision must be a nonempty printable identifier, at most 200 characters')
    inputs = _source_files(root)
    required_assets = {'website/'+name for name in ('index.html','app.js','styles.css')}
    if not required_assets <= inputs.keys():
        raise ProtocolError('missing website source assets: '+repr(sorted(required_assets-inputs.keys())))
    contract = Contract(root)
    runs,histories = validate_sources(root,contract,metadata_only=True)
    data = leaderboard(runs,histories)
    source_digest = digest(canonical_json({name:digest(raw) for name,raw in sorted(inputs.items())}))
    builder_digest = digest(canonical_json({path.name:digest(safe_read(path)) for path in sorted(Path(__file__).parent.glob('*.py'))}))
    identity = {'builder_digest':builder_digest,'bundle_schema_digest':contract.schema_digest,
                'source_revision':source_revision,'source_digest':source_digest,'build_format':1}
    header = {**identity,'build_id':digest(canonical_json(identity))}
    files = {name:inputs[f'website/{name}'] for name in ('index.html','app.js','styles.css')}
    files['.nojekyll'] = b''
    policy = _policy(root,contract)
    artifacts = {}
    for rid,docs in sorted(runs.items()):
        metadata, complete = _artifact_metadata(root,rid,docs,policy,files)
        artifacts[rid] = metadata
        data['runs'][rid]['complete_metrics'] = complete
    if policy:
        raise ProtocolError('public artifact selection references an unknown run or undeclared artifact: '+repr(sorted(policy)))
    # Internal implementation data for the page, not a versioned consumer API.
    # All detail views share this document instead of separate endpoints.
    data.update(build=header,artifact_metadata=artifacts)
    files[PAGE_DATA] = canonical_json(data)
    files['scores.csv'] = _csv(data)
    _validate_generated(files,contract)
    return files,data


class _HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.links=[]; self.ids=set()
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value is not None: self.links.append(value)
            if key=='id': self.ids.add(value)


def safe_link(value, files):
    """Site links are relative to the site root, including on project Pages."""
    if not isinstance(value,str) or not value or any(ord(c)<32 for c in value) or '\\' in value:
        raise ProtocolError(f'unsafe site link: {value!r}')
    try: parsed = urlsplit(value)
    except ValueError as exc: raise ProtocolError(f'unsafe site link: {value!r}') from exc
    if parsed.scheme:
        if parsed.scheme not in ('https','http') or not parsed.hostname or parsed.username or parsed.password:
            raise ProtocolError(f'unsafe external link: {value!r}')
        return
    if parsed.netloc or parsed.query or '%' in parsed.path:
        raise ProtocolError(f'unsafe relative site link: {value!r}')
    name = parsed.path or 'index.html'
    canonical_path(name)
    if name not in files:
        raise ProtocolError(f'broken site link: {value!r}')
    if parsed.fragment:
        if not name.endswith('.html'):
            raise ProtocolError('fragment links must refer to an HTML anchor')
        parser = _HTMLLinks();parser.feed(files[name].decode('utf-8'))
        if parsed.fragment not in parser.ids: raise ProtocolError(f'unknown HTML anchor: {value}')


def _validate_generated(files,contract):
    required = {'index.html','app.js','styles.css','.nojekyll',PAGE_DATA,'scores.csv'}
    if not required <= files.keys():
        raise ProtocolError('incomplete site build; missing files: '+repr(sorted(required-files.keys())))
    for name,raw in files.items():
        if name != '.nojekyll': canonical_path(name)
        if name not in required and not name.startswith('artifacts/'):
            raise ProtocolError(f'unexpected site file: {name}')
    data = read_json(files[PAGE_DATA])
    if (not isinstance(data,dict) or not isinstance(data.get('entries'),list)
            or any(not isinstance(data.get(key),dict) for key in ('build','history','runs','artifact_metadata'))
            or not {'source_revision','source_digest','build_id'} <= data['build'].keys()):
        raise ProtocolError('incomplete leaderboard page data; rebuild the preview')
    for history in data['history'].values():
        for record in history:
            for href in re.findall(r'https?://[^\s<>]+',record['citation'] or ''):
                safe_link(href.rstrip('.,;)'),files)
    for entry in data['entries']:
        for href in re.findall(r'https?://[^\s<>]+',entry['citation'] or ''):
            safe_link(href.rstrip('.,;)'),files)
    for metadata in data['artifact_metadata'].values():
        for item in metadata:
            if item['href']: safe_link(item['href'],files)
    parser = _HTMLLinks();parser.feed(files['index.html'].decode('utf-8'))
    for href in parser.links: safe_link(href,files)


def _output(root,output):
    root = root.resolve()
    # Limit replacements to disposable, documented directories in this checkout.
    # Check the lexical path before resolve so a symlink is never followed.
    output = Path(os.path.abspath(output))
    output = output.parent.resolve()/output.name
    if output.parent != root or output.name not in OUTPUTS:
        raise ProtocolError('build output must be site-build/ (or legacy preview/) directly inside the results checkout')
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ProtocolError('site output must be a directory, never a symlink or file')
    return output


def _build_files(directory):
    # .nojekyll is the only intentional dotfile in derived output.
    marker = directory/'.nojekyll'
    files = tree_files_excluding_marker(directory)
    if marker.exists() or marker.is_symlink(): files['.nojekyll'] = safe_read(marker)
    return files


def tree_files_excluding_marker(directory):
    files = {}
    if directory.is_symlink(): raise ProtocolError('site output must not be a symlink')
    if not directory.is_dir(): raise ProtocolError('site output does not exist; run site build first')
    for path in sorted(directory.iterdir()):
        if path.name == '.nojekyll': continue
        canonical_path(path.name)
        if path.is_dir() and not path.is_symlink():
            files.update({f'{path.name}/{name}':raw for name,raw in tree_files(path).items()})
        else: files[path.name] = safe_read(path)
    return files


def build(root: Path, output: Path, *, source_revision=None):
    root = root.resolve(); output = _output(root,output)
    files,data = generate(root,source_revision=source_revision)
    temporary = Path(tempfile.mkdtemp(prefix='.perseval-site-',dir=root))
    stage,backup = temporary/'output',temporary/'previous'
    cleanup = True
    try:
        stage.mkdir()
        for name,raw in files.items():
            path = stage/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
        staged = _build_files(stage)
        _validate_generated(staged,Contract(root))
        if staged != files:
            raise ProtocolError('staged site does not match the generated output')
        # Detect source edits during the build before replacing the previous preview.
        if digest(canonical_json({name:digest(raw) for name,raw in sorted(_source_files(root).items())})) != data['build']['source_digest']:
            raise ProtocolError('authoritative sources changed during build; retry from a stable checkout')
        if output.is_symlink(): raise ProtocolError('site output became a symlink during build')
        if output.exists(): os.replace(output,backup)
        try: os.replace(stage,output)
        except BaseException:
            if backup.exists():
                try: os.replace(backup,output)
                except OSError as exc:
                    cleanup = False
                    raise ProtocolError(f'site replacement and rollback failed; previous build is preserved at {backup}; restore it manually') from exc
            raise
    finally:
        if cleanup: shutil.rmtree(temporary)
    return data


def validate_preview(root: Path, output: Path, contract: Contract | None = None):
    output = _output(root,output)
    files = _build_files(output)
    _validate_generated(files,contract or Contract(root))
    data = read_json(files[PAGE_DATA])
    source_revision = data['build']['source_revision']
    expected,_ = generate(root,source_revision=source_revision)
    if files != expected:
        raise ProtocolError('site data, ranks or website do not match authoritative records; rebuild the preview')
    return {**data['build'],'counts':{'published':len(data['entries'])}}
