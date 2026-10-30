"""Read-only audit of recovered evidence; never interpret it as measured runs."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from collections import Counter

from .lfs import pointer
from .protocol import Contract, ProtocolError, canonical_json, digest, read_json

HISTORY = 'historical/perseval-97cf3ae'
REVISION = '97cf3aefe75a34f96ac1e37eecbfa6fd37429ced'


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path)!=value or '\\' in value
            or any(part in ('.','..','.git') for part in path.parts)
            or any(ord(char)<32 for char in value)):
        raise ProtocolError(f'unsafe historical evidence path: {value!r}')
    return value


def load_inventory(directory, contract):
    from .storage import safe_read
    directory = Path(directory)
    if directory.is_symlink():
        raise ProtocolError('historical directory must not be a symlink')
    manifest = read_json(safe_read(directory/'manifest.json'))
    inventory_bytes = safe_read(directory/'inventory.json')
    inventory = read_json(inventory_bytes)
    contract.validate('inventory',inventory,kind='history')
    contract.validate('manifest',manifest,kind='history')
    if manifest['inventory_sha256']!=digest(inventory_bytes):
        raise ProtocolError('historical inventory digest differs from manifest')
    entries = inventory['entries'];names = [entry['original_path'] for entry in entries]
    if names!=sorted(set(names)):
        raise ProtocolError('historical paths must be unique and sorted')
    for entry in entries:
        relative_path(entry['original_path']);relative_path(entry['path'])
        if entry['path']!='snapshot/'+entry['original_path']:
            raise ProtocolError('historical original and recovered paths disagree')
    if (manifest['source_commit']!=inventory['source']['commit']
            or manifest['file_count']!=len(entries)
            or manifest['total_bytes']!=sum(entry['size_bytes'] for entry in entries)
            or manifest['categories']!=dict(Counter(entry['artifact_category'] for entry in entries))):
        raise ProtocolError('historical manifest totals/identity differ from inventory')
    commit=safe_read(directory/'source-commit.txt')
    if (digest(commit)!=manifest['source_commit_object_sha256']
            or hashlib.sha1(f'commit {len(commit)}\0'.encode()+commit).hexdigest()!=manifest['source_commit']
            or commit.split(b'\n',1)[0]!=('tree '+inventory['source']['tree']).encode()):
        raise ProtocolError('historical source commit object/hash/tree disagree')
    tree={}
    for entry in entries:
        node=tree;parts=entry['original_path'].split('/')
        for part in parts[:-1]:
            if part in node and not isinstance(node[part],dict):raise ProtocolError('historical path is both file and directory')
            node=node.setdefault(part,{})
        if parts[-1] in node:raise ProtocolError('duplicate/prefix historical path')
        node[parts[-1]]=(entry['git_mode'],entry['git_object_id'])
    def tree_digest(node):
        payload=b''
        for name,value in sorted(node.items(),key=lambda item:(item[0]+('/' if isinstance(item[1],dict) else '')).encode()):
            mode,oid=('40000',tree_digest(value)) if isinstance(value,dict) else value
            payload+=mode.encode()+b' '+name.encode()+b'\0'+bytes.fromhex(oid)
        return hashlib.sha1(f'tree {len(payload)}\0'.encode()+payload).hexdigest()
    if tree_digest(tree)!=inventory['source']['tree']:
        raise ProtocolError('inventory paths/modes/blob IDs do not reconstruct the committed Git tree')
    return manifest,inventory


def audit(directory, contract, *, metadata_only=False):
    """Hash every ordinary payload, or verify unhydrated LFS identities explicitly."""
    directory=Path(directory)
    manifest,inventory=load_inventory(directory,contract)
    expected={entry['path']:entry for entry in inventory['entries']}
    found=set();checked=0;unhydrated=[]
    snapshot=directory/'snapshot'
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise ProtocolError('historical snapshot must be a real directory')
    for path in sorted(snapshot.rglob('*')):
        mode=path.lstat().st_mode
        if stat.S_ISDIR(mode):continue
        name=path.relative_to(directory).as_posix();relative_path(name)
        if not stat.S_ISREG(mode) or name not in expected:
            raise ProtocolError(f'unexpected/link/special historical evidence: {name}')
        found.add(name);entry=expected[name]
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            size=os.fstat(stream.fileno()).st_size
            beginning=stream.read(1024)
            if size!=entry['size_bytes'] and size<=1024:
                ref=pointer(beginning)
                if ref==(entry['sha256'],entry['size_bytes']):
                    if not metadata_only:
                        raise ProtocolError(f'unhydrated historical LFS object: {name}; hydrate it yourself or use --metadata-only')
                    unhydrated.append(name);continue
            sha=hashlib.sha256();git_sha=hashlib.sha1(f'blob {size}\0'.encode())
            sha.update(beginning);git_sha.update(beginning)
            while chunk:=stream.read(1024*1024):sha.update(chunk);git_sha.update(chunk)
        if size!=entry['size_bytes'] or 'sha256:'+sha.hexdigest()!=entry['sha256'] or git_sha.hexdigest()!=entry['git_object_id']:
            raise ProtocolError(f'historical evidence differs from inventory: {name}')
        checked+=1
    if found!=set(expected):
        raise ProtocolError(f'missing historical evidence: {sorted(set(expected)-found)}')
    return {'source_commit':manifest['source_commit'],'files':len(found),'bytes':manifest['total_bytes'],
            'fully_hashed_files':checked,'unhydrated_lfs_paths':unhydrated,
            'mode':'metadata-only' if metadata_only else 'full',
            'legacy_measured_runs_created':0}
