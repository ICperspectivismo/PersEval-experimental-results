"""Recover a complete Git tree using object reads only; never touch the source worktree."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from perseval_results.historical import relative_path, audit
from perseval_results.protocol import Contract, ProtocolError, canonical_json, digest
from perseval_results.lfs import LFS_THRESHOLD, pointer

REVISION='97cf3aefe75a34f96ac1e37eecbfa6fd37429ced'
SOURCE_URL='https://github.com/valeriobasile/PersEval'


def category(path):
    if path.endswith('.DS_Store'):return 'housekeeping'
    first=path.split('/')[0]
    if first=='data':return 'dataset-cache' if path.endswith('.arrow') or '/cache-' in path else 'dataset'
    if first=='data_LaMP':return 'retrieval-intermediate'
    if first.startswith('predictions_'):
        return 'ablation-predictions' if '/ablation_files/' in path else 'normalized-predictions' if '/edited_' in path else 'raw-model-output'
    if first=='results':return 'classification-report'
    if path.endswith('.ipynb'):return 'analysis-notebook'
    if first=='qualitative_analysis':return 'analysis-code' if path.endswith('.py') else 'qualitative-report'
    if path.endswith('config.py') or path in ('requirements.txt','.gitignore'):return 'historical-configuration'
    if first=='tests':return 'historical-test'
    if first=='.github':return 'historical-ci'
    if path.endswith('.py'):return 'historical-code'
    if path.endswith('.md'):return 'historical-documentation'
    return 'housekeeping' if path.endswith('.DS_Store') else 'diagram'


def recover(source,destination,root,revision=REVISION):
    source=Path(source).resolve();destination=Path(destination).resolve();root=Path(root).resolve()
    if root==source or root.is_relative_to(source) or destination.is_relative_to(source):
        raise ProtocolError('recovery must never write into the source PersEval repository')
    contract=Contract(root)
    if 'https://perseval.org/schemas/history/1.0/inventory.schema.json' not in contract.schemas:
        raise ProtocolError('--repo must select the results checkout with historical evidence schemas')
    if (root/'.gitattributes').is_symlink() or not (root/'.gitattributes').is_file():
        raise ProtocolError('results .gitattributes must be an existing regular file')
    if destination.exists():raise ProtocolError('recovery destination already exists; audit it instead of overwriting historical evidence')
    if not destination.is_relative_to(root/'historical'):raise ProtocolError('destination must be under this results checkout historical/')
    env={**os.environ,'GIT_OPTIONAL_LOCKS':'0','GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':'','GIT_TERMINAL_PROMPT':'0'}
    for key in ('GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_COMMON_DIR'):env.pop(key,None)
    def git(*args):
        proc=subprocess.run(['git','-C',str(source),*args],env=env,capture_output=True)
        if proc.returncode:raise ProtocolError(proc.stderr.decode(errors='replace'))
        return proc.stdout
    commit=git('rev-parse',revision+'^{commit}').decode().strip()
    if commit!=revision:raise ProtocolError('use a full immutable source commit ID')
    tree=git('rev-parse',revision+'^{tree}').decode().strip()
    if git('rev-parse','--show-object-format').strip()!=b'sha1':raise ProtocolError('this recovery profile expects SHA-1 Git objects')
    items=[]
    for row in git('ls-tree','-r','-l','-z',commit).split(b'\0'):
        if not row:continue
        metadata,name=row.decode().split('\t',1);mode,kind,oid,size=metadata.split()
        relative_path(name)
        if kind!='blob' or mode not in ('100644','100755'):raise ProtocolError('historical source includes a link/submodule/special object; inspect before recovery')
        items.append((name,mode,oid,int(size)))
    entries=[]
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.recovery-',dir=destination.parent) as temp:
        stage=Path(temp)
        process=subprocess.Popen(['git','-C',str(source),'cat-file','--batch'],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            for name,mode,oid,size in sorted(items):
                process.stdin.write((oid+'\n').encode());process.stdin.flush()
                header=process.stdout.readline().decode().strip().split()
                if header!=[oid,'blob',str(size)]:raise ProtocolError(f'missing or inconsistent Git object: {oid}; no network fetch attempted')
                path=stage/'snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
                sha=hashlib.sha256();git_sha=hashlib.sha1(f'blob {size}\0'.encode());remaining=size;prefix=b''
                with path.open('xb') as stream:
                    while remaining:
                        chunk=process.stdout.read(min(remaining,1024*1024))
                        if not chunk:raise ProtocolError('truncated Git object output')
                        if len(prefix)<1024:prefix+=chunk[:1024-len(prefix)]
                        stream.write(chunk);sha.update(chunk);git_sha.update(chunk);remaining-=len(chunk)
                if process.stdout.read(1)!=b'\n' or git_sha.hexdigest()!=oid:raise ProtocolError('Git blob digest mismatch')
                limits=['Git snapshot identifies file content, not an execution. No per-run environment, immutable run ID, dataset revision or test fingerprint is established.']
                if name.startswith('data/') or name.startswith('data_LaMP/'):
                    limits.append('Historical dataset/cache or prompt/profile evidence; never loaded by normal tests/site builds. Source redistribution terms were not recorded in this tree.')
                if name.startswith('predictions_'):
                    limits.append('Filename flags are hints, not proof of task settings, seed, split coverage, or execution independence. Raw outputs may contain prompts and invalid labels.')
                is_pointer=size<=1024 and pointer(prefix) is not None
                if is_pointer:limits.append('Source Git blob is an LFS pointer; original payload was not downloaded or recovered.')
                entries.append({'original_path':name,'path':'snapshot/'+name,'git_object_id':oid,'git_mode':mode,
                                'sha256':'sha256:'+sha.hexdigest(),'size_bytes':size,'artifact_category':category(name),
                                'associated_experiments':[],'association_basis':None,
                                'recovery_status':'lfs-pointer-only' if is_pointer else 'recovered-git-blob','limitations':limits})
            process.stdin.close()
            if process.wait()!=0:raise ProtocolError(process.stderr.read().decode(errors='replace'))
        finally:
            if process.poll() is None:process.kill();process.wait()
            for stream in (process.stdin,process.stdout,process.stderr):
                if stream is not None and not stream.closed:stream.close()
        inventory={'schema_version':'1.0','source':{'repository':SOURCE_URL,'commit':commit,'tree':tree,'object_format':'sha1'},
                   'selection':'complete-committed-tree','entries':entries}
        raw=canonical_json(inventory)
        manifest={'schema_version':'1.0','source_commit':commit,'inventory_sha256':digest(raw),'file_count':len(entries),
                  'total_bytes':sum(entry['size_bytes'] for entry in entries),'categories':dict(sorted(Counter(e['artifact_category'] for e in entries).items())),
                  'legacy_measured_runs_created':0,'lfs_threshold_bytes':LFS_THRESHOLD}
        commit_object=git('cat-file','commit',commit)
        manifest['source_commit_object_sha256']=digest(commit_object)
        (stage/'source-commit.txt').write_bytes(commit_object)
        (stage/'inventory.json').write_bytes(raw);(stage/'manifest.json').write_bytes(canonical_json(manifest))
        audit(stage,contract)
        # The immutable snapshot and its ledger appear together only after validation.
        os.replace(stage,destination)
    attrs=root/'.gitattributes';old=attrs.read_bytes()
    rules='\n# Recovered historical evidence; exact large payload paths only.\n'
    for entry in entries:
        if entry['size_bytes']>=LFS_THRESHOLD:
            rules+=f'/{destination.relative_to(root).as_posix()}/{entry["path"]} filter=lfs diff=lfs merge=lfs -text\n'
    attrs.write_bytes(old+rules.encode())
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    print(canonical_json(recover(args.source,args.repo/'historical/perseval-97cf3ae',args.repo)).decode())
