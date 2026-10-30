"""Explicit offline cross-checkout integration check; never part of either runtime.

Standard tests in each repository use local fixtures only. This optional maintainer
check receives both checkouts explicitly, copies them into temporary storage and
uses separate isolated Python environments. No real checkout Git writes occur.
"""
from __future__ import annotations
import argparse
import ast
import re
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT=Path(__file__).resolve().parents[1]

def digest(raw):return 'sha256:'+hashlib.sha256(raw).hexdigest()

def compare_contract(producer,results):
    authoritative=results/'schemas/result-bundle/1.0';vendored=producer/'perseval/results/schemas/1.0'
    expected={p.name:p.read_bytes() for p in authoritative.iterdir() if p.is_file()}
    actual={p.name:p.read_bytes() for p in vendored.iterdir() if p.is_file()}
    if expected!=actual:
        changed=sorted(k for k in expected.keys()|actual.keys() if expected.get(k)!=actual.get(k))
        raise ValueError('schema drift: '+', '.join(changed)+'. Vendor the exact reviewed 1.0 directory and source/digest record together; never patch schemas independently.')
    if (results/'schemas/vendor-lock.schema.json').read_bytes() != (producer/'perseval/results/schemas/vendor-lock.schema.json').read_bytes():
        raise ValueError('vendor-lock schema drift; restore the exact reviewed vendor-lock.schema.json copy')
    source=json.loads((producer/'perseval/results/schemas/vendor-source.json').read_bytes())
    if digest(expected['SHA256SUMS'])!=source['schema_digest']:raise ValueError('vendor-source schema digest mismatch; restore the matching contract/source record')
    af=results/'fixtures/conformance';vf=producer/'tests/fixtures/result-bundle-1.0'
    expected={p.relative_to(af).as_posix():p.read_bytes() for p in af.rglob('*') if p.is_file()}
    actual={p.relative_to(vf).as_posix():p.read_bytes() for p in vf.rglob('*') if p.is_file()}
    if expected!=actual:raise ValueError('conformance fixture drift; restore exact reviewed vectors in both repositories')
    inventory=''.join(f'{hashlib.sha256(raw).hexdigest()}  {name}\n' for name,raw in sorted(expected.items())).encode()
    if digest(inventory)!=source['conformance_digest']:raise ValueError('vendor-source conformance digest mismatch')
    return source

def audit_boundaries(producer,results):
    forbidden_commands={'inbox','collect','collector','approve','publish-run','upload','transport','ssh','scp'}
    for root,package,other in ((producer,'perseval','perseval_results'),(results/'src','perseval_results','perseval')):
        for path in (root/package).rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules=[]
                if isinstance(node,ast.Import):modules=[item.name for item in node.names]
                if isinstance(node,ast.ImportFrom) and node.level==0:modules=[node.module or '']
                if any(name==other or name.startswith(other+'.') for name in modules):raise ValueError(f'cross-package runtime import: {path}')
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in {'submit','do_POST','do_PUT','approve'}:raise ValueError(f'unsupported submission/upload API: {path}:{node.lineno}')
                if isinstance(node,ast.Call):
                    name=getattr(node.func,'attr',getattr(node.func,'id',''))
                    literals=[arg.value for arg in node.args if isinstance(arg,ast.Constant) and isinstance(arg.value,str)]
                    if name=='add_parser' and set(literals)&forbidden_commands:raise ValueError(f'unsupported integration command: {path}')
                    if name in {'git','git_process'} and set(literals)&{'push','pull','fetch','reset','switch','checkout','rebase'}:raise ValueError(f'forbidden automatic Git operation: {path}')
                    if package=='perseval' and name=='export_result' and path.name not in {'run.py','cli.py'}:raise ValueError(f'automatic portable export: {path}')
    for path in producer.rglob('*'):
        if '.git' in path.parts or not path.is_file():continue
        if path.name=='scores.csv' or (path.suffix in {'.html','.js'} and 'leaderboard' in path.as_posix().lower()):raise ValueError(f'leaderboard application/result registry in benchmark: {path}')
    js=(results/'website/app.js').read_text()
    if re.search(r'\.(?:rank|ranking_group_id)\s*=(?!=)',js):raise ValueError('frontend assigns ranks instead of displaying backend ranks')
    return 'PASS: runtime imports, Git commands, export boundaries, submission commands and frontend rank assignment audited'

def source_snapshot(root):
    paths=[root/'README.md',root/'pyproject.toml',root/'requirements.txt']
    paths += [p for base in ('perseval','tests','docs') for p in (root/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return {str(p.relative_to(root)):digest(p.read_bytes()) for p in paths if p.exists()}

def interpreter(base,explicit):
    if explicit:return str(Path(explicit).resolve())
    venv.EnvBuilder(system_site_packages=True,with_pip=False).create(base)
    return str(base/('Scripts/python.exe' if os.name=='nt' else 'bin/python'))

def check(producer,results,producer_python=None,results_python=None,capture_fixture=None):
    source=compare_contract(producer,results);print(audit_boundaries(producer,results));before=source_snapshot(producer)
    with tempfile.TemporaryDirectory(prefix='perseval-integration-') as name:
        work=Path(name).resolve();p=work/'producer-project';p.mkdir()
        shutil.copytree(producer/'perseval',p/'perseval',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        r=work/'results-project';r.mkdir()
        for directory in ('schemas','website','catalog','papers','src'):
            shutil.copytree(results/directory,r/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        # Copy only canonical paper sources, never pending measured publications.
        ids=set()
        for spec in (r/'catalog/experiments').glob('*.json'):ids.update(json.loads(spec.read_bytes())['reported_submissions'])
        for sid in sorted(ids):shutil.copytree(results/'registry/submissions'/sid,r/'registry/submissions'/sid)
        historical=Path('historical/perseval-97cf3ae');(r/historical).mkdir(parents=True)
        for item in ('inventory.json','manifest.json','source-commit.txt'):shutil.copyfile(results/historical/item,r/historical/item)
        for item in ('.gitignore','.gitattributes'):shutil.copyfile(results/item,r/item)
        (r/'README.md').write_text('Disposable synthetic integration repository.\n')
        py_producer=interpreter(work/'producer-env',producer_python);py_results=interpreter(work/'results-env',results_python)
        env={**os.environ,'HF_HUB_OFFLINE':'1','HF_DATASETS_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_VISIBLE_DEVICES':'','GIT_NO_LAZY_FETCH':'1','GIT_ALLOW_PROTOCOL':''}
        for script,python,args in [('producer.py',py_producer,[p,work/'experiment']),('consumer.py',py_results,[r,work/'experiment'])]:
            result=subprocess.run([python,'-I',str(ROOT/'tools/integration'/script),*map(str,args)],cwd=work,env=env,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(f'{script} failed:\n{result.stdout}\n{result.stderr}')
            print(result.stdout.strip())
            if script=='producer.py' and capture_fixture:
                # Explicitly requested test-vector capture, not a publication route.
                record=json.loads((work/'experiment/producer.json').read_bytes())
                capture_fixture=Path(capture_fixture);capture_fixture.parent.mkdir(parents=True,exist_ok=True)
                capture_fixture.write_bytes(Path(record['bundles'][0]).read_bytes())
    if source_snapshot(producer)!=before:raise AssertionError('real producer checkout changed during publication')
    print('PASS: independent environments, identical offline contract, export → local publication/commit → Pages build. No push/deployment performed.')
    if source['source_revision'] is None:print('Release limitation: authoritative checkout has no source commit; vendor revision remains honestly null until maintainers pin it.')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perseval',required=True,type=Path)
    parser.add_argument('--results',default=ROOT,type=Path)
    parser.add_argument('--producer-python',type=Path)
    parser.add_argument('--results-python',type=Path)
    parser.add_argument('--capture-fixture',type=Path,help='explicitly save one generated producer test vector')
    args=parser.parse_args(argv)
    try:check(args.perseval.resolve(),args.results.resolve(),args.producer_python,args.results_python,args.capture_fixture)
    except (ValueError,RuntimeError,OSError,AssertionError) as exc:parser.exit(1,f'integration error: {exc}\n')
if __name__=='__main__':main()
