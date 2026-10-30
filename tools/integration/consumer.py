"""Exercise the real CLI and Git in an isolated results checkout. Test tooling only."""
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from contextlib import redirect_stdout

root, producer = map(lambda p:Path(p).resolve(),sys.argv[1:3])
sys.path.insert(0,str(root/'src'))
class NoProducer(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in ('perseval','torch','transformers','datasets','pandas','numpy'):
            raise AssertionError('lightweight consumer imported a benchmark/model dependency: '+fullname)
sys.meta_path.insert(0,NoProducer())
def forbidden(*args,**kwargs):raise AssertionError('network forbidden')
socket.socket.connect=forbidden;socket.create_connection=forbidden
from perseval_results.cli import main
from perseval_results.build import build,validate_preview
from perseval_results.protocol import Contract,read_json
from perseval_results.records import leaderboard,validate_sources
from perseval_results.storage import managed_snapshot
inputs=json.loads((producer/'producer.json').read_text())
def snapshot(directory):
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
before=snapshot(producer)
# Local-only Git commands, in this disposable checkout. The publication command
# itself performs normal commit preflight and exact allowlisted private staging.
for key in list(os.environ):
    if key.startswith('GIT_'):os.environ.pop(key)
os.environ.update(GIT_CONFIG_GLOBAL=os.devnull,GIT_CONFIG_NOSYSTEM='1',GIT_TERMINAL_PROMPT='0',GIT_ALLOW_PROTOCOL='',GIT_NO_LAZY_FETCH='1')
def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.PIPE).strip()
git('init','-q');git('config','user.name','Synthetic integration fixture');git('config','user.email','fixture@example.invalid')
git('config','commit.gpgsign','false');git('add','--','schemas','website','catalog','papers','historical','registry','src','.gitignore','.gitattributes','README.md')
git('commit','-q','-m','Initial isolated integration fixture')
args=['--repo',str(root),'publish-bundle',inputs['bundles'][0],'--name','EPIC RoBERTa reproduction','--submitter','UFMG','--commit','--yes']
with redirect_stdout(io.StringIO()) as output:assert main(args)==0
plan_text=output.getvalue();assert 'git push' in plan_text and 'Durable commit:' in plan_text
head=git('rev-parse','HEAD');changed=git('diff-tree','--no-commit-id','--name-only','-r','HEAD').splitlines()
assert changed and all(name.startswith(('runs/','registry/')) or name=='.gitattributes' for name in changed)
assert not git('status','--porcelain')
runs,histories=validate_sources(root,Contract(root));api=leaderboard(runs,histories)
measured=next(e for e in api['entries'] if e['source_type']=='measured');sid=measured['submission_id']
assert measured['run_count']==1 and measured['metrics']['global_f1']['stddev'] is None
assert any(e['source_type']=='reported' and e['model']=='roberta-base' for e in api['entries'])
persistent=managed_snapshot(root)
with redirect_stdout(io.StringIO()) as output:assert main(args)==0
assert git('rev-parse','HEAD')==head and managed_snapshot(root)==persistent
assert 'No new commit' in output.getvalue()
with redirect_stdout(io.StringIO()):
    assert main(['--repo',str(root),'publish-bundle',inputs['bundles'][1],'--submission',sid,'--commit','--yes'])==0
runs,histories=validate_sources(root,Contract(root));current=histories[sid][-1]
assert len(histories[sid])==2 and current['revision']==2 and set(current['run_ids'])==set(inputs['run_ids'])
api=leaderboard(runs,histories);measured=next(e for e in api['entries'] if e['submission_id']==sid)
assert measured['run_count']==2 and sorted(measured['seeds'])==[42,43]
assert measured['metrics']['global_f1']['mean']==(1+1/3)/2
assert measured['metrics']['global_f1']['stddev'] is not None
validate_preview(root,root/'site-build')
# Same shared build command as Pages: explicit committed revision, no predictions
# or optional report hydration needed to show scores.
for rid in inputs['run_ids']:
    (root/'runs'/rid/'predictions.csv').unlink()
    (root/'runs'/rid/'artifacts/perseval-evaluation.json').unlink()
with redirect_stdout(io.StringIO()):
    assert main(['--repo',str(root),'site','build','--output','site-build/','--source-revision',git('rev-parse','HEAD')])==0
    assert main(['--repo',str(root),'site','check','--output','site-build/'])==0
built=read_json((root/'site-build/data/leaderboard.json').read_bytes())
assert not (root/'site-build/api').exists()
assert [(e['submission_id'],e['rank']) for e in built['entries']]==[(e['submission_id'],e['rank']) for e in api['entries']]
assert any(e['submission_id']==sid for e in built['entries'])
assert snapshot(producer)==before,'publication modified producer data'
assert 'perseval' not in sys.modules
print(json.dumps({'submission_id':sid,'revision':2,'run_count':2,'reported_count':sum(e['source_type']=='reported' for e in api['entries']),
                  'managed_commit_verified':True,'idempotent':True,'backend_ranks':True,'pages_build_verified':True,
                  'producer_unchanged':True,'commit_sha':git('rev-parse','HEAD')},sort_keys=True))
