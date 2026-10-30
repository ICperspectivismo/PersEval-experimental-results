"""Regenerate synthetic conformance vectors after an intentional protocol change.

Run from the repository root with PYTHONPATH=src. Never used during normal imports.
"""
import copy
import io
import gzip
import json
import tarfile
from pathlib import Path

from perseval_results.bundle import pack_files, seal_payload
from perseval_results.protocol import Contract, canonical_json, checksum_bytes, configuration_id, digest

ROOT = Path(__file__).resolve().parents[1]
contract = Contract(ROOT)
fixtures = ROOT / 'fixtures/conformance'
config = {
    'benchmark_version':'synthetic-conformance-v1',
    'dataset':{'name':'toy','revision':'fixture-v1','fingerprint':digest(b'synthetic dataset v1')},
    'label':'sentiment', 'split':{'protocol':'toy-holdout-v1','parameters':{'test_fraction':0.2},'seed':17,'test_fingerprint':digest(b'synthetic test gold v1')},
    'model':{'name':'toy-model','revision':'fixture-model-v1'}, 'task':'baseline','adaptation':'none','extended':False,
    'method':'synthetic','representation':'none', 'hyperparameters':{'training':{'learning_rate':0.001},'inference':{'temperature':0.0}},
}
run = {'schema_version':'1.0','run_id':'run_11111111-1111-4111-8111-111111111111','configuration_id':configuration_id(config),
       'configuration':config,'seed':42,'execution':{'started_at':'2026-01-01T00:00:00Z','finished_at':'2026-01-01T00:01:00Z','command':None,'executor':None},
       'provenance':{'perseval_version':'1.0.0','perseval_git_revision':'7e1962504f0319cb1eb5ef69ea430edadbc525b5','perseval_git_dirty':False,
                     'label_mapping':{'negative':'0','positive':'1'},'split_provenance':{'builder':'synthetic-conformance-only'},'notes':'Synthetic; not an experiment or paper value.'}}
metrics={'schema_version':'1.0','scoring_rule':'perseval.macro-f1.v1','prediction_count':2,'compact':{'global_f1':0.75,'user_f1':0.5,'text_f1':0.5}}
environment={'schema_version':'1.0','python':'3.11.0','platform':None,'machine':None,'packages':None,'accelerator':None,'notes':'Synthetic frozen snapshot; optional fields deliberately unavailable.'}
payload={'run.json':canonical_json(run),'metrics.json':canonical_json(metrics),'environment.json':canonical_json(environment),'predictions.csv':b'user_id,text_id,predictions\nu1,t1,0\nu2,t2,1\n'}
valid=seal_payload(payload,contract)
(fixtures/'valid/minimal.perseval-result.tar.gz').write_bytes(pack_files(valid))
for filename,data in valid.items():
    dest=fixtures/'valid/minimal'/filename;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
selected={'artifacts/summary.txt':b'Explicitly selected synthetic artifact.\n'}
artifact=seal_payload({**payload,**selected},contract,[{'path':'artifacts/summary.txt','media_type':'text/plain','description':'Selected synthetic summary'}])
(fixtures/'valid/selected-artifact.perseval-result.tar.gz').write_bytes(pack_files(artifact))
limited_run=copy.deepcopy(run);limited_run['configuration']['split']['test_fingerprint']=None
limited_run['configuration']['dataset']['revision']=None
limited_run['configuration_id']=configuration_id(limited_run['configuration'])
limited_run['provenance']['label_mapping']=None
limited_run['provenance']['split_provenance']=None
limited=seal_payload({**payload,'run.json':canonical_json(limited_run)},contract)
(fixtures/'valid/limited-provenance.perseval-result.tar.gz').write_bytes(pack_files(limited))
cases=[]
def invalid(name, files, error):
    filename=name+'.perseval-result.tar.gz'
    (fixtures/'invalid'/filename).write_bytes(pack_files(files))
    cases.append({'file':'invalid/'+filename,'expected_error':error})
def reseal(files):
    return seal_payload({k:v for k,v in files.items() if k not in ['manifest.json','checksums.sha256']},contract)
def edit_json(filename, transform):
    files=dict(valid);doc=json.loads(files[filename]);transform(doc);files[filename]=canonical_json(doc);return reseal(files)
invalid('missing-environment',{k:v for k,v in valid.items() if k!='environment.json'},'missing required files')
invalid('checksum-mismatch',{**valid,'predictions.csv':payload['predictions.csv']+b'u3,t3,0\n'},'checksums')
invalid('extra-prediction-column',reseal({**valid,'predictions.csv':b'user_id,text_id,predictions,gold\nu1,t1,0,0\nu2,t2,1,1\n'}),'prediction columns')
invalid('duplicate-pair',reseal({**valid,'predictions.csv':b'user_id,text_id,predictions\nu1,t1,0\nu1,t1,1\n'}),'unique and sorted')
invalid('wrong-count',edit_json('metrics.json',lambda m:m.update(prediction_count=3)),'prediction count')
invalid('out-of-range-metric',edit_json('metrics.json',lambda m:m['compact'].update(global_f1=1.1)),'metrics')
invalid('configuration-mismatch',edit_json('run.json',lambda r:r['configuration']['model'].update(name='changed')),'configuration_id')
invalid('execution-seed-in-configuration',edit_json('run.json',lambda r:r['configuration']['hyperparameters']['training'].update(seed=42)),'run')
invalid('unknown-scoring-rule',edit_json('metrics.json',lambda m:m.update(scoring_rule='best-seed')),'metrics')
invalid('unselected-artifact',reseal({**valid,'artifacts/cache.txt':b'not selected'}),'explicitly declared')
wrong=dict(valid);manifest=json.loads(wrong['manifest.json']);manifest['schema_digest']='sha256:'+'0'*64;wrong['manifest.json']=canonical_json(manifest);wrong['checksums.sha256']=checksum_bytes({k:v for k,v in wrong.items() if k!='checksums.sha256'})
invalid('unknown-schema-digest',wrong,'schema digest')
wrong=dict(valid);manifest=json.loads(wrong['manifest.json']);manifest['payload_digest']='sha256:'+'0'*64;wrong['manifest.json']=canonical_json(manifest);wrong['checksums.sha256']=checksum_bytes({k:v for k,v in wrong.items() if k!='checksums.sha256'})
invalid('payload-digest-mismatch',wrong,'payload digest')
# Invalid archives cannot pass the canonical encoder; construct them deliberately.
for name,kind in [('path-traversal','path'),('absolute-path','absolute'),('symlink','link'),('duplicate-member','duplicate')]:
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w',format=tarfile.USTAR_FORMAT) as tar:
        info=tarfile.TarInfo('../outside.json' if kind=='path' else '/outside.json' if kind=='absolute' else 'run.json')
        if kind=='link':info.type=tarfile.SYMTYPE;info.linkname='../outside'
        tar.addfile(info,io.BytesIO(b''))
        if kind=='duplicate':tar.addfile(info,io.BytesIO(b''))
    filename=name+'.perseval-result.tar.gz';(fixtures/'invalid'/filename).write_bytes(gzip.compress(out.getvalue(),compresslevel=9,mtime=0))
    cases.append({'file':'invalid/'+filename,'expected_error':'path' if kind in ['path','absolute'] else 'regular' if kind=='link' else 'duplicate'})
reported={'schema_version':'1.0','submission_id':'sub_99999999-9999-4999-8999-999999999999','revision':1,'previous_revision_digest':None,
          'status':'published','superseded_by':None,'name':'Synthetic reported-result contract example','submitter':'Fixture authors','notes':'Not a real paper score.',
          'citation':'Synthetic conformance publication, 2026 (not a real paper).','published_at':'2026-01-02T00:00:00Z',
          'provenance':{'source':'Synthetic table transcription','limitations':['Run fingerprints, seeds, predictions, environment and dispersion are unavailable.']},
          'source_type':'reported','model':'toy-model','group':{key:config[key] for key in ['benchmark_version','label','task','adaptation','extended']},
          'paper_variant':{'id':'synthetic-paper-variant-v1','description':'Toy original paper benchmark variant.','mapping_evidence':'Synthetic appendix describes the holdout; no equivalence to measured fingerprints is claimed.','table':'Toy table 1','page':'1'},'values':metrics['compact']}
reported['group'].update(dataset=config['dataset']['name'],split_protocol=config['split']['protocol'],scoring_rule=metrics['scoring_rule'])
(fixtures/'valid/reported.json').write_bytes(canonical_json(reported))
(fixtures/'index.json').write_text(json.dumps({'schema_version':'1.0','synthetic':True,'valid_bundles':['valid/minimal.perseval-result.tar.gz','valid/selected-artifact.perseval-result.tar.gz','valid/limited-provenance.perseval-result.tar.gz'],'invalid_bundles':cases,'valid_reported_records':['valid/reported.json']},indent=2)+'\n')
(ROOT/'schemas/result-bundle/1.0/SHA256SUMS').write_bytes(checksum_bytes({p.name:p.read_bytes() for p in (ROOT/'schemas/result-bundle/1.0').glob('*.schema.json')}))
print('Bundle schema digest:',contract.schema_digest)
