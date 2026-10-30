"""Synthetic real PersEval prediction/record/export driver. Test tooling only."""
import importlib.abc
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import socket
import sys
import types

project, work = map(lambda p:Path(p).resolve(),sys.argv[1:3])
sys.path.insert(0,str(project))
class NoConsumer(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0]=='perseval_results':raise AssertionError('producer imported the results package')
sys.meta_path.insert(0,NoConsumer())
def forbidden(*args,**kwargs):raise AssertionError('network/model/dataset/GPU execution is forbidden')
socket.socket.connect=forbidden
socket.create_connection=forbidden
# Only the dataset/model loaders are doubled: real dataset, prediction, recorder,
# Evaluator, finalization, schema validation and archive code all run below.
for name in ('datasets','transformers','torch'):
    module=types.ModuleType(name);module.__spec__=ModuleSpec(name,loader=None);sys.modules[name]=module
datasets=sys.modules['datasets']
datasets.load_dataset=datasets.load_from_disk=datasets.concatenate_datasets=forbidden
datasets.Dataset=type('Dataset',(),{'from_pandas':staticmethod(forbidden)})
NS=types.SimpleNamespace
transformers=sys.modules['transformers']
for name in ('AutoTokenizer','AutoModelForSequenceClassification','AutoModelForCausalLM','AutoModel'):
    setattr(transformers,name,type(name,(),{'from_pretrained':staticmethod(forbidden)}))
transformers.Trainer=type('Trainer',(),{})
transformers.TrainingArguments=NS;transformers.set_seed=lambda seed:None;transformers.pipeline=forbidden
torch=sys.modules['torch'];torch.__version__='synthetic-no-GPU';torch.version=NS(cuda=None)
torch.cuda=NS(is_available=lambda:False,get_device_name=forbidden)
import numpy as np
from perseval import config
from perseval.data import PerspectivistDataset,PerspectivistSplit,User
from perseval.baselines.encoder import PerspectivistEncoder
from perseval.results import LocalRun
from perseval.results.contract import Contract,digest
from perseval.results.bundle import validate_bundle
work.mkdir(parents=True,exist_ok=True)
config.output_root=work/'outputs';config.prediction_dir=str(work/'legacy-predictions')
dataset=PerspectivistDataset();dataset.name='EPIC';dataset.label='irony'
dataset.named=False;dataset.extended=False;dataset.user_adaptation=False
dataset.labels={'irony':{0,1}};dataset.result_dataset_revision='synthetic-integration-v1'
dataset.dataset=[{'user':'01','text':'t1','irony':0},{'user':'02','text':'t2','irony':1}]
dataset._result_split_definition={'protocol':'perseval.get_splits.v1','parameters':{'test_fraction':.2},'seed':17,'baseline':True}
for name in ('training','adaptation','test'):setattr(dataset,name+'_set',PerspectivistSplit(name))
dataset.test_set.annotation=[{'user':'01','text':'t1','label':{'irony':0}},{'user':'02','text':'t2','label':{'irony':1}}]
dataset.test_set.users={'01':User('01'),'02':User('02')}
for key,user in dataset.test_set.users.items():user.traits={'group':['a' if key=='01' else 'b']}
dataset.test_set.texts={'t1':{'text':'first synthetic text'},'t2':{'text':'second synthetic text'}}
loaded=NS(config=NS(num_labels=2,_commit_hash='a'*40))
transformers.AutoTokenizer.from_pretrained=staticmethod(lambda *a,**k:NS())
transformers.AutoModelForSequenceClassification.from_pretrained=staticmethod(lambda *a,**k:loaded)
model=PerspectivistEncoder('roberta-base',dataset,'irony')
model._PerspectivistEncoder__generate_data=lambda split:([{},{}],[('01','t1'),('02','t2')])
exports=[];run_ids=[]
for seed,logits in [(42,[[1,0],[0,1]]),(43,[[0,1],[0,1]])]:
    trainer=NS(args=NS(seed=seed,learning_rate=.001,num_train_epochs=2),model=loaded,
               predict=lambda data,values=logits:NS(predictions=np.array(values)))
    before=set(work.rglob('*.tar.gz'))
    assert model.predict(trainer) is None
    run=model.last_run
    assert isinstance(run,LocalRun) and not run.finalized
    assert set(work.rglob('*.tar.gz'))==before,'prediction automatically exported'
    bundle=run.export_result(output_dir=work/'exports')
    original=bundle.read_bytes()
    again=LocalRun(run.path).export_result(output_dir=work/'reexport')
    assert again.read_bytes()==original and LocalRun(run.path).run_id==run.run_id
    _,docs=validate_bundle(bundle,Contract())
    assert docs['run']['seed']==seed
    run_ids.append(run.run_id);exports.append(str(bundle))
assert run_ids[0]!=run_ids[1]
assert 'perseval_results' not in sys.modules
result={'bundles':exports,'run_ids':run_ids,'schema_digest':Contract().schema_digest,
        'archive_digests':[digest(Path(p).read_bytes()) for p in exports],
        'evidence':'Synthetic inference only; real PersEval recorder, Evaluator and explicit exporter; no models, datasets or GPU loaded.'}
(work/'producer.json').write_text(json.dumps(result,sort_keys=True)+'\n')
print(json.dumps(result,sort_keys=True))
