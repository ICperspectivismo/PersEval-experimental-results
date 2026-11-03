"""Offline catalog and evidence audits. Real Git mutations are temporary fixtures only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from perseval_results.build import build, validate_preview
from perseval_results.bundle import validate_bundle, seal_payload, pack_files
from perseval_results.catalog import validate_catalog
from perseval_results.cli import main
from perseval_results.historical import HISTORY, audit, load_inventory
from perseval_results.lfs import pointer_bytes
from perseval_results.protocol import Contract, ProtocolError, canonical_json, configuration_id, digest, read_json
from perseval_results.publish import publish
from perseval_results.records import validate_sources
from perseval_results.storage import revision_path, current_path, current_pointer

ROOT=Path(__file__).resolve().parents[1]


def copy_paper_fixture(source, destination):
    """Future measured publications in the checkout must not become test inputs."""
    for name in ('schemas','website','catalog','papers'):
        shutil.copytree(source/name,destination/name)
    paper_ids=set()
    for path in (source/'catalog/experiments').glob('*.json'):
        paper_ids.update(read_json(path.read_bytes())['reported_submissions'])
    (destination/'registry/submissions').mkdir(parents=True)
    for sid in sorted(paper_ids):
        shutil.copytree(source/'registry/submissions'/sid,destination/'registry/submissions'/sid)
    historical=destination/HISTORY;historical.mkdir(parents=True)
    for name in ('manifest.json','inventory.json','source-commit.txt'):
        shutil.copyfile(source/HISTORY/name,historical/name)
    for name in ('.gitignore','.gitattributes'):shutil.copyfile(source/name,destination/name)


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='perseval-catalog-test-');self.addCleanup(self.temp.cleanup)
        self.root=(Path(self.temp.name)/'results').resolve();self.root.mkdir()
        copy_paper_fixture(ROOT,self.root)
        self.contract=Contract(self.root)

    def sources(self):return validate_sources(self.root,self.contract,metadata_only=True)

    def specs(self):
        _,histories=self.sources()
        return validate_catalog(self.root,self.contract,histories)

    def test_complete_main_and_ablation_catalog_and_honest_paper_records(self):
        runs,histories=self.sources();specs=validate_catalog(self.root,self.contract,histories)
        self.assertEqual(runs,{})
        self.assertEqual(len(specs),284);self.assertEqual(len(histories),284)
        self.assertEqual(sum(spec['scope']=='main' for spec in specs.values()),72)
        self.assertEqual(sum(spec['scope']=='ablation' for spec in specs.values()),212)
        for history in histories.values():
            record=history[0];self.assertEqual(record['source_type'],'reported')
            for absent in ('run_id','run_ids','seed','seeds','environment','configuration_id','stddev'):
                self.assertNotIn(absent,record)
            self.assertIsNone(record['experiment']['model_revision'])
            self.assertIn('Run IDs',record['provenance']['limitations'][0])
        for spec in specs.values():
            self.assertIsNone(spec['seed_policy']['historical_execution_seed'])
            self.assertIsNone(spec['split']['historical_test_fingerprint'])
            self.assertFalse(spec['reporting']['wandb_required'])
            self.assertEqual(spec['execution']['kind'],'declarative-only')

    def test_known_cells_traits_positive_class_and_macro_f1_are_not_confused(self):
        _,histories=self.sources()
        rows={history[0]['experiment']['source_document']['row']:history[0] for history in histories.values()}
        self.assertEqual(rows['t3-r001']['values'],{'global_f1':0.555,'user_f1':0.538,'text_f1':0.376})
        self.assertEqual(rows['t3-r017']['values']['text_f1'],0.594)  # printed anomaly retained
        self.assertEqual(rows['t5-r002']['values'],{'global_f1':0.484,'user_f1':0.467,'text_f1':0.322})
        self.assertEqual(rows['t5-r002']['experiment']['trait_f1'],{'Gender':0.465,'Nationality':0.492,'Generation':0.463})
        self.assertEqual(rows['t4-r001']['values']['global_f1'],0.340)
        self.assertEqual(rows['t6-r003']['values']['global_f1'],0.365)
        self.assertEqual(rows['t5-r001']['group']['scoring_rule'],'perseval.positive-class-f1.v1')
        self.assertEqual(rows['t6-r001']['group']['scoring_rule'],'perseval.macro-f1.v1')
        self.assertEqual(rows['t10-r001']['values'],{'global_f1':0.481,'user_f1':None,'text_f1':None})
        self.assertEqual(rows['t10-r001']['experiment']['additional_reported_metrics'],{'precision':0.473,'recall':0.489})

    def test_schema_rejects_fake_execution_metadata(self):
        _,histories=self.sources();record=copy.deepcopy(next(iter(histories.values()))[0]);record['seed']=42
        with self.assertRaises(ProtocolError):self.contract.validate('reported',record,record=True)
        spec=copy.deepcopy(next(iter(self.specs().values())));spec['seed_policy']['historical_execution_seed']=42
        with self.assertRaises(ProtocolError):self.contract.validate('experiment',spec,kind='catalog')

    def test_known_parameters_distinguish_historical_decoder_implementations(self):
        specs=self.specs()
        llama=specs['paper2025-llama-epic-perspective-named-none-e0-main']['hyperparameters']['historical_implementation']
        ipa=specs['paper2025-llama-epic-ipa-named-test-e0-main']['hyperparameters']['historical_implementation']
        roberta=specs['paper2025-roberta-epic-focal-loss-named-train-e0-main']['hyperparameters']
        self.assertEqual(llama['torch_dtype'],'bfloat16');self.assertEqual(llama['device'],'cuda');self.assertNotIn('device_map',llama)
        self.assertEqual(ipa['torch_dtype'],'float16');self.assertEqual(ipa['num_profiles'],5);self.assertEqual(ipa['device_map'],'auto')
        self.assertEqual(roberta['paper']['learning_rate'],5e-6);self.assertEqual(roberta['paper']['per_device_train_batch_size'],16)
        self.assertEqual(roberta['historical_implementation']['focal_loss_gamma'],5.0)

    def test_catalog_rejects_unknown_evidence(self):
        path=next((self.root/'catalog/experiments').glob('*.json'));spec=read_json(path.read_bytes())
        spec['historical_evidence'][0]['path']=HISTORY+'/snapshot/nonexistent.csv';path.write_bytes(canonical_json(spec))
        with self.assertRaisesRegex(ProtocolError,'uninventoried evidence'):self.sources()

    def test_catalog_rejects_missing_report(self):
        path=next((self.root/'catalog/experiments').glob('*.json'));spec=read_json(path.read_bytes())
        spec['reported_submissions']=['sub_11111111-1111-4111-8111-111111111111'];path.write_bytes(canonical_json(spec))
        with self.assertRaisesRegex(ProtocolError,'missing reported'):self.sources()

    def test_paper_record_value_tampering_is_detected(self):
        _,histories=self.sources();record=copy.deepcopy(next(iter(histories.values()))[0]);record['values']['global_f1']=0.999
        (self.root/revision_path(record)).write_bytes(canonical_json(record))
        (self.root/current_path(record['submission_id'])).write_bytes(canonical_json(current_pointer(record)))
        with self.assertRaisesRegex(ProtocolError,'value mismatch'):self.sources()

    def test_swapped_trait_columns_rejected_even_after_table_digest_update(self):
        path=self.root/'papers/perseval-2025/tables.json';tables=read_json(path.read_bytes())
        row=next(row for row in tables['rows'] if row['row_id']=='t5-r002')
        row['trait_f1']['Gender'],row['trait_f1']['Nationality']=row['trait_f1']['Nationality'],row['trait_f1']['Gender']
        raw=canonical_json(tables);path.write_bytes(raw)
        source_path=self.root/'papers/perseval-2025/source.json';source=read_json(source_path.read_bytes());source['tables_sha256']=digest(raw);source_path.write_bytes(canonical_json(source))
        with self.assertRaisesRegex(ProtocolError,'printed paper cells'):self.sources()

    def test_inventory_backlinks_are_checked(self):
        path=self.root/HISTORY/'inventory.json';inventory=read_json(path.read_bytes())
        entry=next(entry for entry in inventory['entries'] if entry['associated_experiments']);entry['associated_experiments']=[]
        raw=canonical_json(inventory);path.write_bytes(raw)
        manifest_path=self.root/HISTORY/'manifest.json';manifest=read_json(manifest_path.read_bytes());manifest['inventory_sha256']=digest(raw);manifest_path.write_bytes(canonical_json(manifest))
        with self.assertRaisesRegex(ProtocolError,'backlinks disagree'):self.sources()

    def test_site_build_without_any_historical_payload_or_model_import(self):
        self.assertFalse((self.root/HISTORY/'snapshot').exists())
        real_import=__import__
        def forbid(name,*args,**kwargs):
            if name=='perseval' or name.startswith('perseval.') or name.split('.')[0] in ('torch','transformers','datasets','pandas'):
                raise AssertionError(f'benchmark/model dependency imported: {name}')
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=forbid),patch.object(socket,'socket',side_effect=AssertionError('network forbidden')):
            data=build(self.root,self.root/'site-build');validate_preview(self.root,self.root/'site-build')
        self.assertEqual(len(data['entries']),72);self.assertEqual(data['runs'],{})
        self.assertEqual(len(data['history']),72)
        self.assertTrue(all(entry['experiment']['scope']=='main' for entry in data['entries']))
        archived_ids={sid for sid,history in self.sources()[1].items()
                      if history[-1]['experiment']['scope']=='ablation'}
        self.assertEqual(len(archived_ids),212)
        self.assertTrue(archived_ids.isdisjoint(data['history']))
        import csv
        csv_ids={row['submission_id'] for row in csv.DictReader(
            (self.root/'site-build/scores.csv').read_text().splitlines())}
        self.assertEqual(csv_ids,{entry['submission_id'] for entry in data['entries']})
        self.assertTrue(archived_ids.isdisjoint(csv_ids))
        self.assertTrue(all(entry['comparability']=='documented-paper-variant' for entry in data['entries']))
        self.assertTrue(all(metric['stddev'] is None for entry in data['entries'] for metric in entry['metrics'].values()))
        self.assertEqual(read_json((self.root/'site-build/data/leaderboard.json').read_bytes()),data)
        self.assertFalse((self.root/'site-build/api').exists())

    def test_list_show_help_and_errors_are_read_only(self):
        before={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        args=['--repo',str(self.root),'catalog']
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([*args,'list','--dataset','EPIC','--model','llama','--scope','main','--json']),0)
        specs=json.loads(output.getvalue());self.assertEqual(len(specs),4)
        eid=specs[0]['experiment_id']
        with redirect_stdout(io.StringIO()) as output:self.assertEqual(main([*args,'show',eid]),0)
        self.assertEqual(json.loads(output.getvalue())['experiment_id'],eid)
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):main([*args,'show','unknown'])
        self.assertIn('catalog list',errors.getvalue())
        after={p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_new_measured_run_coexists_with_identical_paper_model_name(self):
        env={**os.environ,'GIT_CONFIG_GLOBAL':os.devnull,'GIT_CONFIG_NOSYSTEM':'1',
             'GIT_AUTHOR_NAME':'Fixture','GIT_AUTHOR_EMAIL':'fixture@example.invalid',
             'GIT_COMMITTER_NAME':'Fixture','GIT_COMMITTER_EMAIL':'fixture@example.invalid'}
        def git(*args):
            proc=subprocess.run(['git','-C',str(self.root),*args],env=env,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr);return proc.stdout.strip()
        git('init','-q');git('config','user.name','Fixture');git('config','user.email','fixture@example.invalid');git('config','commit.gpgsign','false')
        git('add','--','schemas','website','catalog','papers','registry','historical','.gitignore','.gitattributes');git('commit','-q','-m','Temporary paper metadata fixture')
        before={p.relative_to(self.root):p.read_bytes() for p in (self.root/'registry').rglob('*') if p.is_file()}
        files,docs=validate_bundle(ROOT/'fixtures/conformance/valid/minimal.perseval-result.tar.gz',self.contract)
        run=copy.deepcopy(docs['run']);run['configuration']['model']['name']='roberta-base'
        run['configuration_id']=configuration_id(run['configuration'])
        payload={k:v for k,v in files.items() if k not in ('manifest.json','checksums.sha256')};payload['run.json']=canonical_json(run)
        path=Path(self.temp.name)/'new-execution.tar.gz';path.write_bytes(pack_files(seal_payload(payload,self.contract)))
        with patch.dict(os.environ,env):
            result=publish(self.root,[path],name='New measured reproduction',submitter='Fixture group',commit=True,yes=True,output=lambda text:None)
        self.assertEqual(result.status,'committed');self.assertEqual(git('status','--porcelain'),'')
        data=build(self.root,self.root/'site-build')
        entries=[entry for entry in data['entries'] if entry['model']=='roberta-base']
        self.assertEqual({entry['source_type'] for entry in entries},{'measured','reported'})
        for name,data in before.items():self.assertEqual((self.root/name).read_bytes(),data)
        self.assertFalse((self.root/HISTORY/'snapshot').exists())
        # The normal test fixture stays independent after new measured records exist.
        later=Path(self.temp.name)/'later-test-fixture';later.mkdir()
        copy_paper_fixture(self.root,later)
        later_runs,later_histories=validate_sources(later,Contract(later),metadata_only=True)
        self.assertEqual(later_runs,{})
        self.assertEqual(len(later_histories),284)


class HistoricalAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='perseval-history-test-');self.addCleanup(self.temp.cleanup)
        self.root=(Path(self.temp.name)/'results').resolve();self.root.mkdir()
        shutil.copytree(ROOT/'schemas',self.root/'schemas');(self.root/'.gitattributes').write_text('# Fixture attributes\n')
        self.contract=Contract(self.root);self.source=(Path(self.temp.name)/'source').resolve();self.source.mkdir()
        (self.source/'data').mkdir();(self.source/'data/example.csv').write_bytes(b'explicit recovered fixture\n'*100)
        (self.source/'main.py').write_text('raise RuntimeError("historical code must never be executed")\n')
        self.env={**os.environ,'GIT_CONFIG_GLOBAL':os.devnull,'GIT_CONFIG_NOSYSTEM':'1',
                  'GIT_AUTHOR_NAME':'Fixture','GIT_AUTHOR_EMAIL':'fixture@example.invalid',
                  'GIT_COMMITTER_NAME':'Fixture','GIT_COMMITTER_EMAIL':'fixture@example.invalid'}
        def git(*args):
            proc=subprocess.run(['git','-C',str(self.source),*args],env=self.env,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr);return proc.stdout.strip()
        self.git=git;git('init','-q');git('config','commit.gpgsign','false');git('add','--','main.py','data');git('commit','-q','-m','Historical fixture')
        self.head=git('rev-parse','HEAD');self.index=(self.source/'.git/index').read_bytes()
        spec=importlib.util.spec_from_file_location('history_recovery_fixture',ROOT/'tools/recover_history.py')
        self.recovery=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.recovery)
        self.directory=self.root/HISTORY
        self.recovery.recover(self.source,self.directory,self.root,revision=self.head)

    def test_every_recovered_file_is_audited_and_source_is_untouched(self):
        result=audit(self.directory,self.contract)
        self.assertEqual(result['files'],2);self.assertEqual(result['fully_hashed_files'],2)
        self.assertEqual(result['legacy_measured_runs_created'],0)
        self.assertEqual(self.head,self.git('rev-parse','HEAD'));self.assertEqual(self.index,(self.source/'.git/index').read_bytes())
        self.assertEqual(self.git('status','--porcelain'),'')
        with self.assertRaisesRegex(ProtocolError,'already exists'):
            self.recovery.recover(self.source,self.directory,self.root,revision=self.head)

    def test_full_audit_detects_changed_and_missing_payloads(self):
        path=self.directory/'snapshot/data/example.csv';original=path.read_bytes();path.write_bytes(b'changed')
        with self.assertRaisesRegex(ProtocolError,'differs from inventory'):audit(self.directory,self.contract)
        path.write_bytes(original);path.unlink()
        with self.assertRaisesRegex(ProtocolError,'missing historical evidence'):audit(self.directory,self.contract)

    def test_unknown_files_and_links_are_rejected(self):
        path=self.directory/'snapshot/extra.txt';path.write_text('not from the commit')
        with self.assertRaisesRegex(ProtocolError,'unexpected'):audit(self.directory,self.contract)
        path.unlink();path.symlink_to(self.source/'main.py')
        with self.assertRaisesRegex(ProtocolError,'link'):audit(self.directory,self.contract)

    def test_unhydrated_clone_can_audit_pointer_identities_but_full_audit_requires_bytes(self):
        path=self.directory/'snapshot/data/example.csv';original=path.read_bytes();path.write_bytes(pointer_bytes(original))
        with self.assertRaisesRegex(ProtocolError,'unhydrated historical LFS'):audit(self.directory,self.contract)
        result=audit(self.directory,self.contract,metadata_only=True)
        self.assertEqual(result['fully_hashed_files'],1);self.assertEqual(len(result['unhydrated_lfs_paths']),1)
        path.write_bytes(pointer_bytes(b'wrong scientific data'))
        with self.assertRaises(ProtocolError):audit(self.directory,self.contract,metadata_only=True)

    def test_inventory_paths_and_blob_modes_reconstruct_git_tree(self):
        path=self.directory/'inventory.json';inventory=read_json(path.read_bytes());inventory['entries'][0]['git_mode']='100755'
        raw=canonical_json(inventory);path.write_bytes(raw)
        manifest_path=self.directory/'manifest.json';manifest=read_json(manifest_path.read_bytes());manifest['inventory_sha256']=digest(raw);manifest_path.write_bytes(canonical_json(manifest))
        with self.assertRaisesRegex(ProtocolError,'reconstruct the committed Git tree'):load_inventory(self.directory,self.contract)

    def test_original_commit_object_is_verified_independently_of_source_repo(self):
        path=self.directory/'source-commit.txt';raw=path.read_bytes()+b'changed\n';path.write_bytes(raw)
        manifest_path=self.directory/'manifest.json';manifest=read_json(manifest_path.read_bytes());manifest['source_commit_object_sha256']=digest(raw);manifest_path.write_bytes(canonical_json(manifest))
        with self.assertRaisesRegex(ProtocolError,'source commit object'):load_inventory(self.directory,self.contract)

    def test_canonical_inventory_is_deterministic_for_same_git_tree(self):
        second=self.root/'historical/second';self.recovery.recover(self.source,second,self.root,revision=self.head)
        self.assertEqual((self.directory/'inventory.json').read_bytes(),(second/'inventory.json').read_bytes())
        self.assertEqual((self.directory/'manifest.json').read_bytes(),(second/'manifest.json').read_bytes())


if __name__=='__main__':unittest.main()
