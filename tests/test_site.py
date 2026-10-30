"""Offline site tests; no HTTP listeners, model packages or payload downloads."""
from __future__ import annotations

import copy
import csv
import io
import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from perseval_results.build import build, generate, safe_link, validate_preview, _validate_generated
from perseval_results.bundle import seal_payload, validate_bundle, validate_files
from perseval_results.cli import main
from perseval_results.protocol import Contract, ProtocolError, canonical_json, digest, read_json
from perseval_results.records import limitations
from perseval_results.storage import current_path, current_pointer, revision_path, run_index

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT/'fixtures/conformance/valid/minimal.perseval-result.tar.gz'
RID2 = 'run_22222222-2222-4222-8222-222222222222'
SID = 'sub_11111111-1111-4111-8111-111111111111'


class SiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='perseval-site-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        for name in ('schemas','website'): shutil.copytree(ROOT/name,self.root/name)
        self.contract = Contract(self.root)
        self.files,self.docs = validate_bundle(FIXTURE,self.contract)
        self.output = self.root/'site-build'

    def write(self,path,value):
        path=self.root/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else canonical_json(value))

    def install_run(self,rid=None,seed=42,score=.75,artifact=False):
        payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
        run=copy.deepcopy(self.docs['run']);run['run_id']=rid or run['run_id'];run['seed']=seed
        metrics=copy.deepcopy(self.docs['metrics']);metrics['compact']['global_f1']=score
        payload.update({'run.json':canonical_json(run),'metrics.json':canonical_json(metrics)})
        if score != .75: payload['predictions.csv']=b'user_id,text_id,predictions\nu1,t1,1\nu2,t2,1\n'
        selected=[]
        if artifact:
            payload['artifacts/diagnostics.txt']=b'Explicit optional diagnostic.\n'
            selected=[{'path':'artifacts/diagnostics.txt','media_type':'text/plain','description':'Diagnostics'}]
        files=seal_payload(payload,self.contract,selected);docs=validate_files(files,self.contract)
        for name,raw in files.items():self.write(f'runs/{run["run_id"]}/{name}',raw)
        self.write(f'registry/runs/{run["run_id"]}.json',run_index(files,docs))
        return run['run_id']

    def submission(self,ids,sid=SID,revision=1,previous=None,status='published'):
        run=read_json((self.root/'runs'/ids[0]/'run.json').read_bytes())
        record={'schema_version':'1.0','source_type':'measured','submission_id':sid,'revision':revision,
                'previous_revision_digest':digest(canonical_json(previous)) if previous else None,
                'status':status,'superseded_by':None,'name':'Reproduction','submitter':'Research group',
                'notes':'Independent measured evidence.','citation':None,'published_at':f'2026-01-0{revision}T00:00:00Z',
                'configuration_id':run['configuration_id'],'scoring_rule':'perseval.macro-f1.v1','run_ids':ids,
                'provenance':{'source':'explicit result-bundle publication','limitations':limitations(run)}}
        self.write(revision_path(record),record);self.write(current_path(sid),current_pointer(record))
        return record

    def site_files(self):
        return {p.relative_to(self.output).as_posix():p.read_bytes() for p in self.output.rglob('*') if p.is_file()}

    def test_empty_site_has_only_internal_data_and_csv_not_public_api(self):
        data=build(self.root,self.output,source_revision='a'*40)
        self.assertEqual(data['entries'],[])
        metadata=validate_preview(self.root,self.output)
        self.assertEqual(metadata['source_revision'],'a'*40)
        self.assertEqual(set(self.site_files()),{
            'index.html','app.js','styles.css','.nojekyll','data/leaderboard.json','scores.csv'})
        self.assertNotIn('api_version',data)
        self.assertNotIn('resources',data)
        self.assertFalse(list((ROOT/'schemas/api').rglob('*.schema.json')))
        self.assertNotIn('JSON API',(self.root/'website/index.html').read_text())
        self.assertEqual(metadata['counts']['published'],0)

    def test_details_mean_std_n_revisions_and_reported_coexistence(self):
        rid=self.install_run();first=self.submission([rid])
        self.install_run(RID2,seed=7,score=.5)
        second=self.submission([rid,RID2],revision=2,previous=first)
        reported=read_json((ROOT/'fixtures/conformance/valid/reported.json').read_bytes())
        # The display model is deliberately identical to the reproduction.
        reported['model']='toy-model'
        self.write(revision_path(reported),reported);self.write(current_path(reported['submission_id']),current_pointer(reported))
        data=build(self.root,self.output)
        measured=next(e for e in data['entries'] if e['source_type']=='measured')
        paper=next(e for e in data['entries'] if e['source_type']=='reported')
        self.assertEqual(measured['run_count'],2);self.assertEqual(measured['seeds'],[42,7])
        self.assertEqual(measured['metrics']['global_f1']['mean'],.625)
        self.assertAlmostEqual(measured['metrics']['global_f1']['stddev'],.1767766952966369)
        self.assertNotEqual(measured['ranking_group_id'],paper['ranking_group_id'])
        self.assertNotIn('run_count',paper);self.assertNotIn('run_ids',paper)
        self.assertIsNone(paper['metrics']['global_f1']['stddev'])
        page=read_json((self.output/'data/leaderboard.json').read_bytes())
        self.assertEqual(page['history'][SID],[first,second])
        self.assertEqual(page['history'][SID][-1]['revision'],2)
        run=page['runs'][rid]
        self.assertEqual(run['environment'],self.docs['environment'])
        self.assertEqual(measured['run_ids'],[rid,RID2])
        self.assertEqual(measured['configuration'],run['run']['configuration'])
        rows=list(csv.DictReader(io.StringIO((self.output/'scores.csv').read_text())))
        self.assertEqual(next(r for r in rows if r['source_type']=='reported')['n'],'')
        self.assertEqual(next(r for r in rows if r['source_type']=='measured')['n'],'2')

    def test_internal_data_requires_details_and_build_identity(self):
        files,_=generate(self.root)
        document=read_json(files['data/leaderboard.json'])
        for field in ('build','history','runs','artifact_metadata'):
            changed=dict(document);changed.pop(field)
            with self.subTest(field=field),self.assertRaises(ProtocolError):
                _validate_generated({**files,'data/leaderboard.json':canonical_json(changed)},self.contract)

    def test_rebuild_removes_obsolete_api_outputs(self):
        build(self.root,self.output)
        legacy=self.output/'api/v1'
        legacy.mkdir(parents=True)
        (legacy/'metadata.json').write_bytes(b'{}\n')
        build(self.root,self.output)
        self.assertFalse((self.output/'api').exists())
        validate_preview(self.root,self.output)

    def test_deterministic_build_and_explicit_revision(self):
        self.submission([self.install_run()])
        build(self.root,self.output,source_revision='revision-A');first=self.site_files()
        build(self.root,self.output,source_revision='revision-A');self.assertEqual(first,self.site_files())
        original=read_json(first['data/leaderboard.json'])['build']
        self.assertNotIn('timestamp',original)
        build(self.root,self.output,source_revision='revision-B')
        changed=read_json((self.output/'data/leaderboard.json').read_bytes())['build']
        self.assertEqual(original['source_digest'],changed['source_digest'])
        self.assertNotEqual(original['build_id'],changed['build_id'])

    def test_full_precision_competition_ties_and_fingerprint_partitions(self):
        rid=self.install_run();self.submission([rid])
        self.submission([rid],sid='sub_22222222-2222-4222-8222-222222222222')
        self.install_run(RID2,score=.7500000000000001)
        self.submission([RID2],sid='sub_33333333-3333-4333-8333-333333333333')
        data=build(self.root,self.output)
        self.assertEqual([e['rank'] for e in data['entries']],[1,2,2])
        self.assertEqual(len({e['ranking_group_id'] for e in data['entries']}),1)
        self.assertTrue(all(e['metrics']['global_f1']['stddev'] is None for e in data['entries']))

    def test_missing_private_artifacts_and_predictions_need_no_payload_download(self):
        rid=self.install_run(artifact=True);self.submission([rid])
        (self.root/'runs'/rid/'predictions.csv').unlink()
        (self.root/'runs'/rid/'artifacts/diagnostics.txt').unlink()
        data=build(self.root,self.output)
        artifact=data['artifact_metadata'][rid][0]
        self.assertFalse(artifact['public']);self.assertIsNone(artifact['href'])
        self.assertGreater(artifact['size'],0)
        self.assertFalse((self.output/'artifacts').exists())
        self.assertFalse((self.output/'runs').exists())
        from perseval_results.records import validate_sources
        with self.assertRaisesRegex(ProtocolError,'missing required files'):
            validate_sources(self.root,self.contract)

    def policy(self,rid,**changes):
        item={'run_id':rid,'path':'artifacts/diagnostics.txt','license':'CC0-1.0','attribution':'Synthetic fixture','redistribution_permitted':True}
        item.update(changes)
        self.write('site-publication.json',{'schema_version':'1.0','artifacts':[item]})

    def test_only_explicit_licensed_artifacts_are_exposed_and_missing_public_fails(self):
        rid=self.install_run(artifact=True);self.submission([rid]);self.policy(rid)
        data=build(self.root,self.output);item=data['artifact_metadata'][rid][0]
        self.assertTrue(item['public']);self.assertEqual((self.output/item['href']).read_bytes(),b'Explicit optional diagnostic.\n')
        self.assertEqual(item['license'],'CC0-1.0');validate_preview(self.root,self.output)
        first=self.site_files();(self.root/'runs'/rid/'artifacts/diagnostics.txt').unlink()
        with self.assertRaisesRegex(ProtocolError,'public artifact is unavailable'):build(self.root,self.output)
        self.assertEqual(first,self.site_files())

    def test_complete_and_trait_metrics_require_explicit_public_selection(self):
        compact=self.docs['metrics']['compact']
        report={'format':'perseval-evaluation-v1','scoring_rule':'perseval.macro-f1.v1',
                'global':{'macro avg':{'f1-score':compact['global_f1'],'support':self.docs['metrics']['prediction_count']}},
                'user_average':{'macro avg':{'f1-score':compact['user_f1']}},
                'text_average':{'macro avg':{'f1-score':compact['text_f1']}},
                'traits':{'synthetic-trait':{'macro avg':{'f1-score':.625}}},
                'task_score':{'kind':'macro-f1','value':compact['global_f1']}}
        payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
        payload['artifacts/perseval-evaluation.json']=canonical_json(report)
        selected=[{'path':'artifacts/perseval-evaluation.json','media_type':'application/json','description':'Complete synthetic metrics'}]
        files=seal_payload(payload,self.contract,selected);docs=validate_files(files,self.contract)
        rid=docs['run']['run_id']
        for name,raw in files.items():self.write(f'runs/{rid}/{name}',raw)
        self.write(f'registry/runs/{rid}.json',run_index(files,docs));self.submission([rid])
        data=build(self.root,self.output)
        self.assertIsNone(data['runs'][rid]['complete_metrics'])
        self.policy(rid,path='artifacts/perseval-evaluation.json')
        data=build(self.root,self.output)
        self.assertEqual(data['runs'][rid]['complete_metrics'],report)

    def test_terminal_revision_retains_history_and_has_no_rank(self):
        rid=self.install_run();first=self.submission([rid])
        self.submission([rid],revision=2,previous=first,status='withdrawn')
        data=build(self.root,self.output)
        self.assertEqual(data['entries'],[])
        self.assertEqual(len(data['history'][SID]),2)
        self.assertEqual(data['history'][SID][-1]['status'],'withdrawn')

    def test_artifact_selection_requires_license_permission_and_declared_canonical_path(self):
        rid=self.install_run(artifact=True);self.submission([rid])
        for changes in ({'license':''},{'redistribution_permitted':False},{'path':'artifacts/../secret'},{'path':'artifacts/missing.txt'}):
            with self.subTest(changes=changes):
                self.policy(rid,**changes)
                with self.assertRaises(ProtocolError):build(self.root,self.output)
        self.assertFalse(self.output.exists())

    def test_safe_links_and_all_html_assets(self):
        files,_=generate(self.root)
        safe_link('data/leaderboard.json',files);safe_link('#main',files)
        for value in ('/data/leaderboard.json','../secrets','//evil.example/a','javascript:alert(1)',
                      'data:text/html,test','https://user:pass@example.org/a','data%2fleaderboard.json',
                      'index.html#missing','missing.json','data/leaderboard.json?x=1','\\secret','https://example.org/\n'):
            with self.subTest(value=value),self.assertRaises(ProtocolError):safe_link(value,files)
        (self.root/'website/index.html').write_text('<a href="missing.html">broken</a>')
        with self.assertRaisesRegex(ProtocolError,'broken site link'):build(self.root,self.output)

    def test_atomic_build_validation_failure_and_replacement_failure_preserve_previous(self):
        build(self.root,self.output);before=self.site_files()
        from perseval_results import build as service
        real=service._validate_generated
        def fail_stage(files,contract):
            fail_stage.calls+=1
            if fail_stage.calls==2:raise ProtocolError('staged validation failed')
            return real(files,contract)
        fail_stage.calls=0
        with patch.object(service,'_validate_generated',side_effect=fail_stage):
            with self.assertRaisesRegex(ProtocolError,'staged validation'):build(self.root,self.output)
        self.assertEqual(before,self.site_files())
        replace=os.replace
        def fail_replace(source,dest):
            if Path(source).name=='output':raise OSError('replacement failed')
            return replace(source,dest)
        with patch.object(service.os,'replace',side_effect=fail_replace):
            with self.assertRaisesRegex(OSError,'replacement failed'):build(self.root,self.output)
        self.assertEqual(before,self.site_files());self.assertEqual(list(self.root.glob('.perseval-site-*')),[])

    def test_failed_rollback_keeps_previous_build_backup(self):
        build(self.root,self.output);before=self.site_files()
        from perseval_results import build as service
        replace=os.replace
        def fail_replace(source,dest):
            if Path(source).name in ('output','previous'):raise OSError('filesystem unavailable')
            return replace(source,dest)
        with patch.object(service.os,'replace',side_effect=fail_replace):
            with self.assertRaisesRegex(ProtocolError,'previous build is preserved'):build(self.root,self.output)
        backup=next(self.root.glob('.perseval-site-*/previous'))
        recovered={p.relative_to(backup).as_posix():p.read_bytes() for p in backup.rglob('*') if p.is_file()}
        self.assertEqual(before,recovered)

    def test_publisher_preview_double_failure_preserves_previous_backup(self):
        build(self.root,self.output);before=self.site_files()
        candidate=self.root/'candidate';candidate.mkdir()
        shutil.copytree(self.output,candidate/'site-build')
        from perseval_results.publish import _refresh_preview
        replace=os.replace
        def fail_replace(source,dest):
            if Path(dest)==self.output:raise OSError('filesystem unavailable')
            return replace(source,dest)
        with patch('perseval_results.publish.os.replace',side_effect=fail_replace):
            with self.assertRaisesRegex(ProtocolError,'previous preview is preserved'):_refresh_preview(self.root,candidate)
        backup=next(self.root.glob('.perseval-preview-*/previous'))
        recovered={p.relative_to(backup).as_posix():p.read_bytes() for p in backup.rglob('*') if p.is_file()}
        self.assertEqual(before,recovered)

    def test_concurrent_source_change_leaves_previous_site(self):
        build(self.root,self.output);before=self.site_files()
        from perseval_results import build as service
        real=service._source_files
        def modified(root):
            modified.calls+=1
            if modified.calls==2:
                (root/'website/styles.css').write_text('body { color: black; }\n')
            return real(root)
        modified.calls=0
        with patch.object(service,'_source_files',side_effect=modified):
            with self.assertRaisesRegex(ProtocolError,'sources changed'):build(self.root,self.output)
        self.assertEqual(before,self.site_files())

    def test_check_detects_tampered_or_unexpected_outputs(self):
        build(self.root,self.output);(self.output/'data/extra.json').write_bytes(b'{}\n')
        with self.assertRaises(ProtocolError):validate_preview(self.root,self.output)
        (self.output/'data/extra.json').unlink();(self.output/'styles.css').write_text('corrupt')
        with self.assertRaisesRegex(ProtocolError,'authoritative records'):validate_preview(self.root,self.output)

    def test_symlink_outputs_and_source_replacement_protection(self):
        for path in (self.root/'registry',self.root/'website',self.root.parent/'site-build'):
            with self.assertRaisesRegex(ProtocolError,'output'):build(self.root,path)
        target=self.root/'outside';target.mkdir();(target/'keep').write_text('user')
        self.output.symlink_to(target,target_is_directory=True)
        with self.assertRaisesRegex(ProtocolError,'symlink'):build(self.root,self.output)
        self.assertEqual((target/'keep').read_text(),'user')

    def test_cli_help_build_check_errors_and_serve_without_network(self):
        prefix=['--repo',str(self.root),'site']
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main([*prefix,'build','--output','site-build/']),0)
            self.assertEqual(main([*prefix,'check','--output','site-build/']),0)
        self.assertIn('Valid site:',out.getvalue())
        with redirect_stdout(io.StringIO()),self.assertRaises(SystemExit) as exc:main([*prefix,'--help'])
        self.assertEqual(exc.exception.code,0)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main([*prefix,'serve','--port','0'])
        with patch('perseval_results.serve.serve') as serve:
            self.assertEqual(main([*prefix,'serve','--port','8000']),0)
            self.assertEqual(serve.call_args.args[2],8000)

    def test_loopback_server_binding_and_no_directory_listing(self):
        build(self.root,self.output)
        from perseval_results.serve import serve,PreviewHandler
        with patch('perseval_results.serve.ThreadingHTTPServer') as factory,redirect_stdout(io.StringIO()):
            serve(self.root,self.output,8000)
        self.assertEqual(factory.call_args.args[0],('127.0.0.1',8000))
        handler=object.__new__(PreviewHandler)
        handler.send_error=lambda *args:None
        self.assertIsNone(handler.list_directory('/secret'))

    def test_offline_build_never_imports_benchmark_or_model_dependencies(self):
        rid=self.install_run();self.submission([rid])
        real_import=__import__
        def forbidden(name,*args,**kwargs):
            if name.split('.')[0] in ('perseval','torch','datasets','transformers','pandas'):
                raise AssertionError('unexpected benchmark dependency: '+name)
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=forbidden),patch.object(socket,'socket',side_effect=AssertionError('network forbidden')):
            build(self.root,self.output);validate_preview(self.root,self.output)

    def test_frontend_ranks_are_backend_values_and_accessible_controls_exist(self):
        js=(ROOT/'website/app.js').read_text();html=(ROOT/'website/index.html').read_text()
        self.assertNotRegex(js,r'\.(?:rank|ranking_group_id)\s*=(?!=)')
        self.assertNotIn('innerHTML',js)
        self.assertIn('String(entry.rank)',js)
        for control in ('dataset-filter','label-filter','family-encoder','family-decoder',
                        'task-filter','adaptation-filter','extended-filter','search-filter',
                        'source-filter','scope-filter','group-filter','view-mode','copy-results',
                        'submission-dialog','retry'):
            self.assertIn(f'id="{control}"',html)
        self.assertIn('role="status"',html);self.assertIn('role="alert"',html)
        self.assertIn('Skip to results',html)
        self.assertNotIn('http://',html)

    def test_frontend_filters_sort_preserve_ranks_and_reported_display(self):
        if not shutil.which('node'):self.skipTest('optional Node runtime is unavailable')
        result=subprocess.run(['node','--test',str(ROOT/'tests/site-ui.test.cjs')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_workflow_prs_never_deploy_and_build_uses_shared_service(self):
        workflow=(ROOT/'.github/workflows/pages.yml').read_text()
        self.assertIn('pull_request:',workflow)
        self.assertIn("github.event_name != 'pull_request'",workflow)
        self.assertIn("vars.PUBLICATION_BRANCH || 'main'",workflow)
        self.assertIn('perseval-results site build',workflow);self.assertIn('perseval-results site check',workflow)
        self.assertIn('lfs: false',workflow);self.assertNotIn('lfs: true',workflow)
        self.assertNotIn('git push',workflow)


if __name__=='__main__':unittest.main()
