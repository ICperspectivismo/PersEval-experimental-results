"""End-to-end local publication tests; all Git mutation is confined to temporary repos."""
from __future__ import annotations

import copy
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from pathlib import Path
from unittest.mock import patch, Mock

from perseval_results.build import build, validate_preview
from perseval_results.bundle import (MAX_FILE, pack_files, read_archive, seal_payload,
                                    validate_bundle, validate_files)
from perseval_results.cli import main
from perseval_results.gitops import GitRepository
from perseval_results.lfs import pointer_bytes
from perseval_results.protocol import Contract, ProtocolError, canonical_json, configuration_id, digest, read_json
from perseval_results.publish import publish, publisher_lock, validate_git_history
from perseval_results.records import validate_sources
from perseval_results.storage import (current_pointer, current_path, managed_snapshot,
                                     revision_path, run_index)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT/'fixtures/conformance'
BUNDLE = FIXTURES/'valid/minimal.perseval-result.tar.gz'
RID2 = 'run_22222222-2222-4222-8222-222222222222'
RID3 = 'run_33333333-3333-4333-8333-333333333333'


def files_under(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file() and '.git' not in p.relative_to(root).parts}


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='perseval-publication-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = (Path(self.temp.name)/'repo').resolve()
        self.root.mkdir()
        for directory in ('schemas','website'):
            shutil.copytree(ROOT/directory,self.root/directory)
        shutil.copyfile(ROOT/'.gitignore',self.root/'.gitignore')
        shutil.copyfile(ROOT/'.gitattributes',self.root/'.gitattributes')
        (self.root/'README.md').write_text('Synthetic temporary test repository.\n')
        self.env = {**os.environ,'GIT_CONFIG_GLOBAL':os.devnull,'GIT_CONFIG_NOSYSTEM':'1',
                    'GIT_AUTHOR_NAME':'Fixture author','GIT_AUTHOR_EMAIL':'fixture@example.invalid',
                    'GIT_COMMITTER_NAME':'Fixture author','GIT_COMMITTER_EMAIL':'fixture@example.invalid',
                    'GIT_TERMINAL_PROMPT':'0'}
        self.env_patch = patch.dict(os.environ,self.env)
        self.env_patch.start();self.addCleanup(self.env_patch.stop)
        self.git('init','-q')
        self.git('config','user.name','Fixture author')
        self.git('config','user.email','fixture@example.invalid')
        self.git('config','commit.gpgsign','false')
        self.git('add','.')
        self.git('commit','-q','-m','Initial fixture sources')
        self.initial_head = self.git('rev-parse','HEAD')
        self.contract = Contract(self.root)
        self.files,self.docs = validate_bundle(BUNDLE,self.contract)
        self.plans=[]

    def git(self,*args):
        proc = subprocess.run(['git','-C',str(self.root),*args],capture_output=True,text=True,env=self.env)
        if proc.returncode:
            raise AssertionError(proc.stderr)
        return proc.stdout.strip()

    def publication(self,bundles=None,**kwargs):
        return publish(self.root,bundles or [BUNDLE],name=kwargs.pop('name','Experiment'),
                       submitter=kwargs.pop('submitter','Research group'),commit=kwargs.pop('commit',False),
                       yes=kwargs.pop('yes',True),output=kwargs.pop('output',self.plans.append),**kwargs)

    def bundle(self,rid=RID2,seed=43,score=0.5,mutate=None,predictions=None,extra=None):
        run=copy.deepcopy(self.docs['run']);run.update(run_id=rid,seed=seed)
        if mutate:mutate(run)
        run['configuration_id']=configuration_id(run['configuration'])
        metrics=copy.deepcopy(self.docs['metrics']);metrics['compact']['global_f1']=score
        payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
        payload.update({'run.json':canonical_json(run),'metrics.json':canonical_json(metrics)})
        if predictions is None and score != 0.75:
            pair = ('1','1') if score == 0.5 else ('1','0') if score > 0.5 else ('0','0')
            predictions = f'user_id,text_id,predictions\nu1,t1,{pair[0]}\nu2,t2,{pair[1]}\n'.encode()
        if predictions is not None:payload['predictions.csv']=predictions
        if extra:payload.update(extra)
        files=seal_payload(payload,self.contract)
        path=Path(self.temp.name)/f'{rid}-{seed}-{score}.tar.gz';path.write_bytes(pack_files(files))
        return path

    def data(self):
        return build(self.root,self.root/'site-build')

    def add_reported(self):
        record=read_json((FIXTURES/'valid/reported.json').read_bytes())
        path=self.root/revision_path(record);path.parent.mkdir(parents=True)
        path.write_bytes(canonical_json(record))
        (self.root/current_path(record['submission_id'])).write_bytes(canonical_json(current_pointer(record)))
        return record

    def test_single_primary_command_publishes_and_real_commit_is_managed_only(self):
        (self.root/'user-work.txt').write_text('previously committed user work\n')
        self.git('add','user-work.txt')
        self.git('commit','-q','-m','Existing user file')
        result=self.publication(commit=True)
        self.assertEqual(result.status,'committed')
        self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))
        changed=self.git('diff-tree','--no-commit-id','--name-only','-r',result.commit_sha).splitlines()
        self.assertTrue(changed)
        self.assertTrue(all(p.startswith(('runs/','registry/')) for p in changed))
        self.assertEqual(self.git('status','--porcelain'),'')
        self.assertEqual((self.root/'user-work.txt').read_text(),'previously committed user work\n')
        message=self.git('show','-s','--format=%B',result.commit_sha)
        self.assertIn(result.record['submission_id'],message)
        self.assertIn(self.docs['run']['run_id'],message)
        self.assertIn('Global F1 mean: 0.75',message)
        self.assertNotIn('site-build/',changed)
        self.assertEqual(self.git('remote'),'')
        validate_git_history(self.root)
        runs,histories=validate_sources(self.root,self.contract)
        self.assertEqual((len(runs),len(histories)),(1,1))
        sid=result.record['submission_id'];rid=self.docs['run']['run_id']
        self.assertEqual(read_json((self.root/current_path(sid)).read_bytes()),current_pointer(result.record))
        self.assertEqual(run_index(self.files,self.docs),read_json((self.root/f'registry/runs/{rid}.json').read_bytes()))
        self.assertEqual(self.data()['entries'][0]['metrics']['global_f1'],{'mean':0.75,'stddev':None})

    def test_retry_without_submission_is_idempotent(self):
        first=self.publication(commit=True)
        before=files_under(self.root);head=self.git('rev-parse','HEAD')
        second=self.publication(commit=True,yes=False,interactive=False)
        self.assertEqual(second.status,'unchanged')
        self.assertEqual(second.record,first.record)
        self.assertEqual(before,files_under(self.root));self.assertEqual(head,self.git('rev-parse','HEAD'))
        self.assertEqual(len(self.data()['entries']),1)

    def test_retry_changed_name_requires_explicit_reuse(self):
        self.publication()
        before=managed_snapshot(self.root)
        with self.assertRaisesRegex(ProtocolError,'reuse-existing-runs'):
            self.publication(name='Independent reproduction')
        self.assertEqual(before,managed_snapshot(self.root))
        result=self.publication(name='Independent reproduction',reuse_existing_runs=True)
        self.assertEqual(len(self.data()['entries']),2)
        self.assertTrue(any('reused' in v for v in result.record['provenance']['limitations']))

    def test_explicit_reuse_even_same_display_metadata_creates_new_entry(self):
        first=self.publication();second=self.publication(reuse_existing_runs=True)
        self.assertNotEqual(first.record['submission_id'],second.record['submission_id'])
        self.assertEqual(len(self.data()['entries']),2)
        self.assertEqual(len(self.data()['runs']),1)
        with self.assertRaisesRegex(ProtocolError,'ambiguous'):
            self.publication()

    def test_new_execution_same_configuration_creates_independent_entry(self):
        first=self.publication();second=self.publication([self.bundle()])
        self.assertNotEqual(first.record['submission_id'],second.record['submission_id'])
        self.assertEqual(first.record['configuration_id'],second.record['configuration_id'])
        self.assertEqual(len(self.data()['entries']),2)
        self.assertEqual({e['model'] for e in self.data()['entries']},{'toy-model'})

    def test_duplicate_members_in_invocation_do_not_increase_count(self):
        result=self.publication([BUNDLE,BUNDLE])
        self.assertEqual(result.plan['aggregation']['count'],1)
        self.assertEqual(len(result.plan['bundles']),1)

    def test_multirun_mean_sample_stddev_and_no_best_seed(self):
        result=self.publication([BUNDLE,self.bundle()])
        entry=self.data()['entries'][0]
        self.assertEqual(entry['run_count'],2);self.assertEqual(entry['seeds'],[42,43])
        self.assertEqual(entry['metrics']['global_f1']['mean'],0.625)
        self.assertAlmostEqual(entry['metrics']['global_f1']['stddev'],0.1767766952966369)
        self.assertEqual(result.plan['aggregation']['run_ids'],entry['run_ids'])

    def test_revision_addition_inherits_metadata_preserves_all_runs_and_history(self):
        first=self.publication(commit=True);sid=first.record['submission_id']
        before=(self.root/revision_path(first.record)).read_bytes()
        second=self.publication([self.bundle()],submission=sid,name=None,submitter=None,commit=True)
        self.assertEqual(second.record['revision'],2)
        self.assertEqual(second.record['name'],first.record['name'])
        self.assertEqual(second.record['previous_revision_digest'],digest(before))
        self.assertEqual((self.root/revision_path(first.record)).read_bytes(),before)
        self.assertEqual(len(second.record['run_ids']),2)
        self.assertEqual(len(self.data()['history'][sid]),2)
        validate_git_history(self.root)
        repeated=self.publication(submission=sid,name=None,submitter=None)
        self.assertEqual(repeated.status,'unchanged');self.assertEqual(len(repeated.record['run_ids']),2)

    def test_missing_submission_is_not_inferred(self):
        self.publication()
        with self.assertRaisesRegex(ProtocolError,'reuse-existing-runs'):
            self.publication([BUNDLE,self.bundle()])
        with self.assertRaisesRegex(ProtocolError,'existing submission'):
            self.publication(submission='sub_aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa')

    def test_different_payload_same_run_id_conflicts(self):
        self.publication();before=managed_snapshot(self.root)
        changed=self.bundle(rid=self.docs['run']['run_id'],score=0.1)
        with self.assertRaisesRegex(ProtocolError,'different immutable content'):
            self.publication([changed])
        self.assertEqual(before,managed_snapshot(self.root))

    def test_identical_scientific_content_under_new_uuid_blocked_and_reported(self):
        self.publication()
        alias=self.bundle(seed=99,score=0.75)
        before=managed_snapshot(self.root)
        with self.assertRaisesRegex(ProtocolError,'Identical scientific payload') as error:
            self.publication([alias])
        self.assertIn(RID2,str(error.exception));self.assertIn(self.docs['run']['run_id'],str(error.exception))
        self.assertEqual(before,managed_snapshot(self.root))
        result=self.publication([alias],allow_duplicate_content=True)
        self.assertTrue(result.plan['duplicate_scientific_content'])
        self.assertTrue(result.record['provenance']['limitations'])

    def test_identical_content_within_aggregate_needs_explicit_acknowledgement(self):
        alias=self.bundle(seed=42,score=0.75)
        with self.assertRaisesRegex(ProtocolError,'Identical scientific payload'):
            self.publication([BUNDLE,alias])
        result=self.publication([BUNDLE,alias],allow_duplicate_content=True)
        self.assertEqual(result.plan['aggregation']['count'],2)
        self.assertEqual(result.plan['aggregation']['metrics']['global_f1'],{'mean':0.75,'stddev':0.0})
        self.assertTrue(result.record['provenance']['limitations'])
        self.assertEqual(self.publication([BUNDLE,alias]).status,'unchanged')

    def test_duplicate_with_external_alias_still_validates_internal_group(self):
        third=self.bundle(rid=RID3,seed=44,score=0.75)
        self.publication([third])
        alias=self.bundle(seed=43,score=0.75)
        result=self.publication([BUNDLE,alias],allow_duplicate_content=True)
        self.assertEqual(result.plan['aggregation']['count'],2)
        validate_sources(self.root,self.contract)

    def test_same_claimed_test_predictions_and_rule_cannot_have_different_metrics(self):
        self.publication()
        contradictory=self.bundle(predictions=self.files['predictions.csv'],score=0.123)
        with self.assertRaisesRegex(ProtocolError,'inconsistent metrics'):
            self.publication([contradictory],allow_duplicate_content=True)
        self.assertEqual(len(self.data()['runs']),1)

    def test_same_seed_different_predictions_scores_is_not_deduplicated(self):
        self.publication([BUNDLE,self.bundle(seed=42)])
        self.assertEqual(self.data()['entries'][0]['seeds'],[42,42])

    def test_incompatible_configuration_never_installed(self):
        second=self.bundle(mutate=lambda r:r['configuration']['split'].update(seed=999))
        with self.assertRaisesRegex(ProtocolError,'configuration_id'):
            self.publication([BUNDLE,second])
        self.assertEqual(managed_snapshot(self.root),{})

    def test_unknown_test_fingerprint_cannot_aggregate(self):
        first=self.bundle(mutate=lambda r:r['configuration']['split'].update(test_fingerprint=None))
        second=self.bundle(rid=RID3,score=0.4,mutate=lambda r:r['configuration']['split'].update(test_fingerprint=None))
        with self.assertRaisesRegex(ProtocolError,'test fingerprint'):
            self.publication([first,second])

    def test_reported_and_measured_coexist_without_fabricated_runs(self):
        report=self.add_reported();self.publication()
        entries=self.data()['entries'];self.assertEqual(len(entries),2)
        self.assertEqual(len({e['ranking_group_id'] for e in entries}),2)
        entry=next(e for e in entries if e['source_type']=='reported')
        self.assertEqual(entry['submission_id'],report['submission_id'])
        for key in ('run_ids','seeds','run_count','configuration_id'):self.assertNotIn(key,entry)
        self.assertIsNone(entry['metrics']['global_f1']['stddev'])
        with self.assertRaisesRegex(ProtocolError,'active measured'):
            self.publication(submission=report['submission_id'])

    def test_backend_competition_ranking_full_precision(self):
        self.publication();self.publication(reuse_existing_runs=True)
        self.publication([self.bundle(score=0.749999999999)])
        self.assertEqual([e['rank'] for e in self.data()['entries']],[1,1,3])
        self.assertIn('String(entry.rank)',(self.root/'website/app.js').read_text())

    def test_comparability_uses_provenance_of_every_included_run(self):
        incomplete=self.bundle(mutate=lambda r:r['provenance'].update(split_provenance=None))
        result=self.publication([BUNDLE,incomplete])
        self.assertEqual(result.plan['comparability'],'limited-provenance')
        self.assertIn('Unavailable: split provenance',result.plan['limitations'])

    def test_conflicting_label_mappings_are_not_aggregated(self):
        other=self.bundle(mutate=lambda r:r['provenance'].update(label_mapping={'zero':'0','one':'1'}))
        with self.assertRaisesRegex(ProtocolError,'label mapping'):
            self.publication([BUNDLE,other])

    def test_incompatible_test_fingerprints_never_share_ranking(self):
        self.publication()
        second=self.bundle(mutate=lambda r:r['configuration']['split'].update(test_fingerprint=digest(b'other gold')))
        self.publication([second])
        entries=self.data()['entries']
        self.assertEqual(len({e['ranking_group_id'] for e in entries}),2)
        self.assertEqual([e['rank'] for e in entries],[1,1])

    def test_limited_provenance_visible_and_isolated(self):
        path=FIXTURES/'valid/limited-provenance.perseval-result.tar.gz'
        first=self.publication([path]);self.publication([path],reuse_existing_runs=True)
        entries=self.data()['entries']
        self.assertEqual({e['comparability'] for e in entries},{'limited-provenance'})
        self.assertEqual(len({e['ranking_group_id'] for e in entries}),2)
        self.assertIn('Unavailable: test fingerprint',first.plan['limitations'])

    def test_dry_run_has_no_checkout_or_index_writes_or_prompt(self):
        before=files_under(self.root);index=(self.root/'.git/index').read_bytes()
        result=self.publication(commit=True,dry_run=True,yes=False,interactive=False,
                                input_func=Mock(side_effect=AssertionError('must not prompt')))
        self.assertEqual(result.status,'dry-run')
        self.assertTrue(result.plan['planned_files']);self.assertTrue(result.plan['commit']['files'])
        self.assertEqual(before,files_under(self.root));self.assertEqual(index,(self.root/'.git/index').read_bytes())
        self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))
        self.assertFalse((self.root/'.perseval-results.lock').exists())

    def test_plan_contains_required_review_fields(self):
        result=self.publication(dry_run=True)
        plan=result.plan
        self.assertTrue(self.plans)
        summary=plan['bundles'][0]
        for field in ('run_id','dataset','model','task','adaptation','extended','seed','configuration_id','compact_metrics','artifacts','limitations','import_status'):
            self.assertIn(field,summary)
        for field in ('submission','aggregation','planned_files','commit','rank','comparability'):self.assertIn(field,plan)
        self.assertIn(result.record['submission_id'],plan['commit']['message'])
        self.assertTrue(all(not item['path'].startswith('/') for item in plan['planned_files']))

    def test_interactive_confirmation_after_plan_before_persistence(self):
        def answer(prompt):
            self.assertTrue(self.plans);self.assertEqual(managed_snapshot(self.root),{})
            self.assertIn('[y/N]',prompt)
            return 'yes'
        result=self.publication(yes=False,interactive=True,input_func=answer)
        self.assertEqual(result.status,'published')

    def test_interactive_rejection_and_eof_write_no_sources_or_preview(self):
        for answer in ('','n'):
            result=self.publication(yes=False,interactive=True,input_func=lambda _:answer)
            self.assertEqual(result.status,'cancelled');self.assertEqual(managed_snapshot(self.root),{})
            self.assertFalse((self.root/'site-build').exists())
        with patch('builtins.input',side_effect=EOFError):
            result=self.publication(yes=False,interactive=True)
        self.assertEqual(result.status,'cancelled')

    def test_noninteractive_mutation_requires_yes(self):
        with self.assertRaisesRegex(ProtocolError,'requires --yes'):
            self.publication(yes=False,interactive=False)
        self.assertTrue(self.plans);self.assertEqual(managed_snapshot(self.root),{})
        self.assertFalse((self.root/'.perseval-results.lock').exists())

    def test_lock_blocks_another_process_and_dry_run_without_stealing(self):
        with publisher_lock(self.root):
            lock=(self.root/'.perseval-results.lock').read_bytes()
            for dry in (False,True):
                with self.assertRaisesRegex(ProtocolError,'lock|publisher'):
                    self.publication(dry_run=dry)
            script='from pathlib import Path; from perseval_results.publish import publisher_lock; '\
                   f'\nwith publisher_lock(Path({str(self.root)!r})): pass'
            env={**self.env,'PYTHONPATH':str(ROOT/'src')}
            child=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,env=env)
            self.assertNotEqual(child.returncode,0);self.assertIn('publisher',child.stderr)
            self.assertEqual(lock,(self.root/'.perseval-results.lock').read_bytes())
        self.assertFalse((self.root/'.perseval-results.lock').exists())

    def test_invalid_bundle_never_creates_persistent_paths(self):
        before=files_under(self.root)
        with self.assertRaises(ProtocolError):
            self.publication([FIXTURES/'invalid/checksum-mismatch.perseval-result.tar.gz'])
        self.assertEqual(before,files_under(self.root))

    def test_build_validation_failure_before_install_leaves_repo_untouched(self):
        before=files_under(self.root)
        with patch('perseval_results.publish.build',side_effect=ProtocolError('site-build validation failed')):
            with self.assertRaisesRegex(ProtocolError,'site-build validation'):
                self.publication(commit=True)
        self.assertEqual(before,files_under(self.root));self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))

    def test_partial_install_failure_rolls_back_every_created_file(self):
        import perseval_results.publish as module
        original=module._install_file;counter=0
        def fail(path,data):
            nonlocal counter
            counter+=1
            if counter==4:raise OSError('simulated disk failure')
            return original(path,data)
        before=files_under(self.root)
        with patch('perseval_results.publish._install_file',side_effect=fail):
            with self.assertRaisesRegex(OSError,'disk failure'):self.publication()
        self.assertEqual(before,files_under(self.root));self.assertEqual(managed_snapshot(self.root),{})

    def test_commit_hook_failure_rolls_back_sources_index_and_preview(self):
        first=self.publication(commit=True);sid=first.record['submission_id']
        (self.root/'site-build/user-work.txt').write_text('preserve ignored local work')
        before=files_under(self.root);index=(self.root/'.git/index').read_bytes();head=self.git('rev-parse','HEAD')
        hook=self.root/'.git/hooks/pre-commit';hook.write_text('#!/bin/sh\necho "fixture hook rejected" >&2\nexit 1\n');hook.chmod(0o755)
        with self.assertRaisesRegex(ProtocolError,'fixture hook rejected'):
            self.publication([self.bundle()],submission=sid,name=None,submitter=None,commit=True)
        self.assertEqual(before,files_under(self.root));self.assertEqual(index,(self.root/'.git/index').read_bytes())
        self.assertEqual(head,self.git('rev-parse','HEAD'))
        self.assertEqual(self.git('diff','--cached','--name-only'),'')
        self.assertEqual(len(self.data()['history'][sid]),1)

    def test_failed_commit_of_pending_sources_preserves_previous_local_publication(self):
        first=self.publication();before=files_under(self.root)
        hook=self.root/'.git/hooks/pre-commit';hook.write_text('#!/bin/sh\nexit 1\n');hook.chmod(0o755)
        with self.assertRaisesRegex(ProtocolError,'clean checkout'):
            self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))
        self.assertEqual(len(self.data()['history'][first.record['submission_id']]),1)
        self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))

    def test_git_clean_filter_cannot_silently_change_managed_bytes(self):
        (self.root/'.gitattributes').write_text('runs/** filter=fixture\n')
        self.git('config','filter.fixture.clean','sed s/user_id/changed_user_id/g')
        self.git('add','.gitattributes');self.git('commit','-q','-m','Fixture filter')
        head=self.git('rev-parse','HEAD')
        before=files_under(self.root);index=(self.root/'.git/index').read_bytes()
        with self.assertRaisesRegex(ProtocolError,'staged bytes differ'):
            self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))
        self.assertEqual(index,(self.root/'.git/index').read_bytes())
        self.assertEqual(head,self.git('rev-parse','HEAD'))

    def test_first_commit_failure_removes_entire_new_submission(self):
        before=files_under(self.root)
        hook=self.root/'.git/hooks/pre-commit';hook.write_text('#!/bin/sh\nexit 1\n');hook.chmod(0o755)
        with self.assertRaisesRegex(ProtocolError,'git commit failed'):
            self.publication(commit=True)
        self.assertEqual(before,files_under(self.root));self.assertEqual(self.git('diff','--cached','--name-only'),'')

    def test_no_commit_preview_copy_failure_rolls_back_sources(self):
        with patch('perseval_results.publish._refresh_preview',side_effect=OSError('copy failed')):
            with self.assertRaisesRegex(OSError,'copy failed'):self.publication()
        self.assertEqual(managed_snapshot(self.root),{})

    def test_post_install_validation_failure_rolls_back_all_sources(self):
        original=validate_sources
        calls=0
        def fail_second(root,contract,**kwargs):
            nonlocal calls
            calls+=1
            if calls==2:raise ProtocolError('installed registry validation failed')
            return original(root,contract,**kwargs)
        before=files_under(self.root)
        with patch('perseval_results.publish.validate_sources',side_effect=fail_second):
            with self.assertRaisesRegex(ProtocolError,'installed registry'):
                self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))
        self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))

    def test_post_commit_inspection_failure_does_not_rollback_commit(self):
        original=GitRepository.head_files
        def fail_after_commit(repo,head=None):
            if head is not None and head!=self.initial_head:
                raise OSError('post-commit inspection failed')
            return original(repo,head)
        with patch.object(GitRepository,'head_files',new=fail_after_commit):
            result=self.publication(commit=True)
        self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))
        self.assertTrue(result.warnings)
        self.assertEqual(len(validate_sources(self.root,self.contract)[0]),1)

    def test_cli_post_commit_output_failure_reports_durable_sha(self):
        args=['--repo',str(self.root),'publish-bundle',str(BUNDLE),'--name','Experiment',
              '--submitter','Research group','--commit','--yes']
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as errors:
            with patch('builtins.print',side_effect=OSError('display failed')):
                with self.assertRaises(SystemExit):main(args)
        head=self.git('rev-parse','HEAD')
        self.assertNotEqual(head,self.initial_head)
        self.assertIn(head,errors.getvalue());self.assertIn('durable',errors.getvalue())
        self.assertEqual(len(validate_sources(self.root,self.contract)[0]),1)

    def test_post_commit_preview_failure_keeps_durable_commit_and_sources(self):
        with patch('perseval_results.publish._refresh_preview',side_effect=OSError('late preview failure')):
            result=self.publication(commit=True)
        self.assertEqual(result.status,'committed');self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))
        self.assertTrue(result.warnings);self.assertIn('durable',result.warnings[0])
        self.assertEqual(len(validate_sources(self.root,self.contract)[0]),1)
        self.assertNotEqual(self.initial_head,result.commit_sha)

    def test_durable_commit_is_reported_even_if_lock_release_raises(self):
        @contextmanager
        def failing_release(root):
            with publisher_lock(root):
                yield
            raise OSError('simulated lock cleanup failure')
        with patch('perseval_results.publish.publisher_lock',new=failing_release):
            result=self.publication(commit=True)
        self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))
        self.assertTrue(any('lock cleanup' in item for item in result.warnings))
        self.assertEqual(len(validate_sources(self.root,self.contract)[0]),1)

    def test_no_commit_publication_requires_manual_commit_before_commit_retry(self):
        first=self.publication();self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))
        with self.assertRaisesRegex(ProtocolError,'clean checkout'):
            self.publication(commit=True)
        self.git('add','runs','registry');self.git('commit','-q','-m','Manually review and commit local publication')
        head=self.git('rev-parse','HEAD')
        second=self.publication(commit=True)
        self.assertEqual(second.record,first.record);self.assertEqual(second.status,'unchanged')
        self.assertEqual(head,self.git('rev-parse','HEAD'))
        self.assertEqual(len(self.data()['entries']),1);self.assertEqual(len(self.data()['history'][first.record['submission_id']]),1)

    def test_commit_dirty_staged_unstaged_untracked_rejected_without_writes(self):
        for kind in ('staged','unstaged','untracked'):
            with self.subTest(kind=kind):
                path=self.root/('README.md' if kind!='untracked' else 'research-notes.txt')
                original=path.read_bytes() if path.exists() else None
                path.write_text('user work must survive\n')
                if kind=='staged':self.git('add','README.md')
                before=files_under(self.root);index=(self.root/'.git/index').read_bytes()
                with self.assertRaisesRegex(ProtocolError,'clean checkout'):
                    self.publication(commit=True)
                self.assertEqual(before,files_under(self.root));self.assertEqual(index,(self.root/'.git/index').read_bytes())
                self.assertEqual(self.git('rev-parse','HEAD'),self.initial_head)
                if original is None:path.unlink()
                else:path.write_bytes(original)
                if kind=='staged':self.git('add','README.md')

    def test_commit_missing_identity_rejected_before_writes(self):
        self.git('config','--unset','user.email')
        before=files_under(self.root)
        identity_keys=('GIT_AUTHOR_NAME','GIT_AUTHOR_EMAIL','GIT_COMMITTER_NAME','GIT_COMMITTER_EMAIL','EMAIL')
        with patch.dict(os.environ):
            for key in identity_keys:os.environ.pop(key,None)
            with self.assertRaisesRegex(ProtocolError,'identity.*user.email'):
                self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))

    def test_tracked_preview_is_rejected_without_overwriting_it(self):
        path=self.root/'site-build/index.html';path.parent.mkdir();path.write_text('tracked user preview\n')
        self.git('add','-f','site-build/index.html');self.git('commit','-q','-m','Bad tracked generated preview fixture')
        head=self.git('rev-parse','HEAD')
        with self.assertRaisesRegex(ProtocolError,'tracked generated output'):
            self.publication(commit=True)
        self.assertEqual(path.read_text(),'tracked user preview\n');self.assertEqual(head,self.git('rev-parse','HEAD'))
        self.assertEqual(managed_snapshot(self.root),{})

    def test_missing_preview_ignore_rules_rejected_before_writes(self):
        (self.root/'.gitignore').write_text('*.pyc\n')
        self.git('add','.gitignore');self.git('commit','-q','-m','Bad ignore fixture')
        before=files_under(self.root)
        with self.assertRaisesRegex(ProtocolError,'must be ignored by Git'):
            self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))

    def test_unfinished_merge_and_rebase_rejected(self):
        for marker in ('MERGE_HEAD','rebase-merge','rebase-apply','CHERRY_PICK_HEAD'):
            path=self.root/'.git'/marker
            if marker.startswith('rebase'):path.mkdir()
            else:path.write_text(self.initial_head+'\n')
            with self.subTest(marker=marker),self.assertRaisesRegex(ProtocolError,'unfinished Git operation'):
                self.publication(commit=True)
            if path.is_dir():path.rmdir()
            else:path.unlink()
            self.assertEqual(managed_snapshot(self.root),{})

    def test_unresolved_conflicts_rejected_without_index_changes(self):
        oid=self.git('rev-parse','HEAD:README.md')
        payload=f'0 {"0"*40}\tREADME.md\n100644 {oid} 1\tREADME.md\n100644 {oid} 2\tREADME.md\n'
        proc=subprocess.run(['git','-C',str(self.root),'update-index','--index-info'],input=payload,text=True,capture_output=True,env=self.env)
        self.assertEqual(proc.returncode,0,proc.stderr)
        index=(self.root/'.git/index').read_bytes()
        with self.assertRaisesRegex(ProtocolError,'unresolved Git conflicts'):
            self.publication(commit=True)
        self.assertEqual(index,(self.root/'.git/index').read_bytes());self.assertEqual(managed_snapshot(self.root),{})

    def test_known_upstream_behind_or_diverged_requires_user_update(self):
        # Use a local tracking branch and commit-tree: no fetch, network, switch or reset.
        tree=self.git('rev-parse','HEAD^{tree}')
        upstream=self.git('commit-tree',tree,'-p',self.initial_head,'-m','Upstream-only fixture')
        self.git('update-ref','refs/heads/upstream-fixture',upstream)
        self.git('branch','--set-upstream-to=upstream-fixture')
        before=files_under(self.root)
        with self.assertRaisesRegex(ProtocolError,'behind.*update the checkout'):
            self.publication(commit=True)
        local=self.git('commit-tree',tree,'-p',self.initial_head,'-m','Local-only fixture')
        branch=self.git('symbolic-ref','HEAD');self.git('update-ref',branch,local)
        with self.assertRaisesRegex(ProtocolError,'diverged.*update the checkout'):
            self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))

    def test_local_ahead_upstream_is_allowed_and_never_pushed(self):
        self.git('branch','upstream-fixture')
        self.git('branch','--set-upstream-to=upstream-fixture')
        result=self.publication(commit=True)
        self.assertEqual(result.status,'committed')
        self.assertEqual(self.git('rev-parse','upstream-fixture'),self.initial_head)
        self.assertEqual(result.plan['git_preflight']['upstream']['upstream_ahead'],0)
        second=self.publication(commit=True)
        self.assertEqual(second.status,'unchanged')
        self.assertEqual(second.plan['git_preflight']['upstream']['local_ahead'],1)

    def test_concurrent_user_file_after_plan_is_preserved_and_stops_commit(self):
        def output(plan):
            (self.root/'new-user-work.txt').write_text('created while confirming\n')
        with self.assertRaisesRegex(ProtocolError,'changed concurrently'):
            self.publication(commit=True,output=output)
        self.assertEqual((self.root/'new-user-work.txt').read_text(),'created while confirming\n')
        self.assertEqual(managed_snapshot(self.root),{});self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))

    def test_concurrent_staging_after_preflight_is_never_restored_over(self):
        def output(plan):
            (self.root/'new-user-work.txt').write_text('user staged after preflight\n')
            self.git('add','new-user-work.txt')
        with self.assertRaisesRegex(ProtocolError,'changed concurrently'):
            self.publication(commit=True,output=output)
        self.assertEqual(self.git('diff','--cached','--name-only'),'new-user-work.txt')
        self.assertEqual(managed_snapshot(self.root),{})

    def test_concurrent_change_after_staging_rolls_back_only_publication(self):
        import perseval_results.gitops as module
        original=module.git_process
        def inject(root,*args,**kwargs):
            proc=original(root,*args,**kwargs)
            if args and args[0]=='add' and kwargs.get('private_index'):
                (self.root/'user-during-staging.txt').write_text('preserve\n')
            return proc
        index=(self.root/'.git/index').read_bytes()
        with patch('perseval_results.gitops.git_process',side_effect=inject):
            with self.assertRaisesRegex(ProtocolError,'changed concurrently'):
                self.publication(commit=True)
        self.assertEqual((self.root/'user-during-staging.txt').read_text(),'preserve\n')
        self.assertEqual(index,(self.root/'.git/index').read_bytes());self.assertEqual(managed_snapshot(self.root),{})

    def test_concurrent_edit_of_created_publication_file_survives_failure(self):
        original=validate_sources;target=None
        def edit(root,contract,**kwargs):
            nonlocal target
            if root==self.root and (root/'runs').exists():
                target=next((root/'runs').glob('*/metrics.json'))
                target.write_text('concurrent user edit\n')
            return original(root,contract,**kwargs)
        with patch('perseval_results.publish.validate_sources',side_effect=edit):
            with self.assertRaisesRegex(ProtocolError,'concurrent edits preserved'):
                self.publication(commit=True)
        self.assertEqual(target.read_text(),'concurrent user edit\n')
        self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))

    def test_git_commands_never_invoke_transports_or_history_rewrites(self):
        import perseval_results.gitops as module
        original=module.git_process;commands=[]
        def observe(root,*args,**kwargs):
            commands.append(args[0]);return original(root,*args,**kwargs)
        with patch('perseval_results.gitops.git_process',side_effect=observe):
            result=self.publication(commit=True)
        self.assertEqual(result.status,'committed')
        self.assertFalse(set(commands)&{'pull','push','fetch','reset','switch','checkout','rebase'})

    def test_exact_staged_allowlist_is_verified_not_just_expected_blobs(self):
        import perseval_results.gitops as module
        original=module.git_process
        def inject(root,*args,**kwargs):
            proc=original(root,*args,**kwargs)
            if args and args[0]=='add' and kwargs.get('private_index'):
                new_path=next(path for path in args if path.startswith('registry/'))
                oid=original(root,'rev-parse',f':{new_path}',**kwargs).stdout.decode().strip()
                altered=original(root,'update-index','--cacheinfo',f'100644,{oid},README.md',**kwargs)
                self.assertEqual(altered.returncode,0,altered.stderr)
            return proc
        before=files_under(self.root);index=(self.root/'.git/index').read_bytes()
        with patch('perseval_results.gitops.git_process',side_effect=inject):
            with self.assertRaisesRegex(ProtocolError,'exact planned file allowlist'):
                self.publication(commit=True)
        self.assertEqual(before,files_under(self.root));self.assertEqual(index,(self.root/'.git/index').read_bytes())
        self.assertEqual(self.initial_head,self.git('rev-parse','HEAD'))

    def test_private_index_cleanup_failure_after_commit_is_durable(self):
        original=Path.unlink
        def fail(path,*args,**kwargs):
            if path.name.startswith('perseval-index-') and not path.name.endswith('.lock'):
                raise OSError('simulated private index cleanup failure')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',new=fail):
            result=self.publication(commit=True)
        self.assertEqual(result.status,'committed');self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))
        self.assertTrue(any('cleanup failed' in warning for warning in result.warnings))
        self.assertEqual(self.git('status','--porcelain'),'')
        self.assertEqual(len(validate_sources(self.root,self.contract)[0]),1)

    def test_index_finalization_failure_keeps_successful_commit(self):
        original=os.replace
        def fail(source,destination,*args,**kwargs):
            if Path(destination)==self.root/'.git/index':
                raise OSError('simulated real index installation failure')
            return original(source,destination,*args,**kwargs)
        with patch('perseval_results.gitops.os.replace',side_effect=fail):
            result=self.publication(commit=True)
        self.assertEqual(result.status,'committed');self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))
        self.assertNotEqual(self.initial_head,result.commit_sha)
        self.assertTrue(any('durable' in warning for warning in result.warnings))
        self.assertEqual(len(validate_sources(self.root,self.contract)[0]),1)

    def test_new_concurrent_git_lock_after_commit_is_not_removed(self):
        original=GitRepository.head_files
        lock=self.root/'.git/index.lock'
        def after_commit(repo,head=None):
            if head is not None and head!=self.initial_head:
                lock.write_bytes(b'other Git operation owns this new lock\n')
            return original(repo,head)
        with patch.object(GitRepository,'head_files',new=after_commit):
            result=self.publication(commit=True)
        self.assertEqual(result.status,'committed')
        self.assertEqual(lock.read_bytes(),b'other Git operation owns this new lock\n')
        self.assertEqual(result.commit_sha,self.git('rev-parse','HEAD'))

    def test_failing_hook_does_not_erase_user_files_it_created(self):
        hook=self.root/'.git/hooks/pre-commit'
        hook.write_text('#!/bin/sh\nprintf "user hook notes\\n" > hook-notes.txt\nexit 1\n');hook.chmod(0o755)
        index=(self.root/'.git/index').read_bytes()
        with self.assertRaisesRegex(ProtocolError,'git commit failed'):
            self.publication(commit=True)
        self.assertEqual((self.root/'hook-notes.txt').read_text(),'user hook notes\n')
        self.assertEqual(index,(self.root/'.git/index').read_bytes());self.assertEqual(managed_snapshot(self.root),{})

    def test_required_lfs_preflight_failure_has_no_persistent_writes(self):
        import perseval_results.gitops as module
        original=module.git_process
        def no_lfs(root,*args,**kwargs):
            if args[:2]==('lfs','version'):
                return subprocess.CompletedProcess(args,1,b'',b'LFS unavailable')
            return original(root,*args,**kwargs)
        before=files_under(self.root);index=(self.root/'.git/index').read_bytes()
        with patch('perseval_results.gitops.LFS_THRESHOLD',1),patch('perseval_results.gitops.git_process',side_effect=no_lfs):
            with self.assertRaisesRegex(ProtocolError,'Git LFS.*install --local'):
                self.publication(commit=True)
        self.assertEqual(before,files_under(self.root));self.assertEqual(index,(self.root/'.git/index').read_bytes())

    def test_large_metadata_is_never_lfs(self):
        # Lowering the policy threshold must still only select payloads, not JSON.
        repo=GitRepository(self.root)
        with patch('perseval_results.gitops.LFS_THRESHOLD',1):
            writes,paths=repo.plan_lfs({'registry/runs/example.json':b'large metadata','runs/example/metrics.json':b'metrics'})
        self.assertEqual((writes,paths),({},[]))

    def test_lfs_does_not_capture_registry_metadata(self):
        (self.root/'.gitattributes').write_text('registry/** filter=lfs\n')
        self.git('add','.gitattributes');self.git('commit','-q','-m','Bad metadata LFS fixture')
        before=files_under(self.root)
        with self.assertRaisesRegex(ProtocolError,'ordinary Git'):
            self.publication(commit=True)
        self.assertEqual(before,files_under(self.root))

    def test_lfs_requires_pre_push_delegation_without_modifying_hooks(self):
        if subprocess.run(['git','lfs','version'],capture_output=True).returncode:
            self.skipTest('Git LFS not installed')
        self.git('lfs','install','--local')
        hooks=self.root/'.git/custom-hooks';hooks.mkdir()
        self.git('config','core.hooksPath',str(hooks))
        hook=hooks/'pre-push';hook.write_text('#!/bin/sh\nexit 0\n');hook.chmod(0o755)
        before=files_under(self.root)
        with patch('perseval_results.gitops.LFS_THRESHOLD',1):
            with self.assertRaisesRegex(ProtocolError,'pre-push hook.*git lfs pre-push'):
                self.publication(commit=True)
        self.assertEqual(hook.read_text(),'#!/bin/sh\nexit 0\n')
        self.assertEqual(before,files_under(self.root))

    def test_real_lfs_commit_and_unhydrated_clone_build_without_objects(self):
        if subprocess.run(['git','lfs','version'],capture_output=True).returncode:
            self.skipTest('Git LFS not installed; missing-support rejection tested separately')
        self.git('lfs','install','--local')
        artifact='artifacts/selected-diagnostics.txt'
        payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
        payload[artifact]=b'explicit artifact\n'
        files=seal_payload(payload,self.contract,[{'path':artifact,'media_type':'text/plain','description':'explicit fixture'}])
        bundle=Path(self.temp.name)/'selected.tar.gz';bundle.write_bytes(pack_files(files))
        with patch('perseval_results.gitops.LFS_THRESHOLD',1):
            result=self.publication([bundle],commit=True)
        rid=self.docs['run']['run_id']
        paths=self.git('diff-tree','--no-commit-id','--name-only','-r',result.commit_sha).splitlines()
        self.assertIn('.gitattributes',paths)
        self.assertEqual(set(paths),set(result.plan['commit']['files']))
        for name in ('predictions.csv',artifact):
            self.assertEqual(self.git('show',f'HEAD:runs/{rid}/{name}').encode()+b'\n',pointer_bytes(files[name]))
        for name in ('manifest.json','run.json','metrics.json','environment.json','checksums.sha256'):
            self.assertFalse(self.git('show',f'HEAD:runs/{rid}/{name}').startswith('version https://git-lfs'))
        self.assertEqual(self.git('status','--porcelain'),'')
        # Clone using the local filesystem only. No LFS remote/object downloads.
        clone=Path(self.temp.name)/'unhydrated-clone'
        env={**self.env,'GIT_LFS_SKIP_SMUDGE':'1'}
        proc=subprocess.run(['git','clone','--no-local',str(self.root),str(clone)],capture_output=True,env=env)
        self.assertEqual(proc.returncode,0,proc.stderr)
        self.assertFalse(list((clone/'.git/lfs/objects').glob('*/*/*')))
        self.assertEqual((clone/f'runs/{rid}/predictions.csv').read_bytes(),pointer_bytes(files['predictions.csv']))
        with self.assertRaises(ProtocolError):validate_sources(clone,Contract(clone))
        data=build(clone,clone/'site-build');validate_preview(clone,clone/'site-build')
        self.assertEqual(data['entries'][0]['metrics']['global_f1']['mean'],0.75)
        self.assertFalse((clone/'site-build/api').exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--repo',str(clone),'validate','--metadata-only','--history']),0)
        # Pointer OIDs must still match the ordinary-Git checksum inventory.
        pred=clone/f'runs/{rid}/predictions.csv'
        pred.write_bytes(pred.read_bytes().replace(b'oid sha256:',b'oid sha256:0',1))
        with self.assertRaises(ProtocolError):build(clone,clone/'site-build')

    def test_literal_pointer_text_in_explicit_artifact_is_not_a_missing_object(self):
        artifact='artifacts/pointer-example.txt'
        payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
        payload[artifact]=pointer_bytes(b'literal example')
        files=seal_payload(payload,self.contract,[{'path':artifact,'media_type':'text/plain','description':'literal text'}])
        path=Path(self.temp.name)/'literal-pointer.tar.gz';path.write_bytes(pack_files(files))
        result=self.publication([path],commit=True)
        self.assertEqual(result.status,'committed')
        self.assertEqual(len(self.data()['runs']),1)

    def test_cli_success_displays_sha_and_only_remaining_git_push(self):
        args=['--repo',str(self.root),'publish-bundle',str(BUNDLE),'--name','Experiment',
              '--submitter','Group','--commit','--yes']
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(args),0)
        self.assertIn(self.git('rev-parse','HEAD'),output.getvalue())
        self.assertIn('Next: git push',output.getvalue())
        self.assertEqual(self.git('status','--porcelain'),'')
        head=self.git('rev-parse','HEAD')
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(args),0)
        self.assertIn('No new commit',output.getvalue());self.assertIn('Next: git push',output.getvalue())
        self.assertEqual(head,self.git('rev-parse','HEAD'))

    def test_registry_or_current_pointer_tampering_detected(self):
        result=self.publication()
        path=self.root/current_path(result.record['submission_id']);raw=path.read_bytes()
        pointer=read_json(raw);pointer['revision_digest']='sha256:'+'0'*64;path.write_bytes(canonical_json(pointer))
        with self.assertRaisesRegex(ProtocolError,'current pointer'):validate_sources(self.root,self.contract)
        path.write_bytes(raw)
        rid=self.docs['run']['run_id'];index_path=self.root/f'registry/runs/{rid}.json'
        index=read_json(index_path.read_bytes());index['scientific_digest']='sha256:'+'0'*64;index_path.write_bytes(canonical_json(index))
        with self.assertRaisesRegex(ProtocolError,'run registry'):validate_sources(self.root,self.contract)

    def test_git_history_rejects_immutable_edit_even_if_json_is_valid(self):
        result=self.publication(commit=True)
        path=self.root/revision_path(result.record);record=read_json(path.read_bytes());record['notes']='Changed old revision'
        path.write_bytes(canonical_json(record))
        self.git('add',str(path.relative_to(self.root)));self.git('commit','-q','-m','Deliberate invalid history fixture')
        with self.assertRaisesRegex(ProtocolError,'immutable source'):validate_git_history(self.root)

    def test_working_committed_immutable_file_change_is_rejected(self):
        result=self.publication(commit=True)
        path=self.root/revision_path(result.record);record=read_json(path.read_bytes());record['notes']='Changed'
        path.write_bytes(canonical_json(record))
        (self.root/current_path(record['submission_id'])).write_bytes(canonical_json(current_pointer(record)))
        with self.assertRaisesRegex(ProtocolError,'committed immutable'):
            self.publication(submission=record['submission_id'])

    def test_symlink_roots_and_special_source_files_rejected(self):
        outside=Path(self.temp.name)/'outside';outside.mkdir()
        (self.root/'registry').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ProtocolError,'symlink'):
            self.publication()
        self.assertFalse(list(outside.iterdir()))

    def test_symlink_website_source_is_rejected_by_both_build_paths(self):
        outside=Path(self.temp.name)/'outside.js';outside.write_text('private source')
        asset=self.root/'website/app.js';asset.unlink();asset.symlink_to(outside)
        with self.assertRaisesRegex(ProtocolError,'symlink'):
            self.publication()
        with self.assertRaisesRegex(ProtocolError,'symlink'):
            build(self.root,self.root/'site-build')
        self.assertEqual(outside.read_text(),'private source')
        self.assertEqual(managed_snapshot(self.root),{})

    def test_preview_symlink_cannot_write_outside_repository(self):
        outside=Path(self.temp.name)/'outside';outside.mkdir()
        (self.root/'site-build').symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ProtocolError,'symlink'):self.publication()
        self.assertFalse(list(outside.iterdir()));self.assertEqual(managed_snapshot(self.root),{})

    def test_preview_and_publisher_share_validated_site_data(self):
        self.publication();first=files_under(self.root/'site-build')
        self.data();self.assertEqual(first,files_under(self.root/'site-build'))
        data=read_json((self.root/'site-build/data/leaderboard.json').read_bytes())
        self.assertFalse((self.root/'site-build/api').exists())
        data['entries'][0]['rank']=7
        (self.root/'site-build/data/leaderboard.json').write_bytes(canonical_json(data))
        with self.assertRaisesRegex(ProtocolError,'manifest|authoritative records'):
            validate_preview(self.root,self.root/'site-build')

    def test_withdrawn_history_kept_and_cannot_accept_runs(self):
        first=self.publication();record={**first.record,'revision':2,'previous_revision_digest':digest(canonical_json(first.record)),
                                        'status':'withdrawn','notes':'Author withdrew this result'}
        (self.root/revision_path(record)).write_bytes(canonical_json(record))
        (self.root/current_path(record['submission_id'])).write_bytes(canonical_json(current_pointer(record)))
        self.assertEqual(self.data()['entries'],[])
        self.assertEqual(len(self.data()['history'][record['submission_id']]),2)
        with self.assertRaisesRegex(ProtocolError,'active measured'):
            self.publication(submission=record['submission_id'])

    def test_malformed_citation_is_a_helpful_error_without_writes(self):
        for citation in ('file:///tmp/paper','https://[','https://user:secret@example.org/paper'):
            with self.subTest(citation=citation),self.assertRaisesRegex(ProtocolError,'citation'):
                self.publication(citation=citation)
        self.assertEqual(managed_snapshot(self.root),{})

    def test_cli_nested_bundle_inspect_verify_and_dry_run(self):
        for command in (['bundle','inspect',str(BUNDLE)],['bundle','verify',str(BUNDLE)],['validate-bundle',str(BUNDLE)]):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--repo',str(self.root),*command]),0)
            self.assertIn(self.docs['run']['run_id'],output.getvalue())
        before=files_under(self.root)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--repo',str(self.root),'publish-bundle',str(BUNDLE),'--name','Experiment','--submitter','Group','--commit','--dry-run']),0)
        self.assertIn('dry-run',output.getvalue());self.assertEqual(before,files_under(self.root))

    def test_cli_requires_yes_and_supports_submission_update_without_metadata(self):
        args=['--repo',str(self.root),'publish-bundle',str(BUNDLE),'--name','Experiment','--submitter','Group']
        with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):main(args)
        self.assertIn('--yes',errors.getvalue())
        with redirect_stdout(io.StringIO()):self.assertEqual(main([*args,'--yes']),0)
        _,histories=validate_sources(self.root,self.contract);sid=next(iter(histories))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--repo',str(self.root),'publish-bundle',str(self.bundle()),'--submission',sid,'--yes']),0)
        self.assertEqual(len(validate_sources(self.root,self.contract)[1][sid]),2)


class ArchiveSecurity(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='perseval-archive-test-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.contract=Contract(ROOT)
        self.files,self.docs=validate_bundle(BUNDLE,self.contract)

    def test_fifo_input_is_rejected_without_blocking(self):
        path=self.root/'fifo';os.mkfifo(path)
        with self.assertRaisesRegex(ProtocolError,'regular file'):read_archive(path)

    def test_all_special_tar_member_types_rejected_before_extraction(self):
        for kind in (tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.CHRTYPE,tarfile.BLKTYPE,tarfile.FIFOTYPE,
                     tarfile.DIRTYPE,tarfile.GNUTYPE_SPARSE,tarfile.XHDTYPE,tarfile.GNUTYPE_LONGNAME):
            with self.subTest(kind=kind):
                info=tarfile.TarInfo('artifacts/special');info.type=kind;info.linkname='../outside'
                path=self.root/'bad.tar.gz'
                import gzip
                path.write_bytes(gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT)+bytes(1024),mtime=0))
                with self.assertRaises(ProtocolError):read_archive(path)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),['bad.tar.gz'])

    def test_excessive_count_and_expanded_size(self):
        import gzip
        info=tarfile.TarInfo('too-big');info.size=MAX_FILE+1
        path=self.root/'bad.tar.gz';path.write_bytes(gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT),mtime=0))
        with self.assertRaisesRegex(ProtocolError,'size limit'):read_archive(path)
        files={f'artifacts/f{i}':b'' for i in range(257)}
        path.write_bytes(pack_files(files))
        with self.assertRaisesRegex(ProtocolError,'too many'):read_archive(path)
        path.write_bytes(pack_files({'a':b'x'*10,'b':b'x'*10}))
        with patch('perseval_results.bundle.MAX_TOTAL',19):
            with self.assertRaisesRegex(ProtocolError,'size limit'):read_archive(path)
        with patch('perseval_results.bundle.MAX_ARCHIVE',1):
            with self.assertRaisesRegex(ProtocolError,'compressed'):read_archive(path)

    def test_noncanonical_paths_and_duplicate_members(self):
        import gzip
        for name in ('/absolute','../outside','artifacts/../escape','artifacts//bad','./bad','artifacts/\\bad','artifacts/.hidden'):
            with self.subTest(name=name):
                info=tarfile.TarInfo(name);path=self.root/'bad.tar.gz'
                path.write_bytes(gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT)+bytes(1024),mtime=0))
                with self.assertRaises(ProtocolError):read_archive(path)
        info=tarfile.TarInfo('duplicate');path.write_bytes(gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT)*2+bytes(1024),mtime=0))
        with self.assertRaisesRegex(ProtocolError,'duplicate'):read_archive(path)

    def test_missing_duplicate_blank_ids_and_invalid_csv(self):
        for data in (b'user_id,text_id,predictions\n,t1,0\n',b'user_id,text_id,predictions\nu,t,0\nu,t,1\n',
                     b'user_id,text_id,predictions\n ,t,0\n',b'user_id,text_id,predictions\n"unterminated',b'\xff'):
            with self.subTest(data=data):
                payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
                payload['predictions.csv']=data
                with self.assertRaises(ProtocolError):validate_files(seal_payload(payload,self.contract),self.contract)

    def test_recognized_full_report_cannot_disagree_with_compact_metrics(self):
        report={'format':'perseval-evaluation-v1','scoring_rule':'perseval.macro-f1.v1',
                'global':{'macro avg':{'f1-score':0.99,'support':2}},
                'user_average':{'macro avg':{'f1-score':0.5}},'text_average':{'macro avg':{'f1-score':0.5}},
                'task_score':{'kind':'macro-f1','positive_label':None,'value':0.75}}
        name='artifacts/perseval-evaluation.json'
        payload={k:v for k,v in self.files.items() if k not in ('manifest.json','checksums.sha256')}
        payload[name]=canonical_json(report)
        files=seal_payload(payload,self.contract,[{'path':name,'media_type':'application/json','description':'synthetic report'}])
        with self.assertRaisesRegex(ProtocolError,'inconsistent metrics'):validate_files(files,self.contract)
        report['global']['macro avg']['f1-score']=0.75;payload[name]=canonical_json(report)
        files=seal_payload(payload,self.contract,[{'path':name,'media_type':'application/json','description':'synthetic report'}])
        validate_files(files,self.contract)


if __name__=='__main__':
    unittest.main()
