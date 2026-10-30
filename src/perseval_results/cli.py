from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build import build, validate_preview
from .bundle import validate_bundle
from .protocol import Contract, ProtocolError
from .publish import bundle_summary, publish, publisher_lock, validate_git_history
from .records import validate_sources
from .catalog import validate_catalog
from .historical import HISTORY, audit


def main(argv=None):
    parser = argparse.ArgumentParser(prog='perseval-results',description='Offline result-bundle publication in a results checkout')
    parser.add_argument('--repo',type=Path,default=Path.cwd(),help='results checkout root (default: current directory)')
    commands = parser.add_subparsers(dest='command',required=True)
    bundles = commands.add_parser('bundle',help='read-only bundle inspection and verification')
    actions = bundles.add_subparsers(dest='bundle_action',required=True)
    for action in ('inspect','verify'):
        actions.add_parser(action).add_argument('path',type=Path)
    legacy = commands.add_parser('validate-bundle',help='compatibility alias for bundle verify')
    legacy.add_argument('path',type=Path)
    pub = commands.add_parser('publish-bundle',help='validate, review, import, publish and optionally commit in one command')
    pub.add_argument('paths',type=Path,nargs='+')
    pub.add_argument('--name')
    pub.add_argument('--submitter')
    pub.add_argument('--notes')
    pub.add_argument('--citation',help='citation HTTP(S) URL')
    pub.add_argument('--submission','--submission-id',dest='submission',help='explicit existing measured submission ID')
    pub.add_argument('--commit',action='store_true',help='recommended: require clean checkout and commit exact planned persistent files; then run git push yourself')
    pub.add_argument('--yes',action='store_true',help='confirm this publication; required for noninteractive mutation')
    pub.add_argument('--dry-run',action='store_true',help='validate, build in temporary storage and show plan with no checkout writes')
    pub.add_argument('--reuse-existing-runs',action='store_true',help='intentionally reuse imported runs in a new independent submission')
    pub.add_argument('--allow-duplicate-content',action='store_true',help='explicitly attest genuine executions despite identical scientific payloads; disclose the limitation')
    commands.add_parser('build',help='compatibility alias for site build')
    site = commands.add_parser('site',help='build, check or serve the derived local leaderboard offline')
    site_actions = site.add_subparsers(dest='site_action',required=True)
    for action in ('build','serve','check'):
        action_parser = site_actions.add_parser(action)
        action_parser.add_argument('--output',type=Path,default=Path('site-build'),help='site-build/ (default) or legacy preview/ inside this checkout')
        if action=='build': action_parser.add_argument('--source-revision',help='explicit source identifier; CI supplies its immutable commit SHA')
        if action=='serve': action_parser.add_argument('--port',type=int,default=8000)

    catalog = commands.add_parser('catalog',help='offline declarative Task 2.1 experiments; never execute models')
    catalog_actions = catalog.add_subparsers(dest='catalog_action',required=True)
    listing = catalog_actions.add_parser('list')
    listing.add_argument('--dataset')
    listing.add_argument('--model',help='model identifier substring')
    listing.add_argument('--method',choices=['focal-loss','base-zero','perspective','ipa'])
    listing.add_argument('--scope',choices=['main','ablation'])
    listing.add_argument('--json',action='store_true')
    catalog_actions.add_parser('show').add_argument('experiment_id')
    catalog_actions.add_parser('validate')
    history = commands.add_parser('historical',help='audit preserved historical evidence without loading datasets')
    history_actions = history.add_subparsers(dest='historical_action',required=True)
    history_actions.add_parser('audit').add_argument('--metadata-only',action='store_true',help='accept unhydrated LFS objects only when pointer OID/size match the inventory')
    validate = commands.add_parser('validate')
    validate.add_argument('--history',action='store_true',help='also check append-only source records in local Git history')
    validate.add_argument('--metadata-only',action='store_true',help='validate committed metadata and LFS pointer/checksum identities without fetching payload objects')
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    result = None
    try:
        if args.command in ('bundle','validate-bundle'):
            files,docs = validate_bundle(args.path,Contract(root))
            if args.command=='bundle' and args.bundle_action=='inspect':
                print(json.dumps(bundle_summary(files,docs),indent=2,ensure_ascii=False,allow_nan=False))
            else:
                print(f'Valid bundle: {docs["run"]["run_id"]}; payload {docs["manifest"]["payload_digest"]}')
        elif args.command=='publish-bundle':
            result = publish(root,args.paths,name=args.name,submitter=args.submitter,notes=args.notes,citation=args.citation,
                             submission=args.submission,commit=args.commit,yes=args.yes,dry_run=args.dry_run,
                             reuse_existing_runs=args.reuse_existing_runs,allow_duplicate_content=args.allow_duplicate_content)
            record = result.record
            print(f'{record["submission_id"]} revision {record["revision"]}: {result.status}')
            if result.commit_sha:
                print(f'Durable commit: {result.commit_sha}')
                print('Next: git push')
            elif args.commit and result.status=='unchanged':
                print(f'No new commit; current HEAD: {result.plan["git_preflight"]["head"]}')
                print('Next: git push')
            elif result.status=='published':
                print('Validated publication files are available for manual Git review and commit. Use --commit from a clean checkout for the recommended publish, then git push workflow.')
            for warning in result.warnings:
                print(f'Warning: {warning}')
        elif args.command=='historical':
            print(json.dumps(audit(root/HISTORY,Contract(root),metadata_only=args.metadata_only),indent=2))
        elif args.command=='catalog':
            _,histories=validate_sources(root,Contract(root),metadata_only=True)
            specs=validate_catalog(root,Contract(root),histories)
            if args.catalog_action=='show':
                if args.experiment_id not in specs:
                    raise ProtocolError('unknown catalog experiment ID; use catalog list to see available specifications')
                print(json.dumps(specs[args.experiment_id],indent=2,ensure_ascii=False,allow_nan=False))
            elif args.catalog_action=='validate':
                print(f'Validated {len(specs)} declarative experiments and associated reported paper records')
            else:
                selected=[spec for spec in specs.values() if
                          (not args.dataset or spec['dataset']['name'].casefold()==args.dataset.casefold()) and
                          (not args.model or args.model.casefold() in spec['model']['identifier'].casefold()) and
                          (not args.method or spec['method']==args.method) and
                          (not args.scope or spec['scope']==args.scope)]
                if args.json:
                    print(json.dumps(selected,indent=2,ensure_ascii=False,allow_nan=False))
                else:
                    for spec in selected:
                        print(f'{spec["experiment_id"]}\t{spec["dataset"]["name"]}/{spec["dataset"]["label"]}\t{spec["model"]["identifier"]}\t{spec["task"]}/{spec["adaptation"]}\textended={str(spec["extended"]).lower()}')
                    print(f'{len(selected)} specifications; declarative only, no experiment executed.')
        elif args.command in ('build','site'):
            action = 'build' if args.command=='build' else args.site_action
            output = root/'site-build' if args.command=='build' else args.output
            if not output.is_absolute(): output = root/output
            if action=='build':
                Contract(root)
                with publisher_lock(root):
                    data = build(root,output,source_revision=getattr(args,'source_revision',None))
                print(f'Built and validated {len(data["entries"])} entries in {output}')
            elif action=='check':
                metadata = validate_preview(root,output)
                print(f'Valid site: {metadata["build_id"]}; source {metadata["source_revision"] or metadata["source_digest"]}')
            else:
                if not 1 <= args.port <= 65535:
                    raise ProtocolError('port must be between 1 and 65535')
                from .serve import serve
                serve(root,output,args.port)
        else:
            if (root/'.perseval-results.lock').exists() or (root/'.perseval-results.lock').is_symlink():
                raise ProtocolError('publisher is active; retry validation after publication completes')
            runs,histories = validate_sources(root,Contract(root),metadata_only=args.metadata_only)
            if args.history:
                validate_git_history(root)
            scope = 'metadata and pointer identities (payload CSV/artifacts may be unavailable)' if args.metadata_only else 'complete payloads'
            print(f'Validated {len(runs)} immutable runs and {len(histories)} submission histories: {scope}')
    except (ProtocolError,OSError) as exc:
        if result is not None and result.commit_sha:
            parser.exit(1,f'Commit {result.commit_sha} is durable; later output failed: {exc}. No rollback was attempted.\n')
        parser.exit(1,f'error: {exc}\n')
    return 0
