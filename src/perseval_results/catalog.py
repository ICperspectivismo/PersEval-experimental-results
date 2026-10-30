"""Offline declarative experiments and provenance-checked reported paper records."""
from __future__ import annotations

from pathlib import Path
from collections import Counter

from .historical import HISTORY, REVISION, load_inventory, relative_path
from .protocol import Contract, ProtocolError, digest, read_json
from .storage import tree_files

TRAIT_COLUMNS={'EPIC':['Gender','Nationality','Generation'],'BREXIT':['Group'],
               'MHS':['Gender','Generation','Education','Income','Ideology'],
               'MD':[],'DICES':['Gender','Generation','Education','Race']}


def validate_catalog(root, contract=None, histories=None):
    root=Path(root);contract=contract or Contract(root);histories=histories or {}
    directory=root/'catalog'
    if not directory.exists() and not directory.is_symlink():
        if any(record.get('experiment') for history in histories.values() for record in history):
            raise ProtocolError('reported experiment metadata requires its catalog and paper source ledger')
        return {}
    files=tree_files(directory)
    if 'index.json' not in files:raise ProtocolError('catalog index is missing')
    index=read_json(files['index.json']);contract.validate('index',index,kind='catalog')
    ids=index['experiments']
    if ids!=sorted(set(ids)):raise ProtocolError('catalog IDs must be unique and sorted')
    if set(files)!={'index.json'}|{f'experiments/{eid}.json' for eid in ids}:
        raise ProtocolError('catalog file set differs from index')
    tree_files(root/'papers')
    source=read_json((root/index['paper_source']).read_bytes())
    raw=(root/index['paper_tables']).read_bytes();tables=read_json(raw)
    contract.validate('paper-source',source,kind='catalog');contract.validate('paper-tables',tables,kind='catalog')
    if digest(raw)!=source['tables_sha256']:raise ProtocolError('paper table ledger differs from source digest')
    rows={row['row_id']:row for row in tables['rows']}
    if len(rows)!=len(tables['rows']):raise ProtocolError('duplicate paper table row')
    if Counter(row['table'] for row in rows.values())!={'3':28,'4':8,'5':28,'6':8,'10':28,'11':62,'12':62,'13':60}:
        raise ProtocolError('paper F1 table coverage is incomplete or has unexpected rows')
    # Numeric cell provenance must reproduce the exact printed compact/trait values.
    for row in rows.values():
        values=[float(cell['value']) for cell in row['printed_numeric_cells']]
        wanted=[row['compact']['global_f1'],row['compact']['user_f1'],row['compact']['text_f1'],*row['trait_f1'].values()] if row['scope']=='main' else [row['additional_metrics'].get('precision'),row['additional_metrics'].get('recall'),row['compact']['global_f1']]
        if row['scope']=='main':
            columns=TRAIT_COLUMNS[row['dataset']] if row['task']=='named' else []
            if (set(row['trait_f1'])!=set(columns) or values[:3]!=wanted[:3]
                    or values[3:]!=[row['trait_f1'][key] for key in columns]):
                raise ProtocolError(f'printed paper cells differ from score values: {row["row_id"]}')
        elif values!=wanted:raise ProtocolError(f'printed ablation cells differ from values: {row["row_id"]}')
        if (row['dataset']=='DICES')!=(row['scoring_rule']=='perseval.macro-f1.v1'):
            raise ProtocolError('paper binary/positive-class and multiclass/macro scoring rules must remain distinct')
    _,inventory=load_inventory(root/HISTORY,contract)
    if inventory['source']['commit']!=REVISION:
        raise ProtocolError('catalog historical evidence must refer to the pinned PersEval commit')
    evidence={HISTORY+'/'+entry['path']:entry for entry in inventory['entries']}
    reverse={path:set() for path in evidence};specs={};used_rows=set();used_reports=set()
    for eid in ids:
        spec=read_json(files[f'experiments/{eid}.json']);contract.validate('experiment',spec,kind='catalog')
        if spec['experiment_id']!=eid:raise ProtocolError('catalog identity/path mismatch')
        row=rows.get(spec['paper_row_id'])
        if row is None or spec['paper_row_id'] in used_rows:raise ProtocolError('catalog must map each paper row exactly once')
        used_rows.add(spec['paper_row_id'])
        for key in ('task','adaptation','extended','method','representation','trait_subset','scope'):
            if spec[key]!=row[key]:raise ProtocolError(f'catalog setting differs from paper row: {eid}/{key}')
        if spec['dataset']['name']!=row['dataset']:raise ProtocolError('catalog dataset differs from paper')
        if spec['task']=='unnamed' and spec['adaptation']=='none':raise ProtocolError('unnamed task requires adaptation')
        if spec['dataset']['name']=='MD' and spec['task']=='named':raise ProtocolError('MD has no named-trait setting in this paper')
        for link in spec['historical_evidence']:
            relative_path(link['path'])
            if link['path'] not in evidence:raise ProtocolError(f'catalog references uninventoried evidence: {link["path"]}')
            reverse[link['path']].add(eid)
        for sid in spec['reported_submissions']:
            if sid in used_reports or sid not in histories:raise ProtocolError('catalog references duplicate or missing reported submission')
            used_reports.add(sid);record=histories[sid][0]
            if record['source_type']!='reported':raise ProtocolError('catalog paper reference must be a reported submission')
            detail=record.get('experiment',{})
            for key in ('method','representation','scope','trait_subset','trait_f1'):
                if detail.get(key)!=row[key]:raise ProtocolError(f'reported experiment differs from paper: {sid}/{key}')
            if (detail.get('catalog_id')!=eid or detail.get('model_identifier')!=spec['model']['identifier']
                    or detail.get('model_family')!=spec['model']['family'] or record['model']!=spec['model']['identifier']
                    or record['values']!=row['compact'] or detail.get('additional_reported_metrics')!=row['additional_metrics']):
                raise ProtocolError(f'paper submission/catalog/value mismatch: {sid}')
            if detail.get('positive_class')!=(None if row['dataset']=='DICES' else '1'):
                raise ProtocolError('reported positive-class identity differs from paper mapping')
            locator=detail['source_document']
            if (locator!={'id':source['paper_id'],'sha256':source['pdf_sha256'],'pdf_page':row['pdf_page'],'printed_page':row['printed_page'],'table':row['table'],'row':row['row_id']}
                    or record['paper_variant']['table']!=row['table'] or record['paper_variant']['page']!=row['printed_page']):
                raise ProtocolError('reported paper citation/table/page locator differs from source ledger')
            group=record['group']
            if (group['dataset']!=row['dataset'] or group['label']!=spec['dataset']['label']
                    or group['scoring_rule']!=row['scoring_rule']
                    or any(group[key]!=row[key] for key in ('task','adaptation','extended'))):
                raise ProtocolError('reported comparison group differs from documented paper variant')
        specs[eid]=spec
    if used_rows!=set(rows):raise ProtocolError('paper table rows missing from catalog')
    for path,entry in evidence.items():
        if set(entry['associated_experiments'])!=reverse[path]:raise ProtocolError(f'historical inventory/catalog backlinks disagree: {path}')
    return specs
