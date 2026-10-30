"""Deterministically establish/check declarative specs and reported paper rows.

Input is the reviewed numeric-cell ledger, not executable historical code. No models,
datasets or PersEval package are loaded. Existing published files are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import uuid
from collections import Counter
from pathlib import Path

from perseval_results.protocol import Contract, ProtocolError, canonical_json, digest, read_json
from perseval_results.storage import revision_path, current_path, current_pointer
from perseval_results.historical import HISTORY, load_inventory
from recover_history import category

MODELS={
 'roberta':{'family':'encoder','identifier':'roberta-base','paper_identifier':'FacebookAI/roberta-base','revision':None,'notes':'Historical CLI alias for the paper encoder; exact weight revision is unavailable.'},
 'llama':{'family':'decoder','identifier':'meta-llama/Meta-Llama-3.1-8B-Instruct','paper_identifier':'meta-llama/Llama-3.1-8B','revision':None,'notes':'Paper describes instruction tuning but footnote names base Llama; historical CLI names Meta-Llama-3.1-8B-Instruct. Identifier mapping is documented, not a recovered checkpoint identity.'},
 'mixtral':{'family':'decoder','identifier':'mistralai/Mixtral-8x7B-Instruct-v0.1','paper_identifier':'Mixtral-8x7B-Instruct-v0.1','revision':None,'notes':'Paper instruction-tuned family and historical CLI identifier; exact weight revision is unavailable.'},
}
LABELS={'EPIC':'irony','BREXIT':'hs','MHS':'hateful','MD':'offensiveness','DICES':'Q2_harmful_content_overall'}
BASE_LIMITS=[
 'Paper rows are rounded reports, not measured run bundles. Run IDs, actual execution seeds, environments, checkpoint revisions, test fingerprints and standard deviations are unavailable.',
 'The recovered commit postdates the paper. Code defaults are evidence for a proposed reproduction, not proof of each published execution configuration.',
 'Reported paper variants never establish verified comparability with a new measured test fingerprint.',
]


def experiment_id(row):
    return f'paper2025-{row["model_key"]}-{row["dataset"].lower()}-{row["method"]}-{row["task"]}-{row["adaptation"]}-e{int(row["extended"])}-'+(
        'main' if row['scope']=='main' else 'ablation-'+'-'.join(t.lower() for t in row['trait_subset']))


def submission_id(row):
    # Reproducible ledger identity, never an execution identity.
    raw=hashlib.sha256(('perseval-emnlp-2025/'+row['row_id']).encode()).digest()[:16]
    return 'sub_'+str(uuid.UUID(bytes=raw,version=4))


def evidence(row,entries):
    linked={}
    def add(path,relationship,notes):
        if any(e['original_path']==path for e in entries):
            linked[path]={'path':HISTORY+'/snapshot/'+path,'relationship':relationship,'notes':notes}
    for path in ('README.md','main.py','perseval/config.py','perseval/data.py','perseval/model.py','perseval/evaluation.py','requirements.txt'):
        add(path,'implementation-only','Historical implementation/reference, not a per-execution log or proof that this version produced the paper row.')
    if row['model_key']!='roberta':add('perseval/prompts.py','implementation-only','Historical prompt templates; correspondence to a particular paper execution is not established.')
    if row['method']=='ipa':add('perseval/personalized_llms.py','implementation-only','Historical retrieval/profile preparation implementation.')
    if row['model_key']!='roberta' and row['task']=='named':
        add('qualitative_analysis/utils/ablation_study.py','implementation-only','Historical trait ensembling/ablation code, not an attested execution or a recovery of unavailable paper tie-breaking details.')
    dataset=row['dataset'].lower();family=('lamp_' if row['method']=='ipa' else '')+row['model_key']
    traitset=lambda values:{('generation' if dataset=='mhs' and v.lower()=='age' else v.lower()) for v in values}
    chosen=traitset(row['trait_subset'])
    for entry in entries:
        path=entry['original_path'];lower=path.lower();base=Path(lower).stem
        data_matches=(dataset=='md' and lower.startswith('data/md-agreement_dataset/')) or (dataset=='dices' and lower.startswith('data/diverse_safety_'))
        if data_matches:add(path,'shared-data','Dataset/cache association follows the historical path; this does not establish dataset revision or test coverage.')
        if dataset=='epic' and row['method']=='ipa' and lower.startswith('data_lamp/epic'):
            add(path,'shared-data','EPIC retrieval inputs/outputs. Missing profile ranks and exact per-execution profile provenance remain unresolved.')
        if row['model_key']=='roberta':continue
        if not re.search(r'(?:^|[/_])'+re.escape(dataset)+r'(?:_|$|\.)',lower):continue
        family_match=('_'+family+'_') in ('_'+base+'_')
        if row['method']!='ipa' and '_lamp_' in ('_'+base+'_'):family_match=False
        if lower.startswith('predictions_'+family+'/'):
            if '/ablation_files/' in lower:
                tail=base.split('_'+dataset+'_')[-1]
                if row['task']=='named' and traitset(tail.split('_'))==chosen:
                    add(path,'filename-correspondence','Model/dataset/trait subset match the filename; execution identity and split equivalence are unverified.')
            elif row['scope']=='main':
                if row['task']=='baseline' and '_false_' in lower and lower.endswith('_zero.csv'):
                    add(path,'filename-correspondence','Raw zero-shot filename matches dataset/model. The train flag is not treated as proof of adaptation.')
                elif row['task']=='named' and ((row['method']=='ipa' and lower.endswith('_true.csv')) or (row['method']=='perspective' and '_true_' in lower)):
                    add(path,'filename-correspondence','Trait-specific output filename match; final ensembling and exact prompts are not execution-attested.')
                elif row['task']=='unnamed' and row['method']=='ipa' and lower.endswith('_false.csv'):
                    add(path,'filename-correspondence','Unnamed IPA edited output filename match; not promoted to a measured run.')
        if lower.startswith('results/') and family_match and row['scope']=='main':
            if (row['task'] in ('baseline','unnamed') and base.endswith('_false')) or (dataset=='brexit' and row['task']=='named' and base.endswith('_group')):
                add(path,'filename-correspondence','Rounded historical report corresponding by filename; values need not be identical to the published table and do not replace it.')
        if lower.startswith('qualitative_analysis/results_ablation/'+family+'/') and row['task']=='named':
            tail=base.split('_'+family+'_')[-1]
            if traitset(tail.split('_'))==chosen:add(path,'filename-correspondence','Ablation report matches model/dataset/traits. This is intermediate evidence, not an independently verified execution.')
        if lower.startswith(('qualitative_analysis/results_jsd/','qualitative_analysis/results_label_change/')) and family_match and row['task']=='named':
            add(path,'supporting-analysis','Non-F1 demographic analysis; never used as a compact leaderboard score.')
    return [linked[key] for key in sorted(linked)]


def artifacts(root):
    contract=Contract(root)
    source=read_json((root/'papers/perseval-2025/source.json').read_bytes())
    table_bytes=(root/'papers/perseval-2025/tables.json').read_bytes();tables=read_json(table_bytes)
    contract.validate('paper-source',source,kind='catalog');contract.validate('paper-tables',tables,kind='catalog')
    if digest(table_bytes)!=source['tables_sha256']:raise ProtocolError('paper table ledger digest changed; review transcription before generation')
    _,inventory=load_inventory(root/HISTORY,contract);entries=inventory['entries']
    output={};ids=[];reverse={e['original_path']:[] for e in entries}
    for row in tables['rows']:
        eid=experiment_id(row);sid=submission_id(row);ids.append(eid)
        limits=list(BASE_LIMITS)
        if row['dataset']=='BREXIT':limits.append('BREXIT Table 2 calls the positive class Offensiveness, but Appendix B, prompt examples and historical config use hate speech/hs. This record maps to hs with that conflict disclosed.')
        if row['model_key']=='llama':limits.append(MODELS['llama']['notes'])
        if row['method']=='ipa':limits.append('Section 6.2 says Named Adaptation-T in one sentence, while Tables 5/6 and historical LaMP require test/inference adaptation. This catalog maps IPA to test and preserves the inconsistency.')
        if row['model_key']!='roberta':limits.append('Decoder tables omit extended training flags; false is taken from matching historical filenames, not independently attested by the table. Text prompts can still use adaptation examples.')
        if row['model_key']=='roberta':limits.append('No encoder prediction/report artifacts were committed in this tree. Table 9 parameters are reported, but complete per-run training configuration is unrecoverable.')
        if row['model_key']=='roberta' and row['task']=='baseline':limits.append('Paper baseline has no user/trait tokens; historical main.py does not expose that no-token baseline and the historical encoder always injects user IDs. A faithful new baseline needs explicit text-only input construction.')
        if row['scope']=='ablation':limits.append('Ablation table reports global precision/recall/F1 only; user/text F1 are unavailable. Trait subset names are retained as printed, including MHS Age versus Generation in the main tables.')
        if row['table']=='3' and row['dataset']=='EPIC' and row['extended'] and row['adaptation']=='train' and row['task']=='named':limits.append('Printed text F1 .594 is retained exactly; it is not corrected to resemble adjacent rows.')
        split={'user_based_split_percentage':0.2}
        split.update({'text_based_split_percentage_train':0.7,'text_based_split_percentage_dev':0.05} if row['dataset']=='BREXIT' else {'text_based_split_percentage':0.05})
        paper_hyper={'eval_strategy':'epoch','greater_is_better':False,'learning_rate':5e-6,'load_best_model_at_end':True,'metric_for_best_model':'eval_loss','num_train_epochs':5,'per_device_eval_batch_size':32,'per_device_train_batch_size':16} if row['model_key']=='roberta' else ({'retrieval_examples_per_user':5,'retriever':'facebook/contriever'} if row['method']=='ipa' else {})
        if row['model_key']!='roberta' and row['task']=='named':
            paper_hyper.update(trait_ensemble='majority-vote',tie_breaking=None)
        historical_hyper={'max_length':512,'padding':'max_length','truncation':True,'focal_loss_gamma':5.0,'focal_loss_alpha':0.2,'class_weights':'balanced when computable'} if row['model_key']=='roberta' else {'do_sample':False,'max_new_tokens':100,'torch_dtype':'float16','device_map':'auto'}
        if row['model_key']=='llama' and row['method']!='ipa':
            historical_hyper.update(torch_dtype='bfloat16',device='cuda')
            historical_hyper.pop('device_map')
        if row['method']=='ipa':historical_hyper.update(retrieval_batch_size=16,max_length=1024,num_profiles=5)
        links=evidence(row,entries)
        for link in links:reverse[link['path'].removeprefix(HISTORY+'/snapshot/')].append(eid)
        spec={'schema_version':'1.0','experiment_id':eid,'scope':row['scope'],'paper_row_id':row['row_id'],'model':MODELS[row['model_key']],
              'dataset':{'name':row['dataset'],'label':LABELS[row['dataset']],'revision':None,'fingerprint':None},
              **{key:row[key] for key in ('task','adaptation','extended','method','representation','trait_subset')},
              'split':{'protocol':'perseval-paper-2025-historical','parameters':split,'seed_default':42,'historical_test_fingerprint':None,
                       'provenance':[HISTORY+'/snapshot/perseval/config.py',HISTORY+'/snapshot/perseval/data.py','papers/perseval-2025/source.json']},
              'hyperparameters':{'paper':paper_hyper,'historical_implementation':historical_hyper,'provenance':['Paper Appendix D Table 9 (encoder), sections 6.1/6.2; code defaults are a separate evidence source.',HISTORY+'/snapshot/perseval/config.py',HISTORY+'/snapshot/perseval/model.py']},
              'seed_policy':{'historical_execution_seed':None,'historical_config_default':42,'new_execution_seeds':'explicit-user-selection','aggregation':'Record the actual split seed/test fingerprint and each actual execution seed. Export every explicitly included execution; arithmetic mean/sample SD, no best-seed selection.'},
              'optional_dependencies':['torch','transformers','datasets','scikit-learn','accelerate','pandas','numpy']+(['sentence-transformers','rank-bm25'] if row['method']=='ipa' else []),
              'reporting':{'wandb_required':False,'recommended_trainer_report_to':[],'notes':'WandB is not needed for catalog/list/build/validation. New reproduction training should set report_to=[]; historical config says None, but historical train() omits that setting, so installed Trainer defaults may differ.'},
              'expected_hardware':{'historical_reference':'README setup: Ubuntu 22.04.3, Python 3.10.12, AMD EPYC-Rome 24 cores, 503 GiB RAM, 4 NVIDIA A40 GPUs (46 GiB each).','minimum_gpu_memory_gib':None,'notes':'Reference machine, not a measured minimum or per-run allocation. Historical README says multi-GPU is unsupported while decoder code uses device_map=auto. Provision/model placement must be resolved separately; no experiment is run by this package.'},
              'historical_evidence':links,'reported_submissions':[sid],'limitations':limits,
              'execution':{'kind':'declarative-only','automatic_transfer':False,'explicit_result_export_required':True}}
        trait_label=' / '.join(row['trait_subset']) if row['scope']=='ablation' else row['representation']
        record={'schema_version':'1.0','submission_id':sid,'revision':1,'previous_revision_digest':None,'status':'published','superseded_by':None,
                'source_type':'reported','name':f'Paper {row["model_key"]} {row["dataset"]}: {row["method"]}, {row["task"]}/{row["adaptation"]}, extended={str(row["extended"]).lower()} ({trait_label})',
                'submitter':'Lo et al. (EMNLP 2025)','notes':'Published values transcribed without recomputation; unavailable fields are not inferred. '+('Appendix trait-ablation row.' if row['scope']=='ablation' else 'Main experiment table row.'),
                'citation':source['citation']+' '+source['url'],'published_at':source['transcribed_at'],
                'provenance':{'source':f'Paper Table {row["table"]}, printed p. {row["printed_page"]}, PDF p. {row["pdf_page"]}, row {row["row_id"]}; source PDF {source["pdf_sha256"]}. Timestamp is ledger transcription, not experiment execution.','limitations':limits},
                'model':MODELS[row['model_key']]['identifier'],
                'group':{'benchmark_version':'perseval-emnlp-2025','dataset':row['dataset'],'label':LABELS[row['dataset']],'task':row['task'],'adaptation':row['adaptation'],'extended':row['extended'],'split_protocol':'perseval-paper-2025-historical','scoring_rule':row['scoring_rule']},
                'paper_variant':{'id':f'perseval-emnlp2025-{row["dataset"].lower()}-e{int(row["extended"])}','description':'Documented paper benchmark, annotation-level evaluation with disjoint annotators/users, adaptation as stated in tables, and extended training as recorded; fingerprints unavailable.','mapping_evidence':f'Paper sections 3/6/7 and Table {row["table"]}; all discrepancies are disclosed in provenance and the associated catalog spec. No equivalence to modern measured splits is asserted.','table':row['table'],'page':row['printed_page']},
                'values':row['compact'],
                'experiment':{'catalog_id':eid,'model_family':MODELS[row['model_key']]['family'],'model_identifier':MODELS[row['model_key']]['identifier'],'model_revision':None,'method':row['method'],'representation':row['representation'],'scope':row['scope'],'trait_subset':row['trait_subset'],'trait_f1':row['trait_f1'],'positive_class':None if row['dataset']=='DICES' else '1','additional_reported_metrics':row['additional_metrics'],
                              'rounding':{'decimal_places':3,'scale':'0-1','notes':'Three decimal places as printed; dashes mean unavailable, not zero. No unrounded scores or dispersion are inferred.'},
                              'source_document':{'id':source['paper_id'],'sha256':source['pdf_sha256'],'pdf_page':row['pdf_page'],'printed_page':row['printed_page'],'table':row['table'],'row':row['row_id']}}}
        contract.validate('experiment',spec,kind='catalog');contract.validate('reported',record,record=True)
        output[f'catalog/experiments/{eid}.json']=canonical_json(spec)
        output[revision_path(record)]=canonical_json(record)
        output[current_path(sid)]=canonical_json(current_pointer(record))
    output['catalog/index.json']=canonical_json({'schema_version':'1.0','scope':'scholarship-task-2.1','experiments':sorted(ids),'paper_source':'papers/perseval-2025/source.json','paper_tables':'papers/perseval-2025/tables.json','historical_inventory':HISTORY+'/inventory.json'})
    # Inventory relationships describe evidence association, never execution identity.
    for entry in entries:
        entry['artifact_category']=category(entry['original_path'])
        entry['associated_experiments']=sorted(reverse[entry['original_path']])
        entry['association_basis']='Catalog evidence links: relationship-qualified implementation/data/filename associations; no execution identity is claimed.' if entry['associated_experiments'] else None
    inv=canonical_json(inventory);manifest=read_json((root/HISTORY/'manifest.json').read_bytes())
    manifest.update(inventory_sha256=digest(inv),categories=dict(sorted(Counter(e['artifact_category'] for e in entries).items())))
    output[HISTORY+'/inventory.json']=inv;output[HISTORY+'/manifest.json']=canonical_json(manifest)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--check',action='store_true');args=parser.parse_args();root=args.repo.resolve()
    for name,data in artifacts(root).items():
        path=root/name
        if args.check:
            if not path.is_file() or path.read_bytes()!=data:raise ProtocolError(f'catalog/generated record differs: {name}')
        elif path.exists() and path.read_bytes()!=data and name not in (HISTORY+'/inventory.json',HISTORY+'/manifest.json'):
            raise ProtocolError(f'refusing to overwrite an established paper/catalog record: {name}')
        elif not path.exists() or path.read_bytes()!=data:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    print('Validated deterministic catalog/record generation.' if args.check else 'Established catalog and reported paper records; historical snapshot bytes were untouched.')


if __name__=='__main__':main()
