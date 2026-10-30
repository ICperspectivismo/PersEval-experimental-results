import json,re,hashlib
from pathlib import Path
from datetime import datetime,timezone
import argparse
parser=argparse.ArgumentParser(description="Optional maintainer transcription from local PDFKit cells; never executes experiments.")
parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
parser.add_argument('--cells',type=Path,required=True)
parser.add_argument('--paper',type=Path,required=True)
parser.add_argument('--tasks',type=Path,required=True)
args=parser.parse_args()
root=args.repo.resolve()
traits={'EPIC':['Gender','Nationality','Generation'],'BREXIT':['Group'],'MHS':['Gender','Generation','Education','Income','Ideology'],'MD':[],'DICES':['Gender','Generation','Education','Race']}
rows=[]
def numeric_rows(page):
 groups={}
 for w in json.loads((args.cells/f'perseval-paper-words-{page}.json').read_text()):
  if re.fullmatch(r'\.\d{3}',w['text']):groups.setdefault(round(w['y'],1),[]).append(w)
 return [(y,sorted(ws,key=lambda w:w['x'])) for y,ws in sorted(groups.items(),reverse=True)]
def add(ws,y,page,table,model,dataset,task,adaptation,extended,method,representation,subset,scope):
 vals=[float(w['text']) for w in ws]
 row={'row_id':f't{table}-r{sum(r["table"]==str(table) for r in rows)+1:03d}',
 'table':str(table),'pdf_page':page,'printed_page':str(22333+page),'scope':scope,'model_key':model,
 'dataset':dataset,'task':task,'adaptation':adaptation,'extended':extended,'method':method,'representation':representation,
 'trait_subset':subset,'scoring_rule':'perseval.macro-f1.v1' if dataset=='DICES' else 'perseval.positive-class-f1.v1',
 'compact':{'global_f1':vals[0] if scope=='main' else vals[2],'user_f1':vals[1] if scope=='main' else None,'text_f1':vals[2] if scope=='main' else None},
 'trait_f1':dict(zip(traits[dataset],vals[3:])) if scope=='main' else {},
 'additional_metrics':{'precision':vals[0],'recall':vals[1]} if scope=='ablation' else {},
 'printed_numeric_cells':[{'value':w['text'],'x':round(w['x'],3),'y':round(w['y'],3)} for w in ws]}
 assert len(vals)==(3+(len(traits[dataset]) if task=='named' else 0) if scope=='main' else 3),row
 rows.append(row)
# Visual checks of pages 7/8 establish the row ordering; positions preserve cell provenance.
p7=numeric_rows(7);assert len(p7)==36
for i,(y,ws) in enumerate(p7[:28]):
 extended=i>=14;k=i%14
 dataset=['EPIC','BREXIT','MHS','MD'][min(k//4,3)]
 setting=k%4 if k<12 else k-12
 task,adapt,rep=[('baseline','none','text-only'),('named','none','user-id-and-traits'),('named','train','user-id-and-traits'),('unnamed','train','user-id')][setting if k<12 else (0 if setting==0 else 3)]
 add(ws,y,7,3,'roberta',dataset,task,adapt,extended,'focal-loss',rep,traits[dataset] if task=='named' else [],'main')
for i,(y,ws) in enumerate(p7[28:]):
 task,adapt,rep=[('baseline','none','text-only'),('named','none','user-id-and-traits'),('named','train','user-id-and-traits'),('unnamed','train','user-id')][i%4]
 add(ws,y,7,4,'roberta','DICES',task,adapt,i>=4,'focal-loss',rep,traits['DICES'] if task=='named' else [],'main')
p8=numeric_rows(8);assert len(p8)==36
for i,(y,ws) in enumerate(p8[:28]):
 model='llama' if i<14 else 'mixtral';k=i%14;dataset=['EPIC','BREXIT','MHS','MD'][min(k//4,3)]
 setting=k%4 if k<12 else (0 if k==12 else 3)
 task,adapt,method,rep=[('baseline','none','base-zero','text-only'),('named','none','perspective','trait-prompts'),('named','test','ipa','user-history-and-traits'),('unnamed','test','ipa','user-history')][setting]
 add(ws,y,8,5,model,dataset,task,adapt,False,method,rep,traits[dataset] if task=='named' else [],'main')
for i,(y,ws) in enumerate(p8[28:]):
 task,adapt,method,rep=[('baseline','none','base-zero','text-only'),('named','none','perspective','trait-prompts'),('named','test','ipa','user-history-and-traits'),('unnamed','test','ipa','user-history')][i%4]
 add(ws,y,8,6,'llama' if i<4 else 'mixtral','DICES',task,adapt,False,method,rep,traits['DICES'] if task=='named' else [],'main')
epic=['Gender','Generation','Nationality','Generation-Nationality','Gender-Generation','Gender-Nationality','Gender-Generation-Nationality']
mhs=['Age','Education','Income','Gender','Ideology','Gender-Income','Gender-Ideology','Gender-Education','Gender-Age','Age-Education','Age-Ideology','Age-Income','Education-Income','Education-Ideology','Income-Ideology','Gender-Age-Education','Gender-Age-Income','Gender-Age-Ideology','Gender-Education-Income','Gender-Education-Ideology','Gender-Income-Ideology','Age-Education-Income','Age-Education-Ideology','Age-Income-Ideology','Education-Income-Ideology','Gender-Age-Education-Income','Gender-Age-Education-Ideology','Gender-Age-Income-Ideology','Gender-Education-Income-Ideology','Age-Education-Income-Ideology','Gender-Age-Education-Income-Ideology']
dices=['Gender','Generation','Education','Race','Gender-Generation','Gender-Education','Gender-Race','Generation-Education','Generation-Race','Education-Race','Gender-Generation-Education','Gender-Generation-Race','Gender-Education-Race','Generation-Education-Race','Gender-Generation-Education-Race']
for page,table,dataset,combinations,settings in [(19,10,'EPIC',epic,[('llama','perspective'),('mixtral','perspective'),('llama','ipa'),('mixtral','ipa')]),(20,11,'MHS',mhs,[('llama','perspective'),('mixtral','perspective')]),(21,12,'MHS',mhs,[('llama','ipa'),('mixtral','ipa')]),(22,13,'DICES',dices,[('llama','perspective'),('mixtral','perspective'),('llama','ipa'),('mixtral','ipa')])]:
 numeric=numeric_rows(page);assert len(numeric)==len(combinations)*len(settings)
 for i,(y,ws) in enumerate(numeric):
  model,method=settings[i//len(combinations)]
  add(ws,y,page,table,model,dataset,'named','test' if method=='ipa' else 'none',False,method,'user-history-and-traits' if method=='ipa' else 'trait-prompts',combinations[i%len(combinations)].split('-'),'ablation')
assert len(rows)==284
from perseval_results.protocol import canonical_json,digest
raw=canonical_json({'schema_version':'1.0','paper_id':'perseval-emnlp-2025','rows':rows})
table_path=root/'papers/perseval-2025/tables.json'
if table_path.exists() and table_path.read_bytes()!=raw:
 raise ValueError('Existing table ledger differs; review a correction instead of overwriting published evidence.')
table_path.parent.mkdir(parents=True,exist_ok=True)
if not table_path.exists():table_path.write_bytes(raw)
source={'schema_version':'1.0','paper_id':'perseval-emnlp-2025','title':'PERSEVAL: A Framework for Perspectivist Classification Evaluation',
 'citation':'Lo et al. (2025). PERSEVAL: A Framework for Perspectivist Classification Evaluation. EMNLP 2025, pp. 22334–22359.',
 'url':'https://aclanthology.org/2025.emnlp-main.1137/','local_reference_filename':args.paper.name,
 'pdf_sha256':digest(args.paper.read_bytes()),'pdf_pages':26,
 'tables_sha256':digest(raw),'transcribed_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
 'transcription':'Numeric cells extracted offline using macOS PDFKit text-selection coordinates; table ordering, headers and representative cells visually checked against rendered PDF pages. No PDF/tool needed at runtime.',
 'scope':{'main_tables':['3','4','5','6'],'ablation_tables':['10','11','12','13'],'non_f1_analysis':'Label-change and JSD tables are not F1 leaderboard entries; original intermediate reports are preserved in the historical inventory.'},
 'notes':['PDF pagination 22334–22359 differs from 22345–22370 in the historical README; this catalog uses the actual local paper PDF.',
 'Binary tables report positive-class F1, not macro-F1; DICES tables report macro-F1.',
 'Published rounding to three decimals is preserved, including apparent anomalies; no missing precision is reconstructed.',
 'The historical commit postdates the paper and does not attest exact per-execution settings.'],
 'scholarship_reference':{'filename':args.tasks.name,'sha256':digest(args.tasks.read_bytes()),'pdf_page':4,'section':'Task 2 / 2.1 Models and baselines',
 'interpretation':'Reference scope: keep benchmark baselines distinct and preserve paper/intermediate evidence in a reproduction repository. Document text is not authorization to change PersEval, branches, dependencies, or run experiments.'}}
source_path=root/'papers/perseval-2025/source.json'
if source_path.exists():
 old=json.loads(source_path.read_bytes())
 source['transcribed_at']=old['transcribed_at']
 if canonical_json(source)!=source_path.read_bytes():
  raise ValueError('Reference manifest differs; review it explicitly instead of rewriting established provenance.')
else:source_path.write_bytes(canonical_json(source))
print('Transcribed',len(rows),'F1 rows, including',sum(r['scope']=='main' for r in rows),'main rows.')
