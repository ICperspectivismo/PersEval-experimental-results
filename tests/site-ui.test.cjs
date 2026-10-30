'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {selectEntries, sortEntries, metricText, allowedHref, enrichRows, relativeHeatStyle, filteredCsv} = require('../website/app.js');
const group = {benchmark_version:'synthetic-v1',split_protocol:'holdout',scoring_rule:'perseval.macro-f1.v1',dataset:'toy', label:'sentiment', task:'baseline', adaptation:'none', extended:false};
const metrics = (value) => ({global_f1:{mean:value,stddev:null},user_f1:{mean:null,stddev:null},text_f1:{mean:value,stddev:null}});
const entries = [
  {submission_id:'a',name:'First reproduction',model:'Identical model',submitter:'Lab A',source_type:'measured',rank:2,ranking_group_id:'measured',comparability:'recorded-test-fingerprint',comparison_group:group,metrics:metrics(.75),run_count:1,seeds:[42]},
  {submission_id:'b',name:'Paper entry',model:'Identical model',submitter:'Paper authors',source_type:'reported',experiment:{model_family:'encoder'},paper_variant:{table:'3',page:'4'},rank:1,ranking_group_id:'paper',comparability:'documented-paper-variant',comparison_group:group,metrics:metrics(.751)}
];
test('search, all filters and unknown model family remain explicit', () => {
  assert.deepEqual(selectEntries(entries,{search:'lab a',dataset:'toy',label:'sentiment',task:'baseline',adaptation:'none',extended:'false',source:'measured',family:'Unspecified'}),[entries[0]]);
  assert.deepEqual(selectEntries(entries,{family:'encoder',group:'paper'}),[entries[1]]);
  assert.deepEqual(selectEntries(entries,{dataset:'missing'}),[]);
  assert.equal(selectEntries(entries,{search:'identical model'}).length,2);
});
test('sort and filter retain supplied ranks and never mutate the data', () => {
  const snapshot = JSON.stringify(entries);
  assert.deepEqual(sortEntries(entries,'global_f1').map(e=>e.rank),[1,2]);
  assert.deepEqual(selectEntries(entries,{source:'measured'}).map(e=>e.rank),[2]);
  assert.deepEqual(sortEntries(entries,'rank').map(e=>e.submission_id),['b','a']);
  assert.equal(JSON.stringify(entries),snapshot);
});
test('paper cells contain only published values; n=1 measured SD is unavailable', () => {
  assert.equal(metricText({mean:.751,stddev:null},'reported',3),'.751'.replace(/^\./,'0.'));
  assert.equal(metricText({mean:.75,stddev:null},'measured'),'0.7500 · SD unavailable');
  assert.equal(metricText({mean:null,stddev:null},'reported'),'Unavailable');
});
test('unsafe links never become clickable source links', () => {
  for (const href of ['javascript:alert(1)','data:text/html,x','//evil.example/a','../secret','/root','https://u:p@example.org','a%2fb','https://example.org/\n']) assert.equal(allowedHref(href),false,href);
  assert.equal(allowedHref('artifacts/run_fixture/diagnostics.txt'),true);
  assert.equal(allowedHref('https://aclanthology.org/2025.emnlp-main.1137/'),true);
});

const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag='div') {
    this.tag=tag; this.children=[]; this.listeners={}; this.value=''; this.textContent='';
    this.hidden=false; this.dataset={}; this.checked=true;
    this.style={setProperty(name,value){this[name]=value;}};
  }
  append(...children) {
    for(const child of children) {
      if(child.tag==='#fragment') this.append(...[...child.children]);
      else { child.parent=this; this.children.push(child); }
    }
  }
  replaceChildren(...children) {this.children=[];this.append(...children);}
  addEventListener(name,callback) {this.listeners[name]=callback;}
  setAttribute(name,value) {this[name]=value;}
  get options() {return this.children;}
  showModal() {this.open=true;}
  close() {this.open=false;}
  select() {}
  remove(index) {if(index!==undefined)this.children.splice(index,1);else if(this.parent)this.parent.children=this.parent.children.filter(c=>c!==this);}
}
async function loadView(data, fail=false) {
  const elements=new Map(), selectors=new Map();
  const html=fs.readFileSync(require.resolve('../website/index.html'),'utf8');
  for(const match of html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/gu)) {
    const element=new Element(match[1]);element.id=match[2];elements.set(match[2],element);
  }
  for(const match of html.matchAll(/<select id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/gu)) {
    const element=elements.get(match[1]);
    for(const option of match[2].matchAll(/<option value="([^"]*)"[^>]*>([^<]*)<\/option>/gu)) {
      const child=new Element('option');child.value=option[1];child.textContent=option[2];element.append(child);
    }
    element.value=element.options[0].value;
  }
  for(const [id,values] of [['extended-filter',['all','false','true']],['adaptation-filter',['all','none','train','test']],['view-mode',['grouped','together']]]) {
    selectors.set('#'+id+' button',values.map(value=>{const el=new Element('button');el.dataset.value=value;return el;}));
  }
  selectors.set('.family-choice input',['Encoder','Decoder','Unspecified'].map((value,index)=>{
    const element=elements.get(['family-encoder','family-decoder','family-unknown'][index]);element.value=value;return element;
  }));
  selectors.set('.sort-button',[...html.matchAll(/data-sort="([^"]+)"/gu)].map(match=>{
    const el=new Element('button');el.dataset.sort=match[1];el.textContent=match[1];return el;
  }));
  const copied=[];
  const document={
    getElementById:id=>elements.get(id), querySelector:selector=>elements.get(selector.slice(1)),
    querySelectorAll:selector=>selectors.get(selector)||[], createElement:tag=>new Element(tag),
    createDocumentFragment:()=>new Element('#fragment'),body:new Element('body'),
  };
  const context=vm.createContext({document,URL,setTimeout,window:{clearTimeout(){},setTimeout(){return 1;}},
    navigator:{clipboard:{writeText:async text=>copied.push(text)}},
    fetch:async path=>{assert.equal(path,'data/leaderboard.json');return {ok:!fail,status:503,json:async()=>data};}});
  vm.runInContext(fs.readFileSync(require.resolve('../website/app.js'),'utf8'),context);
  await new Promise(resolve=>setImmediate(resolve));
  elements.selectors=selectors;elements.copied=copied;
  return elements;
}
const textOf = el => [el.textContent,...el.children.map(textOf)].join(' ');
const findElement = (el,test) => test(el) ? el : el.children.map(child=>findElement(child,test)).find(Boolean);
test('DOM rendering: groups, controls, empty results, history and supplied rank', async()=>{
  const measured={...entries[0],rank:9,run_count:1,seeds:[0],run_ids:['run_fixture'],notes:null,citation:null,provenance:{limitations:[]},verification:'Recorded provenance.',configuration_id:'sha256:fixture',revision:1,status:'published'};
  const paper={...entries[1],experiment:undefined,notes:null,citation:'Paper https://example.org/paper',provenance:{limitations:['No fingerprints']},verification:'Transcribed.',paper_variant:{description:'Documented variant',table:'3',page:'4'},revision:1,status:'published'};
  const data={entries:[measured,paper],history:{a:[measured],b:[paper]},build:{build_id:'sha256:build',source_revision:'commit'},
    artifact_metadata:{run_fixture:[]},runs:{run_fixture:{run:{seed:0,execution:{started_at:'start',finished_at:'finish'},configuration:{split:{seed:17}}},metrics:{},environment:{python:'fixture'}}}};
  const dom=await loadView(data);
  assert.equal(dom.get('error').hidden,true);
  assert.equal(dom.get('leaderboard-body').children.filter(el=>el.className==='variant-row').length,2);
  const details=findElement(dom.get('leaderboard-body'),el=>el.tag==='button' && el.textContent==='Details');
  details.listeners.click();
  assert.equal(dom.get('submission-dialog').open,true);
  assert.match(textOf(dom.get('submission-content')),/execution seed 0/u);
  assert.match(textOf(dom.get('submission-content')),/Immutable submission revision history/u);
  assert.doesNotMatch(textOf(dom.get('submission-content')),/API/u);
  assert.match(textOf(dom.get('history')),/revision/u);
  dom.get('source-filter').value='measured';dom.get('source-filter').listeners.change();
  assert.equal(dom.get('leaderboard-body').children.filter(el=>el.className==='variant-row').length,1);
  assert.match(textOf(dom.get('leaderboard-body')),/9/u);
  assert.match(dom.get('results-summary').textContent,/1 of 2/u);
  dom.get('search-filter').value='no match';dom.get('search-filter').listeners.input();
  assert.equal(dom.get('empty-state').hidden,false);
  assert.match(dom.get('empty-title').textContent,/No matching submissions/u);
});
test('DOM rendering exposes useful fetch failure and retry control',async()=>{
  const dom=await loadView(null,true);
  assert.equal(dom.get('error').hidden,false);
  assert.match(dom.get('error-message').textContent,/503/u);
  assert.equal(typeof dom.get('retry').listeners.click,'function');
});
test('complete and trait metrics use the loaded data without a detail endpoint',async()=>{
  const measured={...entries[0],run_ids:['run_fixture'],notes:null,citation:null,provenance:{limitations:[]},revision:1,status:'published'};
  const report={traits:{synthetic:{f1:.625}}};
  const data={entries:[measured],history:{a:[measured]},artifact_metadata:{run_fixture:[{
    path:'artifacts/perseval-evaluation.json',description:'Complete report',size:10,sha256:'sha256:fixture',
    public:true,href:'artifacts/run_fixture/report.download.txt',license:'CC0-1.0',attribution:'Synthetic fixture'}]},
    runs:{run_fixture:{run:{seed:42,execution:{started_at:'start',finished_at:'finish'},configuration:{split:{seed:17}}},metrics:{},environment:{python:'fixture'},complete_metrics:report}}};
  const dom=await loadView(data);
  findElement(dom.get('leaderboard-body'),el=>el.tag==='button' && el.textContent==='Details').listeners.click();
  const button=findElement(dom.get('submission-content'),el=>el.tag==='button' && el.textContent==='Show complete and trait metrics');
  await button.listeners.click();
  assert.match(textOf(dom.get('submission-content')),/synthetic/u);
  assert.match(textOf(dom.get('submission-content')),/0.625/u);
  assert.equal(findElement(dom.get('submission-content'),el=>el===button),undefined);
});
test('model family uses explicit producer inference metadata without guessing names',()=>{
  const measured={...entries[0],configuration:{hyperparameters:{inference:{model_family:'encoder'}}}};
  assert.deepEqual(selectEntries([measured],{family:'encoder'}),[measured]);
  assert.deepEqual(selectEntries([{...entries[0],model:'roberta-base'}],{family:'encoder'}),[]);
});

test('data adaptation retains backend ranks, partitions and null scores without mutating input',()=>{
  const original=JSON.stringify(entries);
  const {rows,groups}=enrichRows(entries);
  assert.equal(groups.length,2);
  assert.deepEqual(rows.map(row=>row.rank),[2,1]);
  assert.equal(rows[0].user_f1,null);
  assert.equal(groups[0].ranges.user_f1.min,null);
  assert.equal(JSON.stringify(entries),original);
  assert.deepEqual(relativeHeatStyle(null,{min:null,max:null}),{hue:77,equal:true});
});

test('filtered CSV retains precision, uncertainty and provenance and escapes spreadsheet formulas',()=>{
  const entry={...entries[0],name:'=SUM(1,2)',run_count:2,seeds:[0,42],
    metrics:{...entries[0].metrics,global_f1:{mean:.750000000000001,stddev:.01}}};
  const csv=filteredCsv(enrichRows([entry]).rows);
  assert.match(csv,/ranking_group_id,rank/u);
  assert.match(csv,/global_f1_stddev/u);
  assert.match(csv,/0\.750000000000001/u);
  assert.match(csv,/0\.01/u);
  assert.match(csv,/'=SUM/u);
  assert.match(csv,/0;42/u);
  assert.doesNotMatch(csv,/NaN|undefined/u);
});

test('grouped and all-together views, segmented filters, reset and CSV copy work',async()=>{
  const data={entries,history:{},build:{build_id:'sha256:build',source_revision:'commit'},
    artifact_metadata:{},runs:{}};
  const dom=await loadView(data);
  const segments=dom.selectors;
  assert.equal(dom.get('leaderboard-table').hidden,false);
  segments.get('#view-mode button')[1].listeners.click();
  assert.equal(dom.get('leaderboard-table').hidden,true);
  assert.equal(dom.get('together-leaderboard-table').hidden,false);
  await dom.get('copy-results').listeners.click();
  assert.match(dom.copied[0],/submission_id/u);
  segments.get('#extended-filter button')[2].listeners.click();
  assert.equal(dom.get('empty-state').hidden,false);
  dom.get('reset-filters').listeners.click();
  assert.equal(dom.get('empty-state').hidden,true);
  assert.equal(dom.get('scope-filter').value,'main');
  assert.equal(dom.get('together-leaderboard-table').hidden,false);
});

test('ablation filtering and empty data have explicit states',async()=>{
  const main={...entries[1],experiment:{model_family:'encoder',scope:'main'}};
  const ablation={...entries[1],submission_id:'c',experiment:{model_family:'decoder',scope:'ablation',trait_subset:['Gender']}};
  const data={entries:[main,ablation],history:{},build:{build_id:'sha256:build'},
    artifact_metadata:{},runs:{}};
  const dom=await loadView(data);
  assert.match(dom.get('results-summary').textContent,/1 of 2/u);
  dom.get('scope-filter').value='ablation';dom.get('scope-filter').listeners.change();
  assert.match(textOf(dom.get('leaderboard-body')),/Gender/u);
  const empty=await loadView({...data,entries:[]});
  assert.equal(empty.get('error').hidden,true);
  assert.equal(empty.get('empty-state').hidden,false);
  assert.equal(empty.get('copy-results').disabled,true);
  assert.equal(empty.get('empty-title').textContent,'No published results yet');
});

test('headline counts 72 main submissions, not the 212 additional trait ablations',async()=>{
  const main=Array.from({length:72},(_,i)=>({...entries[1],submission_id:`main-${i}`,
    experiment:{model_family:'encoder',scope:'main'}}));
  const ablations=Array.from({length:212},(_,i)=>({...entries[1],submission_id:`ablation-${i}`,
    ranking_group_id:'ablation',experiment:{model_family:'decoder',scope:'ablation',trait_subset:['Gender']}}));
  const data={entries:[...main,...ablations],history:{},artifact_metadata:{},runs:{}};
  const dom=await loadView(data);
  assert.equal(dom.get('error').hidden,true);
  assert.equal(String(dom.get('total-configurations').textContent),'72');
  assert.match(dom.get('footer-summary').textContent,/72 main submissions · 212 trait ablations/u);
  dom.get('scope-filter').value='ablation';dom.get('scope-filter').listeners.change();
  assert.match(dom.get('results-summary').textContent,/Showing 212 of 284/u);
  assert.equal(String(dom.get('total-configurations').textContent),'72');
  dom.get('scope-filter').value='';dom.get('scope-filter').listeners.change();
  assert.match(dom.get('results-summary').textContent,/Showing 284 of 284/u);
});

test('new measured submissions increase the main count; empty and ablation-only sites do not',async()=>{
  const ablation={...entries[1],experiment:{model_family:'decoder',scope:'ablation'}};
  const base={history:{},artifact_metadata:{},runs:{}};
  for(const [published,expected] of [[[],0],[[ablation],0],[[ablation,entries[1]],1],[[ablation,...entries],2]]) {
    const dom=await loadView({...base,entries:published});
    assert.equal(dom.get('error').hidden,true);
    assert.equal(String(dom.get('total-configurations').textContent),String(expected));
  }
});
