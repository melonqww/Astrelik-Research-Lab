"""Independent standard-library regrading, policy trace replay and paired effects."""
import csv,math
from pathlib import Path
import gzip,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
from tasks import check_pair,grade,canonical

def check_trace(trace,policy,token,tol):
    assert trace['chosen']==token and 0<=trace['base_max']<=1
    if policy=='base':assert token==trace['base_top'];return
    d=trace['neighbor_distances'];v=trace['neighbor_values'];support=trace['neighbor_support']
    assert len(d)==len(v)==len(support)==8 and all(math.isfinite(x) and x>=0 for x in d)
    assert all(type(i) is int and 0<=i<260 for i in v) and all(type(x) is bool for x in support)
    weights=[math.exp(-x+min(d)) for x in d];total=sum(weights);weights=[x/total for x in weights]
    distribution={}
    for value,w in zip(v,weights):distribution[value]=distribution.get(value,0.)+w
    memmax=max(distribution.values());assert abs(memmax-trace['memory_max'])<=tol
    assert abs(sum(w for w,s in zip(weights,support) if s)-trace['support_weight'])<=tol
    assert abs(distribution.get(trace['memory_top'],0.)-memmax)<=tol
    probs={int(k):value for k,value in trace['candidate_base_probs'].items()}
    assert set(probs)==set(v)|{trace['base_top']}
    assert all(math.isfinite(x) and 0<=x<=trace['base_max']+tol for x in probs.values())
    assert abs(probs[trace['base_top']]-trace['base_max'])<=tol
    weight=.25
    if policy=='model_confidence' and trace['base_max']>=.5:weight=0.
    if policy=='neighbor_concentration' and trace['memory_max']<.75:weight=0.
    assert policy in ('fixed','model_confidence','neighbor_concentration') and trace['weight']==weight
    scores={i:(1-weight)*value+weight*distribution.get(i,0.) for i,value in probs.items()}
    assert token in scores and max(scores.values())-scores[token]<=tol

def load_answers(blob,cases,p):
    rows=blob['answers'];lookup={}
    expected={(c['id'],variant,0,'base') for c in cases for variant in ['past','current']}
    expected|={(c['id'],variant,n,policy) for c in cases for variant in ['past','current'] for n in p['densities'] for policy in p['policies']}
    case_map={c['id']:c for c in cases}
    for a in rows:
        key=(a['pair_id'],a['variant'],a['copies'],a['policy']);assert key not in lookup and key in expected
        tokens=a['generated_token_ids'];assert 1<=len(tokens)<=8 and len(tokens)==len(a['trace'])
        assert 259 not in tokens[:-1]
        for token,trace in zip(tokens,a['trace']):check_trace(trace,a['policy'],token,p['reference_tolerance'])
        a['grade']=grade(case_map[a['pair_id']][a['variant']],tokens);lookup[key]=a
    assert set(lookup)==expected and len(rows)==len(expected)
    receipts=blob['support_receipts'];assert len(receipts)==len(cases) and {r['pair_id'] for r in receipts}==set(case_map)
    for r in receipts:
        values=list(case_map[r['pair_id']]['past']['answer'].encode('ascii'))+[259]
        assert r['support_source']=='past_only' and r['values']==values and r['rows']==len(values)
    return lookup

def main():
    prov=read(ROOT/'evidence/provenance.json')
    archive=ROOT/'evidence/measurements.json.gz';assert sha(archive)==prov['measurements_sha256']
    payload=json.loads(gzip.decompress(archive.read_bytes()));assert payload['schema']=='AR-6-traces-v1'
    p=payload['protocol'];cases=read(ROOT/'evidence/cases.json')
    assert hashlib.sha256((ROOT/'evidence/cases.json').read_text(encoding='utf-8').encode()).hexdigest()==prov['cases_text_sha256']
    assert p['densities']==[0,1,8] and p['policies']==['fixed','model_confidence','neighbor_concentration']
    assert set(payload['models'])==set(p['models']) and len(p['models'])==6;assert len(cases)==96 and len({c['id'] for c in cases})==96
    signatures=set()
    for c in cases:
        check_pair(c)
        for variant in ['past','current']:
            key=canonical(c[variant]['prompt']);assert key not in signatures;signatures.add(key)
    signatures_hashed={hashlib.sha256(s.encode()).hexdigest() for s in signatures}
    assert not signatures_hashed&set(payload['known_input_sha256'])
    settings=[(0,'base')]+[(n,policy) for n in p['densities'] for policy in p['policies']]
    scores=[];families=[];paired=[];effects=[];tradeoffs=[];models=[];answers_count=0;trace_count=0;repeated_pilot=0;wall=0.
    def key(c,variant,n,policy):return c['id'],variant,n,policy
    for model in p['models']:
        blob=payload['models'][model];r=blob['report'];assert r['status']=='complete' and r['model_unchanged'] and r['optimizer_steps']==r['GPU_seconds']==0
        assert r['protocol_sha256']==p['technical_protocol_sha256'] and r['initial_model_sha256']==r['final_model_sha256']
        exit=blob['exit'];assert exit['status']=='complete' and exit['exit_code']==0 and exit['wall_seconds']<=p['outer_seconds']
        wall+=exit['wall_seconds'];lookup=load_answers(blob,cases,p);answers_count+=len(lookup);trace_count+=sum(len(a['trace']) for a in lookup.values());models.append(r)
        common=dict(model=model,cohort=p['groups'][model],seed=r['seed'])
        if model in p['pilot_models']:
            small=[c for c in cases if c['id'] in p['pilot_pair_ids']];pilot=load_answers(blob['pilot'],small,p)
            pr=blob['pilot']['report'];assert pr['status']=='complete' and pr['model_unchanged'] and pr['reference_max_error']<=p['reference_tolerance'] and pr['current_pairs_with_support']>=1
            for k,a in pilot.items():assert a['generated_token_ids']==lookup[k]['generated_token_ids'];repeated_pilot+=1
        for split in ['iid','ood']:
            selected=[c for c in cases if c['split']==split];assert len(selected)==48
            def good(c,variant,n,policy):return lookup[key(c,variant,n,policy)]['grade']['passed']
            def stale(c,n,policy):
                g=lookup[key(c,'current',n,policy)]['grade'];return g['natural_EOS'] and g['decode_error'] is None and g['text']==c['past']['answer']
            for n,policy in settings:
                for variant in ['past','current']:
                    arr=[lookup[key(c,variant,n,policy)] for c in selected];first=[a['trace'][0] for a in arr]
                    scores.append(dict(**common,split=split,variant=variant,copies=n,policy=policy,cases=48,correct=sum(a['grade']['passed'] for a in arr),
                        completed=sum(a['grade']['natural_EOS'] for a in arr),first_memory_used=sum(t.get('weight',0)>0 for t in first),
                        mean_first_support_weight=sum(t.get('support_weight',0) for t in first)/48))
                    for family in ['memory_recall','memory_update','execute_counter']:
                        subset=[c for c in selected if c['family']==family];assert len(subset)==16
                        families.append(dict(**common,split=split,variant=variant,copies=n,policy=policy,family=family,cases=16,
                            correct=sum(good(c,variant,n,policy) for c in subset),
                            old_answer_current=sum(stale(c,n,policy) for c in subset) if variant=='current' else None))
                paired.append(dict(**common,split=split,copies=n,policy=policy,pairs=48,
                    past_correct=sum(good(c,'past',n,policy) for c in selected),current_correct=sum(good(c,'current',n,policy) for c in selected),
                    both_correct=sum(good(c,'past',n,policy) and good(c,'current',n,policy) for c in selected),
                    completed_old_answer_current=sum(stale(c,n,policy) for c in selected),
                    old_answer_after_correct_past=sum(stale(c,n,policy) and good(c,'past',n,policy) for c in selected)))
                if policy!='base' and n:
                    effects.append(dict(**common,split=split,copies=n,policy=policy,cases=48,
                        current_gained=sum(not good(c,'current',0,policy) and good(c,'current',n,policy) for c in selected),
                        current_lost=sum(good(c,'current',0,policy) and not good(c,'current',n,policy) for c in selected),
                        old_answer_introduced=sum(not stale(c,0,policy) and stale(c,n,policy) for c in selected),
                        old_answer_removed=sum(stale(c,0,policy) and not stale(c,n,policy) for c in selected)))
                if policy not in ('base','fixed'):
                    tradeoffs.append(dict(**common,split=split,copies=n,policy=policy,cases=48,
                        current_gained=sum(not good(c,'current',n,'fixed') and good(c,'current',n,policy) for c in selected),
                        current_lost=sum(good(c,'current',n,'fixed') and not good(c,'current',n,policy) for c in selected),
                        past_gained=sum(not good(c,'past',n,'fixed') and good(c,'past',n,policy) for c in selected),
                        past_lost=sum(good(c,'past',n,'fixed') and not good(c,'past',n,policy) for c in selected)))
    assert answers_count==11520 and repeated_pilot==480
    (ROOT/'metrics').mkdir(exist_ok=True)
    for name,rows in [('scores.csv',scores),('families.csv',families),('pairs.csv',paired),('stale_effects.csv',effects),('filter_tradeoffs.csv',tradeoffs)]:
        with (ROOT/'metrics'/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Verified',answers_count,'primary answers,',trace_count,'token traces; all 480 pilot outputs reproduce.')
if __name__=='__main__':main()
