"""Portable AR-5 regrading and budget checks. Python standard library only."""
from pathlib import Path
import ast,csv,hashlib,itertools,json,math,operator,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from grader import check_function
ROOT=Path(__file__).resolve().parents[1]
OPS={'plus':('+',operator.add),'minus':('-',operator.sub),'times':('*',operator.mul)}
GRID=list(itertools.product([-7,-1,0,2,9],repeat=2))

def decode_ids(ids,token_bytes):
    eos=bool(ids and ids[-1]==0);body=ids[:-1] if eos else ids
    if any(type(i) is not int or i<2 for i in body):return '',eos,'special_token'
    try:return b''.join(bytes.fromhex(token_bytes[str(i)]) for i in body).decode('utf-8'),eos,None
    except (KeyError,ValueError,UnicodeDecodeError):return '',eos,'invalid_bytes'

def grade_case(c,text,eos,error=None):
    result=check_function(c['prompt']+text,c['name'],c['arguments'],c['examples'])
    try:unknown=sorted({n.id for n in ast.walk(ast.parse(c['prompt']+text)) if isinstance(n,ast.Name)}-set(c['arguments']))
    except (SyntaxError,ValueError,RecursionError):unknown=[]
    return dict(**result,undeclared_names=unknown,passed=bool(eos and error is None and result['correct']))

def validate_case(c):
    a,b=c['arguments'];symbol,fn=OPS[c['operation']]
    assert c['prompt']==f"def {c['name']}({a}, {b}):\n    # Return {a} {c['operation']} {b}.\n    return"
    assert c['text']==c['prompt']+f' {a} {symbol} {b}\n'
    assert c['examples']==[[[x,y],fn(x,y)] for x,y in GRID]
    assert check_function(c['text'],c['name'],c['arguments'],c['examples'])['correct']

def regrade(cases,answers,token_bytes,role):
    mapping={c['id']:c for c in cases};assert len(mapping)==len(cases)==len(answers)
    assert len({a['id'] for a in answers})==len(answers) and {a['id'] for a in answers}==set(mapping)
    for a in answers:
        c=mapping[a['id']];text,eos,error=decode_ids(a['generated_token_ids'],token_bytes)
        assert (text,eos,error)==(a['text'],a['natural_EOS'],a['decode_error']) and len(a['generated_token_ids'])<=16
        assert grade_case(c,text,eos,error)==a['check'] and a['operation']==c['operation']
        if role!='diagnostic':assert a['panel']==c.get('panel','train_fit') and a['triplet_id']==c.get('triplet_id')
    return {a['id']:a for a in answers}

def compute(e):
    assert e['schema']=='AR-5-synthetic-v1';p=e['protocol'];tb=e['token_bytes'];data=e['training_cases'];evaluation=e['evaluation_cases'];diagnostic=e['diagnostic_cases']
    assert p['sampler_seeds']==[17,29,43] and p['conditions']==['fixed','varied'] and p['steps']==256 and p['batch_size']==8
    assert len(evaluation)==81 and len(diagnostic)==18 and len(p['fit_ids'])==12
    assert len(data['fixed'])==len(data['varied'])==36
    for cases in list(data.values())+[evaluation,diagnostic]:
        for c in cases:validate_case(c)
    train_names={c['name'] for c in data['fixed']};assert not train_names&{c['name'] for c in evaluation+diagnostic}
    for condition in p['conditions']:
        for c in data[condition]+evaluation:
            assert c['full_ids'][:len(c['prefix_ids'])]==c['prefix_ids']
            assert decode_ids(c['full_ids'],tb)==(c['text'],True,None)
            assert decode_ids(c['prefix_ids'],tb)==(c['prompt'],False,None)
    for a,b in zip(data['fixed'],data['varied']):
        assert (a['id'],a['name'],a['operation'],len(a['full_ids']),len(a['prefix_ids']),a['suffix_targets'])==(b['id'],b['name'],b['operation'],len(b['full_ids']),len(b['prefix_ids']),b['suffix_targets'])
    assert not {v for c in data['varied'] for v in c['arguments']}&{v for c in evaluation if c['panel']=='unseen' for v in c['arguments']}
    groups={}
    for c in evaluation:groups.setdefault(c['triplet_id'],[]).append(c)
    assert len(groups)==27 and all(len(g)==3 and {c['panel'] for c in g}==set(p['evaluation_panels']) and len({(len(c['full_ids']),len(c['prefix_ids'])) for c in g})==1 for g in groups.values())
    runs={(r['seed'],r['condition']):r for r in e['runs']};assert len(runs)==len(e['runs'])==6
    assert set(runs)=={(s,c) for s in p['sampler_seeds'] for c in p['conditions']}
    scores=[];families=[];training=[];pairs=[];triples=[];extras=[];count=0
    for seed in p['sampler_seeds']:
        matches=[];maps={}
        for condition in p['conditions']:
            r=runs[seed,condition];t=r['training'];h=r['training_history'];indices=r['batch_indices'];cases=data[condition]
            assert t['status']=='complete' and t['steps']==256 and t['fresh_optimizer'] and not t['evaluation_outputs_read']
            assert t['sampler_seed']==seed and t['condition']==condition
            assert [v['step'] for v in h]==list(range(1,257)) and len(indices)==256
            index_text=json.dumps(indices,ensure_ascii=False,indent=2)+'\n'
            # Original Path.write_text used native line endings; preserve its hash
            # while accepting the same JSON records on Windows and Unix.
            hashes={hashlib.sha256(s.encode('utf-8')).hexdigest() for s in (index_text,index_text.replace('\n','\r\n'))}
            assert t['batch_indices_sha256'] in hashes
            targets=0;mask_hash=hashlib.sha256()
            for row,ix in zip(h,indices):
                assert len(ix)==8 and all(type(i) is int and 0<=i<36 for i in ix)
                selected=[cases[i] for i in ix];length=max(len(c['full_ids'])-1 for c in selected);n=0
                for c in selected:
                    prefix=len(c['prefix_ids'])-1;full=len(c['full_ids'])-1
                    assert c['suffix_targets']==full-prefix
                    n+=full-prefix;mask_hash.update(bytes([0]*prefix+[1]*(full-prefix)+[0]*(length-full)))
                assert row['targets']==n and math.isfinite(row['NLL']) and math.isfinite(row['gradient_norm']);targets+=n
            assert targets==t['training_targets']==10240 and mask_hash.hexdigest()==t['supervision_masks_sha256']
            fit=[c for c in cases if c['id'] in p['fit_ids']];regrade(fit,r['fit_answers'],tb,'fit');count+=12
            fits={op:sum(a['check']['passed'] for a in r['fit_answers'] if a['operation']==op) for op in OPS};assert fits==t['fit_counts']
            passed=sum(fits.values())>=p['fit_gate']['minimum_passed'] and min(fits.values())>=p['fit_gate']['minimum_per_operation'];assert passed==t['fit_passed']
            if seed==p['pilot_seed']:assert passed
            training.append(dict(seed=seed,condition=condition,steps=256,sampled_functions=2048,target_tokens=targets,final_batch_NLL=h[-1]['NLL'],fit_correct=sum(fits.values()),fit_cases=12));matches.append(r)
            er=r['evaluation'];assert er['status']=='complete' and er['weights_unchanged'] and er['source_checkpoint_sha256']==t['checkpoint_sha256']
            maps[condition]=regrade(evaluation,r['answers'],tb,'evaluation');count+=81
            for panel in p['evaluation_panels']:
                chosen=[a for a in r['answers'] if a['panel']==panel];assert len(chosen)==27
                correct=sum(a['check']['passed'] for a in chosen);assert correct==er['panel_counts'][panel]
                scores.append(dict(seed=seed,condition=condition,panel=panel,cases=27,functional_correct=correct,syntax_correct=sum(a['check']['syntax'] for a in chosen),natural_EOS=sum(a['natural_EOS'] for a in chosen),undeclared_names=sum(bool(a['check']['undeclared_names']) for a in chosen)))
                for op in OPS:
                    subset=[a for a in chosen if a['operation']==op];assert len(subset)==9
                    families.append(dict(seed=seed,condition=condition,panel=panel,operation=op,cases=9,correct=sum(a['check']['passed'] for a in subset)))
            triples.append(dict(seed=seed,condition=condition,triplets=27,all_three_correct=sum(all(maps[condition][c['id']]['check']['passed'] for c in group) for group in groups.values())))
            regrade(diagnostic,r['diagnostic_answers'],tb,'diagnostic');count+=18
            assert all(a['seed']==seed and a['condition']==condition for a in r['diagnostic_answers'])
            correct=sum(a['check']['passed'] for a in r['diagnostic_answers']);expected=next(d for d in e['diagnostic_summary'] if d['seed']==seed and d['condition']==condition);assert correct==expected['correct']
            extras.append(dict(seed=seed,condition=condition,cases=18,correct=correct,role='posthoc_seen_pairs'))
        assert matches[0]['batch_indices']==matches[1]['batch_indices']
        for k in ['initial_model_sha256','batch_indices_sha256','training_targets','supervision_masks_sha256']:assert matches[0]['training'][k]==matches[1]['training'][k]
        for panel in p['evaluation_panels']:
            ids=[c['id'] for c in evaluation if c['panel']==panel];a,b=maps['fixed'],maps['varied']
            pairs.append(dict(seed=seed,panel=panel,cases=27,fixed=sum(a[i]['check']['passed'] for i in ids),varied=sum(b[i]['check']['passed'] for i in ids),gained=sum(not a[i]['check']['passed'] and b[i]['check']['passed'] for i in ids),lost=sum(a[i]['check']['passed'] and not b[i]['check']['passed'] for i in ids)))
    assert count==666 and len({r['training']['initial_model_sha256'] for r in e['runs']})==1
    return {'scores.csv':scores,'operations.csv':families,'training.csv':training,'paired.csv':pairs,'triplets.csv':triples,'diagnostic.csv':extras}

def main():
    f=ROOT/'evidence/measurements.json';e=json.loads(f.read_text(encoding='utf-8'));prov=json.loads((ROOT/'evidence/provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(f.read_text(encoding='utf-8').encode('utf-8')).hexdigest()==prov['measurements_sha256']
    metrics=compute(e);(ROOT/'metrics').mkdir(exist_ok=True)
    for name,rows in metrics.items():
        with (ROOT/'metrics'/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('Regraded 666 saved responses; synthetic oracles, bytes/EOS, fit gate and paired training budgets verified.')
if __name__=='__main__':main()
