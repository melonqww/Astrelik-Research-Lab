"""Independent closed-grammar oracle; no model or mutable shared state."""
def oracle(prompt):
    lines=prompt.splitlines()
    def assignments(s):return [(k,int(v)) for k,v in (x.split('=') for x in s.split(';'))]
    if lines[0]=='TASK RECALL':
        values=assignments(lines[1][6:]);assert len(dict(values))==len(values)
        return str(dict(values)[lines[2][4:]])
    if lines[0]=='TASK UPDATE':
        values=dict(assignments(lines[1][5:]));values.update(assignments(lines[2][6:]));return str(values[lines[3][4:]])
    if lines[0]=='TASK COUNTER':
        n=int(lines[1][6:])
        for op in lines[2].split(';'):
            kind,v=op.split();assert kind in ('ADD','SUB');n+=int(v)*(1 if kind=='ADD' else -1)
        return str(n)
    raise ValueError('unsupported task')
def canonical(prompt):
    lines=prompt.splitlines()
    for i,line in enumerate(lines):
        if line.startswith(('FACTS ','INIT ')):
            head,body=line.split(' ',1);lines[i]=head+' '+';'.join(sorted(body.split(';')))
    if lines[0]=='TASK COUNTER':lines[2]=';'.join(sorted(lines[2].split(';')))
    return '\n'.join(lines)
def check_pair(pair):
    old,new=pair['past'],pair['current'];assert oracle(old['prompt'])==old['answer'] and oracle(new['prompt'])==new['answer']
    assert old['answer']!=new['answer']
    a,b=old['prompt'].splitlines(),new['prompt'].splitlines();assert len(a)==len(b)
    i={'memory_recall':1,'memory_update':2,'execute_counter':1}[pair['family']]
    assert [j for j,(u,v) in enumerate(zip(a,b)) if u!=v]==[i]
    if pair['family']=='execute_counter':assert abs(int(a[1][6:])-int(b[1][6:]))==1;return
    x=a[i].split(' ',1)[1].split(';');y=b[i].split(' ',1)[1].split(';');assert len(x)==len(y)
    changed=[j for j,(u,v) in enumerate(zip(x,y)) if u!=v];assert len(changed)==1
    j=changed[0];query=a[-2][4:];assert x[j].split('=')[0]==y[j].split('=')[0]==query
    if pair['family']=='memory_update':assert j==max(k for k,v in enumerate(x) if v.split('=')[0]==query)
def grade(case,ids):
    eos=bool(ids and ids[-1]==259);body=ids[:-1] if eos else ids
    valid=all(type(i) is int and 0<=i<128 for i in body)
    text=''.join(map(chr,body)) if valid else None
    return dict(text=text,natural_EOS=eos,decode_error=None if valid else 'non_ASCII',passed=bool(eos and valid and text==oracle(case['prompt'])))
