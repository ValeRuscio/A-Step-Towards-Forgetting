"""Deterministic registry and natural-text sentiment -> news classification."""
import hashlib,json,random
from pathlib import Path
from association_core import make_data

def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def pin_assets(s):
    from huggingface_hub import HfApi
    api=HfApi()
    for item in s['models']:
        if not item['repo'].startswith('tiny-') and not Path(item['repo']).exists():
            item['revision']=api.model_info(item['repo'],revision=item.get('revision','main')).sha
    if 'text' in s['tasks']:
        s['datasets']={name:api.dataset_info(repo,revision=s.get('datasets',{}).get(name,'main')).sha for name,repo in [('sst2','stanfordnlp/sst2'),('ag_news','fancyzhx/ag_news')]}
    return s

def dataset(e,s,seed,task):
    if task=='registry':return make_data(e.tok,e.c,seed)
    if e.c.smoke:return make_data(e.tok,e.c,seed)
    from datasets import load_dataset
    specs=[('A','stanfordnlp/sst2','sst2','sentence','validation',2,'Sentiment: A = negative; B = positive.'),('B','fancyzhx/ag_news','ag_news','text','test',4,'Topic: A = World; B = Sports; C = Business; D = Science/Technology.')]
    data={};seen=set();rng=random.Random(500003+seed)
    for domain,repo,key,field,testsplit,k,instruction in specs:
        revision=s['datasets'][key]
        ds=load_dataset(repo,revision=revision)
        tokens=[e.tok.encode(' '+x,add_special_tokens=False) for x in 'ABCD'[:k]]
        if any(len(x)!=1 for x in tokens) or len(set(x[0] for x in tokens))!=k:raise ValueError('A/B/C/D must be distinct single tokens; refusing label change')
        labels=[x[0] for x in tokens]
        pools={'train':list(range(len(ds['train']))),'test':list(range(len(ds[testsplit])))}
        for pool in pools.values():rng.shuffle(pool)
        for split,count in [('valid',s['text_valid']),('train',s['text_train']),('test',s['text_test'])]:
            source=ds[testsplit] if split=='test' else ds['train'];pool=pools['test' if split=='test' else 'train'];rows=[]
            while len(rows)<count:
                if not pool:raise ValueError('Insufficient unique examples')
                idx=pool.pop();r=source[idx];text=r[field].strip();identity=hashlib.sha256(text.encode()).hexdigest()
                if identity in seen or not 0<=int(r['label'])<k:continue
                # Truncate content, never the instruction or the answer cue. Record exact input.
                prefix=e.tok.encode(instruction+'\nText: ',add_special_tokens=False);suffix=e.tok.encode('\nAnswer:',add_special_tokens=False)
                room=s['max_length']-len(prefix)-len(suffix)
                if room<8:raise ValueError('max_length too short')
                content=e.tok.encode(text,add_special_tokens=False);ids=prefix+content[:room]+suffix
                if tuple(ids) in seen:continue
                seen.add(identity);seen.add(tuple(ids));q=[0.]*k;q[int(r['label'])]=1.
                rows.append(dict(ids=ids,q=q,labels=labels,entity=int(identity[:12],16),example_id=identity,source_index=idx,source_split=testsplit if split=='test' else 'train',source_revision=revision,truncated=len(content)>room))
            data[domain+'_'+split]=rows
    return data
