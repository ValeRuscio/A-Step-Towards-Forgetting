"""Hash-separated development/confirmation text pools, deterministic paired label controls."""
import hashlib,json,random

def partition(text):return 'development' if int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%10<2 else 'confirmation'
def label_mapping(seed,variant):
    if variant=='standard':return {'A':['A','B'],'B':['A','B','C','D']}
    # Paired controls share exactly the same A codebook. B-overlap is the manipulation.
    chars=list('ABCDEF');random.Random(77009+seed).shuffle(chars)
    a=chars[:2];b=(chars[:4] if variant=='counterbalanced_shared' else chars[2:6])
    if variant not in ['counterbalanced_shared','counterbalanced_disjoint']:raise ValueError(variant)
    random.Random(88201+seed).shuffle(b)
    return {'A':a,'B':b}

def dataset(e,s,seed,role,variant='standard'):
    mapping=label_mapping(seed,variant);data={};meta=dict(role=role,seed=seed,variant=variant,mapping=mapping,partition='sha256(text) first32bits mod10; development<2, confirmation>=2')
    specs=[('A','stanfordnlp/sst2','sst2','sentence','validation',['negative','positive'],'Sentiment'),('B','fancyzhx/ag_news','ag_news','text','test',['World','Sports','Business','Science/Technology'],'Topic')]
    for domain,repo,key,field,ts,classes,taskname in specs:
        if s['smoke']:
            labels=[50+ord(x)-65 for x in mapping[domain]]
            for split,count in [('train',s['text_train']),('valid',s['text_valid']),('test',s['text_test'])]:
                rows=[]
                for i in range(count):
                    target=(i+seed)%len(classes);identity=hashlib.sha256(f'{role}/{seed}/{domain}/{split}/{i}'.encode()).hexdigest();q=[0.]*len(classes);q[target]=1.
                    ids=[1,2 if domain=='A' else 3,4+target,10+i%10,20+seed%10,30 if variant=='standard' else 31,labels[target] if split=='train' else 40]
                    rows.append(dict(ids=ids,q=q,labels=labels,entity=int(identity[:12],16),example_id=identity,source_index=i,source_split=split,source_revision='smoke',class_id=target))
                data[domain+'_'+split]=rows
            continue
        from datasets import load_dataset
        ds=load_dataset(repo,revision=s['datasets'][key]);ids=[e.tok.encode(' '+ch,add_special_tokens=False) for ch in mapping[domain]]
        if any(len(t)!=1 for t in ids) or len({t[0] for t in ids})!=len(ids):raise ValueError(f'{domain} codebook is not distinct single tokens: {mapping[domain]}')
        labels=[t[0] for t in ids]
        prefix=e.tok.encode(taskname+': '+'; '.join(ch+' = '+cl for ch,cl in zip(mapping[domain],classes))+'.\nText: ',add_special_tokens=False);suffix=e.tok.encode('\nAnswer:',add_special_tokens=False)
        # Identical content limit across codebooks, despite different tokenized instruction lengths.
        maxprefix=0
        for v in ['standard','counterbalanced_shared','counterbalanced_disjoint']:
            codes=label_mapping(seed,v)[domain];instr=taskname+': '+'; '.join(ch+' = '+cl for ch,cl in zip(codes,classes))+'.\nText: '
            maxprefix=max(maxprefix,len(e.tok.encode(instr,add_special_tokens=False)))
        room=s['max_length']-maxprefix-len(suffix)
        if room<8:raise ValueError('Context too short')
        seen=set();pools={}
        for split in ['train',ts]:
            pools[split]=list(range(len(ds[split])));random.Random(seed*1009+(11 if domain=='A' else 19)+(37 if split=='train' else 53)).shuffle(pools[split])
        for split,count in [('valid',s['text_valid']),('train',s['text_train']),('test',s['text_test'])]:
            source='train' if split!='test' else ts;rows=[]
            while len(rows)<count:
                if not pools[source]:raise ValueError('Insufficient hash-separated unique examples')
                idx=pools[source].pop();r=ds[source][idx];text=r[field].strip();identity=hashlib.sha256(text.encode()).hexdigest()
                if partition(text)!=role or identity in seen:continue
                target=int(r['label'])
                if not 0<=target<len(classes):continue
                content=e.tok.encode(text,add_special_tokens=False);inp=prefix+content[:room]+suffix
                # Content identities, not codebook-dependent instruction, drive paired selection.
                contentkey=tuple(content[:room])
                if contentkey in seen:continue
                seen.add(identity);seen.add(contentkey);q=[0.]*len(classes);q[target]=1.
                rows.append(dict(ids=inp,q=q,labels=labels,entity=int(identity[:12],16),example_id=identity,source_index=idx,source_split=source,source_revision=s['datasets'][key],class_id=target,truncated=len(content)>room))
            data[domain+'_'+split]=rows
    overlap=set(data['A_train'][0]['labels']) & set(data['B_train'][0]['labels']);meta['answer_token_overlap']=sorted(overlap)
    meta['data_sha256']=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    return data,meta
