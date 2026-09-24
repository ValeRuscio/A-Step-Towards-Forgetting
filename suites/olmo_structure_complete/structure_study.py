"""OLMo checkpoint structure study: weights, activations and attention geometry.
Random initialization is an ensemble baseline, not the actual pretraining history.
Uses the supplied forgetting_mechanisms.py core for exact association replay.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/olmo_structure_mpl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse, copy, hashlib, itertools, json, math, subprocess, sys, time
import numpy as np
import torch
from forgetting_mechanisms import (Config, Engine, read, load, make_data, digest, minibatches,
    collate, predictions, atomic_json, write_csv, Progress, sha256_file, create_smoke_source)


def seed_for(name,seed=0):
    return (int(hashlib.sha256(name.encode()).hexdigest()[:8],16)+seed)%2147483647


def filename(name): return hashlib.sha256(name.encode()).hexdigest()[:20]+'.npz'


def points_distance(x):
    x=np.asarray(x,dtype=np.float64)
    sq=(x*x).sum(1);d=np.sqrt(np.maximum(sq[:,None]+sq[None,:]-2*x@x.T,0))
    np.fill_diagonal(d,0);return (d+d.T)/2


def persistent_homology(d,thresholds):
    """Exact H0/H1 of a finite flag filtration over F2, including triangles.
    Works for symmetric dissimilarities too, but then it is not a metric-space claim.
    Pure Python fallback avoids extra dependencies; cap at 32 landmarks.
    """
    d=np.asarray(d,dtype=np.float64);n=len(d)
    if d.shape!=(n,n) or not np.isfinite(d).all() or np.min(d)<-1e-10 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0):
        raise ValueError('Topology input must be finite, nonnegative, symmetric, zero diagonal')
    if n>32:raise ValueError('Built-in H1 supports at most 32 fixed landmarks; use fewer points')
    simplices=[(0.,(i,)) for i in range(n)]
    simplices += [(float(d[i,j]),(i,j)) for i,j in itertools.combinations(range(n),2)]
    simplices += [(max(float(d[i,j]),float(d[i,k]),float(d[j,k])),(i,j,k)) for i,j,k in itertools.combinations(range(n),3)]
    simplices.sort(key=lambda z:(z[0],len(z[1]),z[1]))
    ids={s:i for i,(_,s) in enumerate(simplices)};pivots={};births=set();pairs={}
    for j,(value,s) in enumerate(simplices):
        col=set() if len(s)==1 else {ids[s[:i]+s[i+1:]] for i in range(len(s))}
        while col and max(col) in pivots:col ^= pivots[max(col)]
        if col:
            low=max(col);pivots[low]=col;pairs[low]=j
        else:births.add(j)
    out={}
    for dim in [0,1]:
        intervals=[]
        for b in sorted(births):
            if len(simplices[b][1])-1!=dim:continue
            death=simplices[pairs[b]][0] if b in pairs else None
            if death is not None and death<=simplices[b][0]+1e-12:continue
            intervals.append([simplices[b][0],death])
        finite=[death-birth for birth,death in intervals if death is not None]
        out['H'+str(dim)]=dict(intervals=intervals,finite_total_persistence=float(sum(finite)),
            finite_max_persistence=float(max(finite,default=0)),
            betti=[sum(b<=t and (end is None or t<end) for b,end in intervals) for t in thresholds])
    out['thresholds']=list(thresholds);out['n_points']=n
    return out


def cloud_geometry(x):
    d=points_distance(x);positive=d[d>1e-12];scale=float(np.median(positive)) if len(positive) else 1.
    result=persistent_homology(d/scale,[.25,.5,.75,1.,1.25,1.5,2.])
    result['median_positive_distance']=scale;result['scale_convention']='Each cloud divided by its own median positive pair distance; shape only. Raw distances are saved.'
    return result,d


def entropy_rank(e):
    e=np.maximum(np.asarray(e,dtype=float),0);total=e.sum()
    if total<=1e-30:return 0.
    p=e[e>0]/total;return float(np.exp(-np.sum(p*np.log(p))))


def centered_cka(x,y):
    if x.shape[0]!=y.shape[0]:return None
    x=x-x.mean(0);y=y-y.mean(0);gx=x@x.T;gy=y@y.T
    den=np.linalg.norm(gx)*np.linalg.norm(gy)
    return float(np.sum(gx*gy)/den) if den>1e-20 else None


def exact_norm2(w):
    return sum(float(x.double().square().sum()) for x in w.detach().reshape(-1).split(1048576))


def top_svd(w,k,seed,device):
    """Randomized leading SVD; not a full-spectrum estimator. Residual audit retained."""
    a=w.detach().to(device=device,dtype=torch.float32);q=min(min(a.shape),k+8)
    devices=[a.device.index if a.device.index is not None else torch.cuda.current_device()] if a.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if q==min(a.shape):u,s,vh=torch.linalg.svd(a,full_matrices=False);v=vh.T
        else:u,s,v=torch.svd_lowrank(a,q=q,niter=3)
    keep=min(k,len(s));u=u[:,:keep];v=v[:,:keep];s=s[:keep]
    residual=float(torch.linalg.norm(a@v-u*s)/s.norm().clamp_min(1e-20))
    residual_right=float(torch.linalg.norm(a.T@u-v*s)/s.norm().clamp_min(1e-20))
    return s.cpu().numpy(),u.cpu().numpy(),v.cpu().numpy(),max(residual,residual_right)


def covariance_spectrum(sample):
    x=np.array(sample,dtype=np.float64)
    if x.shape[0]>x.shape[1]:x=x.T
    x-=x.mean();sd=float(x.std())
    if sd<1e-20:return np.zeros(x.shape[0]),x.shape[0]/x.shape[1]
    x/=sd
    return np.maximum(np.linalg.eigvalsh(x@x.T/x.shape[1]),0),x.shape[0]/x.shape[1]


def sample_rmt(sample,seed):
    eig,gamma=covariance_spectrum(sample);rng=np.random.default_rng(seed)
    gaussian=rng.normal(size=sample.shape);shuffled=np.asarray(sample).reshape(-1).copy();rng.shuffle(shuffled);shuffled=shuffled.reshape(sample.shape)
    ge,_=covariance_spectrum(gaussian);pe,_=covariance_spectrum(shuffled)
    lo=(1-math.sqrt(gamma))**2;hi=(1+math.sqrt(gamma))**2
    return dict(aspect_ratio=gamma,mp_lower=lo,mp_upper=hi,
        fraction_above_mp_edge=float(np.mean(eig>hi)),
        gaussian_spectral_W1=float(np.mean(np.abs(eig-ge))),
        shuffled_spectral_W1=float(np.mean(np.abs(eig-pe))),
        covariance_effective_rank=entropy_rank(eig),
        interpretation='Exact spectrum of a fixed matrix SUBSAMPLE after scalar centering/variance normalization. MP is an asymptotic iid covariance null, not a significance test.'),eig,ge,pe


def matrix_inventory(model):
    aliases={}
    for name,p in model.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
    return [(name,p,aliases[id(p)]) for name,p in model.named_parameters() if p.ndim==2]


@torch.no_grad()
def matrix_stats(name,w,s,folder):
    rng=np.random.default_rng(seed_for(name,s['sampling_seed']));m,n=w.shape
    ri=np.sort(rng.choice(m,min(m,s['matrix_sample']),replace=False));ci=np.sort(rng.choice(n,min(n,s['matrix_sample']),replace=False))
    # Rectangular full matrix has a square subsample by default; report actual sample aspect.
    sample=w.detach().cpu().numpy()[np.ix_(ri,ci)].astype(np.float64)
    point_rows=np.sort(rng.choice(m,min(m,s['topology_points']),replace=False))
    cloud=w.detach().cpu().numpy()[point_rows].astype(np.float64)
    sv,u,v,res=top_svd(w,s['top_k'],seed_for(name),s['device']);norm2=exact_norm2(w)
    geom,dist=cloud_geometry(cloud);rmt,eig,gauss,shuffle=sample_rmt(sample,seed_for(name,99))
    result=dict(matrix=name,shape=list(w.shape),frobenius=math.sqrt(norm2),
        mean=float(w.double().mean()),std=float(w.double().std(unbiased=False)),
        top_singular_values=sv.tolist(),top_svd_triplet_residual=res,
        stable_rank_estimate=float(norm2/max(float(sv[0])**2,1e-30)) if len(sv) else None,
        leading_energy_fraction=float(np.sum(sv.astype(float)**2)/max(norm2,1e-30)),
        rmt=rmt,row_cloud=geom)
    np.savez_compressed(folder/filename(name),sample=sample,singular_values=sv,left=u,right=v,
                        point_rows=point_rows,row_distances=dist,sample_eigenvalues=eig,gaussian_eigenvalues=gauss,shuffled_eigenvalues=shuffle)
    return result


def instantiate(settings,spec,progress):
    c=Config(**read(Path(settings['source'])/'config.json'));c.device='cpu';c.threads=settings['threads'];c.validate()
    if spec['kind']!='event':c.attn='eager'
    progress.update(phase='loading model',checkpoint=spec['id']);e=Engine(c)
    if spec['kind']=='random':
        from transformers import AutoModelForCausalLM
        cfg=copy.deepcopy(e.model.config);cfg._attn_implementation='eager'
        with torch.random.fork_rng():
            torch.manual_seed(spec['random_seed']);model=AutoModelForCausalLM.from_config(cfg)
        e.model=model.float().eval();e.params=dict(model.named_parameters());e.mods=dict(model.named_modules())
    elif spec['kind']=='hf':
        from transformers import AutoModelForCausalLM
        kwargs=dict(torch_dtype=torch.float32,attn_implementation='eager',trust_remote_code=False)
        if spec.get('revision'):kwargs['revision']=spec['revision']
        e.model=AutoModelForCausalLM.from_pretrained(spec['path'],**kwargs).eval();e.params=dict(e.model.named_parameters());e.mods=dict(e.model.named_modules())
    elif spec['kind'] in ['anchor','event','checkpoint']:
        seed=spec['seed'];src=Path(settings['source'])/f'seed{seed}'
        data=make_data(e.tok,c,seed)
        if digest(data)!=read(src/'data.json')['hash']:raise ValueError('Source data hash mismatch')
        if spec['kind']=='event':
            t=spec['event'];files=[f for f in src.glob('fork*/fork.pt') if int(f.parent.name[4:])<=t]
            if not files:raise ValueError('No preceding fork for requested event')
            file=max(files,key=lambda f:int(f.parent.name[4:]));z=load(file)
            if int(z['step'])!=int(file.parent.name[4:]):raise ValueError('Source fork label mismatch')
            c.device=settings['device'];e.model.to(c.device);e.restore(z['engine']);start=int(z['step']);del z
            for step in range(start,t+1):
                progress.update(phase='replaying original B updates',step=step)
                e.gradient(minibatches(data,'B',seed,step,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
            e.model.cpu();c.device='cpu'
        else:
            file=src/'anchor.pt' if spec['kind']=='anchor' else Path(spec['path'])
            z=load(file);state=z.get('engine',z);state=state.get('model',state)
            e.model.load_state_dict(state,strict=True);del z,state
    elif spec['kind']!='pretrained':raise ValueError('Unknown checkpoint kind')
    # Release Adam state: this is an observational pass after any exact replay.
    e.opt=None;e.model.eval();e.model.config.use_cache=False
    # Older Transformers selects the attention class during construction. Rebuild
    # only after replay, preserving every trained weight, to observe eager attention.
    if not hasattr(e.model,'set_attn_implementation') and any('SdpaAttention' in type(m).__name__ for m in e.model.modules()):
        from transformers import AutoModelForCausalLM
        cfg=copy.deepcopy(e.model.config);cfg._attn_implementation='eager'
        state=e.model.state_dict();e.model=AutoModelForCausalLM.from_config(cfg).float().eval()
        e.model.load_state_dict(state,strict=True);del state
    if e.model.config.model_type!='olmo':raise ValueError('Original OLMo architecture required')
    e.params=dict(e.model.named_parameters());e.mods=dict(e.model.named_modules())
    return e


def qk_geometry(q,k,attention,inv_freq,position_offset=0):
    """Original OLMo half-split RoPE planes; reconstruct post-RoPE scores and audit."""
    heads,t,dim=q.shape;positions=np.arange(t)+position_offset;half=dim//2
    phase=positions[:,None]*inv_freq[None,:]
    c=np.cos(phase)[None,:,:];sn=np.sin(phase)[None,:,:]
    def rotate(x):return np.concatenate([x[:,:,:half]*c-x[:,:,half:]*sn,x[:,:,half:]*c+x[:,:,:half]*sn],axis=-1)
    qr,kr=rotate(q),rotate(k)
    scores=qr@kr.transpose(0,2,1)/math.sqrt(dim);mask=np.tri(t,dtype=bool)
    masked=np.where(mask,scores,-np.inf);p=np.exp(masked-np.max(masked,axis=-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    error=float(np.max(np.abs(p-attention)))
    qnorm=np.linalg.norm(qr,axis=-1);knorm=np.linalg.norm(kr,axis=-1)
    cos=(qr@kr.transpose(0,2,1))/np.maximum(qnorm[:,:,None]*knorm[:,None,:],1e-30)
    return qr,kr,cos,error


def attention_summary(a,q,k,cos,s):
    """a is one full causal attention head. Graph statistics use explicit symmetrization."""
    t=len(a);mask=np.tri(t,dtype=bool);entropy=-np.sum(np.where(a>0,a*np.log(np.maximum(a,1e-300)),0),axis=1)
    # Position 0 is visible more often: normalize incoming mass by eligible queries.
    incoming=a.sum(0)/np.arange(t,0,-1)
    threshold=float(np.quantile(a[mask],.9));frequency=((a>=threshold)&mask).sum(0)/np.arange(t,0,-1)
    g=(a+a.T)/2;np.fill_diagonal(g,0)
    graph=[]
    for tau in [.001,.005,.01,.02,.05,.1,.2]:
        adj=(g>tau).astype(float);degree=adj.sum(1);lap=np.diag(degree)-adj;eig=np.linalg.eigvalsh(lap)
        central=float(np.sum(degree.max()-degree)/((t-1)*(t-2))) if t>2 else 0.
        hub=int(np.argmax(degree));leaf=np.delete(np.arange(t),hub)
        leaf_density=float(adj[np.ix_(leaf,leaf)].sum()/max((t-1)*(t-2),1))
        star=float(degree[hub]/max(t-1,1)*(1-leaf_density))
        graph.append(dict(threshold=tau,fiedler=float(max(eig[1],0)) if t>1 else 0.,degree_centralization=central,
                          star_likeness_defined=star,edges=int(adj.sum()/2)))
    idx=np.linspace(0,t-1,min(t,s['topology_points']),dtype=int)
    # 1-symmetric affinity is not generally a metric; it defines a flag filtration.
    dis=1-g[np.ix_(idx,idx)];np.fill_diagonal(dis,0)
    topo=persistent_homology(dis,[.5,.75,.9,.95,.99,1.])
    # Hellinger distance between full attention rows is a true metric.
    hdist=points_distance(np.sqrt(np.maximum(a[idx],0)))/math.sqrt(2)
    htopo=persistent_homology(hdist,[.1,.25,.5,.75,1.])
    singular=np.linalg.svd(a,compute_uv=False);energy=singular**2
    eligible=np.arange(t,0,-1)>=max(1,int(math.ceil(.3*t)))
    ref=int(np.argmax(np.where(eligible,incoming,-np.inf)));den=np.maximum(np.linalg.norm(q,axis=-1)*np.linalg.norm(k[ref]),1e-30)
    qref=np.sum(q*k[ref],axis=-1)/den
    first_qk=float(np.mean(cos[1:,0])) if t>1 else float(cos[0,0])
    # Complex-plane cross products summarize relative Q/K orientation, amplitude weighted.
    half=q.shape[-1]//2;qc=q[:,:half]+1j*q[:,half:];kc=k[ref,:half]+1j*k[ref,half:]
    z=qc*np.conj(kc);valid=np.arange(t)>=ref;z=z[valid]
    resultant=z.sum()/max(float(np.abs(z).sum()),1e-30)
    first=float(a[1:,0].mean()) if t>1 else float(a[0,0])
    uniform=float(np.mean(1/np.arange(2,t+1))) if t>1 else 1.
    ordered=np.sort(incoming);gini=float(np.dot(2*np.arange(1,t+1)-t-1,ordered)/max(t*ordered.sum(),1e-30))
    return dict(length=t,mean_entropy=float(entropy.mean()),first_token_mass=first,
        incoming_attention_gini=gini,
        causal_uniform_first_token_mass=uniform,first_token_excess_over_causal_uniform=first-uniform,
        reference_position=ref,reference_eligible_mean_mass=float(incoming[ref]),sink_positions=np.flatnonzero((frequency>=.4)&eligible).tolist(),
        incoming_eligible_mean=incoming.tolist(),mean_first_key_cosine=first_qk,
        reference_key_norm_ratio=float(np.linalg.norm(k[ref])/max(np.linalg.norm(k,axis=-1).mean(),1e-30)),
        reference_qk_cosine=float(qref[valid].mean()),
        rope_relative_resultant_real=float(resultant.real),rope_relative_resultant_imag=float(resultant.imag),
        rope_relative_resultant_length=float(abs(resultant)),
        rope_relative_mean_angle=float(np.angle(resultant)) if abs(resultant)>1e-8 else None,
        attention_singular_gap=float(singular[0]/max(singular[1],1e-30)) if len(singular)>1 else None,
        attention_energy_participation=float(energy.sum()**2/max(float(np.sum(energy**2)),1e-30)),
        symmetrized_graph=graph,affinity_flag_topology=topo,hellinger_row_topology=htopo)


def observe_activations(e,s,folder,progress):
    """Fixed prompts shared across checkpoints, using one prompt per forward.
    All Linear/Embedding outputs are sampled at fixed token indices; q/k capture all positions.
    """
    e.model.to(s['device']);e.c.device=s['device']
    # Eager attention is necessary for observed attention probabilities.
    if hasattr(e.model,'set_attn_implementation'):e.model.set_attn_implementation('eager')
    else:
        e.model.config._attn_implementation='eager'
        for mod in e.mods.values():
            if 'SdpaAttention' in type(mod).__name__:raise RuntimeError('This Transformers version needs model construction with eager attention; set source config attn=eager in a copied config/source or use 4.57.x for this observational run.')
    data=make_data(e.tok,e.c,s['panel_seed']);panels={sp:data[sp][:s['prompts_per_domain']] for sp in ['A_test','B_test']}
    if s['geometry_texts']:
        rows=[]
        for i,text in enumerate(s['geometry_texts']):
            ids=e.tok.encode(text,add_special_tokens=False)
            if not ids or len(ids)>e.c.max_length:raise ValueError('Geometry text is empty or exceeds source max_length; shorten it explicitly')
            rows.append(dict(ids=ids,prompt=text,entity=i,q=[.5,.5],labels=data['A_test'][0]['labels']))
        panels={'unlabelled_text':rows}
    canonical={id(p):n for n,p in e.params.items()};active={};handles=[];captures={};attnrows=[];preds={}
    def hook(name,mod):
        def f(module,args,y):
            # Only last-position logits are materialized, so lm_head receives one point.
            output=y.detach().float().cpu().numpy();x=args[0].detach().cpu().numpy()
            output=output.reshape(-1,output.shape[-1]);indices=np.linspace(0,len(output)-1,min(len(output),s['activation_positions']),dtype=int)
            xx=x.reshape(-1,x.shape[-1])[indices].astype(np.float32) if isinstance(mod,torch.nn.Linear) else None
            active[name]=dict(x=xx,y=output[indices],all_y=output if name.endswith(('q_proj','k_proj')) else None)
        return f
    modules={n:m for n,m in e.mods.items() if isinstance(m,(torch.nn.Linear,torch.nn.Embedding))}
    for n,m in modules.items():handles.append(m.register_forward_hook(hook(n,m)))
    try:
        for sp,rows in panels.items():
            preds[sp]=[]
            for pi,r in enumerate(rows):
                progress.update(phase='activation and attention geometry',split=sp,prompt=pi)
                active.clear();inputs,q,labels=collate([r],e.tok.pad_token_id,e.c.device)
                with torch.no_grad():
                    result=e.model.model(**inputs,use_cache=False,return_dict=True,output_attentions=True)
                    logits=e.model.lm_head(result.last_hidden_state[:,-1,:]);lp=logits.double().log_softmax(-1)[:,labels]
                    full=float((q*(q.log()-lp)).sum());margin=float(lp[0,0]-lp[0,1])
                preds[sp].append(dict(entity=r['entity'],full=full if sp!='unlabelled_text' else None,
                                     margin=margin if sp!='unlabelled_text' else None))
                for n,item in active.items():
                    key=sp+'::'+n;captures.setdefault(key,dict(x=[],y=[],matrix=canonical[id(modules[n].weight)]))
                    if item['x'] is not None:captures[key]['x'].append(item['x'])
                    captures[key]['y'].append(item['y'])
                if result.attentions is None:raise RuntimeError('No attention matrices returned; eager backend required')
                for layer,a in enumerate(result.attentions):
                    if a is None:raise RuntimeError('Missing head attention weights')
                    a=a[0].float().cpu().numpy().astype(float);heads,t,_=a.shape;dim=e.model.config.hidden_size//heads
                    name=f'model.layers.{layer}.self_attn';qq=active[name+'.q_proj']['all_y'];kk=active[name+'.k_proj']['all_y']
                    clip=getattr(e.model.config,'clip_qkv',None)
                    if clip is not None:qq=np.clip(qq,-clip,clip);kk=np.clip(kk,-clip,clip)
                    qq=qq.reshape(t,heads,dim).transpose(1,0,2);kv=e.model.config.num_key_value_heads
                    kk=kk.reshape(t,kv,dim).transpose(1,0,2);kk=np.repeat(kk,heads//kv,axis=0)
                    rotary=getattr(e.model.model,'rotary_emb',None) or getattr(e.mods[name],'rotary_emb',None)
                    inv=rotary.inv_freq.detach().float().cpu().numpy() if rotary is not None else 1/(e.model.config.rope_theta**(np.arange(0,dim,2)/dim))
                    qr,kr,cos,error=qk_geometry(qq.astype(float),kk.astype(float),a,inv)
                    for h in range(heads):
                        row=dict(split=sp,prompt=pi,layer=layer,head=h,rope_attention_reconstruction_error=error,
                                 rope_audit_pass=error<1e-4,
                                 **attention_summary(a[h],qr[h],kr[h],cos[h],s))
                        # Reject phase claims if actual attention cannot be reconstructed.
                        if error>=1e-4:
                            for key in list(row):
                                if key.startswith('rope_relative') or key in ['reference_qk_cosine','mean_first_key_cosine']:row[key]=None
                        attnrows.append(row)
                del result,logits,lp
    finally:
        for handle in handles:handle.remove()
    activations=[]
    for key,cap in captures.items():
        y=np.concatenate(cap['y']).astype(float);x=np.concatenate(cap['x']).astype(float) if cap['x'] else None
        idx=np.linspace(0,len(y)-1,min(s['topology_points'],len(y)),dtype=int);geo,dist=cloud_geometry(y[idx])
        yc=y-y.mean(0);eigen=np.maximum(np.linalg.eigvalsh(yc@yc.T),0)
        activations.append(dict(site=key,matrix=cap['matrix'],points=len(y),centered_effective_rank=entropy_rank(eigen),
                                rms=float(np.sqrt(np.mean(y*y))),topology=geo))
        values=dict(y=y,row_distances=dist)
        if x is not None:values['x']=x
        np.savez_compressed(folder/filename(key),**values)
    atomic_json(activations,folder/'activations.json');atomic_json(attnrows,folder/'attention.json');atomic_json(preds,folder/'probe_predictions.json')
    # Full original test panels for all requested seeds are distinct from small geometry samples.
    risk=[]
    for seed in s['seeds']:
        data=make_data(e.tok,e.c,seed)
        for sp in ['A_test','B_test']:
            vals=predictions(e,data[sp]);risk.append(dict(seed=seed,split=sp,mean_KL=float(np.mean([r['full'] for r in vals]))))
    atomic_json(risk,folder/'risk.json')
    atomic_json({sp:[dict(prompt=r['prompt'],ids=r['ids'],entity=r['entity']) for r in rows] for sp,rows in panels.items()},folder/'probe_panel.json')
    return e


def compare_checkpoint(s,current,reference,folder,progress):
    """Paired same-architecture checkpoints; no random-to-trained displacement interpretation."""
    e=instantiate(s,current,progress);r=instantiate(s,reference,progress)
    root=Path(s['output']);cf=root/current['id'];rf=root/reference['id']
    before=read(rf/'matrices.json');after=read(cf/'matrices.json');lookup={x['matrix']:x for x in before}
    if set(e.params)!=set(r.params):raise ValueError('Checkpoint parameter keys differ')
    acts0={x['site']:x for x in read(rf/'activations.json')};acts1={x['site']:x for x in read(cf/'activations.json')}
    rows=[];arows=[]
    for name,p,aliases in matrix_inventory(e.model):
        progress.update(phase='paired weight changes',matrix=name)
        if p.shape!=r.params[name].shape:raise ValueError('Incompatible matrix shapes')
        w0=r.params[name].detach().to(s['device']);w1=p.detach().to(s['device']);d=w1-w0
        dn=exact_norm2(d);wn=exact_norm2(w0);inner=sum(float((a.double()*b.double()).sum()) for a,b in zip(d.reshape(-1).split(1048576),w0.reshape(-1).split(1048576)))
        with np.load(rf/'matrix_arrays'/filename(name)) as z:
            u0=z['left'];v0=z['right'];sing0=z['singular_values'];sample0=z['sample']
        with np.load(cf/'matrix_arrays'/filename(name)) as z:
            u1=z['left'];v1=z['right'];sample1=z['sample']
        dim=min(u0.shape[1],u1.shape[1]);left=np.clip(np.linalg.svd(u0[:,:dim].T@u1[:,:dim],compute_uv=False),0,1)
        right=np.clip(np.linalg.svd(v0[:,:dim].T@v1[:,:dim],compute_uv=False),0,1)
        u=torch.as_tensor(u0,device=s['device']);v=torch.as_tensor(v0,device=s['device'])
        lfrac=exact_norm2(u.T@d)/max(dn,1e-30);rfrac=exact_norm2(d@v)/max(dn,1e-30)
        ds,_,_,dres=top_svd(d,s['top_k'],seed_for(name,111),s['device'])
        cos=float(np.clip(inner/math.sqrt(max(dn*wn,1e-30)),-1,1)) if dn>1e-25 else None
        rows.append(dict(matrix=name,reference=reference['id'],checkpoint=current['id'],relative_update_norm=math.sqrt(dn/max(wn,1e-30)),
            weight_update_cosine=cos,update_top_singular_values=ds.tolist(),update_svd_residual=dres,
            update_energy_in_reference_left_span=lfrac,update_energy_in_reference_right_span=rfrac,
            left_principal_angles_degrees=np.degrees(np.arccos(left)).tolist(),right_principal_angles_degrees=np.degrees(np.arccos(right)).tolist(),
            sample_signed_entry_correlation=float(np.corrcoef(sample0.reshape(-1),sample1.reshape(-1))[0,1]) if sample0.std()>1e-20 and sample1.std()>1e-20 else None))
        # Exact local matrix accounting for each use of a weight, on fixed actual inputs.
        for site,a1 in acts1.items():
            if a1['matrix']!=name or site not in acts0:continue
            with np.load(rf/'activation_arrays'/filename(site)) as z:y0=z['y'];x0=z['x'] if 'x' in z else None
            with np.load(cf/'activation_arrays'/filename(site)) as z:y1=z['y'];x1=z['x'] if 'x' in z else None
            item=dict(site=site,matrix=name,reference=reference['id'],checkpoint=current['id'],centered_CKA=centered_cka(y0,y1),
                      output_relative_change=float(np.linalg.norm(y1-y0)/max(np.linalg.norm(y0),1e-30)))
            if x0 is not None and x1 is not None:
                x=torch.as_tensor(x0,dtype=torch.float32,device=s['device']);dx=torch.as_tensor(x1-x0,dtype=torch.float32,device=s['device'])
                direct=x@d.T;upstream=dx@w0.T;interaction=dx@d.T
                actual=torch.as_tensor(y1-y0,dtype=torch.float32,device=s['device'])
                # Original OLMo projections are bias-free. Refuse silent bias omission.
                module_name=site.split('::',1)[1]
                if getattr(e.mods[module_name],'bias',None) is not None:raise ValueError('Projection bias requires an additional direct-bias term')
                item.update(direct_weight_rms=math.sqrt(exact_norm2(direct)/direct.numel()),
                    changed_input_rms=math.sqrt(exact_norm2(upstream)/upstream.numel()),interaction_rms=math.sqrt(exact_norm2(interaction)/interaction.numel()),
                    reconstruction_relative_error=math.sqrt(exact_norm2(actual-direct-upstream-interaction)/max(exact_norm2(actual),1e-30)))
            arows.append(item)
        del w0,w1,d,u,v
    atomic_json(rows,folder/'weight_changes.json');atomic_json(arows,folder/'activation_changes.json')
    write_csv([{k:v for k,v in row.items() if not isinstance(v,list)} for row in rows],folder/'weight_changes.csv')
    write_csv(arows,folder/'activation_changes.csv')
    ra=read(rf/'attention.json');ca=read(cf/'attention.json');rmap={(x['split'],x['prompt'],x['layer'],x['head']):x for x in ra};changes=[]
    for x in ca:
        key=(x['split'],x['prompt'],x['layer'],x['head']);b=rmap[key];out=dict(zip(['split','prompt','layer','head'],key))
        for k in ['first_token_mass','mean_entropy','reference_key_norm_ratio','mean_first_key_cosine','rope_relative_resultant_length']:
            out[k+'_change']=x[k]-b[k] if x[k] is not None and b[k] is not None else None
        angle0=b['rope_relative_mean_angle'];angle1=x['rope_relative_mean_angle']
        out['relative_phase_change']=float(np.angle(np.exp(1j*(angle1-angle0)))) if angle0 is not None and angle1 is not None and x['reference_position']==b['reference_position'] else None
        out['same_reference_position']=x['reference_position']==b['reference_position'];changes.append(out)
    atomic_json(changes,folder/'attention_changes.json');write_csv(changes,folder/'attention_changes.csv')
    risk0={(x['seed'],x['split']):x['mean_KL'] for x in read(rf/'risk.json')}
    risks=[dict(**x,KL_change=x['mean_KL']-risk0[(x['seed'],x['split'])],reference=reference['id']) for x in read(cf/'risk.json')]
    atomic_json(risks,folder/'risk_changes.json')


def snapshot(s,spec,folder,progress):
    folder=Path(folder);(folder/'matrix_arrays').mkdir(parents=True,exist_ok=True);(folder/'activation_arrays').mkdir(exist_ok=True)
    e=instantiate(s,spec,progress);rows=[];inventory=matrix_inventory(e.model)
    for index,(name,p,aliases) in enumerate(inventory):
        progress.update(phase='all-matrix spectral/topological analysis',matrix=name,completed=index,total=len(inventory))
        result=matrix_stats(name,p,s,folder/'matrix_arrays');result['aliases']=aliases;rows.append(result)
        atomic_json(rows,folder/'matrices.json')
    atomic_json([dict(parameter=n,shape=list(p.shape)) for n,p in e.params.items() if p.ndim!=2],folder/'non_matrix_parameters.json')
    write_csv([dict(matrix=x['matrix'],frobenius=x['frobenius'],stable_rank_estimate=x['stable_rank_estimate'],
                    leading_energy_fraction=x['leading_energy_fraction'],sample_covariance_effective_rank=x['rmt']['covariance_effective_rank'],
                    gaussian_spectral_W1=x['rmt']['gaussian_spectral_W1'],row_H1_persistence=x['row_cloud']['H1']['finite_total_persistence']) for x in rows],folder/'matrices.csv')
    observe_activations(e,s,folder/'activation_arrays',progress)
    # Human-facing JSON at snapshot root; large sampled arrays stay below their subdirectory.
    for name in ['activations.json','attention.json','probe_predictions.json','probe_panel.json','risk.json']:
        (folder/'activation_arrays'/name).replace(folder/name)
    atomic_json(spec,folder/'checkpoint.json')


def plot_results(output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output);stages=[p for p in sorted(root.iterdir()) if p.is_dir() and (p/'matrices.json').exists() and (p/'risk.json').exists()]
    if not stages:return
    fig,ax=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for stage in stages:
        ms=read(stage/'matrices.json');idx=np.arange(len(ms));label=stage.name
        ax[0,0].plot(idx,[r['leading_energy_fraction'] for r in ms],'.-',lw=.6,ms=2,label=label)
        ax[0,1].plot(idx,[r['rmt']['gaussian_spectral_W1'] for r in ms],'.-',lw=.6,ms=2,label=label)
        rows=read(stage/'attention.json');layers=sorted(set(x['layer'] for x in rows))
        ax[1,0].plot(layers,[np.mean([x['first_token_mass'] for x in rows if x['layer']==l]) for l in layers],'o-',label=label)
        ax[1,1].plot(layers,[np.mean([x['hellinger_row_topology']['H1']['finite_total_persistence'] for x in rows if x['layer']==l]) for l in layers],'o-',label=label)
    ax[0,0].set(xlabel='Matrix index (names in matrices.json)',ylabel='Energy fraction',title='Full-matrix leading singular energy')
    ax[0,1].set(xlabel='Matrix index',ylabel='Empirical spectral W1',title='Fixed covariance subsample vs Gaussian null')
    ax[1,0].set(xlabel='Layer',ylabel='First-position attention mass',title='Mean across fixed prompts and heads')
    ax[1,1].set(xlabel='Layer',ylabel='Total finite H1 persistence',title='Attention-row Hellinger geometry')
    for a in ax.flat:a.grid(alpha=.15)
    ax[0,0].legend(fontsize=6,ncol=2)
    smoke=read(root/'study_metadata.json').get('smoke',False) if (root/'study_metadata.json').exists() else False
    fig.suptitle('TINY RANDOM-MODEL IMPLEMENTATION TEST' if smoke else 'Random baseline, pretraining checkpoint and fine-tuning structure — observational comparison')
    fig.savefig(root/'structure_overview.png',dpi=160);plt.close(fig)
    comparisons=[p for p in root.glob('compare_*') if (p/'weight_changes.json').exists()]
    if comparisons:
        fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for p in comparisons:
            rows=read(p/'weight_changes.json');x=np.arange(len(rows));ax[0].plot(x,[r['relative_update_norm'] for r in rows],'.-',ms=2,label=p.name)
            ax[1].plot(x,[r['update_energy_in_reference_right_span'] for r in rows],'.-',ms=2,label=p.name)
        ax[0].set(xlabel='Matrix index',ylabel='Relative weight displacement',title='Fine-tuning changes relative to declared anchor')
        ax[1].set(xlabel='Matrix index',ylabel='Update energy fraction',title='Update overlap with reference input singular subspace')
        ax[0].legend(fontsize=6);fig.savefig(root/'weight_change_overview.png',dpi=160);plt.close(fig)


def report(output,full=False):
    root=Path(output);lines=['OLMO STRUCTURE STUDY — observational, not causal attribution']
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'risk.json').exists():continue
        matrices=read(folder/'matrices.json');attention=read(folder/'attention.json')
        lines.append(f"\n{folder.name}: {len(matrices)} unique matrices; RoPE audits {sum(x['rope_audit_pass'] for x in attention)}/{len(attention)} heads/prompts.")
        for risk in read(folder/'risk.json'):lines.append(f"  Seed {risk['seed']} {risk['split']}: KL {risk['mean_KL']:.7f}")
        if full:
            lines.append('  matrix | Frobenius | leading energy | sampled covariance effective rank')
            for m in matrices:lines.append(f"  {m['matrix']} | {m['frobenius']:.6g} | {m['leading_energy_fraction']:.6g} | {m['rmt']['covariance_effective_rank']:.6g}")
    for folder in sorted(root.glob('compare_*')):
        if (folder/'risk_changes.json').exists():
            lines.append('\n'+folder.name)
            for r in read(folder/'risk_changes.json'):lines.append(f"  Seed {r['seed']} {r['split']} change: {r['KL_change']:+.7f}")
    lines += ['\nRandom models are fresh initialization controls, not original pretraining checkpoints.',
              'Matrix and topology changes are descriptive. Correlation with A loss does not identify a cause.',
              'Phase here means relative Q/K orientation in RoPE planes, not temporal optimizer phase.',
              'Native matrix coordinates, sampled rows, graph symmetrization and filtration choices are recorded.']
    text='\n'.join(lines)+'\n';(root/('full_report.txt' if full else 'report.txt')).write_text(text);return text


def default_settings(source,output):
    source=Path(source).expanduser().resolve();output=Path(output).expanduser().resolve()
    specs=[dict(id='random_101',kind='random',random_seed=101),dict(id='random_202',kind='random',random_seed=202),dict(id='pretrained',kind='pretrained')]
    return dict(source=str(source),output=str(output),device='cuda:0',threads=8,seeds=[1],events=[21,47],geometry_texts=[],
                panel_seed=1,prompts_per_domain=2,activation_positions=8,topology_points=16,
                top_k=16,matrix_sample=128,sampling_seed=4401,extra_checkpoints=[],base_checkpoints=specs)


def plan(s):
    specs=list(s['base_checkpoints'])+list(s['extra_checkpoints'])
    for seed in s['seeds']:
        specs.append(dict(id=f'seed{seed}_A_anchor',kind='anchor',seed=seed))
        specs += [dict(id=f'seed{seed}_B_after{t:03d}',kind='event',seed=seed,event=t) for t in s['events']]
    pairs=[]
    ids={x['id'] for x in specs}
    for seed in s['seeds']:
        anchor=f'seed{seed}_A_anchor'
        if 'pretrained' in ids:pairs.append((anchor,'pretrained'))
        for t in s['events']:pairs.append((f'seed{seed}_B_after{t:03d}',anchor))
    # Optional references permit actual intermediate pretraining or A-continuation comparisons.
    for spec in s['extra_checkpoints']:
        if spec.get('compare_to'):pairs.append((spec['id'],spec['compare_to']))
    return specs,pairs


def validate(s):
    if set(s)!=set(default_settings('.','./output')):raise ValueError('Unexpected/missing settings field')
    src=Path(s['source']);out=Path(s['output'])
    if not (src/'config.json').exists():raise FileNotFoundError(f'Original association config not found: {src}/config.json')
    if src==out or src in out.parents or out in src.parents:raise ValueError('Use a separate, non-nested output directory')
    c=Config(**read(src/'config.json'));c.validate()
    if not 4<=s['topology_points']<=32:raise ValueError('Use 4..32 topology landmarks')
    if min(s['threads'],s['top_k'],s['matrix_sample'],s['prompts_per_domain'],s['activation_positions'])<1:raise ValueError('Sizes must be positive')
    if s['panel_seed'] not in c.seeds or any(x not in c.seeds for x in s['seeds']):raise ValueError('Seed missing from source')
    if any(t<1 or t>c.b_steps for t in s['events']):raise ValueError('Event outside source schedule')
    if not c.smoke and (c.model!='allenai/OLMo-1B-hf' or len(c.revision)!=40):raise ValueError('Original OLMo with pinned revision required')
    specs,pairs=plan(s);ids=[x['id'] for x in specs]
    if len(set(ids))!=len(ids):raise ValueError('Checkpoint IDs must be unique')
    if any(not x or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in x) for x in ids):raise ValueError('Checkpoint IDs must be simple letters/numbers/underscores/hyphens')
    if any(a not in ids or b not in ids for a,b in pairs):raise ValueError('Comparison reference missing')
    return specs,pairs


def fingerprints(s):
    src=Path(s['source']);files=[src/'config.json']
    for seed in set(s['seeds']+[s['panel_seed']]):
        files += [src/f'seed{seed}'/'data.json',src/f'seed{seed}'/'anchor.pt']
        for t in s['events']:
            fs=[p for p in (src/f'seed{seed}').glob('fork*/fork.pt') if int(p.parent.name[4:])<=t]
            if not fs:raise FileNotFoundError('Missing preceding fork')
            files.append(max(fs,key=lambda p:int(p.parent.name[4:])))
    for spec in s['extra_checkpoints']:
        if spec['kind']=='checkpoint':files.append(Path(spec['path']))
        if spec['kind']=='hf':
            p=Path(spec['path'])
            if p.is_dir():files += sorted(f for f in p.iterdir() if f.suffix in ['.json','.safetensors','.bin'])
            elif len(spec.get('revision',''))!=40:raise ValueError('Remote extra HF checkpoint requires pinned revision')
    return {str(p):sha256_file(p) for p in sorted(set(files))}


def run(s):
    import fcntl,transformers
    specs,pairs=validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'queue.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Progress(out) as progress:
            progress.update(phase='fingerprinting checkpoints')
            design=dict(settings=s,inputs=fingerprints(s),code={p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('forgetting_mechanisms.py')]},
                        versions=dict(torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__))
            if (out/'design.json').exists() and read(out/'design.json')!=design:raise ValueError('Design or code changed; choose a NEW output directory')
            atomic_json(design,out/'design.json');atomic_json(s,out/'settings.json')
            atomic_json(dict(smoke=Config(**read(Path(s['source'])/'config.json')).smoke,
                             comparison='Observational; random baselines are not original initialization'),out/'study_metadata.json')
            jobs=[dict(id=x['id'],kind='snapshot',spec=x) for x in specs]+[dict(id='compare_'+a+'__'+b,kind='compare',current=a,reference=b) for a,b in pairs]
            atomic_json(jobs,out/'jobs.json')
            for i,job in enumerate(jobs):
                folder=out/job['id'];folder.mkdir(exist_ok=True)
                if (folder/'done.json').exists():continue
                progress.update(phase='worker',job=job['id'],completed=i,total=len(jobs))
                with open(folder/'worker.log','w') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--settings',str(out/'settings.json'),'--job',job['id']],stdout=log,stderr=subprocess.STDOUT)
                    code=child.wait()
                if code or not (folder/'done.json').exists():raise RuntimeError(f'Worker failed: {job["id"]}; inspect its worker.log')
                plot_results(out);report(out)
            report(out,True);progress.update(status='complete',phase='finished',completed=len(jobs),total=len(jobs))


def worker(s,jobid):
    folder=Path(s['output'])/jobid
    with Progress(folder) as progress:
        jobs=read(Path(s['output'])/'jobs.json');job=next(x for x in jobs if x['id']==jobid)
        if job['kind']=='snapshot':snapshot(s,job['spec'],folder,progress)
        else:
            specs,_=plan(s);lookup={x['id']:x for x in specs};compare_checkpoint(s,lookup[job['current']],lookup[job['reference']],folder,progress)
        atomic_json(dict(complete=True,job=jobid),folder/'done.json');progress.update(status='complete',phase='finished')


def _process(output):
    """Only recognize the recorded launcher, never an unrelated reused PID."""
    import psutil
    out=Path(output)
    if not (out/'process.json').exists():return None
    try:
        info=read(out/'process.json');p=psutil.Process(info['pid']);cmd=p.cmdline()
        if p.status()==psutil.STATUS_ZOMBIE:return None
        if 'created' in info and abs(p.create_time()-info['created'])>.01:return None
        if not any(Path(x).name=='structure_study.py' for x in cmd):return None
        if 'run' not in cmd or '--settings' not in cmd:return None
        settings=Path(cmd[cmd.index('--settings')+1])
        if not settings.is_absolute():settings=Path(p.cwd())/settings
        if settings.resolve()!=(out/'settings.json').resolve():return None
        return p
    except (psutil.Error, ValueError, IndexError):return None


def status(output):
    out=Path(output);state=read(out/'status.json') if (out/'status.json').exists() else dict(status='not_started')
    state['alive']=_process(out) is not None
    pending=state['status']=='launching' and time.time()-state.get('launched',0)<30
    if state['status'] in ['launching','running'] and not state['alive'] and not pending:state['status']='interrupted_or_failed'
    if state.get('job') and (out/state['job']/'status.json').exists():state['worker']=read(out/state['job']/'status.json')
    return state


def _normalized(s):
    return json.loads(json.dumps(s,default=str))


def _incompatible(out,s):
    if (out/'settings.json').exists() and read(out/'settings.json')!=s:return True
    if (out/'design.json').exists():
        design=read(out/'design.json')
        code={p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('forgetting_mechanisms.py')]}
        import transformers
        versions=dict(torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__)
        if design.get('code')!=code or design.get('versions')!=versions:return True
    return False


def _fresh_output(s):
    import tempfile
    out=Path(s['output']).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    s['output']=tempfile.mkdtemp(prefix=out.name+'_restart_',dir=str(out.parent))


def launch(s):
    """Idempotent launch. Update s['output'] when a fresh folder is needed."""
    import fcntl,psutil
    validate(s)
    s.update(_normalized(s));out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    fresh=False
    with open(out/'launch.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another launch/stop is in progress. Refresh status shortly.');return out
        st=status(out)
        if st['alive']:
            print('Already running. Refresh status to follow progress.' if not _incompatible(out,s)
                  else 'A previous run is still active. Use study.restart(SETTINGS) to apply your changes.')
            print_status(out);return out
        # A held queue lock means a worker still owns this directory.
        with open(out/'queue.lock','a') as queue:
            try:fcntl.flock(queue,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Previous run is still shutting down. Refresh status shortly.');return out
        if _incompatible(out,s):fresh=True
        else:
            atomic_json(s,out/'settings.json');atomic_json(dict(status='launching',launched=time.time()),out/'status.json')
            try:
                with open(out/'launcher.log','a') as log:
                    p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--settings',str((out/'settings.json').resolve())],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:created=psutil.Process(p.pid).create_time()
                except psutil.Error:created=None
                info=dict(pid=p.pid)
                if created is not None:info['created']=created
                atomic_json(info,out/'process.json')
            except Exception as exc:
                atomic_json(dict(status='failed',error=str(exc)),out/'status.json');raise
    if fresh:
        _fresh_output(s)
        print('Settings or code changed; using a new results folder. Previous results are preserved.')
        return launch(s)
    print('Started. Refresh status to check progress. Results:',out)
    return out


def restart(s):
    """Stop this run, wait for its workers, and launch a fresh study."""
    validate(s)
    print(stop(s['output']))
    _fresh_output(s)
    return launch(s)


def print_status(output):
    s=status(output);print('Status:',s['status'],'| alive:',s['alive'],'| jobs:',s.get('completed',0),'/',s.get('total','?'))
    w=s.get('worker',s);print('Phase:',w.get('phase'),'| matrix:',w.get('matrix'),'| prompt:',w.get('prompt'))
    if s.get('error'):print(s['error'])
    if w.get('error'):print(w['error'])
    if 'heartbeat' in w:print('Heartbeat age:',round(time.time()-w['heartbeat'],1),'seconds. Liveness does not guarantee progress.')


def stop(output):
    import fcntl,psutil,signal
    out=Path(output)
    if not out.exists():return 'No active run'
    with open(out/'launch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=_process(out)
        if p is None:return 'No active run'
        # Detached launcher and all workers share this process group.
        if os.getpgid(p.pid)!=p.pid:raise RuntimeError('Cannot verify the background process group')
        try:members=[p]+p.children(recursive=True)
        except (psutil.Error, PermissionError):members=[p]
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        _,alive=psutil.wait_procs(members,timeout=10)
        # Also clear remaining group members when process enumeration is restricted.
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        for member in alive:
            try:member.kill()
            except psutil.NoSuchProcess:pass
        _,alive=psutil.wait_procs(alive,timeout=5)
        if any(x.is_running() and x.status()!=psutil.STATUS_ZOMBIE for x in alive):
            raise RuntimeError('A worker has not stopped yet; restart was not launched')
        atomic_json(dict(status='stopped',heartbeat=time.time()),out/'status.json')
    return 'Stopped. Previous results are preserved.'


def show(output,full=False):
    output=Path(output)
    if not output.exists():print('No output directory yet');return
    print(report(output,full))
    from IPython.display import display,Image
    for name in ['structure_overview.png','weight_change_overview.png']:
        if (output/name).exists():display(Image(filename=str(output/name)))


def self_test():
    # Unit square: one H1 class born at side length 1 and dying at sqrt(2).
    points=np.array([[0.,0.],[1,0],[1,1],[0,1]])
    h=persistent_homology(points_distance(points),[.5,1.1,1.5])
    assert h['H0']['betti']==[4,1,1] and h['H1']['betti']==[0,1,0]
    np.testing.assert_allclose(h['H1']['finite_max_persistence'],math.sqrt(2)-1)
    x=np.random.default_rng(0).normal(size=(12,5));q,_=np.linalg.qr(np.random.default_rng(1).normal(size=(5,5)))
    assert abs(centered_cka(x,x@q+3)-1)<1e-12
    np.testing.assert_allclose(points_distance(x),points_distance(x@q+3),atol=1e-7)
    spec,_=covariance_spectrum(np.random.default_rng(2).normal(size=(20,30)))
    assert np.min(spec)>=0 and abs(spec.mean()-1)<1e-10
    qq=np.random.default_rng(3).normal(size=(2,5,8));kk=np.random.default_rng(4).normal(size=(2,5,8))
    qr,kr,_,_=qk_geometry(qq,kk,np.zeros((2,5,5)),np.zeros(4))
    np.testing.assert_allclose(qr,qq);np.testing.assert_allclose(kr,kk)
    qa,ka,_,_=qk_geometry(qq,kk,np.zeros((2,5,5)),np.array([1.,.1,.01,.001]))
    qb,kb,_,_=qk_geometry(qq,kk,np.zeros((2,5,5)),np.array([1.,.1,.01,.001]),position_offset=17)
    np.testing.assert_allclose(qa@ka.transpose(0,2,1),qb@kb.transpose(0,2,1),atol=1e-12)
    # Common orthogonal Q/K rotation preserves logits (angles matter relationally).
    q,_=np.linalg.qr(np.random.default_rng(5).normal(size=(8,8)))
    np.testing.assert_allclose((qq@q)@(kk@q).transpose(0,2,1),qq@kk.transpose(0,2,1),atol=1e-12)
    model=torch.nn.Module();model.a=torch.nn.Parameter(torch.randn(3,4));model.b=model.a
    assert len(matrix_inventory(model))==1 and len(matrix_inventory(model)[0][2])==2
    print('Passed topology, geometry invariance, covariance normalization, RoPE and tied-weight tests.')


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ['run','worker']:
        a=subs.add_parser(name);a.add_argument('--settings',required=True)
        if name=='worker':a.add_argument('--job',required=True)
    a=subs.add_parser('smoke');a.add_argument('--output',required=True)
    subs.add_parser('self-test');a=p.parse_args()
    if a.command=='self-test':self_test()
    elif a.command=='smoke':
        root=Path(a.output).resolve();create_smoke_source(root/'source');s=default_settings(root/'source',root/'run')
        s.update(device='cpu',threads=1,events=[9],top_k=3,matrix_sample=12,topology_points=6,activation_positions=4,prompts_per_domain=1)
        s['base_checkpoints']=[dict(id='random_101',kind='random',random_seed=101),dict(id='pretrained',kind='pretrained')]
        run(s)
    elif a.command=='run':run(read(a.settings))
    else:worker(read(a.settings),a.job)

if __name__=='__main__':main()
