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
    if x.shape[0]!=y.shape[0] or x.shape[0]<3:return None
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
    extended_compare(s,current,reference,folder,progress)


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
    extended_snapshot(e,s,folder,progress)


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
                panel_seed=1,prompts_per_domain=8,activation_positions=8,topology_points=16,
                top_k=16,matrix_sample=128,sampling_seed=4401,extra_checkpoints=[],base_checkpoints=specs,
                extended=dict(enabled=True,null_repeats=8,ntk_enabled=True,ntk_mode='auto',ntk_prompts_per_domain=32,
                              ntk_sketch_dim=512,ntk_seed=7301,exact_max_elements=2000000,ntk_audit_tolerance=.2),
                trajectory=dict(enabled=True,run_checkpoint_study=False,steps=[20,21],layers=[],fd_epsilons=[.05,.025],label_null_repeats=16),
                path=dict(enabled=True,alphas=[0.,.125,.25,.375,.5,.625,.75,.875,1.],max_refinements=6,alpha_tolerance=.01,
                          fd_epsilons=[.01,.005],fd_atol=1e-5,fd_rtol=.05,attention=True,attention_audit_tolerance=1e-4),overnight=night_defaults())


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
    for seed in s['seeds']:
        ordered=sorted(s['events'])
        pairs += [(f'seed{seed}_B_after{b:03d}',f'seed{seed}_B_after{a:03d}') for a,b in zip(ordered,ordered[1:])]
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
    n=s['overnight']
    if set(n)!=set(night_defaults()):raise ValueError('Unexpected/missing overnight setting')
    if n['enabled']:
        if not s['seeds']:raise ValueError('Select at least one source seed')
        if not math.isfinite(n['hours']) or n['hours']<=0 or n['reserve_minutes']<0 or n['reserve_minutes']*60>=n['hours']*3600:raise ValueError('Invalid overnight time budget/reserve')
        if not n['events'] or len(set(n['events']))!=len(n['events']) or any(not isinstance(v,int) or v<1 or v>c.b_steps for v in n['events']):raise ValueError('Overnight events must be distinct valid source B steps')
        for key in ['alphas','geometry_alphas','intervention_alphas','ntk_alphas','matrix_alphas','gradient_alphas']:
            if not n[key] or len(set(n[key]))!=len(n[key]) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or v>1 for v in n[key]):raise ValueError('Invalid fraction list: '+key)
        for key in ['alphas','geometry_alphas']:
            if min(n[key])!=0 or max(n[key])!=1:raise ValueError(key+' must contain both endpoints')
        if not 0<n['patch_alpha']<=1:raise ValueError('patch_alpha must be in (0,1]')
        if not math.isfinite(n['sham_loss_atol']) or n['sham_loss_atol']<=0:raise ValueError('Invalid sham audit tolerance')
        for key in ['refinements','head_patches','random_patch_repeats']:
            if not isinstance(n[key],int) or n[key]<0:raise ValueError('Invalid count: '+key)
        for key in ['branch_steps','null_repeats','ntk_dimension','matrix_top_k','matrix_sample']:
            if not isinstance(n[key],int) or n[key]<1:raise ValueError('Invalid positive count: '+key)
        if not math.isfinite(n['minimum_free_gb']) or n['minimum_free_gb']<0:raise ValueError('Invalid disk reserve')
        if any(not isinstance(v,int) or v<0 for v in n['patch_layers']):raise ValueError('Invalid patch layer')
    traj=s['trajectory']
    if set(traj)!=set(default_settings('.','./output')['trajectory']):raise ValueError('Missing/unexpected trajectory setting')
    if traj['enabled']:
        if not isinstance(traj['label_null_repeats'],int) or traj['label_null_repeats']<1:raise ValueError('Use at least one label null repeat')
        if not traj['steps'] or any(not isinstance(t,int) or t<1 or t>c.b_steps for t in traj['steps']):raise ValueError('Trajectory steps must be within source B schedule')
        if c.entities>32 or c.entities<4:raise ValueError('All-entity topology requires 4..32 source entities')
        if len(traj['fd_epsilons'])!=2 or any(not 0<e<=.5 for e in traj['fd_epsilons']) or len(set(traj['fd_epsilons']))!=2:raise ValueError('Use two different positive finite-difference spacings <= .5')
        if any(not isinstance(l,int) or l<0 for l in traj['layers']):raise ValueError('Layer indices must be nonnegative integers')
    path=s['path']
    if set(path)!=set(default_settings('.','./output')['path']):raise ValueError('Missing/unexpected path setting')
    if path['enabled']:
        if not traj['enabled']:raise ValueError('Path analysis requires trajectory.enabled')
        grid=path['alphas']
        if not grid or grid!=sorted(set(grid)) or grid[0]!=0 or grid[-1]!=1 or any(not 0<=v<=1 for v in grid):raise ValueError('Path alphas must increase uniquely from 0 to 1')
        if not isinstance(path['max_refinements'],int) or path['max_refinements']<0:raise ValueError('max_refinements must be a nonnegative integer')
        if not 0<path['alpha_tolerance']<1:raise ValueError('alpha_tolerance must lie in (0,1)')
        if len(path['fd_epsilons'])!=2 or len(set(path['fd_epsilons']))!=2 or any(not 0<e<=.5 for e in path['fd_epsilons']):raise ValueError('Path requires two distinct FD spacings in (0,.5]')
        if any(not math.isfinite(path[k]) or path[k]<=0 for k in ['fd_atol','fd_rtol','attention_audit_tolerance']):raise ValueError('Path audit tolerances must be finite and positive')
    ext=s['extended']
    if set(ext)!=set(default_settings('.','./output')['extended']):raise ValueError('Missing/unexpected extended setting')
    if ext['ntk_mode'] not in ['auto','exact','sketch']:raise ValueError('NTK mode must be auto/exact/sketch')
    if min(ext['null_repeats'],ext['ntk_prompts_per_domain'],ext['ntk_sketch_dim'],ext['exact_max_elements'])<1:raise ValueError('Extended sizes must be positive')
    if not 0<ext['ntk_audit_tolerance']<1:raise ValueError('NTK audit tolerance must lie in (0,1)')
    if traj['run_checkpoint_study'] and ext['ntk_prompts_per_domain']>c.entities:raise ValueError('NTK panel exceeds available entities')
    if not 4<=s['topology_points']<=32:raise ValueError('Use 4..32 topology landmarks')
    if min(s['threads'],s['top_k'],s['matrix_sample'],s['prompts_per_domain'],s['activation_positions'])<1:raise ValueError('Sizes must be positive')
    if (traj['run_checkpoint_study'] and s['panel_seed'] not in c.seeds) or any(x not in c.seeds for x in s['seeds']):raise ValueError('Seed missing from source')
    if traj['run_checkpoint_study'] and any(t<1 or t>c.b_steps for t in s['events']):raise ValueError('Event outside source schedule')
    if not c.smoke and (c.model!='allenai/OLMo-1B-hf' or len(c.revision)!=40):raise ValueError('Original OLMo with pinned revision required')
    specs,pairs=plan(s) if traj['run_checkpoint_study'] else ([],[])
    for spec in specs:
        if spec['kind']=='hf':_hf_local_or_remote(spec)
        if spec['kind']=='checkpoint' and not Path(spec['path']).expanduser().is_file():raise FileNotFoundError(f"Extra checkpoint file not found: {spec['path']}")
    ids=[x['id'] for x in specs]
    if len(set(ids))!=len(ids):raise ValueError('Checkpoint IDs must be unique')
    if any(not x or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in x) for x in ids):raise ValueError('Checkpoint IDs must be simple letters/numbers/underscores/hyphens')
    if any(a not in ids or b not in ids for a,b in pairs):raise ValueError('Comparison reference missing')
    return specs,pairs


def _hf_local_or_remote(spec):
    path=str(spec['path']);p=Path(path).expanduser()
    if p.is_dir():return p
    if p.exists() or path.startswith(('/', '~', '.')):
        raise FileNotFoundError(f"Local HF checkpoint directory not found: {p}. Set a real folder in extra_checkpoints, or disable run_checkpoint_study for the trajectory-only experiment.")
    revision=spec.get('revision','')
    if len(revision)!=40 or any(c not in '0123456789abcdefABCDEF' for c in revision):
        raise ValueError(f"Remote HF checkpoint {path!r} needs its 40-character commit revision. If this is a local checkpoint, provide its existing absolute directory path.")
    return None


def fingerprints(s):
    """Hash only inputs consumed by enabled jobs, including the trajectory's actual fork."""
    src=Path(s['source']);files=[src/'config.json'];traj=s['trajectory']
    checkpoint_study=traj['run_checkpoint_study']
    seeds=set(s['seeds'])
    if checkpoint_study:seeds.add(s['panel_seed'])
    for seed in seeds:
        files += [src/f'seed{seed}'/'data.json',src/f'seed{seed}'/'anchor.pt']
    requested=[]
    if traj['enabled']:requested += [(seed,min(traj['steps'])) for seed in s['seeds']]
    if checkpoint_study:
        specs,_=plan(s)
        requested += [(x['seed'],x['event']) for x in specs if x['kind']=='event']
        for spec in specs:
            if spec['kind']=='checkpoint':files.append(Path(spec['path']).expanduser())
            if spec['kind']=='hf':
                p=_hf_local_or_remote(spec)
                if p is not None:files += sorted(f for f in p.rglob('*') if f.is_file() and f.suffix in ['.json','.safetensors','.bin'])
    for seed,t in requested:
        fs=[p for p in (src/f'seed{seed}').glob('fork*/fork.pt') if int(p.parent.name[4:])<=t]
        if not fs:raise FileNotFoundError(f'Missing saved fork for seed {seed} before update {t}')
        files.append(max(fs,key=lambda p:int(p.parent.name[4:])))
    return {str(p):sha256_file(p) for p in sorted(set(files))}


def run(s):
    if s.get("overnight",{}).get("enabled"):return night_run(s)
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
            if not s['trajectory']['run_checkpoint_study']:jobs=[]
            if s['trajectory']['enabled']:jobs += [dict(id=f'trajectory_seed{seed}',kind='trajectory',seed=seed) for seed in s['seeds']]
            if not jobs:raise ValueError('Enable trajectory or checkpoint study')
            atomic_json(jobs,out/'jobs.json')
            for i,job in enumerate(jobs):
                folder=out/job['id'];folder.mkdir(exist_ok=True)
                if (folder/'done.json').exists():continue
                progress.update(phase='worker',job=job['id'],completed=i,total=len(jobs))
                with open(folder/'worker.log','w') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--settings',str(out/'settings.json'),'--job',job['id']],stdout=log,stderr=subprocess.STDOUT)
                    code=child.wait()
                if code or not (folder/'done.json').exists():raise RuntimeError(f'Worker failed: {job["id"]}; inspect its worker.log')
                plot_results(out);report(out);extended_plots(out);extended_report(out)
            report(out,True);progress.update(status='complete',phase='finished',completed=len(jobs),total=len(jobs))


def worker(s,jobid):
    if s.get("overnight",{}).get("enabled"):return night_worker(s,jobid)
    folder=Path(s['output'])/jobid
    with Progress(folder) as progress:
        jobs=read(Path(s['output'])/'jobs.json');job=next(x for x in jobs if x['id']==jobid)
        if job['kind']=='trajectory':run_trajectory(s,job['seed'],folder,progress)
        elif job['kind']=='snapshot':snapshot(s,job['spec'],folder,progress)
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
    if (Path(output)/'session_budget.json').exists():
        budget=read(Path(output)/'session_budget.json');print('Session minutes remaining:',round(max(0,budget['session_deadline']-time.time())/60,1))
        print('Step:',w.get('step'),'| fraction:',w.get('alpha'),'| layer:',w.get('layer'),'| condition:',w.get('condition'))
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
    if (output/'overnight_summary.json').exists() or (output/'session_budget.json').exists():
        print(night_report(output))
        from IPython.display import display,Image
        if (output/'overnight_paths.png').exists():display(Image(filename=str(output/'overnight_paths.png')))
        for p in sorted(output.glob('branches_*/continuations.png')):display(Image(filename=str(p)))
        for p in sorted(output.glob('patches_*_effects.png'))[:None if full else 2]:display(Image(filename=str(p)))
        if (output/'results_share.zip').exists():print('Share this results ZIP:',output/'results_share.zip')
        return
    if list(output.glob('trajectory_seed*')):print(trajectory_report(output))
    if list(output.glob('trajectory_seed*/update*/path/path_summary.json')):print(path_root_report(output))
    if any((p/'risk.json').exists() for p in output.iterdir() if p.is_dir()):
        print(report(output,full))
        print(extended_report(output))
    from IPython.display import display,Image
    for name in ['structure_overview.png','weight_change_overview.png','ntk_overview.png','ph_rmt_frame_overview.png','ph_rmt_example.png','trajectory_updates.png','trajectory_examples.png']:
        if (output/name).exists():display(Image(filename=str(output/name)))

    for p in sorted(output.glob('trajectory_seed*/update*/path/loss_and_slope.png')):
        print(p.parents[2].name+'/'+p.parents[1].name)
        display(Image(filename=str(p)))
        if (p.parent/'geometry_along_path.png').exists():display(Image(filename=str(p.parent/'geometry_along_path.png')))

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


# ---------------- PH / RMT / reference-frame / empirical NTK extension ----------------

def bottleneck_finite(a,b):
    """Exact bottleneck matching of finite persistence intervals, allowing diagonals."""
    a=np.asarray([x for x in a if x[1] is not None],float).reshape(-1,2)
    b=np.asarray([x for x in b if x[1] is not None],float).reshape(-1,2)
    n,m=len(a),len(b)
    if n+m==0:return 0.
    cost=np.full((n+m,n+m),np.inf)
    if n and m:cost[:n,:m]=np.max(np.abs(a[:,None,:]-b[None,:,:]),axis=2)
    for i in range(n):cost[i,m+i]=(a[i,1]-a[i,0])/2
    for j in range(m):cost[n+j,j]=(b[j,1]-b[j,0])/2
    cost[n:,m:]=0
    candidates=np.unique(cost[np.isfinite(cost)])
    def feasible(threshold):
        match=[-1]*(n+m);edges=[np.flatnonzero(row<=threshold).tolist() for row in cost]
        def assign(i,seen):
            for j in edges[i]:
                if j in seen:continue
                seen.add(j)
                if match[j]<0 or assign(match[j],seen):match[j]=i;return True
            return False
        return all(assign(i,set()) for i in range(n+m))
    lo,hi=0,len(candidates)-1
    while lo<hi:
        mid=(lo+hi)//2
        if feasible(candidates[mid]):hi=mid
        else:lo=mid+1
    return float(candidates[lo])


def _distance_scale(d):
    x=d[np.triu_indices(len(d),1)];x=x[x>1e-12]
    return float(np.median(x)) if len(x) else 1.


def _ph(d,scale=None):
    return persistent_homology(d/(_distance_scale(d) if scale is None else scale),np.linspace(0,2,21).tolist())


def _correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>1e-12 and y.std()>1e-12 else None


def geometry_nulls(x,repeats,seed):
    """Same cloud for spectrum and PH. Isospectral null rotates POINT indices,
    fixing the all-ones direction; it is not a feature-space gauge rotation."""
    x=np.asarray(x,float);n,d=x.shape;yc=x-x.mean(0)
    distance=points_distance(yc);scale=_distance_scale(distance)
    observed=_ph(distance,scale);gram=yc@yc.T/max(d,1)
    spectrum=np.maximum(np.linalg.eigvalsh(gram),0)
    eig,gamma=covariance_spectrum(yc)
    # Complete orthonormal basis whose first vector is constant.
    basis=np.linalg.qr(np.column_stack([np.ones(n),np.eye(n)[:,:n-1]]))[0][:,1:]
    rng=np.random.default_rng(seed);nulls=[]
    for repeat in range(repeats):
        rot=np.linalg.qr(rng.normal(size=(n-1,n-1)))[0]
        iso=basis@rot@basis.T@yc
        shuffled=yc.reshape(-1).copy();rng.shuffle(shuffled);shuffled=shuffled.reshape(yc.shape)
        gaussian=rng.normal(size=yc.shape)*max(float(yc.std()),1e-30)
        for kind,z in [('isospectral_points',iso),('entry_shuffle',shuffled),('gaussian',gaussian)]:
            z=z-z.mean(0);zd=points_distance(z);p=_ph(zd,scale)
            ze,zgamma=covariance_spectrum(z)
            nulls.append(dict(kind=kind,repeat=repeat,H1_total=p['H1']['finite_total_persistence'],
                H1_bottleneck=bottleneck_finite(observed['H1']['intervals'],p['H1']['intervals']),
                normalized_spectral_W1=float(np.mean(np.abs(eig-ze))),
                raw_gram_spectrum_relative_error=float(np.linalg.norm(np.sort(np.linalg.eigvalsh(z@z.T/max(d,1)))-spectrum)/max(np.linalg.norm(spectrum),1e-30))))
    gaussian_w1=[v['normalized_spectral_W1'] for v in nulls if v['kind']=='gaussian']
    iso_h1=[v['H1_total'] for v in nulls if v['kind']=='isospectral_points']
    return dict(n_points=n,n_features=d,reference_distance_scale=scale,PH=observed,
        gram_eigenvalues=spectrum.tolist(),effective_rank=entropy_rank(spectrum),
        scalar_standardized_covariance_eigenvalues=eig.tolist(),aspect_ratio=gamma,
        mp_upper=(1+math.sqrt(gamma))**2,fraction_above_mp_edge=float(np.mean(eig>(1+math.sqrt(gamma))**2)),
        gaussian_spectral_W1_mean=float(np.mean(gaussian_w1)),
        isospectral_H1_min=float(min(iso_h1)),isospectral_H1_max=float(max(iso_h1)),nulls=nulls,
        caveat='MP is a descriptive asymptotic reference; centered, dependent activations violate iid assumptions. Null ranges are not calibrated significance tests. PH uses the observed cloud distance scale for every null.')


def frame_alignment(x0,x1,train_mask):
    """Fit one orthogonal map on calibration rows only, evaluate separate rows.
    The map is identity outside the joint calibration span. No task labels used."""
    x0=np.asarray(x0,float);x1=np.asarray(x1,float);train=np.asarray(train_mask,bool);test=~train
    if train.sum()<2 or test.sum()<2:return dict(available=False,reason='Need at least two calibration and evaluation observations')
    joined=np.concatenate([x0[train],x1[train]],axis=0)
    _,sv,vh=np.linalg.svd(joined,full_matrices=False)
    rank=int(np.sum(sv>max(sv[0],1e-30)*1e-10));basis=vh[:rank].T
    if rank==0:return dict(available=False,reason='Zero calibration span')
    a=x1[train]@basis;b=x0[train]@basis;u,_,vt=np.linalg.svd(a.T@b,full_matrices=False);rotation=u@vt
    z=x1[test];aligned=z+(z@basis)@(rotation-np.eye(rank))@basis.T
    den=max(np.linalg.norm(x0[test]),1e-30)
    scale=float(np.sum(x1[train]*x0[train])/max(np.sum(x1[train]**2),1e-30))
    return dict(available=True,calibration_points=int(train.sum()),evaluation_points=int(test.sum()),
        identified_span_rank=rank,ambient_dimension=x0.shape[1],
        raw_relative_error=float(np.linalg.norm(z-x0[test])/den),
        orthogonal_relative_error=float(np.linalg.norm(aligned-x0[test])/den),
        scale_only_relative_error=float(np.linalg.norm(scale*z-x0[test])/den),
        interpretation='Held-out geometric alignment, not a causal loss rescue. Underdetermined outside calibration span; no translation fitted.')


def extended_snapshot(e,s,folder,progress):
    cfg=s['extended'];folder=Path(folder);rows=[]
    if cfg['enabled']:
        for item in read(folder/'activations.json'):
            progress.update(phase='joint PH/RMT activation controls',site=item['site'])
            with np.load(folder/'activation_arrays'/filename(item['site'])) as z:x=z['y']
            ix=np.linspace(0,len(x)-1,min(len(x),s['topology_points']),dtype=int)
            if len(ix)<4:continue
            result=geometry_nulls(x[ix],cfg['null_repeats'],seed_for(item['site'],s['sampling_seed']))
            rows.append(dict(site=item['site'],matrix=item['matrix'],point_indices=ix.tolist(),**result))
        atomic_json(rows,folder/'ph_rmt_joint.json')
        write_csv([dict(site=x['site'],matrix=x['matrix'],effective_rank=x['effective_rank'],
            H1_total=x['PH']['H1']['finite_total_persistence'],gaussian_spectral_W1=x['gaussian_spectral_W1_mean'],
            isospectral_H1_min=x['isospectral_H1_min'],isospectral_H1_max=x['isospectral_H1_max']) for x in rows],folder/'ph_rmt_joint.csv')
    if cfg['ntk_enabled']:ntk_snapshot(e,s,folder,progress)


def _gradient_sketch(grad,name,dimension,seed):
    """CountSketch with unscaled +/-1 entries, one nonzero per parameter column."""
    rng=np.random.default_rng(seed_for(name,seed));out=np.zeros(dimension,dtype=np.float64)
    flat=grad.detach().reshape(-1)
    for block in flat.split(1048576):
        values=block.float().cpu().numpy().astype(float)
        indices=rng.integers(0,dimension,size=len(values));signs=2*rng.integers(0,2,size=len(values))-1
        out+=np.bincount(indices,weights=values*signs,minlength=dimension)
    return out


def _parameter_group(name):
    if name.startswith('model.layers.'):return '.'.join(name.split('.')[:3])
    return 'embeddings_and_other'


def _task_coordinates(e,row):
    inputs,_,labels=collate([row],e.tok.pad_token_id,e.c.device)
    h=e.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
    logits=e.model.lm_head(h)[0].double()
    label_ix=torch.as_tensor(labels,device=logits.device)
    chosen=logits[label_ix];mask=torch.ones_like(logits,dtype=torch.bool);mask[label_ix]=False
    margin=chosen[0]-chosen[1]
    support=torch.logsumexp(chosen,0)-torch.logsumexp(logits[mask],0)
    return torch.stack([margin,support])


def kernel_summary(K,rows):
    n=len(rows);ai=np.array([i for i,r in enumerate(rows) if r['split']=='A_test']);bi=np.array([i for i,r in enumerate(rows) if r['split']=='B_test'])
    ca=np.ravel(np.column_stack([2*ai,2*ai+1]));cb=np.ravel(np.column_stack([2*bi,2*bi+1]))
    coords=np.array([r['coordinates'] for r in rows]);sigmoid=np.exp(-np.logaddexp(0,-coords))
    errors=sigmoid-np.array([[r['q'][0],1.] for r in rows])
    ga=np.zeros(2*n);gb=ga.copy();ga[ca]=errors[ai].ravel()/len(ai);gb[cb]=errors[bi].ravel()/len(bi)
    aa=float(ga@K@ga);bb=float(gb@K@gb);ab=float(ga@K@gb)
    cross=K[np.ix_(ca,cb)];KA=K[np.ix_(ca,ca)];KB=K[np.ix_(cb,cb)]
    eig=np.linalg.eigvalsh((K+K.T)/2)
    return dict(trace=float(np.trace(K)),minimum_eigenvalue=float(eig.min()),effective_rank=entropy_rank(eig),
        cross_block_relative_frobenius=float(np.linalg.norm(cross)/max(math.sqrt(np.linalg.norm(KA)*np.linalg.norm(KB)),1e-30)),
        loss_gradient_cosine=ab/math.sqrt(max(aa*bb,1e-30)),
        SGD_first_order_A_loss_change_per_unit_lr=-ab,
        SGD_first_order_B_loss_change_per_unit_lr=-bb,
        interpretation='Finite-network empirical kernel of [answer log-odds, support log-odds], all trainable parameters. SGD infinitesimal prediction on this panel, not the actual Adam step or full-vocabulary NTK.')


def ntk_snapshot(e,s,folder,progress):
    cfg=s['extended'];e.model.to(s['device']);e.c.device=s['device'];e.model.eval()
    data=make_data(e.tok,e.c,s['panel_seed']);rows=[]
    for split in ['A_test','B_test']:
        for row in data[split][:cfg['ntk_prompts_per_domain']]:rows.append(dict(row,split=split))
    params=[(n,p) for n,p in e.model.named_parameters() if p.requires_grad]
    groups=sorted(set(_parameter_group(n) for n,p in params));gi={g:i for i,g in enumerate(groups)}
    count=sum(p.numel() for n,p in params);size=2*len(rows)
    exact=cfg['ntk_mode']=='exact' or (cfg['ntk_mode']=='auto' and count*size<=cfg['exact_max_elements'])
    if exact and count*size>cfg['exact_max_elements']:raise ValueError('Exact NTK exceeds exact_max_elements; use auto/sketch or increase the explicit memory limit')
    dim=cfg['ntk_sketch_dim'];features=np.zeros((2,len(groups),size,dim));exact_rows=[];diag=np.zeros(size)
    group_diag=np.zeros((len(groups),size));metadata=[]
    for i,row in enumerate(rows):
        coords=_task_coordinates(e,row)
        metadata.append(dict(split=row['split'],entity=row['entity'],prompt=row['prompt'],ids=row['ids'],q=row['q'],coordinates=coords.detach().cpu().tolist()))
        for channel in range(2):
            index=2*i+channel;progress.update(phase='NTK backward and projection',prompt=i+1,total_prompts=len(rows),coordinate=['answer','support'][channel])
            grads=torch.autograd.grad(coords[channel],[p for _,p in params],retain_graph=channel==0,allow_unused=True)
            full=[]
            for (name,p),grad in zip(params,grads):
                group=gi[_parameter_group(name)]
                if grad is None:
                    if exact:full.append(np.zeros(p.numel(),dtype=np.float32))
                    continue
                norm2=exact_norm2(grad);diag[index]+=norm2;group_diag[group,index]+=norm2
                for repeat in range(2):features[repeat,group,index]=features[repeat,group,index]+_gradient_sketch(grad,name,dim,cfg['ntk_seed']+repeat*1000003)
                if exact:full.append(grad.detach().reshape(-1).float().cpu().numpy().copy())
            if exact:exact_rows.append(np.concatenate(full))
            del grads,full
        del coords
    group_kernels=features@np.swapaxes(features,-1,-2);kernels=group_kernels.sum(axis=1)
    if exact:
        J=np.stack(exact_rows).astype(float);K=J@J.T;del J,exact_rows
    else:K=kernels.mean(axis=0)
    if not np.isfinite(K).all() or not np.isfinite(features).all():raise FloatingPointError('Nonfinite NTK measurements')
    disagreement=float(np.linalg.norm(kernels[0]-kernels[1])/max(np.linalg.norm(K),1e-30))
    diagerror=float(np.linalg.norm(np.diag(K)-diag)/max(np.linalg.norm(diag),1e-30))
    summaries=[dict(group=g,**kernel_summary(group_kernels[:,i].mean(axis=0),metadata)) for i,g in enumerate(groups)]
    result=dict(mode='exact' if exact else 'two_independent_CountSketch_average',parameter_count=count,
        coordinate_order='prompt-major: answer_log_odds, support_log_odds',rows=metadata,
        sketch_dimension_per_group_per_repeat=dim,groups=groups,independent_sketch_relative_disagreement=disagreement,
        exact_diagonal_relative_error=diagerror,projection_audit_pass=disagreement<=cfg['ntk_audit_tolerance'] and diagerror<=cfg['ntk_audit_tolerance'],
        approximation_warning='Two sketches and exact diagonal are diagnostics, not a rigorous off-diagonal error bound. Increase sketch dimension if audit fails.',
        summary=kernel_summary(K,metadata),group_summaries=summaries)
    if exact:result['sketch_vs_exact_relative_error']=float(np.linalg.norm(kernels.mean(axis=0)-K)/max(np.linalg.norm(K),1e-30))
    np.savez_compressed(folder/'ntk_arrays.npz',kernel=K,sketch_kernels=kernels,sketch_features=features,
                        exact_diagonal=diag,group_exact_diagonals=group_diag)
    atomic_json(result,folder/'ntk.json')
    e.model.cpu();e.c.device='cpu'


def extended_compare(s,current,reference,folder,progress):
    root=Path(s['output']);a=root/reference['id'];b=root/current['id'];folder=Path(folder);rows=[]
    if (a/'ph_rmt_joint.json').exists() and (b/'ph_rmt_joint.json').exists():
        before={x['site']:x for x in read(a/'ph_rmt_joint.json')};after=read(b/'ph_rmt_joint.json')
        panels=read(a/'probe_panel.json')
        for item in after:
            site=item['site'];old=before[site];progress.update(phase='paired PH/RMT and frame alignment',site=site)
            with np.load(a/'activation_arrays'/filename(site)) as z:x0=z['y']
            with np.load(b/'activation_arrays'/filename(site)) as z:x1=z['y']
            if x0.shape!=x1.shape:raise ValueError('Activation panels differ')
            idx=np.asarray(old['point_indices']);d0=points_distance(x0[idx]);d1=points_distance(x1[idx]);scale=_distance_scale(d0)
            p0=_ph(d0,scale);p1=_ph(d1,scale)
            split,module=site.split('::');ps=panels[split]
            counts=[1 if module=='lm_head' else min(len(p['ids']),s['activation_positions']) for p in ps]
            mask=np.repeat(np.arange(len(ps))<len(ps)//2,counts)
            if len(mask)!=len(x0):raise ValueError('Prompt/token alignment mismatch')
            align=frame_alignment(x0,x1,mask)
            rows.append(dict(site=site,matrix=item['matrix'],reference=reference['id'],checkpoint=current['id'],
                H0_bottleneck_reference_scale=bottleneck_finite(p0['H0']['intervals'],p1['H0']['intervals']),
                H1_bottleneck_reference_scale=bottleneck_finite(p0['H1']['intervals'],p1['H1']['intervals']),
                H1_total_change=p1['H1']['finite_total_persistence']-p0['H1']['finite_total_persistence'],
                effective_rank_change=item['effective_rank']-old['effective_rank'],
                gaussian_spectral_W1_change=item['gaussian_spectral_W1_mean']-old['gaussian_spectral_W1_mean'],
                heldout_alignment=align))
        atomic_json(rows,folder/'ph_rmt_frame_changes.json')
        write_csv([{k:v for k,v in x.items() if not isinstance(v,dict)} for x in rows],folder/'ph_rmt_frame_changes.csv')
    # Attention PH comparisons are available from the original head-level records.
    am={(x['split'],x['prompt'],x['layer'],x['head']):x for x in read(a/'attention.json')};att=[]
    for x in read(b/'attention.json'):
        key=(x['split'],x['prompt'],x['layer'],x['head']);old=am[key]
        for field in ['hellinger_row_topology','affinity_flag_topology']:
            if field not in x or field not in old:continue
            att.append(dict(split=key[0],prompt=key[1],layer=key[2],head=key[3],geometry=field,
                H1_bottleneck=bottleneck_finite(old[field]['H1']['intervals'],x[field]['H1']['intervals']),
                H1_total_change=x[field]['H1']['finite_total_persistence']-old[field]['H1']['finite_total_persistence']))
    atomic_json(att,folder/'attention_ph_changes.json')
    if (a/'ntk.json').exists() and (b/'ntk.json').exists():
        m0=read(a/'ntk.json');m1=read(b/'ntk.json')
        if [(r['split'],r['ids']) for r in m0['rows']]!=[(r['split'],r['ids']) for r in m1['rows']]:raise ValueError('NTK panels differ')
        with np.load(a/'ntk_arrays.npz') as z:k0=z['kernel'];f0=z['sketch_features']
        with np.load(b/'ntk_arrays.npz') as z:k1=z['kernel'];f1=z['sketch_features']
        atomic_json(dict(reference=reference['id'],checkpoint=current['id'],
            kernel_relative_change=float(np.linalg.norm(k1-k0)/max(np.linalg.norm(k0),1e-30)),
            normalized_kernel_cosine=float(np.sum(k0*k1)/max(np.linalg.norm(k0)*np.linalg.norm(k1),1e-30)),
            reference_audit_pass=m0['projection_audit_pass'],current_audit_pass=m1['projection_audit_pass'],
            reference_summary=m0['summary'],current_summary=m1['summary']),folder/'ntk_changes.json')
    # Link group kernel drift to PH/RMT changes at sites in that group.
        layer_join=[]
        for i,g in enumerate(m0['groups']):
            kg0=(f0[:,i]@np.swapaxes(f0[:,i],-1,-2)).mean(axis=0)
            kg1=(f1[:,i]@np.swapaxes(f1[:,i],-1,-2)).mean(axis=0)
            sites=[r for r in rows if _parameter_group(r['matrix'])==g]
            if sites:
                layer_join.append(dict(group=g,sites=len(sites),
                    sketched_kernel_relative_change=float(np.linalg.norm(kg1-kg0)/max(np.linalg.norm(kg0),1e-30)),
                    median_H1_bottleneck=float(np.median([r['H1_bottleneck_reference_scale'] for r in sites])),
                    median_abs_spectral_W1_change=float(np.median([abs(r['gaussian_spectral_W1_change']) for r in sites]))))
        atomic_json(dict(rows=layer_join,warning='Layer-level association only; all group kernels are sketched even when full kernel is exact.'),folder/'ph_rmt_ntk_layer_join.json')
    # Descriptive matrix-level join; tied weights can have multiple activation sites.
    if rows:
        weights={x['matrix']:x for x in read(folder/'weight_changes.json')};joined=[]
        for row in rows:
            w=weights[row['matrix']];joined.append(dict(site=row['site'],matrix=row['matrix'],
                relative_update_norm=w['relative_update_norm'],H1_bottleneck=row['H1_bottleneck_reference_scale'],
                spectral_W1_change=row['gaussian_spectral_W1_change'],
                input_span_update_energy=w['update_energy_in_reference_right_span'],
                rotation_residual=row['heldout_alignment'].get('orthogonal_relative_error')))
        atomic_json(dict(rows=joined,descriptive_only=True,
            H1_vs_abs_spectral_change=_correlation([x['H1_bottleneck'] for x in joined],[abs(x['spectral_W1_change']) for x in joined]),
            warning='Sites, layers, and heads are dependent observations, not independent experimental seeds.'),folder/'geometry_join.json')


def extended_report(root):
    root=Path(root);lines=['PH / RMT / REFERENCE FRAMES / TASK-COORDINATE NTK']
    for p in sorted(root.iterdir()):
        if not p.is_dir():continue
        if (p/'ntk.json').exists():
            x=read(p/'ntk.json');v=x['summary']
            lines.append(f"{p.name}: NTK={x['mode']}; gradient cosine={v['loss_gradient_cosine']:+.5f}; sketch disagreement={x['independent_sketch_relative_disagreement']:.3f}; audit={x['projection_audit_pass']}")
            if not x['projection_audit_pass']:lines.append('  Increase ntk_sketch_dim before interpreting small kernel differences.')
        if (p/'ph_rmt_frame_changes.json').exists():
            rows=read(p/'ph_rmt_frame_changes.json');fits=[r['heldout_alignment'] for r in rows if r['heldout_alignment']['available']]
            if fits:
                lines.append(f"{p.name}: alignment improves {sum(r['orthogonal_relative_error']<r['raw_relative_error'] for r in fits)}/{len(fits)} held-out sites; median H1 bottleneck={np.median([r['H1_bottleneck_reference_scale'] for r in rows]):.5f}")
    lines+=['NTK predictions concern infinitesimal SGD, not historical Adam. Alignment is geometric, not a causal intervention.',
            'PH is rotation invariant. A spectrum-preserving point-mixing null tests geometry beyond covariance eigenvalues.']
    (root/'extended_report.txt').write_text('\n'.join(lines)+'\n');return '\n'.join(lines)


def extended_plots(root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);snapshots=[p for p in sorted(root.iterdir()) if p.is_dir() and (p/'ntk.json').exists()]
    if snapshots:
        fig,axes=plt.subplots(1,len(snapshots),figsize=(4*len(snapshots),4),squeeze=False,constrained_layout=True)
        for ax,p in zip(axes[0],snapshots):
            with np.load(p/'ntk_arrays.npz') as z:k=z['kernel']
            norm=np.sqrt(np.maximum(np.diag(k),1e-30));k=k/norm[:,None]/norm[None,:]
            im=ax.imshow(k,vmin=-1,vmax=1,cmap='coolwarm');ax.set_title(p.name,fontsize=9);ax.set_xlabel('Prompt / output coordinate');ax.set_ylabel('Prompt / output coordinate')
        fig.colorbar(im,ax=list(axes[0]),label='Normalized task-coordinate kernel');fig.savefig(root/'ntk_overview.png',dpi=140);plt.close(fig)
    comparisons=[p for p in sorted(root.glob('compare_*')) if (p/'ph_rmt_frame_changes.json').exists()]
    if comparisons:
        fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
        for p in comparisons:
            rows=read(p/'ph_rmt_frame_changes.json');axes[0].scatter([abs(x['gaussian_spectral_W1_change']) for x in rows],[x['H1_bottleneck_reference_scale'] for x in rows],s=8,label=p.name)
            fits=[x['heldout_alignment'] for x in rows if x['heldout_alignment']['available']]
            axes[1].scatter([x['raw_relative_error'] for x in fits],[x['orthogonal_relative_error'] for x in fits],s=8)
            if (p/'ph_rmt_ntk_layer_join.json').exists():
                jr=read(p/'ph_rmt_ntk_layer_join.json')['rows']
                axes[2].scatter([x['sketched_kernel_relative_change'] for x in jr],[x['median_H1_bottleneck'] for x in jr],s=14)
        axes[0].set(xlabel='Absolute change in spectral distance to Gaussian',ylabel='H1 bottleneck (reference scale)',title='Same activation clouds: spectrum vs topology')
        axes[1].set(xlabel='Raw held-out relative difference',ylabel='After orthogonal alignment',title='Does a common frame change explain the drift?')
        lim=max(axes[1].get_xlim()[1],axes[1].get_ylim()[1]);axes[1].plot([0,lim],[0,lim],'k--',lw=.6)
        axes[2].set(xlabel='Layer kernel relative change (sketched)',ylabel='Median activation H1 bottleneck',title='Layer association: NTK vs topology')
        axes[0].legend(fontsize=5);fig.savefig(root/'ph_rmt_frame_overview.png',dpi=150);plt.close(fig)
    geometry_example_plot(root)



def geometry_example_plot(root):
    import matplotlib.pyplot as plt
    site='A_test::model.layers.0.self_attn.q_proj'
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True);found=False
    for p in sorted(Path(root).iterdir()):
        if not (p/'ph_rmt_joint.json').is_file():continue
        row=next((x for x in read(p/'ph_rmt_joint.json') if x['site']==site),None)
        if row is None:continue
        found=True
        for dim,ax in enumerate(axes[:2]):
            intervals=[v for v in row['PH']['H'+str(dim)]['intervals'] if v[1] is not None]
            ax.scatter([v[0] for v in intervals],[v[1] for v in intervals],s=18,label=p.name)
            ax.set(xlabel='Birth (cloud median scale)',ylabel='Death',title='H'+str(dim)+' finite persistence diagram')
        eig=np.sort(row['gram_eigenvalues'])[::-1];eig=eig/max(eig.sum(),1e-30)
        axes[2].plot(np.arange(1,len(eig)+1),eig,'.-',label=p.name)
    if found:
        for ax in axes[:2]:
            lim=max(ax.get_xlim()[1],ax.get_ylim()[1],1);ax.plot([0,lim],[0,lim],'k--',lw=.5)
        axes[2].set(xlabel='Eigenvalue index',ylabel='Fraction of centered Gram trace',title='Spectrum of the same point cloud')
        axes[2].legend(fontsize=6);fig.suptitle(site+' — empty H1 diagrams mean no detected finite loops')
        fig.savefig(Path(root)/'ph_rmt_example.png',dpi=150)
    plt.close(fig)

# ---------------- Task-aware trajectory geometry ----------------

def _sigmoid(x):return np.exp(-np.logaddexp(0.,-np.asarray(x,dtype=float)))


def task_topology(distance,labels):
    """Identity-preserving H0 mixing events and nearest opposite/same target gaps.
    Distances must use a fixed reference scale. Labels are only annotations."""
    d=np.asarray(distance,float);labels=np.asarray(labels);n=len(labels)
    parent=list(range(n));members={i:[i] for i in range(n)};mixed=[None]*n
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for value,i,j in sorted((float(d[i,j]),i,j) for i in range(n) for j in range(i+1,n)):
        a,b=find(i),find(j)
        if a==b:continue
        parent[b]=a;members[a]+=members.pop(b)
        group=members[a]
        if len(set(labels[group].tolist()))>1:
            for k in group:
                if mixed[k] is None:mixed[k]=value
    gaps=[]
    for i in range(n):
        same=[d[i,j] for j in range(n) if j!=i and labels[j]==labels[i]]
        other=[d[i,j] for j in range(n) if labels[j]!=labels[i]]
        gaps.append(float(min(other)-min(same)) if same and other else None)
    return dict(first_opposite_component_merge=mixed,nearest_opposite_minus_same=gaps,
                label_counts={str(k):int(sum(labels==k)) for k in set(labels.tolist())})


def trajectory_state(e,s,data,folder,progress):
    """All entities, validation and test forms. Local Jacobians hold other positions fixed."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cfg=s['trajectory'];layers=cfg['layers']
    if not layers:layers=sorted(set([0,e.model.config.num_hidden_layers//2,e.model.config.num_hidden_layers-1]))
    sites=[f'model.layers.{i}' for i in layers]+['final_hidden']
    captured={};handles=[]
    def hook(name):
        def f(module,args,out):captured[name]=out[0] if isinstance(out,tuple) else out
        return f
    for name in sites[:-1]:handles.append(e.mods[name].register_forward_hook(hook(name)))
    arrays={site:{k:[] for k in ['h','jm','jt']} for site in sites};rows=[]
    labels=data['A_test'][0]['labels'];decoder=e.model.get_output_embeddings().weight
    direction=(decoder[labels[0]]-decoder[labels[1]]).detach().double().cpu().numpy();dn=float(np.linalg.norm(direction))
    try:
        for split in ['A_valid','A_test','B_valid','B_test']:
            for row in data[split]:
                progress.update(phase='task geometry and local sensitivities',split=split,entity=row['entity'])
                captured.clear();inputs,_,lab=collate([row],e.tok.pad_token_id,e.c.device)
                result=e.model.model(**inputs,use_cache=False,return_dict=True)
                hidden=result.last_hidden_state;captured['final_hidden']=hidden
                logits=e.model.lm_head(hidden[:,-1,:])[0].double();selected=logits[lab]
                mask=torch.ones_like(logits,dtype=torch.bool);mask[lab]=False
                m=selected[0]-selected[1];t=torch.logsumexp(selected,0)-torch.logsumexp(logits[mask],0)
                targets=[captured[name] for name in sites]
                gm=torch.autograd.grad(m,targets,retain_graph=True)
                gt=torch.autograd.grad(t,targets)
                for name,x,jm,jt in zip(sites,targets,gm,gt):
                    for key,value in [('h',x),('jm',jm),('jt',jt)]:arrays[name][key].append(value[0,-1].detach().float().cpu().numpy().copy())
                mv,tv=float(m.detach()),float(t.detach());q=float(row['q'][0]);c,sp=map(float,_sigmoid([mv,tv]))
                relation=float(q*np.logaddexp(0.,-mv)+(1-q)*np.logaddexp(0.,mv)+q*math.log(q)+(1-q)*math.log(1-q))
                h=arrays['final_hidden']['h'][-1].astype(float);hn=float(np.linalg.norm(h))
                rows.append(dict(split=split,entity=row['entity'],prompt=row['prompt'],q0=q,target_class=int(q>.5),
                    answer_log_odds=mv,support_log_odds=tv,conditional_first_probability=c,support_probability=sp,
                    relation_KL=relation,support_penalty=float(np.logaddexp(0.,-tv)),KL=relation+float(np.logaddexp(0.,-tv)),
                    hidden_norm=hn,decoder_direction_norm=dn,answer_cosine=float(np.clip(mv/max(dn*hn,1e-30),-1,1))))
                del result,hidden,logits,selected,m,t,targets,gm,gt
    finally:
        for handle in handles:handle.remove()
    for site,values in arrays.items():np.savez_compressed(folder/(filename(site)),**{k:np.stack(v) for k,v in values.items()})
    np.save(folder/'decoder_direction.npy',direction)
    atomic_json(rows,folder/'examples.json');atomic_json(dict(sites=sites,rows=len(rows),all_entities=e.c.entities,
        sensitivity='Derivative of answer/support log-odds with respect to final-position output of each selected block. Other positions held fixed; not a complete attribution of upstream changes.'),folder/'metadata.json')
    return rows


def trajectory_topology(anchor,folder,progress,label_null_repeats=16):
    """Frozen A-validation sensitivity metric; no per-checkpoint renormalization."""
    anchor=Path(anchor);folder=Path(folder);ar=read(anchor/'examples.json');rows=read(folder/'examples.json')
    if [(r['split'],r['entity']) for r in rows]!=[(r['split'],r['entity']) for r in ar]:raise ValueError('Trajectory entity ordering changed')
    sites=read(anchor/'metadata.json')['sites'];out=[];ids=np.array([r['split']=='A_valid' for r in ar])
    for site in sites:
        progress.update(phase='task-aware topology',site=site)
        with np.load(anchor/filename(site)) as z:ha=z['h'].astype(float);jm=z['jm'].astype(float);jt=z['jt'].astype(float)
        with np.load(folder/filename(site)) as z:h=z['h'].astype(float)
        c=np.array([r['conditional_first_probability'] for r in ar])[ids];p=np.array([r['support_probability'] for r in ar])[ids]
        # Fisher of the coarsened [answer0, answer1, other] distribution.
        # Its local coordinate Fisher is diag(p*c*(1-c), p*(1-p)).
        F=np.concatenate([np.sqrt(p*c*(1-c))[:,None]*jm[ids],np.sqrt(p*(1-p))[:,None]*jt[ids]])/math.sqrt(sum(ids))
        for metric in ['euclidean','frozen_answer_support_Fisher']:
            x0=ha if metric=='euclidean' else ha@F.T
            x=h if metric=='euclidean' else h@F.T
            scale=_distance_scale(points_distance(x0[ids]))
            for split in ['A_valid','A_test','B_valid','B_test']:
                ix=np.array([r['split']==split for r in rows]);d=points_distance(x[ix])/scale
                target=[r['target_class'] for r in rows if r['split']==split]
                annotated=task_topology(d,target)
                rng=np.random.default_rng(seed_for(site+'::'+split,4109));controls=[]
                for repeat in range(label_null_repeats):
                    shuffled=rng.permutation(target);control=task_topology(d,shuffled)
                    valid=[v for v in control['nearest_opposite_minus_same'] if v is not None]
                    controls.append(float(np.mean(valid)) if valid else None)
                annotated['label_permutation_mean_neighbor_gaps']=controls
                annotated['label_null_note']='Target counts preserved. Same permutation seed across states. Descriptive controls; no significance claim.'
                # All 32 source entities fit; no unlabelled token subsampling.
                ph=persistent_homology(d,np.linspace(0,3,31).tolist())
                out.append(dict(site=site,split=split,metric=metric,reference_scale=scale,
                    entities=[r['entity'] for r in rows if r['split']==split],PH=ph,**annotated))
            # Same entity in validation and test form, compared with other entities.
            for domain in ['A','B']:
                vi=np.flatnonzero([r['split']==domain+'_valid' for r in rows]);ti=np.flatnonzero([r['split']==domain+'_test' for r in rows])
                pair=np.linalg.norm(x[vi,None,:]-x[ti][None,:,:],axis=2)/scale
                for entry in out:
                    if entry['site']==site and entry['metric']==metric and entry['split']==domain+'_test':
                        entry['same_entity_across_prompt_distance']=np.diag(pair).tolist()
                        entry['nearest_validation_entity']= [rows[vi[j]]['entity'] for j in np.argmin(pair,axis=0)]
    atomic_json(out,folder/'task_topology.json')


def _mean_task_gradient(e,rows):
    """Exact full-panel mean loss gradient, with NO clipping or optimizer update."""
    e.opt.zero_grad(set_to_none=True)
    for i in range(0,len(rows),e.c.microbatch):
        batch=rows[i:i+e.c.microbatch];(e.loss_rows(batch).sum()/len(rows)).backward()
    result={name:(p.grad.detach().cpu().clone() if p.grad is not None else None) for name,p in e.params.items()}
    e.opt.zero_grad(set_to_none=True);return result


def _dot_cpu(a,b):return float(np.dot(np.asarray(a,dtype=np.float64).reshape(-1),np.asarray(b,dtype=np.float64).reshape(-1)))


def measured_adam_step(e,data,seed,step,progress):
    """Measure the actual displacement and native Adam numerator channels.
    Host memory: old weights + full A gradient; no per-entity full Jacobians."""
    progress.update(phase='full A-test gradient before actual update',step=step)
    ga=_mean_task_gradient(e,data['A_test'])
    before={name:p.detach().cpu().clone() for name,p in e.params.items()}
    audit=e.gradient(minibatches(data,'B',seed,step,e.c));e.opt.step()
    groups={};after={};delta={}
    group_options={id(p):group for group in e.opt.param_groups for p in group['params']}
    for name,p in e.params.items():
        group=_parameter_group(name);r=groups.setdefault(group,dict(A_gradient_norm2=0.,actual_norm2=0.,actual_A_linear=0.,
            history_norm2=0.,current_norm2=0.,history_A_linear=0.,current_A_linear=0.,channel_reconstruction_norm2=0.))
        after[name]=p.detach().cpu().clone();delta[name]=after[name]-before.pop(name)
        a=ga.pop(name)
        if a is None:continue
        r['A_gradient_norm2']+=exact_norm2(a);r['actual_norm2']+=exact_norm2(delta[name]);r['actual_A_linear']+=_dot_cpu(a,delta[name])
        if p.grad is None:continue
        options=group_options[id(p)]
        if options.get('weight_decay',0)!=0 or options.get('amsgrad',False):raise ValueError('Native-channel accounting expects original Adam without weight decay/AMSGrad')
        b1,b2=options['betas'];state=e.opt.state[p];t=int(state['step'].item());lr=options['lr'];eps=options['eps']
        # Stream chunks; calculate from updated state and actual clipped gradient.
        flat=[v.reshape(-1) for v in [state['exp_avg'],state['exp_avg_sq'],p.grad]]
        ac=a.reshape(-1);dc=delta[name].reshape(-1)
        for start in range(0,p.numel(),1048576):
            stop=start+1048576;m,v,g=[z[start:stop].detach().double().cpu().numpy() for z in flat]
            denom=np.sqrt(v/(1-b2**t))+eps
            current=-lr*((1-b1)*g)/(1-b1**t)/denom
            history=-lr*(m-(1-b1)*g)/(1-b1**t)/denom
            av=ac[start:stop].numpy().astype(float);dv=dc[start:stop].numpy().astype(float)
            r['current_norm2']+=_dot_cpu(current,current);r['history_norm2']+=_dot_cpu(history,history)
            r['current_A_linear']+=_dot_cpu(av,current);r['history_A_linear']+=_dot_cpu(av,history)
            r['channel_reconstruction_norm2']+=_dot_cpu(dv-history-current,dv-history-current)
    del before,ga
    e.opt.zero_grad(set_to_none=True)
    totals={k:sum(r[k] for r in groups.values()) for k in next(iter(groups.values()))}
    totals['actual_update_A_gradient_cosine']=totals['actual_A_linear']/math.sqrt(max(totals['A_gradient_norm2']*totals['actual_norm2'],1e-30))
    totals['channel_reconstruction_relative_error']=math.sqrt(totals['channel_reconstruction_norm2']/max(totals['actual_norm2'],1e-30))
    return dict(step=step,totals=totals,groups=groups,B_gradient_raw_norm=audit['raw_norm'],B_gradient_clip=audit['clip'],
                meaning='Positive A_linear predicts harm to full A-test mean. Native channels are not norm-matched. Reconstruction includes floating-point update-rounding error.'),after,delta


def directional_loss_audit(e,data,after,delta,epsilons,progress):
    """Per-entity symmetric finite differences around BEFORE using the actual update.
    Restore every after-weight bit-for-bit, including on failure. Optimizer state untouched."""
    rows=data['A_test']+data['B_test'];estimates=[]
    try:
        for eps in epsilons:
            values=[]
            for sign in [1,-1]:
                progress.update(phase='per-entity actual-update directional audit',epsilon=eps,sign=sign)
                with torch.no_grad():
                    for name,p in e.params.items():p.copy_((after[name].double()+(sign*eps-1)*delta[name].double()).to(device=p.device,dtype=p.dtype))
                values.append(np.array(e.evaluate(rows)))
            estimates.append((values[0]-values[1])/(2*eps))
    finally:
        with torch.no_grad():
            for name,p in e.params.items():p.copy_(after[name].to(p.device))
    return np.stack(estimates)


def trajectory_compare(before,after,folder,derivatives,epsilons,update):
    before=Path(before);after=Path(after);folder=Path(folder);a=read(before/'examples.json');b=read(after/'examples.json');sites=read(before/'metadata.json')['sites']
    ta={(r['site'],r['split'],r['metric']):r for r in read(before/'task_topology.json')}
    tb={(r['site'],r['split'],r['metric']):r for r in read(after/'task_topology.json')}
    d0=np.load(before/'decoder_direction.npy');d1=np.load(after/'decoder_direction.npy');rows=[]
    for site in sites:
        with np.load(before/filename(site)) as z:h0=z['h'].astype(float);jm0=z['jm'].astype(float);jt0=z['jt'].astype(float)
        with np.load(after/filename(site)) as z:h1=z['h'].astype(float);jm1=z['jm'].astype(float);jt1=z['jt'].astype(float)
        for i,(old,new) in enumerate(zip(a,b)):
            dh=h1[i]-h0[i];c=old['conditional_first_probability'];p=old['support_probability'];q=old['q0']
            mm=float(jm0[i]@dh);tt=float(jt0[i]@dh)
            item=dict(site=site,split=old['split'],entity=old['entity'],target_class=old['target_class'],
                loss_before=old['KL'],loss_after=new['KL'],loss_change=new['KL']-old['KL'],
                relation_loss_change=new['relation_KL']-old['relation_KL'],support_loss_change=new['support_penalty']-old['support_penalty'],
                representation_relative_change=float(np.linalg.norm(dh)/max(np.linalg.norm(h0[i]),1e-30)),
                local_old_loss_linear=(c-q)*mm+(p-1)*tt,
                local_coarse_Fisher_squared_length=p*c*(1-c)*mm**2+p*(1-p)*tt**2,
                answer_sensitivity_cosine=float(jm0[i]@jm1[i]/max(np.linalg.norm(jm0[i])*np.linalg.norm(jm1[i]),1e-30)),
                answer_sensitivity_norm_ratio=float(np.linalg.norm(jm1[i])/max(np.linalg.norm(jm0[i]),1e-30)))
            for metric in ['euclidean','frozen_answer_support_Fisher']:
                t0=ta[(site,old['split'],metric)];t1=tb[(site,old['split'],metric)];j=t0['entities'].index(old['entity'])
                for field in ['first_opposite_component_merge','nearest_opposite_minus_same']:
                    v0,v1=t0[field][j],t1[field][j]
                    item[metric+'_'+field+'_before']=v0
                    item[metric+'_'+field+'_change']=v1-v0 if v0 is not None and v1 is not None else None
            if site=='final_hidden':
                item.update(answer_log_odds_change=new['answer_log_odds']-old['answer_log_odds'],
                    hidden_angle_change_degrees=math.degrees(math.acos(new['answer_cosine'])-math.acos(old['answer_cosine'])),
                    margin_representation_term=float(d0@dh),margin_decoder_term=float((d1-d0)@h0[i]),margin_interaction_term=float((d1-d0)@dh))
                predicted=item['margin_representation_term']+item['margin_decoder_term']+item['margin_interaction_term']
                item['margin_decomposition_absolute_error']=abs(predicted-item['answer_log_odds_change'])
            if old['split'] in ['A_test','B_test']:
                n=sum(x['split']=='A_test' for x in a);index=old['entity']+(n if old['split']=='B_test' else 0)
                item['actual_update_loss_derivative_eps1']=float(derivatives[0,index]);item['actual_update_loss_derivative_eps2']=float(derivatives[1,index])
                item['finite_difference_absolute_disagreement']=float(abs(derivatives[0,index]-derivatives[1,index]))
            rows.append(item)
    atomic_json(rows,folder/'example_geometry_changes.json');write_csv(rows,folder/'example_geometry_changes.csv')
    ph=[]
    for key,old in ta.items():
        new=tb[key];ph.append(dict(site=key[0],split=key[1],metric=key[2],
            H0_bottleneck=bottleneck_finite(old['PH']['H0']['intervals'],new['PH']['H0']['intervals']),
            H1_bottleneck=bottleneck_finite(old['PH']['H1']['intervals'],new['PH']['H1']['intervals'])))
    atomic_json(ph,folder/'task_ph_changes.json')
    n=sum(x['split']=='A_test' for x in a);fd=np.mean(derivatives[:,:n],axis=1);exact=update['totals']['actual_A_linear']
    update['directional_audit']=dict(epsilons=epsilons,FD_A_mean=fd.tolist(),exact_A_mean=exact,
        absolute_error_vs_exact=np.abs(fd-exact).tolist(),
        relative_error_vs_exact=(np.abs(fd-exact)/max(abs(exact),1e-8)).tolist(),
        interpretation='Finite differences can be unreliable at tiny FP32 steps; use two spacings and the exact full-panel gradient check.')
    update['observed_A_change']=float(np.mean([y['KL']-x['KL'] for x,y in zip(a,b) if x['split']=='A_test']))
    update['observed_B_change']=float(np.mean([y['KL']-x['KL'] for x,y in zip(a,b) if x['split']=='B_test']))
    update['A_nonlinear_remainder']=update['observed_A_change']-exact
    if not all(math.isfinite(v) for v in update['totals'].values()):raise FloatingPointError('Nonfinite actual update geometry')
    atomic_json(update,folder/'actual_update_geometry.json')


def run_trajectory(s,seed,folder,progress):
    cfg=s['trajectory'];steps=sorted(set(cfg['steps']));folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    c=Config(**read(Path(s['source'])/'config.json'));c.device=s['device'];c.threads=s['threads'];e=Engine(c)
    data=make_data(e.tok,c,seed);src=Path(s['source'])/f'seed{seed}'
    if digest(data)!=read(src/'data.json')['hash']:raise ValueError('Source data hash mismatch')
    anchor=folder/'A_anchor'
    if not (anchor/'task_topology.json').exists():
        z=load(src/'anchor.pt');e.model.load_state_dict(z['model']);del z
        trajectory_state(e,s,data,anchor,progress);trajectory_topology(anchor,anchor,progress,cfg['label_null_repeats'])
    forks=[p for p in src.glob('fork*/fork.pt') if int(p.parent.name[4:])<=steps[0]]
    if not forks:raise ValueError('No saved B fork before the first requested step')
    file=max(forks,key=lambda p:int(p.parent.name[4:]));z=load(file);start=int(z['step'])
    if start!=int(file.parent.name[4:]):raise ValueError('Fork step mismatch')
    e.restore(z['engine']);del z
    atomic_json(dict(seed=seed,steps=steps,source_fork=str(file),all_entities=c.entities,
        calibration='All A_valid entities define the frozen metric. Test prompts are never used to fit it.',
        finite_difference_epsilons=cfg['fd_epsilons'],scope='Exact original Adam replay; measurements do not update model state.'),folder/'trajectory_metadata.json')
    for step in range(start,steps[-1]+1):
        progress.update(phase='trajectory replay',step=step)
        target=folder/f'update{step:03d}'
        if step not in steps or (target/'done.json').exists():
            e.gradient(minibatches(data,'B',seed,step,c));e.opt.step();e.opt.zero_grad(set_to_none=True);continue
        target.mkdir(exist_ok=True);pre=folder/f'state{step-1:03d}';post=folder/f'state{step:03d}'
        if not (pre/'task_topology.json').exists():
            trajectory_state(e,s,data,pre,progress);trajectory_topology(anchor,pre,progress,cfg['label_null_repeats'])
        update,after,delta=measured_adam_step(e,data,seed,step,progress)
        derivatives=directional_loss_audit(e,data,after,delta,cfg['fd_epsilons'],progress)
        if s['path']['enabled']:analyze_update_path(e,s,data,after,delta,anchor,target,progress)
        del after,delta
        trajectory_state(e,s,data,post,progress);trajectory_topology(anchor,post,progress,cfg['label_null_repeats'])
        trajectory_compare(pre,post,target,derivatives,cfg['fd_epsilons'],update)
        atomic_json(dict(complete=True),target/'done.json');trajectory_report(s['output']);trajectory_plots(s['output'])
    e.opt.zero_grad(set_to_none=True)


def trajectory_report(output):
    root=Path(output);lines=['TASK-AWARE TRAJECTORY — all entities, actual Adam updates'];summaries=[]
    for folder in sorted(root.glob('trajectory_seed*')):
        for p in sorted(folder.glob('update*/actual_update_geometry.json')):
            x=read(p);t=x['totals'];r=dict(seed=int(folder.name.replace('trajectory_seed','')),step=x['step'],
                A_change=x['observed_A_change'],B_change=x['observed_B_change'],A_linear=t['actual_A_linear'],
                A_nonlinear_remainder=x['A_nonlinear_remainder'],history_A_linear=t['history_A_linear'],current_A_linear=t['current_A_linear'],
                update_A_gradient_cosine=t['actual_update_A_gradient_cosine'],channel_error=t['channel_reconstruction_relative_error'])
            summaries.append(r);lines.append(f"seed {r['seed']} step {r['step']}: A {r['A_change']:+.6f}, B {r['B_change']:+.6f}; A linear {r['A_linear']:+.6f}, remainder {r['A_nonlinear_remainder']:+.6f}; history {r['history_A_linear']:+.6f}, current {r['current_A_linear']:+.6f}")
    lines += ['History/current are native Adam numerator channels under the same updated denominator; their norms are not equalized.',
              'Local hidden sensitivities hold other token positions fixed. Frozen coarse Fisher is a semimetric, not full-vocabulary Fisher or the loss Hessian.',
              'Topology records preserve entity identities and target annotations. These are descriptive observations, not a demonstrated cause.']
    trajectory_lagged_table(root)
    text='\n'.join(lines)+'\n';(root/'trajectory_report.txt').write_text(text);atomic_json(summaries,root/'trajectory_summary.json');write_csv(summaries,root/'trajectory_summary.csv');return text


def trajectory_plots(output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output);rows=read(root/'trajectory_summary.json') if (root/'trajectory_summary.json').exists() else []
    if not rows:return
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for seed in sorted(set(r['seed'] for r in rows)):
        rs=[r for r in rows if r['seed']==seed];x=np.arange(len(rs));labels=[str(r['step']) for r in rs]
        ax[0].plot(x,[r['A_change'] for r in rs],'o-',label=f'seed {seed}: observed A')
        ax[0].plot(x,[r['A_linear'] for r in rs],'x--',label=f'seed {seed}: linear prediction')
        ax[1].plot(x,[r['history_A_linear'] for r in rs],'o-',label=f'seed {seed}: history')
        ax[1].plot(x,[r['current_A_linear'] for r in rs],'x-',label=f'seed {seed}: current gradient')
        for a in ax:a.set_xticks(x,labels);a.axhline(0,color='black',lw=.5);a.set_xlabel('Measured B update');a.legend(fontsize=7)
    ax[0].set_ylabel('Full A-test mean loss change');ax[1].set_ylabel('Contribution to first-order A loss change')
    fig.savefig(root/'trajectory_updates.png',dpi=150);plt.close(fig)
    files=sorted(root.glob('trajectory_seed*/update*/example_geometry_changes.json'))
    if files:
        fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for file in files:
            rs=[r for r in read(file) if r['split']=='A_test' and r['site']=='final_hidden']
            ax[0].scatter([r['local_old_loss_linear'] for r in rs],[r['loss_change'] for r in rs],s=14,label=file.parent.name)
            pairs=[r for r in rs if r['frozen_answer_support_Fisher_nearest_opposite_minus_same_change'] is not None]
            ax[1].scatter([r['frozen_answer_support_Fisher_nearest_opposite_minus_same_change'] for r in pairs],[r['loss_change'] for r in pairs],s=14)
        ax[0].set(xlabel='Old loss-gradient dot final-hidden displacement',ylabel='Actual per-entity loss change',title='Task-sensitive motion (local, not full attribution)')
        ax[1].set(xlabel='Change in opposite-minus-same neighbor distance',ylabel='Actual per-entity loss change',title='Fixed-metric task separation vs loss')
        from matplotlib.ticker import MaxNLocator
        for a in ax:
            a.xaxis.set_major_locator(MaxNLocator(4))
            a.ticklabel_format(style='sci',axis='both',scilimits=(-3,3),useOffset=False)
        if not any(len(collection.get_offsets()) for collection in ax[1].collections):
            ax[1].text(.5,.5,'No evaluable same/opposite-label neighbors\nin this panel',ha='center',va='center',transform=ax[1].transAxes)
        ax[0].legend(fontsize=6);fig.savefig(root/'trajectory_examples.png',dpi=150);plt.close(fig)



def trajectory_lagged_table(root):
    """Join features observed at t to loss changes at t+1, no fitted predictor."""
    root=Path(root);rows=[]
    for folder in sorted(root.glob('trajectory_seed*')):
        paths={int(p.parent.name[6:]):p for p in folder.glob('update*/example_geometry_changes.json')}
        for t in sorted(paths):
            if t+1 not in paths:continue
            before={(r['site'],r['split'],r['entity']):r for r in read(paths[t])}
            for new in read(paths[t+1]):
                if new['split']!='A_test':continue
                old=before[(new['site'],new['split'],new['entity'])]
                rows.append(dict(seed=int(folder.name.replace('trajectory_seed','')),feature_step=t,target_step=t+1,
                    site=new['site'],entity=new['entity'],target_class=new['target_class'],
                    known_loss=old['loss_after'],previous_loss_change=old['loss_change'],next_loss_change=new['loss_change'],
                    previous_answer_sensitivity_cosine=old['answer_sensitivity_cosine'],
                    previous_functional_squared_motion=old['local_coarse_Fisher_squared_length'],
                    previous_separation_change=old['frozen_answer_support_Fisher_nearest_opposite_minus_same_change']))
    atomic_json(dict(rows=rows,interpretation='Prospective temporal join only. No model fitted and no held-out predictive validity claimed; prompt sites and entities are dependent.'),root/'lagged_example_predictions.json')
    write_csv(rows,root/'lagged_example_predictions.csv')

# ---------------- Within-update path analysis ----------------

def path_set_weights(e,after,delta,alpha):
    """Absolute assignment from saved tensors; never cumulative interpolation."""
    with torch.no_grad():
        for name,p in e.params.items():
            if alpha==1:p.copy_(after[name].to(p.device))
            else:p.copy_((after[name].double()+(float(alpha)-1)*delta[name].double()).to(device=p.device,dtype=p.dtype))


def path_exact_slope(e,rows,delta):
    """Exact full-panel gradient dot fixed displacement, streamed without a saved Jacobian."""
    e.opt.zero_grad(set_to_none=True)
    try:
        for i in range(0,len(rows),e.c.microbatch):
            (e.loss_rows(rows[i:i+e.c.microbatch]).sum()/len(rows)).backward()
        total=0.
        for name,p in e.params.items():
            if p.grad is None:continue
            gf=p.grad.detach().reshape(-1);df=delta[name].reshape(-1)
            for start in range(0,p.numel(),1048576):
                total+=_dot_cpu(gf[start:start+1048576].float().cpu().numpy(),df[start:start+1048576].numpy())
        if not math.isfinite(total):raise FloatingPointError('Nonfinite exact path slope')
        return total
    finally:e.opt.zero_grad(set_to_none=True)


def reversal_intervals(nodes,tolerance):
    keys=sorted(nodes)
    return [(a,b) for a,b in zip(keys,keys[1:]) if b-a>tolerance and nodes[a]['A_slope']*nodes[b]['A_slope']<0]


def adaptive_path_nodes(evaluate,grid,max_refinements,tolerance):
    nodes={float(x):evaluate(float(x)) for x in grid}
    for _ in range(max_refinements):
        brackets=reversal_intervals(nodes,tolerance)
        if not brackets:break
        a,b=max(brackets,key=lambda v:v[1]-v[0]);mid=(a+b)/2;nodes[mid]=evaluate(mid)
    return nodes


@torch.no_grad()
def path_attention(e,s,data,folder,progress):
    """Reconstruct last-query attention from Q/K/V without changing the training backend.
    Audit P@V against the actual o_proj input. Fixed first-key reference avoids anchor switching."""
    folder=Path(folder);layers=s['trajectory']['layers'] or sorted(set([0,e.model.config.num_hidden_layers//2,e.model.config.num_hidden_layers-1]))
    captures={};handles=[];out=[]
    def hook(key,is_input=False):
        def f(module,args,y):captures[key]=(args[0] if is_input else y)[0].detach().float().cpu().numpy().copy()
        return f
    for layer in layers:
        name=f'model.layers.{layer}.self_attn'
        for kind in ['q','k','v']:handles.append(e.mods[name+'.'+kind+'_proj'].register_forward_hook(hook((layer,kind))))
        handles.append(e.mods[name+'.o_proj'].register_forward_hook(hook((layer,'o_input'),True)))
    try:
        for split in ['A_test','B_test']:
            for row in data[split]:
                progress.update(phase='path attention audit',split=split,entity=row['entity'])
                captures.clear();inputs,_,_=collate([row],e.tok.pad_token_id,e.c.device)
                result=e.model.model(**inputs,use_cache=False,return_dict=True)
                for layer in layers:
                    q,k,v=[captures[(layer,kind)].astype(float) for kind in ['q','k','v']]
                    length=len(q);heads=e.model.config.num_attention_heads;kv=e.model.config.num_key_value_heads;dim=q.shape[-1]//heads
                    clip=getattr(e.model.config,'clip_qkv',None)
                    if clip is not None:q,k,v=[np.clip(x,-clip,clip) for x in [q,k,v]]
                    q=q.reshape(length,heads,dim).transpose(1,0,2)
                    k=np.repeat(k.reshape(length,kv,dim).transpose(1,0,2),heads//kv,axis=0)
                    v=np.repeat(v.reshape(length,kv,dim).transpose(1,0,2),heads//kv,axis=0)
                    module=e.mods[f'model.layers.{layer}.self_attn']
                    rotary=getattr(e.model.model,'rotary_emb',None) or getattr(module,'rotary_emb',None)
                    inv=rotary.inv_freq.detach().float().cpu().numpy() if rotary is not None else 1/(e.model.config.rope_theta**(np.arange(0,dim,2)/dim))
                    phase=np.arange(length)[:,None]*inv[None,:];cos=np.cos(phase)[None];sin=np.sin(phase)[None];half=dim//2
                    def rotate(x):return np.concatenate([x[:,:,:half]*cos-x[:,:,half:]*sin,x[:,:,:half]*sin+x[:,:,half:]*cos],axis=-1)
                    qr,kr=rotate(q),rotate(k);scores=qr@kr.transpose(0,2,1)/math.sqrt(dim)
                    scores=np.where(np.tri(length,dtype=bool),scores,-np.inf);weights=np.exp(scores-scores.max(-1,keepdims=True));weights/=weights.sum(-1,keepdims=True)
                    reconstructed=(weights@v).transpose(1,0,2).reshape(length,-1);actual=captures[(layer,'o_input')]
                    error=float(np.linalg.norm(reconstructed-actual)/max(np.linalg.norm(actual),1e-30));passed=error<s['path']['attention_audit_tolerance']
                    head_rows=[]
                    if passed:
                        for head in range(heads):
                            qq,kk=qr[head,-1],kr[head,0];qkc=float(qq@kk/max(np.linalg.norm(qq)*np.linalg.norm(kk),1e-30))
                            cross=(qq[:half]+1j*qq[half:])*np.conj(kk[:half]+1j*kk[half:]);resultant=cross.sum()/max(np.abs(cross).sum(),1e-30)
                            prob=weights[head,-1]
                            head_rows.append(dict(head=head,probabilities=prob.tolist(),first_key_cosine=qkc,
                                entropy=float(-np.sum(prob*np.log(np.maximum(prob,1e-300)))),
                                relative_phase=float(np.angle(resultant)) if abs(resultant)>1e-6 else None,resultant_length=float(abs(resultant))))
                    out.append(dict(split=split,entity=row['entity'],layer=layer,audit_pass=passed,
                        value_output_relative_error=error,heads=head_rows,reference_key_position=0,
                        interpretation='Reconstructed causal attention; P@V audited against actual o_proj input. Phase is spatial, not temporal optimizer lag.'))
                del result
    finally:
        for h in handles:h.remove()
    atomic_json(out,folder/'path_attention.json')


def path_geometry_features(base,current,alpha):
    base=Path(base);current=Path(current);a=read(base/'examples.json');b=read(current/'examples.json');out=[]
    ta={(x['site'],x['split'],x['metric']):x for x in read(base/'task_topology.json')}
    tb={(x['site'],x['split'],x['metric']):x for x in read(current/'task_topology.json')}
    att0={(x['split'],x['entity'],x['layer']):x for x in read(base/'path_attention.json')} if (base/'path_attention.json').exists() else {}
    att1={(x['split'],x['entity'],x['layer']):x for x in read(current/'path_attention.json')} if (current/'path_attention.json').exists() else {}
    d0=np.load(base/'decoder_direction.npy');d1=np.load(current/'decoder_direction.npy')
    for site in read(base/'metadata.json')['sites']:
        with np.load(base/filename(site)) as z:h0=z['h'].astype(float);j0=z['jm'].astype(float);t0=z['jt'].astype(float)
        with np.load(current/filename(site)) as z:h1=z['h'].astype(float);j1=z['jm'].astype(float)
        for i,(old,new) in enumerate(zip(a,b)):
            if old['split'] not in ['A_test','B_test']:continue
            dh=h1[i]-h0[i];c=old['conditional_first_probability'];sp=old['support_probability'];dm=float(j0[i]@dh);dt=float(t0[i]@dh)
            r=dict(alpha=alpha,site=site,split=old['split'],entity=old['entity'],target_class=old['target_class'],
                baseline_loss=old['KL'],loss=new['KL'],loss_change=new['KL']-old['KL'],baseline_margin=old['answer_log_odds'],
                margin_change=new['answer_log_odds']-old['answer_log_odds'],
                raw_relative_displacement=float(np.linalg.norm(dh)/max(np.linalg.norm(h0[i]),1e-30)),
                old_loss_projection=(c-old['q0'])*dm+(sp-1)*dt,
                old_local_Fisher_length2=sp*c*(1-c)*dm*dm+sp*(1-sp)*dt*dt,
                answer_sensitivity_cosine=float(np.clip(j0[i]@j1[i]/max(np.linalg.norm(j0[i])*np.linalg.norm(j1[i]),1e-30),-1,1)))
            for metric in ['euclidean','frozen_answer_support_Fisher']:
                x,y=ta[(site,old['split'],metric)],tb[(site,old['split'],metric)];idx=x['entities'].index(old['entity'])
                for field in ['first_opposite_component_merge','nearest_opposite_minus_same','same_entity_across_prompt_distance']:
                    if field not in x:continue
                    u,v=x[field][idx],y[field][idx];r[metric+'_'+field+'_change']=v-u if u is not None and v is not None else None
            if site=='final_hidden':
                r.update(hidden_norm_ratio=new['hidden_norm']/max(old['hidden_norm'],1e-30),
                    decoder_norm_ratio=new['decoder_direction_norm']/max(old['decoder_direction_norm'],1e-30),
                    answer_angle_change_degrees=math.degrees(math.acos(new['answer_cosine'])-math.acos(old['answer_cosine'])),
                    margin_representation_term=float(d0@dh),margin_decoder_term=float((d1-d0)@h0[i]),margin_interaction_term=float((d1-d0)@dh))
            else:
                layer=int(site.split('.')[-1]);key=(old['split'],old['entity'],layer)
                if key in att0 and key in att1 and att0[key]['audit_pass'] and att1[key]['audit_pass']:
                    js=[];angles=[];phases=[]
                    for x,y in zip(att0[key]['heads'],att1[key]['heads']):
                        p,q=np.array(x['probabilities']),np.array(y['probabilities']);m=(p+q)/2
                        js.append(float(.5*np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300)))+.5*np.sum(q*np.log(np.maximum(q,1e-300)/np.maximum(m,1e-300)))))
                        angles.append(abs(y['first_key_cosine']-x['first_key_cosine']))
                        if x['relative_phase'] is not None and y['relative_phase'] is not None:
                            phases.append(abs(float(np.angle(np.exp(1j*(y['relative_phase']-x['relative_phase']))))))
                    r.update(attention_JS_mean=float(np.mean(js)),first_key_cosine_abs_change=float(np.mean(angles)),
                             phase_abs_change_mean=float(np.mean(phases)) if phases else None)
            out.append(r)
    atomic_json(out,current/'path_geometry_features.json');write_csv(out,current/'path_geometry_features.csv')
    ph=[]
    for key,x in ta.items():
        y=tb[key];ph.append(dict(alpha=alpha,site=key[0],split=key[1],metric=key[2],
            H0_bottleneck=bottleneck_finite(x['PH']['H0']['intervals'],y['PH']['H0']['intervals']),
            H1_bottleneck=bottleneck_finite(x['PH']['H1']['intervals'],y['PH']['H1']['intervals'])))
    atomic_json(ph,current/'path_PH_changes.json')


def analyze_update_path(e,s,data,after,delta,anchor,folder,progress):
    cfg=s['path'];folder=Path(folder)/'path';folder.mkdir(exist_ok=True);curves=folder/'curve_points';curves.mkdir(exist_ok=True)
    n=len(data['A_test']);panel=data['A_test']+data['B_test'];epsilons=cfg['fd_epsilons']
    def point(alpha):
        file=curves/(float(alpha).hex().replace('+','p').replace('-','m')+'.json')
        if file.exists():return read(file)
        progress.update(phase='within-update loss path',alpha=alpha)
        path_set_weights(e,after,delta,alpha);loss=np.asarray(e.evaluate(panel));slope=path_exact_slope(e,data['A_test'],delta)
        derivatives=[]
        for eps in epsilons:
            path_set_weights(e,after,delta,alpha+eps);plus=np.asarray(e.evaluate(panel))
            path_set_weights(e,after,delta,alpha-eps);minus=np.asarray(e.evaluate(panel))
            derivatives.append((plus-minus)/(2*eps))
        fd=np.stack(derivatives);means=fd[:,:n].mean(axis=1)
        result=dict(alpha=alpha,A_loss=float(loss[:n].mean()),B_loss=float(loss[n:].mean()),A_slope=slope,
            FD_A_slopes=means.tolist(),FD_absolute_errors=np.abs(means-slope).tolist(),
            FD_audit_pass=bool(np.all(np.abs(means-slope)<=cfg['fd_atol']+cfg['fd_rtol']*abs(slope))),
            examples=[dict(split='A_test' if i<n else 'B_test',entity=panel[i]['entity'],loss=float(loss[i]),
                           FD_slopes=fd[:,i].tolist(),FD_spacing_disagreement=float(abs(fd[0,i]-fd[1,i]))) for i in range(len(panel))])
        if not all(math.isfinite(result[k]) for k in ['A_loss','B_loss','A_slope']):raise FloatingPointError('Nonfinite path result')
        atomic_json(result,file);return result
    try:
        nodes=adaptive_path_nodes(point,cfg['alphas'],cfg['max_refinements'],cfg['alpha_tolerance'])
        ordered=sorted(nodes);atomic_json([nodes[x] for x in ordered],folder/'loss_curve.json')
        write_csv([{k:v for k,v in nodes[x].items() if not isinstance(v,list)} for x in ordered],folder/'loss_curve.csv')
        # Independently measure geometry at every sampled path location.
        geometry=[]
        for i,alpha in enumerate(ordered):
            p=folder/f'point{i:03d}';geometry.append((alpha,p))
            if (p/'path_geometry_features.json').exists():continue
            path_set_weights(e,after,delta,alpha)
            trajectory_state(e,s,data,p,progress)
            trajectory_topology(anchor,p,progress,s['trajectory']['label_null_repeats'])
            if cfg['attention']:path_attention(e,s,data,p,progress)
            path_geometry_features(geometry[0][1],p,alpha)
            atomic_json(dict(alpha=alpha),p/'path_position.json')
        brackets=[dict(left=a,right=b,left_slope=nodes[a]['A_slope'],right_slope=nodes[b]['A_slope']) for a,b in zip(ordered,ordered[1:]) if nodes[a]['A_slope']*nodes[b]['A_slope']<0]
        summary=dict(points=len(nodes),slope_reversal_brackets=brackets,sampled_A_minimum_alpha=min(nodes,key=lambda x:nodes[x]['A_loss']),
            A_endpoint_change=nodes[1.]['A_loss']-nodes[0.]['A_loss'],B_endpoint_change=nodes[1.]['B_loss']-nodes[0.]['B_loss'],
            A_initial_slope=nodes[0.]['A_slope'],A_final_slope=nodes[1.]['A_slope'],FD_audits_passed=sum(x['FD_audit_pass'] for x in nodes.values()),
            meaning='Fractions of ONE fixed actual Adam update, not separate training steps. Refinement searches observed sign changes; unsampled reversals may be missed. FD probes extend slightly outside [0,1] only for slope audits.')
        atomic_json(summary,folder/'path_summary.json')
        # Keep future labels separate from geometry to avoid pretending a fitted predictor exists.
        final={(x['split'],x['entity']):x['loss'] for x in nodes[1.]['examples']};initial={(x['split'],x['entity']):x['loss'] for x in nodes[0.]['examples']}
        features=[];prospective=[]
        for i,(alpha,p) in enumerate(geometry):
            rows=read(p/'path_geometry_features.json')
            for row in rows:
                key=(row['split'],row['entity']);row['endpoint_loss_change']=final[key]-initial[key];features.append(row)
                if i+1<len(geometry):
                    nxt=geometry[i+1][0];future={(x['split'],x['entity']):x['loss'] for x in nodes[nxt]['examples']}
                    prospective.append(dict({k:v for k,v in row.items() if k!='endpoint_loss_change'},next_alpha=nxt,next_loss_increment_per_alpha=(future[key]-row['loss'])/(nxt-alpha)))
        write_csv(features,folder/'all_entity_features.csv');atomic_json(features,folder/'all_entity_features.json')
        write_csv(prospective,folder/'next_interval_features.csv')
        path_plots(folder);path_root_report(s['output'])
    finally:
        path_set_weights(e,after,delta,1.);e.opt.zero_grad(set_to_none=True)


def path_plots(folder):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);rows=read(folder/'loss_curve.json');x=[r['alpha'] for r in rows]
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for key in ['A_loss','B_loss']:ax[0].plot(x,[r[key]-rows[0][key] for r in rows],'o-',label=key)
    ax[1].plot(x,[r['A_slope'] for r in rows],'o-',label='Exact A slope')
    ax[1].plot(x,[r['FD_A_slopes'][1] for r in rows],'x--',label='Finite difference check')
    for a in ax:a.axhline(0,color='black',lw=.7);a.set_xlabel('Fraction of actual update');a.legend()
    ax[0].set_ylabel('Loss change from pre-update');ax[1].set_ylabel('d A loss / d alpha')
    fig.savefig(folder/'loss_and_slope.png',dpi=150);plt.close(fig)
    features=read(folder/'all_entity_features.json');sites=sorted(set(r['site'] for r in features));fig,ax=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    keys=['old_loss_projection','answer_sensitivity_cosine','frozen_answer_support_Fisher_nearest_opposite_minus_same_change']
    for site in sites:
        for harmed,style in [(True,'-'),(False,'--')]:
            group=[r for r in features if r['site']==site and r['split']=='A_test' and (r['endpoint_loss_change']>1e-6)==harmed]
            if not group:continue
            for axis,key in zip(ax,keys):
                means=[]
                for alpha in x:
                    values=[r[key] for r in group if r['alpha']==alpha and r.get(key) is not None]
                    means.append(float(np.mean(values)) if values else np.nan)
                axis.plot(x,means,style,label=site+(' harmed' if harmed else 'not harmed'))
    titles=['Motion projected onto old loss sensitivity','Orientation of answer sensitivity','Labelled separation in fixed metric']
    for a,title in zip(ax,titles):a.set(xlabel='Fraction of actual update',title=title)
    for axis,key in zip(ax,keys):
        if not any(r.get(key) is not None for r in features if r['split']=='A_test'):
            axis.text(.5,.5,'Undefined: panel lacks required target neighbours',ha='center',va='center',transform=axis.transAxes,wrap=True,fontsize=8)
        axis.ticklabel_format(axis='y',style='plain',useOffset=False)
    ax[0].legend(fontsize=5);fig.suptitle('Endpoint groups are retrospective; these curves do not establish prediction or causation')
    fig.savefig(folder/'geometry_along_path.png',dpi=150);plt.close(fig)


def path_root_report(output):
    root=Path(output);lines=['WITHIN-UPDATE PATH — actual displacement, exact A slopes'];rows=[]
    for p in sorted(root.glob('trajectory_seed*/update*/path/path_summary.json')):
        r=read(p);label=p.parents[2].name+'/'+p.parents[1].name
        rows.append(dict(run=label,**r));lines.append(f"{label}: {r['points']} path points; A {r['A_endpoint_change']:+.6f}; initial slope {r['A_initial_slope']:+.6f}, final {r['A_final_slope']:+.6f}; FD checks {r['FD_audits_passed']}/{r['points']}")
        lines.append('  Slope-reversal brackets: '+str([(v['left'],v['right']) for v in r['slope_reversal_brackets']]))
    lines.append('Geometry and entity tables are in each update/path folder. No inverse-rotation rescue or validated predictor is performed.')
    text='\n'.join(lines)+'\n';(root/'path_report.txt').write_text(text);atomic_json(rows,root/'path_summary.json');return text


# ---------------- Overnight campaign: all-layer measurement and interventions ----------------
class NightPause(Exception):
    pass


def night_defaults():
    return dict(enabled=True,hours=11.5,reserve_minutes=6,minimum_free_gb=8.,
        events=[21,20,47,46],alphas=sorted(set([round(i/40,6) for i in range(41)]+[round(.50+i*.005,6) for i in range(31)])),
        refinements=12,geometry_alphas=[0.,.234375,.5,.55,.575,.6,.625,.71875,1.],
        intervention_alphas=[.5,.575,.625,1.],patch_alpha=.625,patch_layers=[],
        null_repeats=32,random_patch_repeats=2,head_patches=4,sham_loss_atol=1e-4,branch_steps=16,
        ntk_alphas=[0.,.575,.625,1.],ntk_dimension=1024,
        matrix_alphas=[0.,1.],matrix_top_k=16,matrix_sample=128,
        gradient_alphas=[0.,.575,.625,1.],export_arrays=False)


def night_channel(name):
    if '.self_attn.q_proj.' in name or '.self_attn.k_proj.' in name:return 'QK'
    if '.self_attn.v_proj.' in name or '.self_attn.o_proj.' in name:return 'OV'
    if '.mlp.gate_proj.' in name:return 'gate'
    if '.mlp.up_proj.' in name or '.mlp.down_proj.' in name:return 'MLP_content'
    if 'norm' in name:return 'normalization'
    return 'embedding_readout_other'


def night_check(s):
    root=Path(s['output']);f=root/'session_budget.json'
    if f.exists() and time.time()>=read(f)['work_deadline']:raise NightPause('Time budget reached; saved units can be resumed')
    import shutil
    if shutil.disk_usage(root).free<s['overnight']['minimum_free_gb']*1024**3:raise NightPause('Minimum free disk reserve reached')


class NightProgress:
    def __init__(self,s,progress):self.s=s;self.progress=progress
    def update(self,**kw):night_check(self.s);self.progress.update(**kw)


def night_npz(file,**arrays):
    file=Path(file);file.parent.mkdir(parents=True,exist_ok=True);tmp=file.with_name(file.name+'.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
    os.replace(tmp,file)


def night_points(s):return sorted(set([float(x) for x in s['overnight']['geometry_alphas']]))


def night_plan(s):
    c=Config(**read(Path(s['source'])/'config.json'));events=[t for t in s['overnight']['events'] if t<=c.b_steps]
    jobs=[]
    def add(kind,seed,event):
        jid=f'{kind}_seed{seed}_step{event:03d}'
        if not any(j['id']==jid for j in jobs):jobs.append(dict(id=jid,kind=kind,seed=seed,event=event))
    seeds=s['seeds'];primary=events[:2]
    for event in primary:add('path',seeds[0],event)
    for seed in seeds[1:]:add('path',seed,events[0])
    for kind in ['geometry','factorial']:
        for event in primary:add(kind,seeds[0],event)
    for kind in ['branches','gradients','matrices','patches','ntk']:add(kind,seeds[0],events[0])
    for seed in seeds[1:]:
        for kind in ['geometry','factorial']:add(kind,seed,events[0])
    add('heads',seeds[0],events[0])
    for event in events:
        for seed in seeds:
            for kind in ['path','geometry','factorial','patches','branches','gradients','matrices','ntk','heads']:add(kind,seed,event)
    return jobs


def night_hardware():
    import platform,psutil
    d=dict(python=sys.version,platform=platform.platform(),cpu_count=psutil.cpu_count(),RAM_bytes=psutil.virtual_memory().total,
        torch=torch.__version__,cuda_version=torch.version.cuda,gpus=[])
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i);d['gpus'].append(dict(index=i,name=p.name,total_memory=p.total_memory))
    return d


def night_prepare(s,job,progress,do_step=True):
    c=Config(**read(Path(s['source'])/'config.json'));c.device=s['device'];c.threads=s['threads'];e=Engine(c)
    config_file=Path(s['output'])/'model_config.json'
    if not config_file.exists():atomic_json(json.loads(e.model.config.to_json_string()),config_file)
    data=make_data(e.tok,c,job['seed']);src=Path(s['source'])/f"seed{job['seed']}"
    if digest(data)!=read(src/'data.json')['hash']:raise ValueError('Source data hash mismatch')
    forks=[p for p in src.glob('fork*/fork.pt') if int(p.parent.name[4:])<=job['event']]
    if not forks:raise ValueError('No preceding source fork for event')
    f=max(forks,key=lambda p:int(p.parent.name[4:]));z=load(f);start=int(z['step'])
    if start!=int(f.parent.name[4:]):raise ValueError('Fork label disagrees with stored step')
    e.restore(z['engine']);del z
    for step in range(start,job['event']):
        progress.update(phase='reconstructing exact source trajectory',step=step)
        e.gradient(minibatches(data,'B',job['seed'],step,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
    if not do_step:return e,data
    update,after,delta=measured_adam_step(e,data,job['seed'],job['event'],progress)
    atomic_json(update,Path(s['output'])/job['id']/'actual_update.json')
    return e,data,after,delta


def night_eval(e,data):
    result={}
    for split in ['A_valid','A_test','B_valid','B_test']:
        loss,rel,support=e.evaluate_details(data[split]);result[split]=dict(mean=float(np.mean(loss)),
            entities=[dict(entity=r['entity'],q0=r['q'][0],loss=float(l),relation=float(a),support=float(b)) for r,l,a,b in zip(data[split],loss,rel,support)])
    return result


def night_slope(e,rows,delta):
    e.opt.zero_grad(set_to_none=True)
    try:
        for i in range(0,len(rows),e.c.microbatch):(e.loss_rows(rows[i:i+e.c.microbatch]).sum()/len(rows)).backward()
        records=[]
        for name,p in e.params.items():
            if p.grad is None:continue
            dot=0.;gn=0.;dn=0.
            for g,d in zip(p.grad.detach().reshape(-1).split(1048576),delta[name].reshape(-1).split(1048576)):
                gg=g.float().cpu().numpy();dd=d.numpy();dot+=_dot_cpu(gg,dd);gn+=_dot_cpu(gg,gg);dn+=_dot_cpu(dd,dd)
            records.append(dict(parameter=name,channel=night_channel(name),layer=_parameter_group(name),slope=dot,gradient_norm2=gn,update_norm2=dn))
        total=sum(r['slope'] for r in records)
        if not math.isfinite(total):raise FloatingPointError('Nonfinite path slope')
        return dict(total=total,parameters=records)
    finally:e.opt.zero_grad(set_to_none=True)


def night_path(e,s,data,after,delta,folder,progress):
    cfg=s['overnight'];points=folder/'points';points.mkdir(exist_ok=True)
    def evaluate(alpha):
        f=points/(float(alpha).hex().replace('-','m').replace('+','p')+'.json')
        if f.exists():return read(f)
        progress.update(phase='dense loss path and matrix-resolved slopes',alpha=alpha)
        path_set_weights(e,after,delta,alpha);values=night_eval(e,data)
        slopes={split:night_slope(e,data[split],delta) for split in ['A_test','B_test']}
        eps=s['path']['fd_epsilons'];checks=[]
        for h in eps:
            path_set_weights(e,after,delta,alpha+h);plus={k:np.array(e.evaluate(data[k])) for k in slopes}
            path_set_weights(e,after,delta,alpha-h);minus={k:np.array(e.evaluate(data[k])) for k in slopes}
            checks.append({k:((plus[k]-minus[k])/(2*h)).tolist() for k in slopes})
        audits={k:dict(exact=slopes[k]['total'],finite_difference_means=[float(np.mean(v[k])) for v in checks],
            passed=all(abs(np.mean(v[k])-slopes[k]['total'])<=s['path']['fd_atol']+s['path']['fd_rtol']*abs(slopes[k]['total']) for v in checks)) for k in slopes}
        result=dict(alpha=alpha,losses=values,slopes=slopes,FD=checks,audits=audits);atomic_json(result,f);return result
    nodes={float(a):evaluate(float(a)) for a in cfg['alphas']}
    # Refine finite-interval nonlinearity as well as slope reversals, avoiding a sign-only blind spot.
    for _ in range(cfg['refinements']):
        ordered=sorted(nodes);scores=[]
        for a,b in zip(ordered,ordered[1:]):
            if b-a<.001:continue
            ra,rb=nodes[a],nodes[b];sa=ra['slopes']['A_test']['total'];sb=rb['slopes']['A_test']['total']
            change=rb['losses']['A_test']['mean']-ra['losses']['A_test']['mean'];err=abs(change-(b-a)*(sa+sb)/2)
            scores.append((err+(.001*(b-a) if sa*sb<0 else 0),a,b))
        if not scores:break
        _,a,b=max(scores);m=(a+b)/2;nodes[m]=evaluate(m)
    ordered=[nodes[a] for a in sorted(nodes)];atomic_json(ordered,folder/'curve.json')
    write_csv([dict(alpha=r['alpha'],A=r['losses']['A_test']['mean'],B=r['losses']['B_test']['mean'],
        A_slope=r['slopes']['A_test']['total'],B_slope=r['slopes']['B_test']['total'],A_audit=r['audits']['A_test']['passed'],B_audit=r['audits']['B_test']['passed']) for r in ordered],folder/'curve.csv')


def night_geometry(e,s,data,after,delta,folder,progress,seed):
    local=json.loads(json.dumps(s));local['trajectory']['layers']=list(range(e.model.config.num_hidden_layers))
    local['trajectory']['label_null_repeats']=s['overnight']['null_repeats']
    anchor=Path(s['output'])/f'calibration_seed{seed}';anchor.mkdir(exist_ok=True)
    if not (anchor/'task_topology.json').exists():
        progress.update(phase='all-layer A-anchor calibration')
        z=load(Path(s['source'])/f'seed{seed}'/'anchor.pt');e.model.load_state_dict(z['model']);del z
        trajectory_state(e,local,data,anchor,progress);trajectory_topology(anchor,anchor,progress,s['overnight']['null_repeats'])
    for i,alpha in enumerate(night_points(s)):
        p=folder/f'point{i:03d}';p.mkdir(exist_ok=True)
        if (p/'done.json').exists():continue
        progress.update(phase='all-layer hidden geometry and PH',alpha=alpha)
        path_set_weights(e,after,delta,alpha);trajectory_state(e,local,data,p,progress)
        trajectory_topology(anchor,p,progress,s['overnight']['null_repeats']);path_attention(e,local,data,p,progress)
        path_geometry_features(folder/'point000',p,alpha)
        atomic_json(dict(alpha=alpha),p/'position.json');atomic_json(dict(complete=True),p/'done.json')


def night_assign(e,after,delta,alpha,channels=None,layer=None,restore_group=None):
    # All groups outside a specified intervention retain the common alpha background.
    with torch.no_grad():
        for name,p in e.params.items():
            coeff=alpha
            if channels is not None:coeff=channels.get(night_channel(name),alpha)
            if layer is not None and name.startswith(f'model.layers.{layer}.'):
                if restore_group=='all' or night_channel(name)==restore_group:coeff=0.
            p.copy_((after[name].double()+(coeff-1)*delta[name].double()).to(device=p.device,dtype=p.dtype))


def night_factorial(e,s,data,after,delta,folder,progress):
    sets=[('QK_vs_OV',['QK'],['OV']),('gate_vs_MLP',['gate'],['MLP_content']),
          ('selection_vs_content',['QK','gate'],['OV','MLP_content','embedding_readout_other'])]
    summaries=[]
    for alpha in s['overnight']['intervention_alphas']:
        for label,a,b in sets:
            results={}
            for ia,ib in [(0,0),(1,0),(0,1),(1,1)]:
                key=f'{label}_{alpha}_{ia}{ib}';f=folder/(key+'.json')
                if f.exists():r=read(f)
                else:
                    progress.update(phase='parameter-channel factorial intervention',condition=key)
                    coeff={k:alpha*ia for k in a};coeff.update({k:alpha*ib for k in b})
                    night_assign(e,after,delta,alpha,channels=coeff);r=dict(alpha=alpha,condition=key,coefficients=coeff,background_alpha=alpha,losses=night_eval(e,data));atomic_json(r,f)
                results[(ia,ib)]=r
            interaction={split:results[(1,1)]['losses'][split]['mean']-results[(1,0)]['losses'][split]['mean']-results[(0,1)]['losses'][split]['mean']+results[(0,0)]['losses'][split]['mean'] for split in ['A_valid','A_test','B_valid','B_test']}
            summaries.append(dict(alpha=alpha,factorial=label,loss_interaction=interaction,
                meaning='Finite loss non-additivity with all other parameters fixed at the same alpha; not uniquely a routing/content cause.'))
    atomic_json(summaries,folder/'interactions.json')
    # Every layer is tested; this does not select a causal site from raw attention change.
    for alpha in sorted(set([s['overnight']['patch_alpha'],1.])):
        for layer in range(e.model.config.num_hidden_layers):
            for group in ['QK','OV','gate','MLP_content','all']:
                f=folder/f'restore_alpha{alpha}_layer{layer:02d}_{group}.json'
                if f.exists():continue
                progress.update(phase='layer/group weight restoration',layer=layer,channel=group,alpha=alpha)
                night_assign(e,after,delta,alpha,layer=layer,restore_group=group)
                atomic_json(dict(layer=layer,group=group,background_alpha=alpha,losses=night_eval(e,data)),f)


@torch.no_grad()
def night_attention_arrays(e,row,layers):
    # Original bias-free OLMo only; audit reconstruction before using a patch.
    captures={};handles=[]
    def hook(key,inp=False):
        def f(m,args,out):captures[key]=(args[0] if inp else out)[0].detach().float().cpu().numpy().copy()
        return f
    for layer in layers:
        stem=f'model.layers.{layer}.self_attn'
        for k in ['q','k','v']:handles.append(e.mods[stem+'.'+k+'_proj'].register_forward_hook(hook((layer,k))))
        handles.append(e.mods[stem+'.o_proj'].register_forward_hook(hook((layer,'out'),True)))
        handles.append(e.mods[stem+'.q_proj'].register_forward_hook(hook((layer,'input'),True)))
    try:
        inputs,_,_=collate([row],e.tok.pad_token_id,e.c.device);e.model.model(**inputs,use_cache=False,return_dict=True)
    finally:
        for h in handles:h.remove()
    out={}
    for layer in layers:
        q,k,v=[captures[(layer,a)].astype(np.float64) for a in ['q','k','v']];length=len(q)
        heads=e.model.config.num_attention_heads;kv=e.model.config.num_key_value_heads;dim=q.shape[-1]//heads
        clip=getattr(e.model.config,'clip_qkv',None)
        if clip is not None:q,k,v=[np.clip(a,-clip,clip) for a in [q,k,v]]
        q=q.reshape(length,heads,dim).transpose(1,0,2);k=np.repeat(k.reshape(length,kv,dim).transpose(1,0,2),heads//kv,axis=0);v=np.repeat(v.reshape(length,kv,dim).transpose(1,0,2),heads//kv,axis=0)
        rotary=getattr(e.model.model,'rotary_emb',None) or getattr(e.mods[f'model.layers.{layer}.self_attn'],'rotary_emb',None)
        inv=rotary.inv_freq.detach().float().cpu().numpy() if rotary is not None else 1/(e.model.config.rope_theta**(np.arange(0,dim,2)/dim))
        phase=np.arange(length)[:,None]*inv[None,:];co=np.cos(phase)[None];si=np.sin(phase)[None];half=dim//2
        def rotate(a):return np.concatenate([a[:,:,:half]*co-a[:,:,half:]*si,a[:,:,:half]*si+a[:,:,half:]*co],-1)
        q,k=rotate(q),rotate(k);scores=q@k.transpose(0,2,1)/math.sqrt(dim);scores=np.where(np.tri(length,dtype=bool),scores,-np.inf)
        P=np.exp(scores-scores.max(-1,keepdims=True));P/=P.sum(-1,keepdims=True)
        Y=(P@v).transpose(1,0,2).reshape(length,-1);native=captures[(layer,'out')]
        err=float(np.linalg.norm(Y-native)/max(np.linalg.norm(native),1e-30))
        out[layer]=dict(P=P.astype(np.float32),V=v.astype(np.float32),native=native,error=err,
            X_normalized=captures[(layer,'input')],Q_projected=captures[(layer,'q')],K_projected=captures[(layer,'k')])
    return out


def night_capture(e,s,data,after,delta,alpha,folder,progress,layers):
    folder.mkdir(exist_ok=True);path_set_weights(e,after,delta,alpha)
    for split in ['A_valid','A_test','B_valid','B_test']:
        for row in data[split]:
            f=folder/f"{split}_{row['entity']:03d}.npz"
            if f.exists():continue
            progress.update(phase='capture audited routing/value operands',alpha=alpha,split=split,entity=row['entity'])
            arrays=night_attention_arrays(e,row,layers);stored={}
            for layer,v in arrays.items():
                for k,val in v.items():stored[f'{layer}_{k}']=val
            night_npz(f,**stored)


def night_patch_eval(e,s,data,early,late,layer,kind,head,repeat,progress):
    rows=[];module=e.mods[f'model.layers.{layer}.self_attn.o_proj']
    for split in ['A_valid','A_test','B_valid','B_test']:
        for row in data[split]:
            progress.update(phase='routing/value intervention',layer=layer,condition=kind,head=head,split=split,entity=row['entity'])
            f=f"{split}_{row['entity']:03d}.npz"
            with np.load(early/f) as a,np.load(late/f) as b:
                er=max(float(a[f'{layer}_error']),float(b[f'{layer}_error']))
                if er>s['path']['attention_audit_tolerance']:
                    rows.append(dict(split=split,entity=row['entity'],skipped=True,audit_error=er));continue
                p0,v0=a[f'{layer}_P'].astype(float),a[f'{layer}_V'].astype(float)
                p1,v1=b[f'{layer}_P'].astype(float),b[f'{layer}_V'].astype(float);native=b[f'{layer}_native'].copy()
            Y00=p0@v0;Y10=p1@v0;Y01=p0@v1;Y11=p1@v1
            dy_routing=(p1-p0)@v0;dy_content=p0@(v1-v0);dy_interaction=(p1-p0)@(v1-v0)
            decomposition=dict(routing_norm=float(np.linalg.norm(dy_routing)),content_norm=float(np.linalg.norm(dy_content)),interaction_norm=float(np.linalg.norm(dy_interaction)),
                total_norm=float(np.linalg.norm(Y11-Y00)),reconstruction_error=float(np.linalg.norm(Y11-Y00-dy_routing-dy_content-dy_interaction)))
            chosen={'earlier_routing':Y01,'earlier_values':Y10,'earlier_both':Y00,'sham':Y11}.get(kind,Y00)
            if head is not None:
                mixed=Y11.copy();mixed[head]=chosen[head];chosen=mixed
            shape=chosen.shape;replacement=chosen.transpose(1,0,2).reshape(shape[1],-1)
            if kind=='random_matched':
                norm=np.linalg.norm(replacement-native);rng=np.random.default_rng(seed_for(f'{split}/{row["entity"]}/{layer}/{head}',771+repeat))
                noise=rng.normal(size=native.shape)
                if head is not None:
                    shaped=noise.reshape(shape[1],shape[0],shape[2]);mask=np.zeros_like(shaped);mask[:,head]=shaped[:,head];noise=mask.reshape(native.shape)
                replacement=native+noise*norm/max(np.linalg.norm(noise),1e-30)
            replacement=torch.tensor(replacement,dtype=torch.float32,device=e.c.device)[None]
            def replace(m,args):return (replacement,)+args[1:]
            h=module.register_forward_pre_hook(replace)
            try:loss,rel,support=e.evaluate_details([row])
            finally:h.remove()
            rows.append(dict(split=split,entity=row['entity'],q0=row['q'][0],loss=loss[0],relation=rel[0],support=support[0],audit_error=er,skipped=False,output_decomposition=decomposition))
    means={k:float(np.mean([r['loss'] for r in rows if r['split']==k and not r['skipped']])) if any(r['split']==k and not r['skipped'] for r in rows) else None for k in ['A_valid','A_test','B_valid','B_test']}
    return dict(layer=layer,kind=kind,head=head,repeat=repeat,means=means,rows=rows,
        meaning='All query positions replaced at attention output BEFORE the current O projection. Routing and V content are crossed; O and downstream network stay fixed. Captures are input-matched. Random controls match activation displacement, not weight norm.')


def night_patches(e,s,data,after,delta,folder,progress,head_only=False):
    cfg=s['overnight'];layers=cfg['patch_layers'] or list(range(e.model.config.num_hidden_layers));early=folder/'earlier';late=folder/'later'
    night_capture(e,s,data,after,delta,0.,early,progress,layers);night_capture(e,s,data,after,delta,cfg['patch_alpha'],late,progress,layers)
    path_set_weights(e,after,delta,cfg['patch_alpha'])
    baseline=folder/'background.json'
    if not baseline.exists():atomic_json(night_eval(e,data),baseline)
    # Head ranking uses A_valid only. Test entities are included in all evaluations, never a first-four panel.
    ranked={}
    for layer in layers:
        changes=[]
        for row in data['A_valid']:
            name=f"A_valid_{row['entity']:03d}.npz"
            with np.load(early/name) as a,np.load(late/name) as b:
                p=a[f'{layer}_P'].astype(float)[:,-1];q=b[f'{layer}_P'].astype(float)[:,-1];m=(p+q)/2
                changes.append(.5*np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300))+q*np.log(np.maximum(q,1e-300)/np.maximum(m,1e-300)),axis=-1))
        score=np.mean(changes,axis=0);ranked[layer]=np.argsort(score)[::-1][:cfg['head_patches']].tolist()
    atomic_json(dict(ranking_split='A_valid',heads=ranked),folder/'head_selection.json')
    background=read(baseline)
    for head_mode in (['individual'] if head_only else ['all']):
        for layer in layers:
            heads=[None] if head_mode=='all' else ranked[layer]
            for head in heads:
                cases=[('sham',0),('earlier_routing',0),('earlier_values',0),('earlier_both',0)]+[('random_matched',r) for r in range(cfg['random_patch_repeats'])]
                for kind,repeat in cases:
                    f=folder/f'layer{layer:02d}_head{head}_{kind}_{repeat}.json'
                    if f.exists():continue
                    r=night_patch_eval(e,s,data,early,late,layer,kind,head,repeat,progress)
                    if kind=='sham':
                        ref={(split,x['entity']):x['loss'] for split,part in background.items() for x in part['entities']}
                        errors=[abs(x['loss']-ref[(x['split'],x['entity'])]) for x in r['rows'] if not x['skipped']]
                        r['sham_max_loss_error']=max(errors) if errors else None
                        r['sham_audit_pass']=len(errors)==sum(len(v['entities']) for v in background.values()) and max(errors)<=cfg['sham_loss_atol']
                    else:
                        sham=read(folder/f'layer{layer:02d}_head{head}_sham_0.json')
                        r['sham_audit_pass']=sham['sham_audit_pass']
                    r['causal_comparison_audited']=r['sham_audit_pass'] and all(not x['skipped'] for x in r['rows'])
                    atomic_json(r,f)


def night_branches(s,job,folder,progress):
    cases=['original','reset_first_moment','reset_second_moment','reset_both_keep_clock','fresh_optimizer','first_step_fraction_0375','all_steps_lr_025']
    for case in cases:
        f=folder/(case+'.json')
        if f.exists() and read(f).get('complete'):continue
        try:
            e,data=night_prepare(s,job,progress,do_step=False)
            start=job['event'];stop=min(e.c.b_steps,start+s['overnight']['branch_steps']-1)
            if case=='fresh_optimizer':e.reset_optimizer()
            elif case.startswith('reset_'):
                for state in e.opt.state.values():
                    if case in ['reset_first_moment','reset_both_keep_clock']:state['exp_avg'].zero_()
                    if case in ['reset_second_moment','reset_both_keep_clock']:state['exp_avg_sq'].zero_()
            rows=[]
            for step in range(start,stop+1):
                progress.update(phase='new controlled continuation',condition=case,step=step)
                if case=='all_steps_lr_025':
                    for g in e.opt.param_groups:g['lr']=e.c.lr*.25
                before={n:p.detach().cpu().clone() for n,p in e.params.items()} if case=='first_step_fraction_0375' and step==start else None
                audit=e.gradient(minibatches(data,'B',job['seed'],step,e.c));e.opt.step();e.opt.zero_grad(set_to_none=True)
                if before is not None:
                    with torch.no_grad():
                        for n,p in e.params.items():p.copy_((before[n].to(p.device)+.375*(p-before[n].to(p.device))))
                    del before
                rows.append(dict(step=step,clip=audit['clip'],gradient_norm=audit['raw_norm'],losses=night_eval(e,data)))
                atomic_json(dict(condition=case,complete=False,rows=rows),f)
            atomic_json(dict(condition=case,complete=True,rows=rows,meaning='New training branch using original scheduled minibatches. Moment resets retain the original Adam clock except fresh_optimizer. Fractional first step retains the original moment update; not equivalent to resetting optimizer history.'),f)
            del e
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except NightPause:raise
        except Exception as exc:
            import traceback
            atomic_json(dict(condition=case,complete=False,failed=True,error=str(exc),traceback=traceback.format_exc()),f)
            if 'e' in locals():del e
            if torch.cuda.is_available():torch.cuda.empty_cache()


def night_gradients(e,s,data,after,delta,folder,progress,seed,event):
    # Reconstruct the pre-update first moment from the native post-update state and source gradient.
    path_set_weights(e,after,delta,0.);training=e.gradient(minibatches(data,'B',seed,event,e.c));old_m={}
    for name,p in e.params.items():
        st=e.opt.state.get(p,{})
        if 'exp_avg' in st and p.grad is not None:old_m[name]=((st['exp_avg'].detach().cpu()-(1-e.c.beta1)*p.grad.detach().cpu())/e.c.beta1).float()
    e.opt.zero_grad(set_to_none=True)
    for alpha in s['overnight']['gradient_alphas']:
        for split in ['A_test','B_test']:
            file=folder/f'gradient_{alpha}_{split}.json'
            if file.exists():continue
            progress.update(phase='gradient/momentum alignment for every parameter',alpha=alpha,split=split)
            path_set_weights(e,after,delta,alpha);e.opt.zero_grad(set_to_none=True)
            rows=data[split]
            for i in range(0,len(rows),e.c.microbatch):(e.loss_rows(rows[i:i+e.c.microbatch]).sum()/len(rows)).backward()
            results=[]
            for name,p in e.params.items():
                if p.grad is None or name not in old_m:continue
                stats=dict(gradient_norm2=0.,memory_norm2=0.,gradient_memory_dot=0.,actual_update_dot=0.,frozen_preconditioned_history_dot=0.)
                state=e.opt.state[p];t=int(state['step'].item())
                arrays=[p.grad.detach().reshape(-1),old_m[name].reshape(-1),delta[name].reshape(-1),state['exp_avg_sq'].detach().reshape(-1)]
                for start in range(0,p.numel(),1048576):
                    g,m,d,v=[a[start:start+1048576].double().cpu().numpy() for a in arrays]
                    stats['gradient_norm2']+=_dot_cpu(g,g);stats['memory_norm2']+=_dot_cpu(m,m);stats['gradient_memory_dot']+=_dot_cpu(g,m);stats['actual_update_dot']+=_dot_cpu(g,d)
                    history=-e.c.lr*e.c.beta1*m/(1-e.c.beta1**t)/(np.sqrt(v/(1-e.c.beta2**t))+e.c.eps)
                    stats['frozen_preconditioned_history_dot']+=_dot_cpu(g,history)
                stats['gradient_memory_cosine']=stats['gradient_memory_dot']/math.sqrt(max(stats['gradient_norm2']*stats['memory_norm2'],1e-30))
                results.append(dict(parameter=name,channel=night_channel(name),layer=_parameter_group(name),**stats))
            atomic_json(dict(alpha=alpha,split=split,source_training_clip=training['clip'],parameters=results,
                meaning='Unclipped full-test-panel gradient compared with reconstructed native pre-update first moment. Updated second moment held fixed for history projection. A geometric lag diagnostic, not a time-series phase estimate.'),file)
            e.opt.zero_grad(set_to_none=True)
    del old_m


def night_matrices(e,s,data,after,delta,folder,progress):
    local=json.loads(json.dumps(s));local.update(top_k=s['overnight']['matrix_top_k'],matrix_sample=s['overnight']['matrix_sample'],topology_points=32)
    for alpha in s['overnight']['matrix_alphas']:
        p=folder/f'alpha_{alpha}';p.mkdir(exist_ok=True);path_set_weights(e,after,delta,alpha)
        for name,w,aliases in matrix_inventory(e.model):
            f=p/(filename(name)+'.json')
            if f.exists():continue
            progress.update(phase='all-matrix spectrum/RMT and row topology',matrix=name,alpha=alpha)
            r=matrix_stats(name,w,local,p);r.update(alpha=alpha,aliases=aliases,channel=night_channel(name))
            with np.load(p/filename(name)) as z:u=z['left'];v=z['right']
            # top_svd right vectors are columns (see implementation); radial certificate is truncated.
            dw=delta[name].numpy().astype(float);coeff=np.diag(u.T@dw@v)
            r['leading_radial_coefficients']=coeff.tolist();r['leading_radial_energy_fraction_of_actual_update']=float(np.sum(coeff**2)/max(_dot_cpu(dw,dw),1e-30))
            r['radial_note']='Leading randomized singular triplets only; a truncated first-order certificate, not exact finite isospectral motion. Degenerate singular vectors are basis-dependent.'
            atomic_json(r,f)
    if 0. in s['overnight']['matrix_alphas'] and 1. in s['overnight']['matrix_alphas']:
        comparisons=[]
        for name,w,_ in matrix_inventory(e.model):
            a=folder/'alpha_0.0'/filename(name);b=folder/'alpha_1.0'/filename(name)
            if not (a.exists() and b.exists()):continue
            with np.load(a) as x,np.load(b) as y:
                left=np.linalg.svd(x['left'].T@y['left'],compute_uv=False);right=np.linalg.svd(x['right'].T@y['right'],compute_uv=False)
                sx=x['singular_values'];sy=y['singular_values']
                comparisons.append(dict(matrix=name,channel=night_channel(name),left_principal_angles_degrees=np.degrees(np.arccos(np.clip(left,-1,1))).tolist(),
                    right_principal_angles_degrees=np.degrees(np.arccos(np.clip(right,-1,1))).tolist(),
                    relative_top_spectrum_change=float(np.linalg.norm(sy-sx)/max(np.linalg.norm(sx),1e-30))))
        atomic_json(comparisons,folder/'matrix_changes.json')


def night_ntk(e,s,data,after,delta,folder,progress):
    # Disk shards checkpoint each coordinate. No full per-example Jacobian is stored.
    rows=[dict(r,split=split) for split in ['A_test','B_test'] for r in data[split]]
    groups=sorted({night_channel(n)+'/'+_parameter_group(n) for n in e.params});gi={g:i for i,g in enumerate(groups)}
    dim=s['overnight']['ntk_dimension'];count=2*len(rows)
    for alpha in s['overnight']['ntk_alphas']:
        out=folder/f'alpha_{alpha}';out.mkdir(exist_ok=True)
        if (out/'ntk.json').exists():continue
        path_set_weights(e,after,delta,alpha);metadata=[]
        for i,row in enumerate(rows):
            progress.update(phase='all-entity, channel-resolved task-coordinate NTK',alpha=alpha,prompt=i+1,total_prompts=len(rows))
            # Cached coordinates avoid repeating backwards after a budget stop.
            meta=out/f'row{i:03d}.json'
            files=[out/f'coordinate{2*i+k:03d}.npz' for k in range(2)]
            if all(f.exists() for f in files) and meta.exists():metadata.append(read(meta));continue
            coords=_task_coordinates(e,row);m=dict(split=row['split'],entity=row['entity'],q=row['q'],ids=row['ids'],coordinates=coords.detach().cpu().tolist());metadata.append(m);atomic_json(m,meta)
            for k in range(2):
                if files[k].exists():continue
                progress.update(phase='NTK backward and streamed sketches',coordinate=2*i+k,alpha=alpha)
                grads=torch.autograd.grad(coords[k],list(e.params.values()),retain_graph=k==0,allow_unused=True)
                feats=np.zeros((2,len(groups),dim));diag=np.zeros(len(groups))
                for (name,p),grad in zip(e.params.items(),grads):
                    if grad is None:continue
                    g=gi[night_channel(name)+'/'+_parameter_group(name)];diag[g]+=exact_norm2(grad)
                    for rep in range(2):feats[rep,g]+=_gradient_sketch(grad,name,dim,7301+rep*1000003)
                night_npz(files[k],features=feats,exact_group_diagonal=diag);del grads
            del coords
        features=[];diagonal=[]
        for i in range(count):
            with np.load(out/f'coordinate{i:03d}.npz') as z:features.append(z['features']);diagonal.append(z['exact_group_diagonal'])
        features=np.stack(features,axis=2);diagonal=np.stack(diagonal,axis=1)
        gK=features@features.swapaxes(-1,-2);Ks=gK.sum(axis=1);K=Ks.mean(axis=0)
        disagreement=float(np.linalg.norm(Ks[0]-Ks[1])/max(np.linalg.norm(K),1e-30));diagerror=float(np.linalg.norm(np.diag(K)-diagonal.sum(0))/max(np.linalg.norm(diagonal.sum(0)),1e-30))
        coordinates=np.array([m['coordinates'] for m in metadata]);errors=_sigmoid(coordinates)-np.array([[m['q'][0],1.] for m in metadata])
        coherence=[]
        for j,g in enumerate(groups):
            values=features[:,j].reshape(2,len(rows),2,dim);loss_features=np.sum(values*errors[None,:,:,None],axis=2)
            estimates=[]
            for rep in range(2):
                for split in ['A_test','B_test']:
                    ix=[i for i,m in enumerate(metadata) if m['split']==split];F=loss_features[rep,ix]
                    estimates.append(dict(repeat=rep,split=split,coherence=float(np.linalg.norm(F.mean(0))/max(np.linalg.norm(F,axis=1).mean(),1e-30))))
            coherence.append(dict(group=g,estimates=estimates))
        result=dict(alpha=alpha,mode='two_independent_CountSketch',groups=groups,rows=metadata,summary=kernel_summary(K,metadata),
            independent_sketch_disagreement=disagreement,exact_diagonal_relative_error=diagerror,
            audit_pass=disagreement<=.2 and diagerror<=.2,
            group_summaries=[dict(kernel_summary(gK[:,j].mean(0),metadata),group=g,interpretation='Task-coordinate kernel restricted to this parameter channel/layer; infinitesimal SGD diagnostic.') for j,g in enumerate(groups)],
            per_entity_loss_gradient_coherence_sketch_estimates=coherence,
            meaning='Full trainable parameter task-coordinate Jacobian, all A/B test entities. Approximate off-diagonal kernel, exact diagonal audit. Coherence is entity-gradient coherence, not winner-sign coherence or temporal persistence. Raw SGD geometry, not historical Adam.')
        night_npz(out/'ntk_arrays.npz',kernel=K,group_sketch_kernels=gK,features=features,exact_group_diagonal=diagonal)
        atomic_json(result,out/'ntk.json')


def night_intervention_summary(output):
    root=Path(output);rows=[]
    for folder in sorted(root.glob('patches_seed*')):
        if not (folder/'background.json').exists():continue
        background=read(folder/'background.json')
        for f in sorted(folder.glob('layer*.json')):
            r=read(f)
            if 'means' not in r:continue
            values={k:(r['means'][k]-background[k]['mean']) if r['means'][k] is not None else None for k in background}
            rows.append(dict(job=folder.name,layer=r['layer'],head=r['head'],condition=r['kind'],repeat=r['repeat'],audited=r.get('causal_comparison_audited',False),
                A_valid_change=values['A_valid'],A_test_change=values['A_test'],B_valid_change=values['B_valid'],B_test_change=values['B_test']))
    atomic_json(rows,root/'intervention_summary.json');write_csv(rows,root/'intervention_summary.csv')
    return rows


def night_intervention_plots(output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output);rows=night_intervention_summary(root)
    for job in sorted({r['job'] for r in rows}):
        part=[r for r in rows if r['job']==job and r['head'] is None and r['audited']]
        if not part:continue
        layers=sorted({r['layer'] for r in part});kinds=['earlier_routing','earlier_values','earlier_both','random_matched','sham']
        values=[]
        for split in ['A_test_change','B_test_change']:
            grid=np.full((len(layers),len(kinds)),np.nan)
            for i,layer in enumerate(layers):
                for j,kind in enumerate(kinds):
                    v=[r[split] for r in part if r['layer']==layer and r['condition']==kind and r[split] is not None]
                    if v:grid[i,j]=np.mean(v)
            values.append(grid)
        vmax=max(1e-8,max(float(np.nanmax(abs(v))) for v in values));fig,ax=plt.subplots(1,2,figsize=(12,max(4,len(layers)*.3)),constrained_layout=True)
        for a,v,title in zip(ax,values,['A-test loss change','B-test loss change']):
            im=a.imshow(v,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto');a.set_yticks(range(len(layers)),layers);a.set_xticks(range(len(kinds)),['Old routing','Old values','Old both','Random','Sham'],rotation=25);a.set(title=title,ylabel='Layer')
        fig.colorbar(im,ax=list(ax),label='Change in KL (negative improves)');fig.suptitle(job+'; audited whole-layer patches only')
        fig.savefig(root/(job+'_effects.png'),dpi=150);plt.close(fig)


def night_report(output):
    root=Path(output);jobs=read(root/'jobs.json') if (root/'jobs.json').exists() else [];done=[j for j in jobs if (root/j['id']/'done.json').exists()]
    lines=[f'OVERNIGHT STUDY — {len(done)}/{len(jobs)} jobs completed'];findings=[]
    for j in jobs:
        p=root/j['id']
        if j['kind']=='path':
            fs=sorted((p/'points').glob('*.json')) if (p/'points').exists() else []
            if not fs:continue
            rows=sorted([read(f) for f in fs],key=lambda x:x['alpha']);a,b=rows[0],rows[-1]
            lines.append(f"seed {j['seed']} step {j['event']}: {len(rows)} saved fractions [{a['alpha']:.4f}, {b['alpha']:.4f}]; A {a['losses']['A_test']['mean']:.6f} -> {b['losses']['A_test']['mean']:.6f}; B {a['losses']['B_test']['mean']:.6f} -> {b['losses']['B_test']['mean']:.6f}")
            findings.append(dict(job=j['id'],points=len(rows),last_fraction=b['alpha'],A_last=b['losses']['A_test']['mean'],B_last=b['losses']['B_test']['mean']))
        elif j['kind']=='branches':
            for f in p.glob('*.json'):
                r=read(f)
                if 'rows' in r and r['rows']:
                    last=r['rows'][-1];lines.append(f"  {j['id']} {r['condition']}: step {last['step']}, A {last['losses']['A_test']['mean']:.6f}, B {last['losses']['B_test']['mean']:.6f}; complete={r['complete']}")
    intervention_rows=night_intervention_summary(root)
    for job in sorted({r['job'] for r in intervention_rows}):
        candidates=[r for r in intervention_rows if r['job']==job and r['audited'] and r['head'] is None and r['condition'] in ['earlier_routing','earlier_values','earlier_both'] and r['A_valid_change'] is not None]
        if candidates:
            best=min(candidates,key=lambda r:r['A_valid_change'])
            lines.append(f"  {job}: largest A_valid rescue among completed whole-layer patches: layer {best['layer']} {best['condition']}; test A change {best['A_test_change']:+.6f}, B {best['B_test_change']:+.6f}")
    failures=[dict(job=j['id'],**read(root/j['id']/'failure.json')) for j in jobs if (root/j['id']/'failure.json').exists()]
    if failures:lines.append('Failures (other jobs may still have completed): '+', '.join(r['job'] for r in failures))
    branch_failures=[]
    for f in root.glob('branches_*/*.json'):
        r=read(f)
        if r.get('failed'):branch_failures.append(dict(job=f.parent.name,condition=r.get('condition'),error=r.get('error')))
    if branch_failures:lines.append('Failed continuation variants: '+', '.join(r['condition'] for r in branch_failures))
    lines.append('Progress is saved by unit. An unfinished job is not a negative scientific result. Read audits and per-entity records before interpreting effects.')
    text='\n'.join(lines)+'\n';(root/'overnight_report.txt').write_text(text)
    atomic_json(dict(completed=len(done),planned=len(jobs),findings=findings,failures=failures,branch_failures=branch_failures),root/'overnight_summary.json')
    return text


def night_plots(output):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output);night_intervention_plots(root);jobs=read(root/'jobs.json');available=[]
    for j in jobs:
        if j['kind']!='path':continue
        rows=[read(f) for f in (root/j['id']/'points').glob('*.json')]
        if rows:available.append((j,sorted(rows,key=lambda r:r['alpha'])))
    if available:
        fig,axes=plt.subplots(len(available),2,figsize=(11,3*len(available)),squeeze=False,constrained_layout=True)
        for (j,rows),ax in zip(available,axes):
            x=[r['alpha'] for r in rows]
            for split in ['A_test','B_test']:
                ax[0].plot(x,[r['losses'][split]['mean'] for r in rows],'.-',label=split)
                ax[1].plot(x,[r['slopes'][split]['total'] for r in rows],'.-',label=split)
            for a in ax:a.set_xlabel('Fraction of actual update');a.legend();a.grid(alpha=.2)
            ax[0].set(title=f"Seed {j['seed']}, step {j['event']}",ylabel='Mean KL');ax[1].set_ylabel('Exact directional slope')
        fig.savefig(root/'overnight_paths.png',dpi=140);plt.close(fig)
    for j in jobs:
        if j['kind']!='branches':continue
        rows=[read(f) for f in (root/j['id']).glob('*.json')];rows=[r for r in rows if 'condition' in r and r.get('rows')]
        if not rows:continue
        fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
        for r in rows:
            for a,split in zip(ax,['A_test','B_test']):a.plot([x['step'] for x in r['rows']],[x['losses'][split]['mean'] for x in r['rows']],label=r['condition']);a.set(title=split,xlabel='Actual B step',ylabel='Mean KL')
        ax[0].legend(fontsize=6);fig.savefig(root/j['id']/'continuations.png',dpi=140);plt.close(fig)


def night_export(output,include_arrays=False):
    import zipfile
    root=Path(output);dest=root/'results_share.zip';tmp=root/'results_share.zip.tmp';files=[];omitted=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p in [dest,tmp] or p.suffix in ['.lock','.tmp']:continue
        if '.ipynb_checkpoints' in p.parts:continue
        if p.suffix in ['.json','.csv','.txt','.png','.log'] or (include_arrays and p.suffix=='.npz'):files.append(p)
        else:omitted.append(str(p.relative_to(root)))
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(root)))
        z.writestr('EXPORT_SCOPE.json',json.dumps(dict(raw_arrays_included=include_arrays,omitted=omitted,note='Raw arrays remain in the run directory. Source checkpoints are never copied into the report archive.'),indent=2))
    os.replace(tmp,dest);return dest


def night_inputs(s):
    src=Path(s['source']);paths={src/'config.json'}
    for seed in s['seeds']:
        base=src/f'seed{seed}';paths.update([base/'data.json',base/'anchor.pt'])
        for t in s['overnight']['events']:
            candidates=[p for p in base.glob('fork*/fork.pt') if int(p.parent.name[4:])<=t]
            if not candidates:raise ValueError(f'Seed {seed}: no saved fork before update {t}')
            paths.add(max(candidates,key=lambda p:int(p.parent.name[4:])))
    return {str(p):sha256_file(p) for p in sorted(paths)}


def night_run(s):
    import fcntl,transformers
    validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'queue.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Progress(out) as progress:
            started=time.time();cfg=s['overnight'];end=started+cfg['hours']*3600
            atomic_json(dict(started=started,work_deadline=end-cfg['reserve_minutes']*60,session_deadline=end),out/'session_budget.json')
            atomic_json(night_hardware(),out/'hardware.json')
            atomic_json(read(Path(s['source'])/'config.json'),out/'source_config.json')
            for seed in s['seeds']:atomic_json(read(Path(s['source'])/f'seed{seed}'/'data.json'),out/f'source_data_seed{seed}.json')
            progress.update(phase='fingerprinting all active checkpoint inputs')
            design=dict(settings=s,inputs=night_inputs(s),code={p.name:sha256_file(p) for p in [Path(__file__),Path(__file__).with_name('forgetting_mechanisms.py')]},versions=dict(torch=torch.__version__,numpy=np.__version__,transformers=transformers.__version__))
            if (out/'design.json').exists() and read(out/'design.json')!=design:raise ValueError('Source, settings, code or dependency versions changed; restart in a new output folder')
            atomic_json(design,out/'design.json');atomic_json(s,out/'settings.json');jobs=night_plan(s);atomic_json(jobs,out/'jobs.json')
            consecutive_errors=0;paused=False
            for j in jobs:
                folder=out/j['id'];folder.mkdir(exist_ok=True)
                if (folder/'done.json').exists():continue
                try:night_check(s)
                except NightPause:paused=True;break
                progress.update(phase='overnight worker',job=j['id'],completed=sum((out/x['id']/'done.json').exists() for x in jobs),total=len(jobs))
                with open(folder/'worker.log','a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker','--settings',str(out/'settings.json'),'--job',j['id']],stdout=log,stderr=subprocess.STDOUT)
                    while True:
                        try:code=child.wait(timeout=5);break
                        except subprocess.TimeoutExpired:
                            if time.time()>=end-cfg['reserve_minutes']*60:
                                child.terminate()
                                try:child.wait(timeout=20)
                                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=20)
                                code=child.returncode;paused=True;break
                if (folder/'paused.json').exists():paused=True
                if paused:break
                if code or not (folder/'done.json').exists():
                    consecutive_errors+=1
                    if not (folder/'failure.json').exists():atomic_json(dict(error=f'Worker exit {code}; see worker.log'),folder/'failure.json')
                else:consecutive_errors=0
                night_report(out);night_plots(out)
                if consecutive_errors>=3:break
            progress.update(phase='saving report and share archive')
            night_report(out);night_plots(out)
            completed=sum((out/j['id']/'done.json').exists() for j in jobs)
            summary=read(out/'overnight_summary.json')
            state='complete' if completed==len(jobs) and not summary['failures'] and not summary['branch_failures'] else ('budget_paused' if paused else 'complete_with_failures')
            # Final scientific status is included in the exported archive itself.
            progress.update(status=state,phase='finished; preparing share archive',completed=completed,total=len(jobs),share_archive=str(out/'results_share.zip'))
            try:
                archive=str(night_export(out,cfg['export_arrays']))
                progress.update(phase='finished; resume continues unfinished work',share_archive=archive)
            except (OSError,RuntimeError) as exc:
                atomic_json(dict(error=str(exc)),out/'export_error.json');progress.update(phase='finished; share archive failed',share_archive=None)



def night_worker(s,jobid):
    folder=Path(s['output'])/jobid;job=next(j for j in read(Path(s['output'])/'jobs.json') if j['id']==jobid)
    for name in ['paused.json','failure.json']:
        if (folder/name).exists():(folder/name).unlink()
    with Progress(folder) as raw:
        progress=NightProgress(s,raw)
        try:
            if job['kind']=='branches':night_branches(s,job,folder,progress)
            else:
                e,data,after,delta=night_prepare(s,job,progress)
                try:
                    if job['kind']=='path':night_path(e,s,data,after,delta,folder,progress)
                    elif job['kind']=='geometry':night_geometry(e,s,data,after,delta,folder,progress,job['seed'])
                    elif job['kind']=='factorial':night_factorial(e,s,data,after,delta,folder,progress)
                    elif job['kind'] in ['patches','heads']:
                        shared=Path(s['output'])/f"patches_seed{job['seed']}_step{job['event']:03d}";shared.mkdir(exist_ok=True)
                        night_patches(e,s,data,after,delta,shared,progress,head_only=job['kind']=='heads')
                    elif job['kind']=='gradients':night_gradients(e,s,data,after,delta,folder,progress,job['seed'],job['event'])
                    elif job['kind']=='matrices':night_matrices(e,s,data,after,delta,folder,progress)
                    elif job['kind']=='ntk':night_ntk(e,s,data,after,delta,folder,progress)
                    else:raise ValueError('Unknown overnight job')
                finally:path_set_weights(e,after,delta,1.);e.opt.zero_grad(set_to_none=True)
            atomic_json(dict(complete=True,elapsed_seconds=time.time()-raw.started,peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),folder/'done.json');raw.update(status='complete',phase='finished')
        except NightPause as exc:
            atomic_json(dict(reason=str(exc)),folder/'paused.json');raw.update(status='budget_paused',phase=str(exc))
        except Exception as exc:
            import traceback
            atomic_json(dict(error=str(exc),traceback=traceback.format_exc()),folder/'failure.json');raise


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
        s.update(device='cpu',threads=1,events=[9],top_k=3,matrix_sample=12,topology_points=6,activation_positions=4,prompts_per_domain=4)
        s['extended'].update(null_repeats=2,ntk_prompts_per_domain=2,ntk_sketch_dim=256)
        s['trajectory'].update(steps=[8,9],layers=[0,1],label_null_repeats=2)
        s['overnight'].update(hours=.5,reserve_minutes=1,events=[8],alphas=[0.,.5,1.],refinements=1,geometry_alphas=[0.,1.],intervention_alphas=[1.],patch_alpha=1.,null_repeats=2,random_patch_repeats=1,head_patches=1,branch_steps=2,ntk_alphas=[0.,1.],ntk_dimension=256,matrix_top_k=3,matrix_sample=12,gradient_alphas=[0.,1.],minimum_free_gb=.1)
        s['base_checkpoints']=[dict(id='random_101',kind='random',random_seed=101),dict(id='pretrained',kind='pretrained')]
        run(s)
    elif a.command=='run':run(read(a.settings))
    else:worker(read(a.settings),a.job)

if __name__=='__main__':main()
