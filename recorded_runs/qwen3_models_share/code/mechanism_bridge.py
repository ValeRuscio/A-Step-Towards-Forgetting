"""Controlled selection/content interaction, activation repair and trajectory study.
Uses the original FP32 OLMo association replay; see README for estimands/limitations.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/olmo_bridge_mpl')
import argparse, copy, hashlib, json, math, signal, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np
import torch
import bridge_core as core
from bridge_replay import atomic_json, read, collate, Config, Engine, make_data, digest, minibatches

VERSION='bridge-1.0'
SPLITS=['A_valid','B_valid','A_test','B_test']
SELECTION=['QK','gate']; CONTENT=['OV','MLP_content','embedding_readout_other']
CORNERS=['00','10','01','11']

class Pause(Exception):pass

def defaults(source, output):
    source=Path(source).expanduser().resolve(); c=Config(**read(source/'config.json'))
    seeds=[s for s in c.seeds if list((source/f'seed{s}').glob('fork*/fork.pt'))]
    return dict(version=VERSION,source=str(source),output=str(Path(output).expanduser().resolve()),
        device='cuda:0',threads=8,seeds=seeds,events=[t for t in [21,20,47,46] if t<=c.b_steps],
        fractions=[.625,.575,.55,.6,.5,.675,1.],hours=11.5,reserve_minutes=5,minimum_free_gb=10.,
        capture_layers=[],patch_layers=[0,1,9],calibration_seed=8041,calibration_fraction=.5,
        fit_positions=4,fit_max_rank=64,random_repeats=4,topology_nulls=4,
        attention_tolerance=1e-4,sham_atol=1e-4,projection_fd_epsilon=.02,projection_fd_rtol=.08,
        branch_steps=16,branches=['original','reset_first_moment','all_steps_lr_025'],
        branch_event=21,branch_trace_every=1,matched_direction_fractions=[.25,1.],
        export_arrays=False)

def discover_source():
    candidates=[Path('/home/ubuntu/1/runs/olmo_association_v1')]
    candidates += [p/'runs'/'olmo_association_v1' for p in [Path.cwd(),Path.cwd().parent,Path.home()/'1']]
    return next((p.resolve() for p in candidates if (p/'config.json').exists() and list(p.glob('seed*/fork*/fork.pt'))),None)

def validate(s):
    if s.get('version')!=VERSION:raise ValueError('Create settings with this version of defaults().')
    src=Path(s['source']);c=Config(**read(src/'config.json'))
    if not s['seeds'] or not s['events']:raise ValueError('Choose available seeds and B update numbers.')
    if not (0<s['calibration_fraction']<1):raise ValueError('Calibration fraction must be between 0 and 1.')
    if not s['fractions'] or any(not 0<a<=1 for a in s['fractions']):raise ValueError('Fractions must be in (0,1].')
    if s['fit_max_rank']<2 or s['fit_positions']<1:raise ValueError('Invalid alignment rank/position count.')
    if s['branch_steps']<1 or s['branch_trace_every']<1:raise ValueError('Invalid continuation length/cadence.')
    if set(s['branches'])-{'original','reset_first_moment','all_steps_lr_025'}:raise ValueError('Unknown branch.')
    if s['hours']*60<=s['reserve_minutes']:raise ValueError('Time budget must exceed report reserve.')
    for seed in s['seeds']:
        forks=list((src/f'seed{seed}').glob('fork*/fork.pt'))
        for step in set(s['events']+[s['branch_event']]):
            if step>c.b_steps or not any(int(p.parent.name[4:])<=step for p in forks):
                raise ValueError(f'No usable source fork for seed {seed}, step {step}.')
    if not c.smoke and not Path(c.model).exists() and (not c.revision or len(c.revision)!=40):
        raise ValueError('Original remote model must use its recorded 40-character pinned revision.')
    return c

def layers_for(e,s,patch=False):
    n=e.model.config.num_hidden_layers; requested=s['patch_layers' if patch else 'capture_layers']
    result=sorted(set(range(n) if not requested else [i for i in requested if 0<=i<n]))
    if not result:raise ValueError('No requested layers exist in this model.')
    return result

def partition(data,s):
    ids=sorted(set(r['entity'] for r in data['A_valid']))
    if len(ids)<4:raise ValueError('Need at least four entity identities for disjoint calibration/evaluation.')
    order=np.random.default_rng(s['calibration_seed']).permutation(ids)
    n=max(2,min(len(ids)-2,round(len(ids)*s['calibration_fraction'])))
    return set(map(int,order[:n]))

def progress_check(s,**kw):
    out=Path(s['output'])
    if (out/'STOP').exists():raise Pause('Stop requested; completed units are preserved.')
    budget=read(out/'budget.json') if (out/'budget.json').exists() else {}
    if time.time()>budget.get('deadline',float('inf')):raise Pause('Session time budget reached.')
    import shutil
    if shutil.disk_usage(out).free<s['minimum_free_gb']*1024**3:raise Pause('Disk reserve reached.')
    old=read(out/'status.json') if (out/'status.json').exists() else {}
    old.update(kw,status='running',heartbeat=time.time());atomic_json(old,out/'status.json')

class Progress:
    def __init__(self,s):self.s=s
    def update(self,**kw):progress_check(self.s,**kw)

def settings_for_core(s):
    r=core.default_settings(s['source'],s['output']);r.update(device=s['device'],threads=s['threads'])
    return r

def prepare(s,seed,event,folder):
    folder.mkdir(parents=True,exist_ok=True)
    job=dict(seed=seed,event=event,id=str(folder.relative_to(Path(s['output']))))
    return core.night_prepare(settings_for_core(s),job,Progress(s))

def assign(e,after,delta,alpha,corner):
    coeff={k:alpha*int(corner[0]) for k in SELECTION}
    coeff.update({k:alpha*int(corner[1]) for k in CONTENT})
    core.night_assign(e,after,delta,alpha,channels=coeff)

def key(split,row):return f'{split}_{row["entity"]:03d}'

def eval_row(e,row,replacement=None,layer=None,return_hidden=False):
    handle=None
    if replacement is not None:
        t=torch.as_tensor(replacement,device=e.c.device,dtype=torch.float32)[None]
        def replace(m,args):
            if t.shape!=args[0].shape:raise ValueError('Patch shape differs from live activation.')
            return (t,)+args[1:]
        handle=e.mods[f'model.layers.{layer}.self_attn.o_proj'].register_forward_pre_hook(replace)
    try:
        with torch.no_grad():
            inputs,q,labels=collate([row],e.tok.pad_token_id,e.c.device)
            h=e.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1]
            lp=e.model.lm_head(h).double().log_softmax(-1);p=lp[:,labels]
            loss=(q.double()*(q.double().log()-p)).sum().item()
            margin=(p[0,0]-p[0,1]).item();support=-p.logsumexp(-1).item()
            result=dict(loss=loss,margin=margin,target_margin=math.log(row['q'][0]/row['q'][1]),support=support,relation=loss-support)
            if return_hidden:result['_final_hidden']=h[0].detach().float().cpu().numpy()
            if not all(math.isfinite(result[k]) for k in ['loss','margin','support']):raise FloatingPointError('Nonfinite evaluation result.')
            return result
    finally:
        if handle is not None:handle.remove()


def trace(e,row,layers):
    """Two exact local derivatives at all token positions; no parameter gradients accumulated."""
    values={};handles=[]
    def pre(name):
        def h(m,args):values[name]=args[0]
        return h
    def post(name):
        def h(m,args,result):values[name]=result[0] if isinstance(result,tuple) else result
        return h
    for l in layers:
        root=f'model.layers.{l}'
        handles.append(e.mods[root+'.self_attn.o_proj'].register_forward_pre_hook(pre(f'{l}_Y')))
        handles.append(e.mods[root+'.mlp.gate_proj'].register_forward_hook(post(f'{l}_gate')))
        handles.append(e.mods[root].register_forward_hook(post(f'{l}_residual')))
    try:
        inputs,q,labels=collate([row],e.tok.pad_token_id,e.c.device)
        h=e.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1]
        lp=e.model.lm_head(h).double().log_softmax(-1);p=lp[:,labels]
        loss=(q.double()*(q.double().log()-p)).sum();margin=p[0,0]-p[0,1]
        names=list(values);targets=[values[n] for n in names]
        gm=torch.autograd.grad(margin,targets,retain_graph=True,allow_unused=False)
        gl=torch.autograd.grad(loss,targets,allow_unused=False)
        arrays={}
        for name,x,a,b in zip(names,targets,gm,gl):
            arrays[name]=x[0].detach().float().cpu().numpy()
            arrays[name+'_margin_grad']=a[0].detach().float().cpu().numpy()
            arrays[name+'_loss_grad']=b[0].detach().float().cpu().numpy()
        return arrays
    finally:
        for h in handles:h.remove()
        e.opt.zero_grad(set_to_none=True)


def capture_state(e,s,data,folder,layers):
    folder.mkdir(parents=True,exist_ok=True)
    for split in SPLITS:
        for row in data[split]:
            name=key(split,row);f=folder/(name+'.json');a=folder/(name+'.npz')
            if f.exists() and a.exists():continue
            progress_check(s,phase='capture computation and local task derivatives',split=split,entity=row['entity'])
            attention=core.night_attention_arrays(e,row,layers)
            if any(v['error']>s['attention_tolerance'] for v in attention.values()):
                raise ArithmeticError(f'Attention reconstruction audit failed: {name}')
            arrays=trace(e,row,layers)
            for l,v in attention.items():
                for k,value in v.items():arrays[f'{l}_{k}']=value
                if not np.allclose(arrays[f'{l}_Y'],v['native'],rtol=1e-5,atol=1e-5):
                    raise ArithmeticError('Separate capture forwards disagree.')
            r=eval_row(e,row,return_hidden=True);arrays['final_hidden']=r.pop('_final_hidden')
            for value in arrays.values():
                if not np.isfinite(value).all():raise FloatingPointError('Nonfinite capture array.')
            r.update(entity=row['entity'],split=split,q=row['q'])
            core.night_npz(a,**arrays);atomic_json(r,f)


def fit_frame(early,late,max_rank):
    """Uncentered proper rotation; identity outside a calibration-defined joint span."""
    early=np.asarray(early,dtype=float);late=np.asarray(late,dtype=float)
    joined=np.concatenate([early,late]);_,sv,vh=np.linalg.svd(joined,full_matrices=False)
    rank=min(max_rank,int(np.sum(sv>max(sv[0],1e-30)*1e-9)))
    B=vh[:rank].T
    if rank:
        aa=np.einsum('nd,dr->nr',late,B);bb=np.einsum('nd,dr->nr',early,B)
        u,_,vt=np.linalg.svd(np.einsum('nr,ns->rs',aa,bb),full_matrices=False)
        fix=np.ones(rank);fix[-1]=np.linalg.det(u@vt);R=(u*fix)@vt
    else:R=np.zeros((0,0))
    scale=max(0.,float(np.sum(late*early)/max(np.sum(late*late),1e-30)))
    return dict(basis=B,rotation=R,scale=np.array(scale),shift=(early-late).mean(0),
                fit_relative_error=np.array(np.linalg.norm(rotate(late,B,R)-early)/max(np.linalg.norm(early),1e-30)))

def rotate(x,B,R):
    a=np.einsum('nd,dr->nr',x,B);b=np.einsum('nr,rs->ns',a,R-np.eye(R.shape[0]))
    return x+np.einsum('nr,dr->nd',b,B)

def fitted_map(s,data,calibration,baseline,late,layer,file):
    if file.exists():return dict(np.load(file))
    xs=[];ys=[]
    for split in ['A_valid','B_valid']:
        for row in data[split]:
            if row['entity'] not in calibration:continue
            with np.load(baseline/(key(split,row)+'.npz')) as a,np.load(late/(key(split,row)+'.npz')) as b:
                x=a[f'{layer}_Y'];y=b[f'{layer}_Y']
                ix=np.unique(np.linspace(0,len(x)-1,min(s['fit_positions'],len(x))).astype(int))
                xs.append(x[ix]);ys.append(y[ix])
    result=fit_frame(np.concatenate(xs),np.concatenate(ys),s['fit_max_rank'])
    core.night_npz(file,**result);return result


def replacement(kind,early,current,frame,layer,repeat,rowkey):
    native=current[f'{layer}_Y'].astype(float);p0=early[f'{layer}_P'].astype(float);v0=early[f'{layer}_V'].astype(float)
    p=current[f'{layer}_P'].astype(float);v=current[f'{layer}_V'].astype(float)
    def flat(y):return y.transpose(1,0,2).reshape(native.shape)
    if kind=='native':return native.copy()
    if kind=='sham':return flat(np.einsum('hij,hjk->hik',p,v))
    if kind=='earlier_routing':return flat(np.einsum('hij,hjk->hik',p0,v))
    if kind=='earlier_values':return flat(np.einsum('hij,hjk->hik',p,v0))
    if kind=='earlier_both':return flat(np.einsum('hij,hjk->hik',p0,v0))
    if kind=='rotation':return rotate(native,frame['basis'],frame['rotation'])
    if kind=='scale':return native*float(frame['scale'])
    if kind=='mean_shift':return native+frame['shift']
    if kind=='random_orthogonal':
        B,R=frame['basis'],frame['rotation'];rank=R.shape[0]
        rng=np.random.default_rng(core.seed_for(f'orthogonal/{layer}',8121+repeat))
        Q,_=np.linalg.qr(rng.normal(size=(rank,rank)))
        random_R=np.einsum('ij,jk,lk->il',Q,R,Q)
        return rotate(native,B,random_R)
    if kind.startswith('random_'):
        reference=kind[len('random_'):]
        target=replacement(reference,early,current,frame,layer,repeat,rowkey)
        rng=np.random.default_rng(core.seed_for(f'{rowkey}/{layer}/{reference}',9041+repeat))
        noise=rng.normal(size=native.shape)
        return native+noise*np.linalg.norm(target-native)/max(np.linalg.norm(noise),1e-30)
    raise ValueError(kind)


def summarize_rows(rows,calibration):
    out={}
    for split in SPLITS:
        for population in ['all','heldout_entities']:
            rr=[r for r in rows if r['split']==split and (population=='all' or r['entity'] not in calibration)]
            if rr:out[split+'/'+population]={k:float(np.mean([r[k] for r in rr])) for k in ['loss','margin','support','relation']}
    return out


def intervention(e,s,data,baseline,current,layer,kind,repeat,frame,folder,calibration):
    folder.mkdir(parents=True,exist_ok=True);rows=[]
    for split in SPLITS:
        for row in data[split]:
            name=key(split,row);f=folder/(name+'.json')
            if f.exists() and (folder/(name+'.npz')).exists():r=read(f)
            else:
                progress_check(s,phase='factorial activation intervention',layer=layer,condition=kind,split=split,entity=row['entity'])
                with np.load(baseline/(name+'.npz')) as a,np.load(current/(name+'.npz')) as b:
                    y=replacement(kind,a,b,frame,layer,repeat,name);native=b[f'{layer}_Y'].astype(float)
                    change=y-native;r=eval_row(e,row,y,layer,return_hidden=True)
                    final_hidden=r.pop('_final_hidden')
                    core.night_npz(folder/(name+'.npz'),patched_Y=y[-1],final_hidden=final_hidden)
                    r.update(entity=row['entity'],split=split,displacement_norm=float(np.linalg.norm(change)),
                        loss_linear=float(np.sum(change*b[f'{layer}_Y_loss_grad'])),
                        margin_linear=float(np.sum(change*b[f'{layer}_Y_margin_grad'])))
                    old=read(current/(name+'.json'));r['loss_change']=r['loss']-old['loss'];r['margin_change']=r['margin']-old['margin']
                    r['loss_remainder']=r['loss_change']-r['loss_linear']
                    r['margin_remainder']=r['margin_change']-r['margin_linear']
                    # Audit local activation directional derivatives on calibration prompts only.
                    if kind=='rotation' and split=='A_valid' and row['entity'] in calibration:
                        eps=s['projection_fd_epsilon'];plus=eval_row(e,row,native+eps*change,layer);minus=eval_row(e,row,native-eps*change,layer)
                        fd=(plus['margin']-minus['margin'])/(2*eps);expected=r['margin_linear']
                        r['margin_fd']=fd;r['margin_fd_pass']=abs(fd-expected)<=1e-4+s['projection_fd_rtol']*max(abs(fd),abs(expected))
                if kind in ['sham','native'] and abs(r['loss_change'])>s['sham_atol']:
                    raise ArithmeticError('Sham intervention failed; refusing mechanistic interpretation.')
                atomic_json(r,f)
            rows.append(r)
    result=dict(layer=layer,kind=kind,repeat=repeat,means=summarize_rows(rows,calibration),rows=rows)
    atomic_json(result,folder/'summary.json')
    patch_topology(s,data,calibration,baseline,current,folder,layer)
    return result



def patch_topology(s,data,calibration,baseline,current,folder,layer):
    """Measure repaired-site and final-hidden geometry on held-out entities."""
    for split in SPLITS:
        f=folder/(split+'_topology.json')
        if f.exists():continue
        progress_check(s,phase='topology after intervention',layer=layer,split=split)
        clouds={k:[] for k in ['early','native','patched','early_final','native_final','patched_final']};ids=[]
        for row in sorted(data[split],key=lambda r:r['entity']):
            if row['entity'] in calibration or len(ids)>=32:continue
            name=key(split,row)
            with np.load(baseline/(name+'.npz')) as a,np.load(current/(name+'.npz')) as b,np.load(folder/(name+'.npz')) as c:
                for name2,v in [('early',a[f'{layer}_Y'][-1]),('native',b[f'{layer}_Y'][-1]),('patched',c['patched_Y']),
                                ('early_final',a['final_hidden']),('native_final',b['final_hidden']),('patched_final',c['final_hidden'])]:clouds[name2].append(v)
            ids.append(row['entity'])
        results={}
        for suffix in ['', '_final']:
            anchor=np.stack(clouds['early'+suffix]);d0=core.points_distance(anchor);positive=d0[d0>1e-12]
            scale=float(np.median(positive)) if len(positive) else 1.
            for label in ['early','native','patched']:
                x=np.stack(clouds[label+suffix]);d=core.points_distance(x)
                results[label+suffix]=core.persistent_homology(d/scale,[.25,.5,1.,1.5,2.])
            results['scale'+suffix]=scale
        atomic_json(dict(entities=ids,results=results,interpretation='Fixed identities and baseline scales; final-hidden topology can change downstream even when the patched-site rotation preserves all distances.'),f)


def cloud_stats(x,x0,metric_basis,nulls,seed):
    """Fixed-identity landmarks, baseline distance scale, empirical covariance-preserving null."""
    x=np.asarray(x,float);x0=np.asarray(x0,float);xc=x-x.mean(0)
    ev=np.linalg.eigvalsh(xc@xc.T/max(len(x)-1,1));ev=np.maximum(ev,0)
    d=core.points_distance(x);d0=core.points_distance(x0);positive=d0[d0>1e-12];scale=float(np.median(positive)) if len(positive) else 1.
    ph=core.persistent_homology(d/scale,[.25,.5,1.,1.5,2.])
    taskd=core.points_distance(x@metric_basis);task0=core.points_distance(x0@metric_basis)
    positive=task0[task0>1e-12];tscale=float(np.median(positive)) if len(positive) else 1.
    tph=core.persistent_homology(taskd/tscale,[.25,.5,1.,1.5,2.])
    null=[];rng=np.random.default_rng(seed);n=len(x);unit=np.ones((n,1))/math.sqrt(n)
    # Q fixes the constant vector, preserving centering and X_centered^T X_centered.
    basis=np.linalg.svd(unit.T,full_matrices=True)[2][1:].T
    for _ in range(nulls):
        q,_=np.linalg.qr(rng.normal(size=(n-1,n-1)));mix=unit@unit.T+basis@q@basis.T
        xn=mix@xc+x.mean(0);dn=core.points_distance(xn)
        np.testing.assert_allclose(xn-xn.mean(0),mix@xc,atol=1e-8)
        pn=core.persistent_homology(dn/scale,[.25,.5,1.,1.5,2.])
        null.append(pn['H1']['finite_total_persistence'])
    return dict(covariance_eigenvalues=ev.tolist(),effective_rank=core.entropy_rank(ev),
        PH=ph,task_semimetric_PH=tph,baseline_distance_scale=scale,task_baseline_scale=tscale,
        covariance_preserving_H1_null=null,interpretation='Empirical spectrum-preserving point mixing, not a Marchenko-Pastur test. Task metric uses frozen calibration margin gradients at the final token; not the full Fisher or Hessian.')


def geometry(s,data,calibration,baseline,current,layers,folder):
    folder.mkdir(exist_ok=True)
    for layer in layers:
        # Fit metric on valid calibration entities, never on held-out losses.
        grads=[]
        for split in ['A_valid','B_valid']:
            for row in data[split]:
                if row['entity'] in calibration:
                    with np.load(baseline/(key(split,row)+'.npz')) as a:
                        grads.append(a[f'{layer}_Y_margin_grad'][-1]*math.sqrt(row['q'][0]*row['q'][1]))
        metric=np.stack(grads).T/math.sqrt(len(grads))
        for split in SPLITS:
            file=folder/f'layer{layer:02d}_{split}.json'
            if file.exists():continue
            progress_check(s,phase='task geometry and topology',layer=layer,split=split)
            xs=[];ys=[];ids=[];effects=[]
            rr=[r for r in data[split] if r['entity'] not in calibration]
            # Exact built-in PH is bounded to 32 landmarks; deterministic entity order.
            for row in sorted(rr,key=lambda r:r['entity'])[:32]:
                name=key(split,row)
                with np.load(baseline/(name+'.npz')) as a,np.load(current/(name+'.npz')) as b:
                    x=a[f'{layer}_Y'].astype(float);y=b[f'{layer}_Y'].astype(float)
                    p0=a[f'{layer}_P'].astype(float);v0=a[f'{layer}_V'].astype(float);p=b[f'{layer}_P'].astype(float);v=b[f'{layer}_V'].astype(float)
                    terms=[np.einsum('hij,hjk->hik',p-p0,v0),np.einsum('hij,hjk->hik',p0,v-v0),np.einsum('hij,hjk->hik',p-p0,v-v0)]
                    terms=[t.transpose(1,0,2).reshape(x.shape) for t in terms]
                    reconstruction=float(np.linalg.norm(sum(terms)-(y-x))/max(np.linalg.norm(y-x),1e-12))
                    effect=dict(entity=row['entity'],q0=row['q'][0],reconstruction_relative_error=reconstruction)
                    for label,t in zip(['routing','content','cross'],terms):
                        effect[label]=dict(norm=float(np.linalg.norm(t)),
                            margin_projection_before=float(np.sum(t*a[f'{layer}_Y_margin_grad'])),
                            margin_projection_current=float(np.sum(t*b[f'{layer}_Y_margin_grad'])),
                            loss_projection_before=float(np.sum(t*a[f'{layer}_Y_loss_grad'])))
                    ga=a[f'{layer}_gate'];gb=b[f'{layer}_gate'];effect['gate_sign_change_fraction']=float(np.mean((ga>0)!=(gb>0)))
                    effect['residual_change_norm']=float(np.linalg.norm(b[f'{layer}_residual']-a[f'{layer}_residual']))
                    effect['attention_entropy_before']=float(np.mean(-np.sum(p0*np.log(np.maximum(p0,1e-30)),-1)))
                    effect['attention_entropy_current']=float(np.mean(-np.sum(p*np.log(np.maximum(p,1e-30)),-1)))
                    effects.append(effect);xs.append(x[-1]);ys.append(y[-1]);ids.append(row['entity'])
            result=cloud_stats(np.stack(ys),np.stack(xs),metric,s['topology_nulls'],core.seed_for(str(file),19))
            result.update(entities=ids,per_entity=effects);atomic_json(result,file)


def factorial_job(s,job):
    out=Path(s['output']);folder=out/job['id'];folder.mkdir(parents=True,exist_ok=True)
    e,data,after,delta=prepare(s,job['seed'],job['event'],folder)
    layers=sorted(set(layers_for(e,s)+layers_for(e,s,True)));patch_layers=layers_for(e,s,True)
    cal=partition(data,s);atomic_json(dict(calibration_entities=sorted(cal),fit_splits=['A_valid','B_valid'],evaluation='All splits also report identities excluded from calibration.'),folder/'partition.json')
    baseline=folder.parent/f"baseline_seed{job['seed']}_step{job['event']:03d}"
    core.path_set_weights(e,after,delta,0);capture_state(e,s,data,baseline,layers)
    alpha=job['alpha'];summaries={}
    # First capture every corner; the map is fitted on 11 and then held fixed for all corners.
    for corner in CORNERS:
        assign(e,after,delta,alpha,corner);capture_state(e,s,data,folder/corner/'capture',layers)
    for corner in CORNERS:
        current=folder/corner/'capture';assign(e,after,delta,alpha,corner)
        geometry(s,data,cal,baseline,current,layers,folder/corner/'geometry')
        raw=[read(current/(key(split,row)+'.json')) for split in SPLITS for row in data[split]]
        summaries.setdefault('unpatched',{})[corner]=summarize_rows(raw,cal)
        for layer in patch_layers:
            frame=fitted_map(s,data,cal,baseline,folder/'11'/'capture',layer,folder/f'frame_layer{layer:02d}.npz')
            for kind in ['sham','earlier_routing','earlier_values','earlier_both','rotation','scale','mean_shift',
                         'random_rotation','random_earlier_both','random_orthogonal']:
                count=s['random_repeats'] if kind.startswith('random_') else 1
                for repeat in range(count):
                    label=f'layer{layer:02d}_{kind}_{repeat}'
                    r=intervention(e,s,data,baseline,current,layer,kind,repeat,frame,folder/corner/label,cal)
                    summaries.setdefault(label,{})[corner]=r['means']
    interactions=[]
    for label,corners in summaries.items():
        for split in corners['00']:
            vals={k:corners[k][split]['loss'] for k in CORNERS}
            interaction=vals['11']-vals['10']-vals['01']+vals['00']
            interactions.append(dict(intervention=label,population=split,losses=vals,interaction=interaction))
    atomic_json(interactions,folder/'interactions.json')
    entity_interactions=[]
    for label in summaries:
        for split in SPLITS:
            for row in data[split]:
                values={}
                for corner in CORNERS:
                    file=folder/corner/('capture' if label=='unpatched' else label)/(key(split,row)+'.json')
                    values[corner]=read(file)['loss']
                entity_interactions.append(dict(intervention=label,split=split,entity=row['entity'],heldout=row['entity'] not in cal,
                    interaction=values['11']-values['10']-values['01']+values['00'],losses=values))
    atomic_json(entity_interactions,folder/'entity_interactions.json')
    del e,after,delta
    if torch.cuda.is_available():torch.cuda.empty_cache()


def matched_directions_job(s,job):
    """Reset-m vs original direction at equal parameter displacement from the same weights."""
    folder=Path(s['output'])/job['id'];folder.mkdir(parents=True,exist_ok=True)
    c=settings_for_core(s);j=dict(seed=job['seed'],event=job['event'],id=job['id'])
    e,data=core.night_prepare(c,j,Progress(s),do_step=False)
    _,orig_after,orig_delta=core.measured_adam_step(e,data,job['seed'],job['event'],Progress(s))
    norm0=math.sqrt(sum(core.exact_norm2(d) for d in orig_delta.values()));del e
    if torch.cuda.is_available():torch.cuda.empty_cache()
    e,data=core.night_prepare(c,j,Progress(s),do_step=False)
    for st in e.opt.state.values():
        if 'exp_avg' in st:st['exp_avg'].zero_()
    _,reset_after,reset_delta=core.measured_adam_step(e,data,job['seed'],job['event'],Progress(s))
    norm1=math.sqrt(sum(core.exact_norm2(d) for d in reset_delta.values()))
    scale=norm0/max(norm1,1e-30);dot=sum(core._dot_cpu(orig_delta[n],reset_delta[n]) for n in orig_delta)
    # Audit identical origins before comparing directions.
    err=math.sqrt(sum(core.exact_norm2((orig_after[n]-orig_delta[n])-(reset_after[n]-reset_delta[n])) for n in orig_delta))
    if err>1e-6*max(norm0,1e-8):raise ArithmeticError('Direction comparisons have different starting weights.')
    results=[]
    for name,d in [('original',orig_delta),('reset_m_norm_matched',reset_delta)]:
        factor=1. if name=='original' else scale
        for fraction in s['matched_direction_fractions']:
            progress_check(s,phase='equal-norm optimizer direction intervention',condition=name,alpha=fraction)
            with torch.no_grad():
                for n,p in e.params.items():
                    base=orig_after[n].double()-orig_delta[n].double()
                    p.copy_((base+fraction*factor*d[n].double()).to(p.device,dtype=p.dtype))
            results.append(dict(direction=name,fraction=fraction,displacement_norm=fraction*norm0,losses=core.night_eval(e,data)))
    atomic_json(dict(original_norm=norm0,reset_m_native_norm=norm1,reset_scale=scale,
        direction_cosine=dot/max(norm0*norm1,1e-30),origin_error=err,results=results,
        meaning='Single parameter displacement interventions. Equal norm, fixed origin; not optimizer continuation or equal B-learning comparisons.'),folder/'directions.json')
    del e,orig_after,orig_delta,reset_after,reset_delta
    if torch.cuda.is_available():torch.cuda.empty_cache()


def branch_job(s,job):
    """Full state committed after every traced update; interrupted steps replay deterministically."""
    from bridge_replay import save,load
    folder=Path(s['output'])/job['id'];folder.mkdir(parents=True,exist_ok=True)
    j=dict(seed=job['seed'],event=job['event'],id=job['id'])
    e,data=core.night_prepare(settings_for_core(s),j,Progress(s),do_step=False)
    cal=partition(data,s);layers=layers_for(e,s,True)
    baseline=folder/'baseline';capture_state(e,s,data,baseline,layers)
    checkpoint=folder/'resume.pt';rows=[];start=job['event']
    if checkpoint.exists():
        saved=load(checkpoint);e.restore(saved['engine']);rows=saved['rows'];start=saved['step']+1;del saved
    elif job['branch']=='reset_first_moment':
        for state in e.opt.state.values():
            if 'exp_avg' in state:state['exp_avg'].zero_()
    if job['branch']=='all_steps_lr_025':
        for g in e.opt.param_groups:g['lr']=e.c.lr*.25
    stop=min(e.c.b_steps,job['event']+s['branch_steps']-1)
    for step in range(start,stop+1):
        progress_check(s,phase='optimizer continuation with computational traces',step=step,condition=job['branch'])
        update,after,delta=core.measured_adam_step(e,data,job['seed'],step,Progress(s))
        del after,delta
        losses=core.night_eval(e,data);stepfolder=folder/f'step{step:03d}';stepfolder.mkdir(exist_ok=True)
        atomic_json(update,stepfolder/'update.json')
        if (step-job['event'])%s['branch_trace_every']==0 or step==stop:
            capture_state(e,s,data,stepfolder/'capture',layers)
            geometry(s,data,cal,baseline,stepfolder/'capture',layers,stepfolder/'geometry')
        rows.append(dict(step=step,losses=losses,optimizer_channels=update['totals']))
        # Source checkpoint is never modified. Two checkpoint-sized files may coexist during commit.
        import shutil
        required=2*e.pb*3+s['minimum_free_gb']*1024**3
        if shutil.disk_usage(folder).free<required:raise Pause('Insufficient space for atomic continuation checkpoint.')
        save(dict(engine=e.pack(),rows=rows,step=step),checkpoint)
        atomic_json(dict(branch=job['branch'],complete=step==stop,rows=rows),folder/'curve.json')
    del e
    if torch.cuda.is_available():torch.cuda.empty_cache()


def plan(s):
    jobs=[]
    def factorial(seed,event,alpha):
        jobs.append(dict(id=f'factorial_seed{seed}_step{event:03d}_alpha{alpha:g}',kind='factorial',seed=seed,event=event,alpha=alpha))
    # Complete a core causal contrast and continuation per seed before broadening the grid.
    for seed in s['seeds']:
        factorial(seed,s['events'][0],s['fractions'][0])
        event=s['branch_event']
        jobs.append(dict(id=f'directions_seed{seed}_step{event:03d}',kind='directions',seed=seed,event=event))
        for branch in s['branches']:
            jobs.append(dict(id=f'branch_seed{seed}_step{event:03d}_{branch}',kind='branch',seed=seed,event=event,branch=branch))
    for alpha in s['fractions']:
        for event in s['events']:
            for seed in s['seeds']:
                if alpha==s['fractions'][0] and event==s['events'][0]:continue
                factorial(seed,event,alpha)
    return jobs


def report(output,display=False):
    import fcntl
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    with open(out/'report.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            if display:print('A report refresh is already in progress.')
            return out/'REPORT.txt'
        return _report(out,display)


def _report(output,display=False):
    out=Path(output);all_interactions=[];branches=[]
    for f in sorted(out.glob('factorial_*/interactions.json')):
        for row in read(f):all_interactions.append(dict(job=f.parent.name,**row))
    for f in sorted(out.glob('branch_*/curve.json')):
        r=read(f)
        for split in SPLITS:
            vals=[v['losses'][split]['mean'] for v in r['rows']]
            if vals:branches.append(dict(job=f.parent.name,split=split,complete=r['complete'],steps=len(vals),mean=float(np.mean(vals)),peak=max(vals),final=vals[-1]))
    atomic_json(all_interactions,out/'interaction_summary.json');atomic_json(branches,out/'branch_summary.json')
    if all_interactions:core.write_csv([dict(job=r['job'],intervention=r['intervention'],population=r['population'],interaction=r['interaction'],**{'loss_'+k:v for k,v in r['losses'].items()}) for r in all_interactions],out/'interaction_summary.csv')
    if branches:core.write_csv(branches,out/'branch_summary.csv')
    trajectory=[]
    for file in sorted(out.glob('branch_*/step*/geometry/layer*_*.json')):
        geometry_row=read(file);step=int(file.parent.parent.name[4:]);branch=file.parents[2]
        split=file.stem.split('_',1)[1];layer=int(file.stem.split('_')[0][5:])
        for feature in geometry_row['per_entity']:
            entity=feature['entity'];prediction=read(file.parent.parent/'capture'/f'{split}_{entity:03d}.json')
            trajectory.append(dict(job=branch.name,step=step,layer=layer,split=split,entity=entity,loss=prediction['loss'],margin=prediction['margin'],
                routing_norm=feature['routing']['norm'],content_norm=feature['content']['norm'],cross_norm=feature['cross']['norm'],
                cross_margin_projection=feature['cross']['margin_projection_current'],gate_sign_change=feature['gate_sign_change_fraction'],
                residual_change=feature['residual_change_norm']))
    if trajectory:core.write_csv(trajectory,out/'trajectory_features.csv')
    lines=['MECHANISM BRIDGE — controlled interventions; no universal mechanism assumed']
    state=read(out/'status.json') if (out/'status.json').exists() else {}
    lines.append(f"Status: {state.get('status','unknown')}; completed {state.get('completed',0)}/{state.get('total','?')} jobs")
    for r in all_interactions:
        if r['population']=='A_test/heldout_entities' and (r['intervention']=='unpatched' or '_rotation_0' in r['intervention'] and 'random_' not in r['intervention']):
            lines.append(f"{r['job']} {r['intervention']}: held-out A interaction {r['interaction']:+.6f}")
    for r in branches:
        if r['split']=='A_test':lines.append(f"{r['job']}: A final {r['final']:.6f}; step-average {r['mean']:.6f}; peak {r['peak']:.6f}; complete={r['complete']}")
    lines.extend(['Calibration uses only valid prompts from disjoint entity identities. All-entity and held-out-entity means are separate.',
        'Rotation map is fixed across the four corners. A fitted rotation is not evidence until its intervention and controls succeed.',
        'Additive random patches match each condition-specific activation displacement; orthogonal controls instead match rotation eigenangles. No p-values are inferred from a few controls.',
        'PH is rotation invariant; a rotation rescue need not change Euclidean PH.',
        'Gradient projections are local; finite patch remainders include downstream nonlinear effects.',
        'This queue may require multiple sessions. Completed units resume; missing units are not negative results.'])
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    if display:print('\n'.join(lines[:35]));print('Full report:',out/'REPORT.txt')
    try:plots(out)
    except Exception as exc:atomic_json(dict(error=str(exc)),out/'plot_error.json')
    return out/'REPORT.txt'


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=sorted(out.glob('branch_*/curve.json'))
    if files:
        fig,axs=plt.subplots(1,2,figsize=(12,4))
        for f in files:
            r=read(f);rows=r['rows']
            for ax,split in zip(axs,['A_test','B_test']):
                ax.plot([x['step'] for x in rows],[x['losses'][split]['mean'] for x in rows],label=f.parent.name.replace('branch_',''))
                ax.set_title(split);ax.set_xlabel('B update');ax.set_ylabel('Mean target KL');ax.grid(alpha=.2)
        axs[1].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'continuations.png',dpi=150);plt.close(fig)
    rows=read(out/'interaction_summary.json');rr=[r for r in rows if r['population']=='A_test/heldout_entities' and r['intervention']=='unpatched']
    if rr:
        fig,ax=plt.subplots(figsize=(max(6,len(rr)*.35),4))
        ax.bar(range(len(rr)),[r['interaction'] for r in rr]);ax.set_xticks(range(len(rr)),[r['job'].replace('factorial_','') for r in rr],rotation=70,ha='right',fontsize=7)
        ax.set_ylabel('Held-out A-test loss interaction');ax.axhline(0,color='black',lw=.7)
        fig.tight_layout();fig.savefig(out/'interactions.png',dpi=150);plt.close(fig)
        first=rr[0]['job'];base=rr[0]['interaction']
        patches=[r for r in rows if r['job']==first and r['population']=='A_test/heldout_entities' and r['intervention']!='unpatched']
        if patches:
            fig,ax=plt.subplots(figsize=(10,max(4,len(patches)*.16)))
            ax.barh(range(len(patches)),[r['interaction']-base for r in patches]);ax.set_yticks(range(len(patches)),[r['intervention'] for r in patches],fontsize=6)
            ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Patched interaction minus unpatched interaction');ax.set_title(first+' · held-out A test')
            fig.tight_layout();fig.savefig(out/'repair_interactions.png',dpi=150);plt.close(fig)


def export(output,include_arrays=False):
    import uuid
    out=Path(output);file=out/'results_share.zip';tmp=out/('results_share.'+uuid.uuid4().hex+'.zip.tmp')
    atomic_json(dict(raw_arrays_included=include_arrays,checkpoints_included=False,version=VERSION),out/'EXPORT_SCOPE.json')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in out.rglob('*'):
            if f.is_file() and f.suffix in ({'.json','.csv','.txt','.png','.npz'} if include_arrays else {'.json','.csv','.txt','.png'}):
                if f.name not in ['process.json']:z.write(f,f.relative_to(out))
    os.replace(tmp,file);return file


def signature(s):
    value={k:v for k,v in s.items() if k not in ['hours','reserve_minutes','output','minimum_free_gb','export_arrays']}
    h=hashlib.sha256(json.dumps(value,sort_keys=True).encode())
    for name in ['mechanism_bridge.py','bridge_core.py','bridge_replay.py']:
        h.update((Path(__file__).parent/name).read_bytes())
    # Metadata/content hashes for small source records; checkpoint hashes recorded by worker below.
    h.update((Path(s['source'])/'config.json').read_bytes())
    for seed in s['seeds']:h.update((Path(s['source'])/f'seed{seed}'/'data.json').read_bytes())
    return h.hexdigest()


def fingerprint(s):
    out=Path(s['output']);f=out/'source_fingerprints.json';previous=read(f) if f.exists() else None;records=[]
    src=Path(s['source'])
    paths=[src/'config.json']
    for seed in s['seeds']:
        paths.append(src/f'seed{seed}'/'data.json')
        paths.extend(sorted((src/f'seed{seed}').glob('fork*/fork.pt')))
    for p in paths:
        progress_check(s,phase='fingerprinting original source',file=str(p))
        h=hashlib.sha256()
        with open(p,'rb') as stream:
            while True:
                block=stream.read(16*1024**2)
                if not block:break
                h.update(block)
        records.append(dict(path=str(p),sha256=h.hexdigest(),bytes=p.stat().st_size))
    if previous is not None and previous!=records:raise ValueError('Original source changed; use restart() to create a fresh result directory.')
    atomic_json(records,f)


def run(s):
    import fcntl
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'worker.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(Pause('Stop requested.')))
        jobs=plan(s);completed=sum((out/j['id']/'DONE.json').exists() for j in jobs)
        atomic_json(dict(status='running',started=time.time(),heartbeat=time.time(),completed=completed,total=len(jobs)),out/'status.json')
        status='complete';error=None
        try:
            fingerprint(s)
            import transformers
            hardware=core.night_hardware();hardware.update(numpy=np.__version__,transformers=transformers.__version__,package_version=VERSION)
            atomic_json(hardware,out/'hardware.json')
            for j in jobs:
                folder=out/j['id'];folder.mkdir(exist_ok=True)
                if (folder/'DONE.json').exists():continue
                progress_check(s,job=j['id'],phase='starting job',completed=completed,total=len(jobs))
                {'factorial':factorial_job,'directions':matched_directions_job,'branch':branch_job}[j['kind']](s,j)
                atomic_json(dict(finished=time.time(),job=j),folder/'DONE.json');completed+=1
                progress_check(s,completed=completed);report(out)
        except Pause as exc:status='stopped' if (out/'STOP').exists() else 'budget_paused';error=str(exc)
        except Exception as exc:
            status='failed';error=str(exc);atomic_json(dict(error=error,traceback=traceback.format_exc()),out/'failure.json')
        finally:
            atomic_json(dict(status=status,error=error,heartbeat=time.time(),completed=completed,total=len(jobs)),out/'status.json')
            report(out);export(out,s['export_arrays'])


def process(output):
    import psutil
    f=Path(output)/'process.json'
    if not f.exists():return None
    r=read(f)
    try:
        p=psutil.Process(r['pid'])
        if abs(p.create_time()-r['created'])>.1 or p.status()==psutil.STATUS_ZOMBIE:return None
        cmd=p.cmdline()
        if not any(Path(x).name=='mechanism_bridge.py' for x in cmd):return None
        if str((Path(output)/'settings.json').resolve()) not in cmd:return None
        return p
    except psutil.Error:return None


def status(output):
    out=Path(output);r=read(out/'status.json') if (out/'status.json').exists() else dict(status='not_started')
    r['alive']=process(out) is not None
    if not r['alive'] and r['status'] in ['running','launching']:r['status']='interrupted; ready to resume'
    return r


def launch(s):
    """Idempotent; changed experiment settings create a sibling directory, preserving results."""
    import fcntl,psutil,uuid
    validate(s);out=Path(s['output']).expanduser().resolve();s['output']=str(out);out.mkdir(parents=True,exist_ok=True)
    with open(out/'launch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if process(out) is not None:return out
        sig=signature(s);old=read(out/'signature.json') if (out/'signature.json').exists() else None
        if old and old['signature']!=sig:
            s['output']=str(out.with_name(out.name+'_new_'+uuid.uuid4().hex[:8]));return launch(s)
        # Ensure a previous worker has actually released ownership.
        with open(out/'worker.lock','a') as worker:
            try:fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return out
        (out/'STOP').unlink(missing_ok=True)
        atomic_json(s,out/'settings.json');atomic_json(dict(signature=sig),out/'signature.json')
        atomic_json(dict(deadline=time.time()+(s['hours']*60-s['reserve_minutes'])*60),out/'budget.json')
        atomic_json(dict(status='launching',heartbeat=time.time()),out/'status.json')
        with open(out/'worker.log','a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--settings',str(out/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:created=psutil.Process(p.pid).create_time()
        except psutil.Error:created=0
        atomic_json(dict(pid=p.pid,created=created),out/'process.json')
    return out


def stop(output,force_after=20):
    """Stop only the verified worker belonging to this output; resumable completed units remain."""
    import psutil
    out=Path(output);out.mkdir(parents=True,exist_ok=True);(out/'STOP').touch();p=process(out)
    if p is None:return 'No active worker. Completed units are preserved.'
    try:p.wait(timeout=force_after)
    except psutil.TimeoutExpired:
        p.terminate()
        try:p.wait(timeout=10)
        except psutil.TimeoutExpired:p.kill();p.wait(timeout=10)
    return 'Stopped. Run launch(SETTINGS) to resume, or restart(SETTINGS) for a fresh run.'


def restart(s):
    import uuid
    stop(s['output']);old=Path(s['output']);s['output']=str(old.with_name(old.name+'_restart_'+uuid.uuid4().hex[:8]));return launch(s)


def refresh(output):
    r=status(output);print(f"Status: {r['status']} | alive: {r['alive']} | jobs: {r.get('completed',0)}/{r.get('total','?')}")
    for k in ['job','phase','step','alpha','layer','condition','error']:
        if r.get(k) is not None:print(k+':',r[k])
    if r.get('heartbeat'):print('Last progress:',round(time.time()-r['heartbeat'],1),'seconds ago. Liveness does not guarantee progress.')
    print('Saved prompt/diagnostic records:',sum(1 for _ in Path(output).rglob('*.json')))
    print('Results:',Path(output).resolve())
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','report','export']);p.add_argument('--settings');p.add_argument('--output');a=p.parse_args()
    if a.command=='run':run(read(a.settings))
    elif a.command=='report':report(a.output,True)
    else:print(export(a.output))

if __name__=='__main__':main()
