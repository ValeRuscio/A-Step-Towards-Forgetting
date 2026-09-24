"""Matched prompt/entity derivatives and finite-update geometry for natural OLMo.
Diagnostic interpolation is restored afterward; no modified training branches.
"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/olmo_precise_forgetting_mpl')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, gc, hashlib, json, math, shutil, sqlite3, subprocess, sys, time, traceback, uuid, zipfile
from pathlib import Path
import numpy as np
import torch
import natural_geometry as ng
from association_core import Config,Engine,make_data,minibatches,digest,load,collate
from geometry_math import points_distance,persistent_homology
VERSION='precise-forgetting-1.0'
SPLITS=['A_valid','A_test','B_valid','B_test']
atomic_json=ng.atomic_json
read=ng.read


def defaults(source=None,output=None):
    source=source or ng.discover_source()
    if not source:raise FileNotFoundError('Set SOURCE to the original olmo_association_v1 directory.')
    cfg=read(Path(source)/'config.json')
    priority=[(1,21),(2,11),(3,7),(1,48),(2,35),(3,121),(1,20),(2,96),(3,6),(1,47),(2,10),(3,8)]
    jobs=[dict(seed=seed,step=step) for seed,step in priority if seed in cfg['seeds'] and step<=cfg['b_steps']]
    return dict(version=VERSION,source=str(Path(source).resolve()),output=str(Path(output or Path.cwd()/'runs/olmo_precise_forgetting_v1').resolve()),
        jobs=jobs,device='cuda:0',threads=8,hours=11.5,reserve_minutes=10,minimum_free_gb=25.,max_output_gb=30.,
        splits=SPLITS.copy(),layers=[],partition_seed=8041,frame_rank=8,topology_landmarks=16,
        path_levels=[3,5,9,17],integration_atol=1e-5,integration_rtol=.005,
        fd_epsilon=.01,fd_atol=5e-4,fd_rtol=.05,fd_entities_per_split=2,
        selection_note='Retrospective diagnostic events selected from the previous three trajectories; not independent confirmatory samples.')


def validate(s):
    if s.get('version')!=VERSION:raise ValueError('Use defaults() from this release.')
    c=Config(**read(Path(s['source'])/'config.json'));c.device=s['device'];c.threads=s['threads']
    if not s['jobs'] or len({(j['seed'],j['step']) for j in s['jobs']})!=len(s['jobs']):raise ValueError('Provide unique seed/update jobs.')
    for j in s['jobs']:
        if not 1<=j['step']<=c.b_steps:raise ValueError('Update outside original B trajectory.')
        for f in ['anchor.pt','data.json']:
            if not (Path(s['source'])/f"seed{j['seed']}"/f).exists():raise FileNotFoundError(f)
    if set(s['splits'])!=set(SPLITS):raise ValueError('This paired design requires all four original valid/test task splits.')
    if s['path_levels'][0]!=3 or any(n<3 or (n-1)&(n-2) for n in s['path_levels']) or s['path_levels']!=sorted(set(s['path_levels'])):raise ValueError('Use increasing nested grids: 3,5,9,17,33,...')
    if s['hours']*60<=s['reserve_minutes'] or s['minimum_free_gb']<0 or s['max_output_gb']<=0:raise ValueError('Invalid budget.')
    if not 2<=s['topology_landmarks']<=min(32,c.entities//2):raise ValueError('Need 2..32 held-out topology landmarks, at most half the entities.')
    if not 0<s['fd_epsilon']<.5 or s['fd_entities_per_split']<1:raise ValueError('Invalid finite-difference audit.')
    if min(s['integration_atol'],s['integration_rtol'],s['fd_atol'],s['fd_rtol'])<=0:raise ValueError('Tolerances must be positive.')
    if not c.smoke and not Path(c.model).exists() and (len(c.revision)!=40 or any(x not in '0123456789abcdef' for x in c.revision.lower())):raise ValueError('Use the original pinned HF revision; do not replace it with main.')
    return c


def job_id(job):return f"seed{job['seed']}_update{job['step']:03d}"


class Control:
    def __init__(self,s):
        self.s=s;self.out=Path(s['output']);self.deadline=time.time()+s['hours']*3600-s['reserve_minutes']*60
    def check(self):
        if (self.out/'STOP').exists():raise ng.Pause('Stop requested; completed prompt records are preserved.')
        if time.time()>=self.deadline:raise ng.Pause('Session time budget reached.')
        if shutil.disk_usage(self.out).free<self.s['minimum_free_gb']*1024**3:raise ng.Pause('Disk reserve reached.')
    def pulse(self,**fields):
        self.check();atomic_json(dict(status='running',last_progress=time.time(),**fields),self.out/'status.json')
    def storage(self):
        size=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
        if size>self.s['max_output_gb']*1024**3:raise ng.Pause('Output storage limit reached; increase max_output_gb to continue.')


def connect(output):
    con=sqlite3.connect(Path(output)/'measurements.sqlite',timeout=60);con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS probes(job TEXT, alpha REAL, split TEXT, entity INTEGER, record TEXT NOT NULL, PRIMARY KEY(job,alpha,split,entity))')
    con.execute('CREATE TABLE IF NOT EXISTS geometry(job TEXT, split TEXT, record TEXT NOT NULL, PRIMARY KEY(job,split))')
    con.commit();return con


def tensor_hash(weights):
    h=hashlib.sha256()
    for n,w in weights.items():h.update(n.encode());h.update(w.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(e,s,job,ctl):
    """Reconstruct from the nearest original fork, never from a patched branch."""
    src=Path(s['source'])/f"seed{job['seed']}";data=make_data(e.tok,e.c,job['seed'])
    if digest(data)!=read(src/'data.json')['hash']:raise ValueError('Original task-data hash mismatch.')
    forks=[p for p in src.glob('fork*/fork.pt') if int(p.parent.name[4:])<=job['step']]
    if forks:
        p=max(forks,key=lambda p:int(p.parent.name[4:]));z=load(p);start=int(z['step'])
        if start!=int(p.parent.name[4:]):raise ValueError('Fork label/state disagree.')
        e.restore(z['engine']);del z
    else:
        p=src/'anchor.pt';e.restore(load(p));start=1
        if e.c.transition=='reset':e.reset_optimizer()
    for step in range(start,job['step']):
        ctl.pulse(job=job_id(job),phase='replaying original trajectory',step=step)
        e.gradient(minibatches(data,'B',job['seed'],step,e.c));e.opt.step();e.opt.zero_grad(set_to_none=True)
    ctl.pulse(job=job_id(job),phase='forming actual Adam displacement',step=job['step'])
    before={n:p.detach().cpu().clone() for n,p in e.params.items()}
    training=e.gradient(minibatches(data,'B',job['seed'],job['step'],e.c));hc,cc=ng.channels(e)
    e.opt.step();e.opt.zero_grad(set_to_none=True)
    after={n:p.detach().cpu().clone() for n,p in e.params.items()};delta={n:after[n]-before[n] for n in before}
    norms={'h2':0.,'c2':0.,'hc':0.,'delta2':0.,'rounding2':0.};matrices={}
    for n in delta:
        h,c,d=hc[n],cc[n],delta[n];err=d-h-c
        row=dict(h2=ng.dot(h,h),c2=ng.dot(c,c),hc=ng.dot(h,c),delta2=ng.dot(d,d),rounding2=ng.dot(err,err),
            before_norm=math.sqrt(ng.dot(before[n],before[n])),after_norm=math.sqrt(ng.dot(after[n],after[n])),group=ng.group(n))
        matrices[n]=row
        for k in norms:norms[k]+=row[k]
    norms['relative_channel_rounding_error']=math.sqrt(norms['rounding2']/max(norms['delta2'],1e-30))
    norms['history_current_cosine']=norms['hc']/max(math.sqrt(norms['h2']*norms['c2']),1e-30)
    dirs={label:{n:v.to(e.c.device) for n,v in vec.items()} for label,vec in [('history',hc),('current',cc),('delta',delta)]}
    meta=dict(job=job,source_checkpoint=str(p),source_data_hash=digest(data),before_hash=tensor_hash(before),after_hash=tensor_hash(after),
        channels=norms,matrices=matrices,training=dict(raw_gradient_norm=training['raw_norm'],clip=training['clip']),
        convention='Native Adam history/current share updated denominator; roundoff=actual_delta-history-current. Gradients are unclipped derivatives of each evaluated loss.')
    folder=Path(s['output'])/job_id(job);folder.mkdir(exist_ok=True)
    old=read(folder/'update.json')
    if old and any(old[k]!=meta[k] for k in ['before_hash','after_hash','source_data_hash']):raise ValueError('Reconstructed event differs from cached event. Use a new output directory.')
    atomic_json(meta,folder/'update.json');del hc,cc,delta;return data,before,after,dirs


@torch.no_grad()
def assign(e,before,after,alpha):
    for n,p in e.params.items():
        if alpha==0:p.copy_(before[n])
        elif alpha==1:p.copy_(after[n])
        else:p.copy_(before[n].double().add(after[n].double()-before[n].double(),alpha=float(alpha)).float())


def gpu_dot(a,b):return torch.sum(a*b,dtype=torch.float64)


def probe(e,row,dirs,layers,geometry=False):
    """One exact full-parameter gradient per prompt. No gradient sketch or task proxy."""
    taps={};handles=[]
    if geometry:
        for l in layers:
            def hook(mod,inp,out,k=str(l)):taps[k]=out[0] if isinstance(out,tuple) else out
            handles.append(e.model.model.layers[l].register_forward_hook(hook))
    try:
        inp,q,labels=collate([row],e.tok.pad_token_id,e.c.device)
        final=e.model.model(**inp,use_cache=False,return_dict=True).last_hidden_state
        if geometry:taps['final']=final
        lp=e.model.lm_head(final[:,-1,:]).double().log_softmax(-1)
        loss=(q*(q.log()-lp[:,labels])).sum();margin=(lp[:,labels[0]]-lp[:,labels[1]]).sum();support=-lp[:,labels].logsumexp(-1).sum()
        names=list(e.params);keys=list(taps);params=tuple(e.params.values());targets=params+tuple(taps.values())
        gradients=torch.autograd.grad(loss,targets,retain_graph=geometry)
        pg=gradients[:len(params)];hg=gradients[len(params):]
        values=torch.stack([gpu_dot(g,dirs[label][n]) for label in ['delta','history','current'] for n,g in zip(names,pg)]).detach().cpu().numpy().reshape(3,-1)
        projections={k:float(v.sum()) for k,v in zip(['delta','history','current'],values)}
        projections['roundoff']=projections['delta']-projections['history']-projections['current']
        groups={};by_layer={}
        for i,n in enumerate(names):
            group=ng.group(n);d=groups.setdefault(group,dict(delta=0.,history=0.,current=0.))
            for k,v in zip(['delta','history','current'],values[:,i]):d[k]+=float(v)
            layer=n.split('.')[2] if n.startswith('model.layers.') else 'embedding_readout_other'
            ld=by_layer.setdefault(layer,dict(delta=0.,history=0.,current=0.))
            for k,v in zip(['delta','history','current'],values[:,i]):ld[k]+=float(v)
        arrays={}
        if geometry:
            mg=torch.autograd.grad(margin,tuple(taps.values()))
            for k,gl,gm in zip(keys,hg,mg):
                # All positions of this exact prompt; no padding or position mismatch.
                arrays[k+'__x']=taps[k][0].detach().float().cpu().numpy().copy()
                arrays[k+'__lossgrad']=gl[0].detach().float().cpu().numpy().copy()
                arrays[k+'__margin_grad']=gm[0].detach().float().cpu().numpy().copy()
        record=dict(entity=row['entity'],q0=row['q'][0],loss=float(loss.detach()),margin=float(margin.detach()),support=float(support.detach()),
            projections=projections,parameter_groups=groups,layer_parameter_projections=by_layer,gradient_population='This exact single prompt and target distribution.')
        return record,arrays
    finally:
        for handle in handles:handle.remove()
        e.opt.zero_grad(set_to_none=True)


def get_rows(con,jid,alpha,split):
    return [json.loads(r[0]) for r in con.execute('SELECT record FROM probes WHERE job=? AND alpha=? AND split=? ORDER BY entity',(jid,float(alpha),split))]


def geometry_split(e,s,job,split,data,folder,ctl):
    ids=[r['entity'] for r in data[split]];q0=[r['q'][0] for r in data[split]]
    if len(ids)!=len(set(ids)):raise ValueError('Paired evaluation requires one prompt per entity/form.')
    order=np.random.default_rng(s['partition_seed']).permutation(sorted(ids));fitids=set(map(int,order[:len(ids)//2]));heldids=set(map(int,order[len(ids)//2:len(ids)//2+s['topology_landmarks']]))
    mask=np.array([i in heldids for i in ids]);fit=np.array([i in fitids for i in ids]);sites={};saved={}
    cache={entity:(ng.npz_read(folder/'scratch'/f'0_{split}_{entity:03d}.npz'),ng.npz_read(folder/'scratch'/f'1_{split}_{entity:03d}.npz')) for entity in ids}
    first=cache[ids[0]][0];keys=[k[:-3] for k in first if k.endswith('__x')]
    for site in keys:
        ctl.pulse(job=job_id(job),phase='paired layer geometry and topology',split=split,layer=site)
        xs=[];ys=[];g0=[];g1=[];m0=[];m1=[];local=[]
        for entity in ids:
            a,b=cache[entity]
            x,y=a[site+'__x'],b[site+'__x'];dx=y.astype(float)-x
            gl0=a[site+'__lossgrad'];gl1=b[site+'__lossgrad']
            local.append(dict(entity=entity,all_positions_old_gradient_dot_displacement=float(np.sum(gl0*dx)),
                all_positions_current_gradient_dot_displacement=float(np.sum(gl1*dx)),
                all_positions_gradient_change_dot_displacement=float(np.sum((gl1.astype(float)-gl0)*dx)),
                per_position_old_projection=np.sum(gl0*dx,axis=1).tolist(),displacement_norm=float(np.linalg.norm(dx)),
                note='Local activation derivative; other computational sites held fixed. Do not sum this projection across layers.'))
            xs.append(x[-1]);ys.append(y[-1]);g0.append(gl0[-1]);g1.append(gl1[-1]);m0.append(a[site+'__margin_grad'][-1]);m1.append(b[site+'__margin_grad'][-1])
        x,y,g0,g1,m0,m1=map(np.asarray,[xs,ys,g0,g1,m0,m1])
        metric=m0[fit].astype(float)/math.sqrt(sum(fit))
        stat=ng.frame_metrics(y,x,g1,g0,s['frame_rank'])
        topo_before=ng.topology_record(x[mask].astype(float),x[mask].astype(float),metric,np.array(ids)[mask],np.array(q0)[mask],0,0)
        topo_after=ng.topology_record(y[mask].astype(float),x[mask].astype(float),metric,np.array(ids)[mask],np.array(q0)[mask],0,0)
        # Fit within this prompt form on disjoint entity identities; allow reflection, label it.
        a,b=y[fit].astype(float),x[fit].astype(float);_,sv,v=np.linalg.svd(np.concatenate([a,b]),full_matrices=False)
        rank=int(np.sum(sv>max(sv[0],1e-30)*1e-7));basis=v[:rank].T;u,_,vh=np.linalg.svd((a@basis).T@(b@basis),full_matrices=False)
        aligned=y[mask]+(y[mask]@basis)@(u@vh-np.eye(rank))@basis.T
        stat['alignment']=dict(native_mse=float(np.mean((y[mask]-x[mask])**2)),heldout_orthogonal_mse=float(np.mean((aligned-x[mask])**2)),includes_reflections=True)
        sites[site]=dict(last_token_frame=stat,all_token_local=local,topology_before=topo_before,topology_after=topo_after,
            entities=ids,heldout_topology_entities=np.array(ids)[mask].tolist(),reference='Immediately before this natural update; distance scales fixed within this update.')
        for label,arr in [('x_before',x),('x_after',y),('lossgrad_before',g0),('lossgrad_after',g1),('margin_grad_before',m0),('margin_grad_after',m1)]:saved[site+'__'+label]=arr
    ng.npz_save(folder/(split+'_last_token.npz'),saved)
    return dict(sites=sites,entities=ids,fit_entities=sorted(fitids),heldout_entities=sorted(heldids),prompt_form=split)


def endpoints(e,s,job,con,ctl):
    jid=job_id(job);folder=Path(s['output'])/jid
    if read(folder/'ENDPOINTS_DONE.json'):return
    ctl.pulse(job=jid,phase='preparing natural endpoint diagnostics')
    data,before,after,dirs=prepare(e,s,job,ctl);layers=s['layers'] or list(range(e.model.config.num_hidden_layers))
    if any(l<0 or l>=e.model.config.num_hidden_layers for l in layers):raise ValueError('Geometry layer does not exist.')
    try:
        for alpha in [0.,1.]:
            assign(e,before,after,alpha)
            for split in s['splits']:
                geometry_done=con.execute('SELECT 1 FROM geometry WHERE job=? AND split=?',(jid,split)).fetchone()
                for row in data[split]:
                    scratch=folder/'scratch'/f'{int(alpha)}_{split}_{row["entity"]:03d}.npz'
                    existing=con.execute('SELECT 1 FROM probes WHERE job=? AND alpha=? AND split=? AND entity=?',(jid,alpha,split,row['entity'])).fetchone()
                    if existing and (scratch.exists() or geometry_done):continue
                    ctl.pulse(job=jid,phase='exact per-prompt endpoint gradient and all-token sensitivities',alpha=alpha,split=split,entity=row['entity'])
                    record,arrays=probe(e,row,dirs,layers,True);ng.npz_save(scratch,arrays)
                    con.execute('INSERT OR REPLACE INTO probes VALUES (?,?,?,?,?)',(jid,alpha,split,row['entity'],json.dumps(record,allow_nan=False)));con.commit();ctl.storage()
        for split in s['splits']:
            if con.execute('SELECT 1 FROM geometry WHERE job=? AND split=?',(jid,split)).fetchone():
                for p in (folder/'scratch').glob(f'*_{split}_*.npz'):p.unlink()
                continue
            g=geometry_split(e,s,job,split,data,folder,ctl)
            con.execute('INSERT OR REPLACE INTO geometry VALUES (?,?,?)',(jid,split,json.dumps(g,allow_nan=False)));con.commit()
            # Only this runner's reconstructible all-token scratch; original inputs untouched.
            for p in (folder/'scratch').glob(f'*_{split}_*.npz'):p.unlink()
        atomic_json(dict(done=True,completed=time.time()),folder/'ENDPOINTS_DONE.json')
    finally:
        assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True)
        del before,after,dirs;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def path_level(e,s,job,n,con,ctl):
    jid=job_id(job);folder=Path(s['output'])/jid
    summary=read(folder/'path_summary.json',{})
    if summary.get('resolved') or summary.get('grid_points',0)>=n:return
    data,before,after,dirs=prepare(e,s,job,ctl)
    try:
        for alpha in np.linspace(0,1,n):
            assign(e,before,after,float(alpha))
            for split in s['splits']:
                for row in data[split]:
                    if con.execute('SELECT 1 FROM probes WHERE job=? AND alpha=? AND split=? AND entity=?',(jid,float(alpha),split,row['entity'])).fetchone():continue
                    ctl.pulse(job=jid,phase='exact derivative along actual displacement',alpha=float(alpha),split=split,entity=row['entity'],grid_points=n)
                    rec,_=probe(e,row,dirs,[],False)
                    con.execute('INSERT INTO probes VALUES (?,?,?,?,?)',(jid,float(alpha),split,row['entity'],json.dumps(rec,allow_nan=False)));con.commit()
        audits=read(folder/'finite_difference.json')
        if audits is None:
            audits=[];eps=s['fd_epsilon']
            for alpha in [0.,.5,1.]:
                for split in s['splits']:
                    for row in data[split][:s['fd_entities_per_split']]:
                        ctl.pulse(job=jid,phase='directional finite-difference audit',alpha=alpha,split=split,entity=row['entity'])
                        slopes=json.loads(con.execute('SELECT record FROM probes WHERE job=? AND alpha=? AND split=? AND entity=?',(jid,alpha,split,row['entity'])).fetchone()[0])['projections']['delta']
                        estimates=[]
                        for width in [eps,eps/2]:
                            vals=[]
                            for a in [alpha-width,alpha+width]:
                                assign(e,before,after,a)
                                with torch.no_grad():vals.append(float(e.loss_rows([row])[0]))
                            estimates.append((vals[1]-vals[0])/(2*width))
                        tol=s['fd_atol']+s['fd_rtol']*max(abs(slopes),abs(estimates[-1]))
                        audits.append(dict(alpha=alpha,split=split,entity=row['entity'],autograd=slopes,fd=estimates,
                            passed=abs(estimates[-1]-slopes)<=tol,epsilon=eps,endpoint_extrapolation=True))
            atomic_json(audits,folder/'finite_difference.json')
        analyze_path(s,job,n,con,audits)
    finally:
        assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True);del before,after,dirs;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def simpson(values):
    a=np.asarray(values,dtype=float);n=len(a)
    return float((a[0]+a[-1]+4*a[1:-1:2].sum()+2*a[2:-1:2].sum())/(3*(n-1)))


def paired_components(matrix,entities):
    """Balanced 2-form x entity descriptive decomposition, no random-effects claim."""
    x=np.asarray(matrix,float);mu=float(x.mean());prompt=x.mean(1)-mu;entity=x.mean(0)-mu;interaction=x-mu-prompt[:,None]-entity[None,:]
    return dict(grand_mean=mu,prompt_effects=prompt.tolist(),entity_effects=dict(zip(map(str,entities),entity.tolist())),
        interaction=interaction.tolist(),variance_prompt=float(np.mean(prompt**2)),variance_entity=float(np.mean(entity**2)),variance_interaction=float(np.mean(interaction**2)),total_variance=float(np.var(x)),
        prompt_names=['valid','test'],note='Two fixed prompt forms, fully crossed with the same entities/targets. Descriptive variance partition, not population-level causal attribution.')


def paired_report(s,job,con):
    jid=job_id(job);out={}
    for task in ['A','B']:
        changes=[];linear=[];nonlinear=[];ids=None
        for form in ['valid','test']:
            pre=get_rows(con,jid,0.,task+'_'+form);post=get_rows(con,jid,1.,task+'_'+form)
            if not pre or len(pre)!=len(post):return None
            current=[r['entity'] for r in pre]
            if ids is not None and current!=ids:raise ValueError('Prompt forms have unequal entity sets.')
            if current!=[r['entity'] for r in post]:raise ValueError('Endpoint entity sets disagree.')
            if any(a['q0']!=b['q0'] for a,b in zip(pre,post)):raise ValueError('Endpoint targets changed.')
            if form=='valid':targets=[a['q0'] for a in pre]
            elif targets!=[a['q0'] for a in pre]:raise ValueError('Targets differ between paired prompt forms.')
            ids=current;d=np.array([b['loss']-a['loss'] for a,b in zip(pre,post)]);g=np.array([a['projections']['delta'] for a in pre])
            changes.append(d);linear.append(g);nonlinear.append(d-g)
        out[task]=dict(loss_change=paired_components(changes,ids),linear=paired_components(linear,ids),nonlinear_remainder=paired_components(nonlinear,ids),
            per_entity_prompt_difference=(changes[1]-changes[0]).tolist(),entities=ids)
    atomic_json(out,Path(s['output'])/jid/'prompt_entity.json');return out


def analyze_path(s,job,n,con,audits):
    jid=job_id(job);folder=Path(s['output'])/jid;alphas=np.linspace(0,1,n);pop={};allpassed=n>=min(9,max(s['path_levels']))
    for split in s['splits']:
        nodes=[get_rows(con,jid,a,split) for a in alphas];ids=[r['entity'] for r in nodes[0]]
        if any([r['entity'] for r in node]!=ids for node in nodes):raise ValueError('Incomplete path grid.')
        rows=[]
        for i,entity in enumerate(ids):
            rr=[node[i] for node in nodes];slopes=[r['projections']['delta'] for r in rr];change=rr[-1]['loss']-rr[0]['loss'];integral=simpson(slopes)
            coarse=simpson(slopes[::2]) if n>=5 else None
            tol=s['integration_atol']+s['integration_rtol']*max(abs(change),abs(integral))
            closure=integral-change;convergence=abs(integral-coarse) if coarse is not None else None
            passed=coarse is not None and abs(closure)<=tol and convergence<=tol
            allpassed=allpassed and passed
            contrib={k:simpson([r['projections'][k] for r in rr]) for k in ['history','current','roundoff']}
            rows.append(dict(entity=entity,q0=rr[0]['q0'],loss_change=change,initial_slope=slopes[0],final_slope=slopes[-1],
                exact_finite_remainder=change-slopes[0],integrated_slope=integral,integration_closure_error=closure,
                nested_grid_difference=convergence,tolerance=tol,integration_passed=passed,
                path_contributions=contrib,path_nonlinear_contributions={k:contrib[k]-rr[0]['projections'][k] for k in contrib},
                slope_sign_change_brackets=[[float(alphas[j]),float(alphas[j+1])] for j in range(n-1) if slopes[j]*slopes[j+1]<0],
                path_losses=[r['loss'] for r in rr],path_slopes=slopes))
        pop[split]=dict(n=len(rows),mean_loss_change=float(np.mean([r['loss_change'] for r in rows])),mean_initial_slope=float(np.mean([r['initial_slope'] for r in rows])),
            mean_finite_remainder=float(np.mean([r['exact_finite_remainder'] for r in rows])),integration_pass_count=sum(r['integration_passed'] for r in rows),rows=rows)
    audit_ok=all(a['passed'] for a in audits)
    summary=dict(grid_points=n,alphas=alphas.tolist(),populations=pop,resolved=allpassed and audit_ok,
        quadrature_resolved=allpassed,fd_passes=sum(a['passed'] for a in audits),fd_total=len(audits),
        maximum_grid_reached=n==max(s['path_levels']),
        interpretation='Finite remainder is endpoint loss change minus its own exact initial directional derivative. Channel path integrals are quadrature estimates along the chosen straight actual-displacement path, not optimizer-reset counterfactuals. Convergence tests are numerical checks, not rigorous error bounds; sampled sign-change brackets can miss additional roots.')
    atomic_json(summary,folder/'path_summary.json');paired_report(s,job,con)


def plan(s):return [dict(job=j,stage='endpoints') for j in s['jobs']]+[dict(job=j,stage='path',n=n) for n in s['path_levels'] for j in s['jobs']]


def run(s):
    import fcntl
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        con=None;ctl=Control(s)
        try:
            c=validate(s);ctl.pulse(phase='loading original OLMo');e=Engine(c);con=connect(out)
            atomic_json(dict(torch=torch.__version__,python=sys.version,cuda=torch.version.cuda,device=str(c.device),model_parameter_bytes=e.pb),out/'hardware.json')
            for item in plan(s):
                ctl.check()
                if item['stage']=='endpoints':endpoints(e,s,item['job'],con,ctl);paired_report(s,item['job'],con)
                else:path_level(e,s,item['job'],item['n'],con,ctl)
                report(out);ctl.storage()
            final='complete';message='All configured endpoint measurements and requested path grids finished. Check numerical resolution flags.'
        except ng.Pause as exc:final='paused';message=str(exc)
        except Exception:final='failed';message=traceback.format_exc();(out/'error.txt').write_text(message)
        finally:
            if con:con.close()
        try:report(out)
        except Exception:(out/'report_error.txt').write_text(traceback.format_exc())
        atomic_json(dict(status=final,message=message,last_progress=time.time()),out/'status.json')
        try:export(out)
        except Exception:(out/'export_error.txt').write_text(traceback.format_exc())


def report(output):
    out=Path(output);s=read(out/'settings.json')
    if not s or not (out/'measurements.sqlite').exists():return
    con=connect(out);lines=['PRECISE NATURAL FORGETTING — matched prompts, entities and derivatives'];table=[]
    try:
        for job in s['jobs']:
            jid=job_id(job);p=read(out/jid/'path_summary.json')
            for split in s['splits']:
                a=get_rows(con,jid,0.,split);b=get_rows(con,jid,1.,split)
                if len(a)!=len(b) or not a:continue
                change=float(np.mean([y['loss']-x['loss'] for x,y in zip(a,b)]));lin=float(np.mean([x['projections']['delta'] for x in a]));h=float(np.mean([x['projections']['history'] for x in a]));c=float(np.mean([x['projections']['current'] for x in a]))
                row=dict(job=jid,split=split,n=len(a),loss_change=change,linear=lin,history=h,current=c,remainder=change-lin,
                    path_points=p['grid_points'] if p else 2,path_resolved=p['resolved'] if p else False)
                table.append(row);lines.append(f'{jid} {split}: dL {change:+.6f}; linear {lin:+.6f}; remainder {change-lin:+.6f}; path {row["path_points"]} points, resolved={row["path_resolved"]}')
        lines += ['Each loss/gradient/remainder uses exactly the same entity-prompt population.',
            'Endpoint measurements across all events are prioritized before path-grid refinement.',
            'All events are retrospective diagnostic selections, not independent confirmation.',
            'A complete queue can still contain unresolved quadrature or failed derivative audits; inspect path_summary.json.',
            'No activation repair or optimizer-reset branch is run. Diagnostic interpolated weights are restored to the actual post-update state.']
        (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        if table:
            import csv
            with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
            plot(out,table)
    finally:con.close()


def plot(out,table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks=['A_valid','A_test','B_valid','B_test'];jobs=list(dict.fromkeys(r['job'] for r in table));fig,axes=plt.subplots(2,2,figsize=(13,8))
    for ax,split in zip(axes.flat,tasks):
        rr=[next((r for r in table if r['job']==j and r['split']==split),None) for j in jobs];xx=np.arange(len(jobs))
        for shift,key,label in [(-.25,'loss_change','Observed'),(0,'linear','Linear'),(.25,'remainder','Finite remainder')]:ax.bar(xx+shift,[r[key] if r else np.nan for r in rr],width=.24,label=label)
        ax.set_title(split);ax.set_xticks(xx);ax.set_xticklabels(jobs,rotation=60,ha='right',fontsize=7);ax.axhline(0,color='gray',lw=.7);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'matched_gradients.png',dpi=150);plt.close(fig)
    # Separate plot for actual within-update loss and derivative, retaining population labels.
    paths=[(j,read(out/j/'path_summary.json')) for j in jobs];paths=[(j,p) for j,p in paths if p]
    if paths:
        fig,axes=plt.subplots(len(paths),2,figsize=(11,3*len(paths)),squeeze=False)
        for i,(j,p) in enumerate(paths):
            for split in ['A_valid','A_test']:
                rows=p['populations'][split]['rows'];al=p['alphas']
                changes=np.array([r['path_losses'] for r in rows]);changes-=changes[:,0:1]
                axes[i,0].plot(al,changes.mean(0),label=split);axes[i,1].plot(al,np.mean([r['path_slopes'] for r in rows],axis=0),label=split)
            for ax in axes[i]:ax.set_xlabel('Fraction of actual displacement');ax.axhline(0,color='gray',lw=.7);ax.legend();ax.set_title(j)
            axes[i,0].set_ylabel('Mean loss change');axes[i,1].set_ylabel('Mean exact directional derivative')
        fig.tight_layout();fig.savefig(out/'within_update_paths.png',dpi=120);plt.close(fig)


def export(output):
    out=Path(output);dest=out/'precise_forgetting_share.zip';tmp=dest.with_suffix('.tmp');snapshot=out/'export_snapshot.sqlite'
    if (out/'measurements.sqlite').exists():
        with sqlite3.connect(out/'measurements.sqlite') as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix in ['.json','.txt','.csv','.png'] and 'scratch' not in p.parts:z.write(p,p.relative_to(out))
        if snapshot.exists():z.write(snapshot,'measurements.sqlite')
        z.writestr('EXPORT_SCOPE.txt','All stored scalar/per-entity/layer/topology measurements included. Last-token raw NPZ and reconstructible all-token scratch excluded. No model checkpoints created by this runner.\n')
    os.replace(tmp,dest);snapshot.unlink(missing_ok=True);return str(dest)


def signature(s):return hashlib.sha256(json.dumps({k:v for k,v in s.items() if k not in ['hours','reserve_minutes','minimum_free_gb','max_output_gb','output']},sort_keys=True).encode()).hexdigest()


def launch(s):
    import fcntl
    validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if ng.alive(out):print('Already running. Refresh status.');return str(out)
        old=read(out/'settings.json')
        if old and signature(old)!=signature(s):
            out=out.with_name(out.name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);out.mkdir();s['output']=str(out);print('Changed scientific settings: new output',out)
        atomic_json(s,out/'settings.json');(out/'STOP').unlink(missing_ok=True)
        for p in Path(__file__).parent.glob('*.py'):
            q=out/'code'/p.name;q.parent.mkdir(exist_ok=True);shutil.copy2(p,q)
        atomic_json(dict(source_config=read(Path(s['source'])/'config.json'),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}),out/'provenance.json')
        atomic_json(dict(status='launching',last_progress=time.time()),out/'status.json')
        with (out/'worker.log').open('a') as log:p=subprocess.Popen([sys.executable,str(out/'code/precise_forgetting.py'),'--worker',str(out/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            if ng.alive(out) or p.poll() is not None:break
            time.sleep(.1)
        print('Started/resumed:',out);return str(out)


def status(output):
    out=Path(output);s=read(out/'settings.json',{});st=read(out/'status.json',{'status':'not_started'});st['alive']=ng.alive(out)
    st['seconds_since_worker_progress']=round(max(0,time.time()-st.get('last_progress',time.time())),1)
    st['endpoint_jobs_completed']=sum(bool(read(out/job_id(j)/'ENDPOINTS_DONE.json')) for j in s.get('jobs',[]));st['jobs_total']=len(s.get('jobs',[]))
    st['paths_resolved']=sum(bool(read(out/job_id(j)/'path_summary.json',{}).get('resolved')) for j in s.get('jobs',[]))
    if out.exists():st['free_gib']=round(shutil.disk_usage(out).free/1024**3,1)
    return st


def refresh(output):st=status(output);print(json.dumps(st,indent=2));return st

def stop(output):ng.stop(output)

def restart(s):
    if ng.alive(s['output']):stop(s['output']);print('Wait for alive=False, then call restart again.');return
    p=Path(s['output']);s['output']=str(p.with_name(p.name+'_fresh_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]));return launch(s)

def show(output):
    report(output);p=Path(output)/'REPORT.txt';print(p.read_text() if p.exists() else 'No complete endpoint pairs yet.')
    from IPython.display import display,Image
    for name in ['matched_gradients.png','within_update_paths.png']:
        p=Path(output)/name
        if p.exists():display(Image(filename=str(p)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker');args=ap.parse_args()
    if args.worker:run(read(args.worker))
    else:ap.print_help()
