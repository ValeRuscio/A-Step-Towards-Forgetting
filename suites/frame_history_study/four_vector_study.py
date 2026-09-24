"""Exact four-vector dynamics, finite-path orientation/magnitude accounting and HVPs.
Natural training is restored after probes; no gradient sketches or optimizer resets.
"""
from __future__ import annotations
import argparse, collections, gc, hashlib, json, math, os, shutil, sqlite3, subprocess, sys, time, traceback, uuid
from pathlib import Path
import numpy as np
import torch
import precise_forgetting as p
import refine_paths as refine
ng=p.ng
read=p.read
atomic_json=p.atomic_json
NAMES=['history','current','gA','gB']
VERSION='four-vector-1.0'


def defaults(source=None,output=None):
    s=p.defaults(source,output or Path.cwd()/'runs/olmo_four_vector_v1');c=read(Path(s['source'])/'config.json')
    s.update(version=VERSION,seeds=list(c['seeds']),end_step=min(128,c['b_steps']),chunk_steps=8,hours=11.5,
        trajectory_budget_fraction=.55,path_budget_fraction=.30,gradient_splits=['A_test','B_test'],gradient_batch=2,
        diagnostic_jobs=[j for j in [dict(seed=1,step=21),dict(seed=1,step=48),dict(seed=2,step=11),dict(seed=3,step=7)] if j['seed'] in c['seeds'] and j['step']<=c['b_steps']],
        max_path_points=129,max_path_depth=10,curvature_points_per_event=1,curvature_batch=1,
        curvature_symmetry_atol=1e-5,curvature_symmetry_rtol=.01,minimum_free_gb=20.,max_output_gb=15.)
    return s


def validate(s):
    if s.get('version')!=VERSION:raise ValueError('Use defaults() from this release.')
    q=dict(s,version=p.VERSION,jobs=s['diagnostic_jobs']);c=p.validate(q)
    if len(s['gradient_splits'])!=2 or len(set(s['gradient_splits']))!=2 or set(s['gradient_splits'])-set(p.SPLITS):raise ValueError('Choose two distinct fixed evaluation populations for gA/gB.')
    if not s['seeds'] or len(set(s['seeds']))!=len(s['seeds']) or not set(s['seeds']).issubset(c.seeds):raise ValueError('Choose unique available source seeds.')
    if not 1<=s['end_step']<=c.b_steps or s['chunk_steps']<1 or min(s['gradient_batch'],s['curvature_batch'])<1:raise ValueError('Invalid trajectory/batch configuration.')
    if s['max_path_points']<33 or s['max_path_depth']<4 or s['curvature_points_per_event']<1:raise ValueError('Invalid path/curvature limits.')
    if not (0<s['trajectory_budget_fraction']<1 and 0<s['path_budget_fraction']<1 and s['trajectory_budget_fraction']+s['path_budget_fraction']<1):raise ValueError('Leave a positive session fraction for curvature.')
    return c


def connect(output):
    con=sqlite3.connect(Path(output)/'measurements.sqlite',timeout=60);con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS trajectory(seed INTEGER,step INTEGER,record TEXT,PRIMARY KEY(seed,step))')
    con.execute('CREATE TABLE IF NOT EXISTS path(job TEXT,alpha REAL,record TEXT,PRIMARY KEY(job,alpha))')
    con.execute('CREATE TABLE IF NOT EXISTS curvature(job TEXT,alpha REAL,split TEXT,record TEXT,PRIMARY KEY(job,alpha,split))')
    con.commit();return con


class StagePause(Exception):pass
class Control(p.Control):
    stage_deadline=float('inf')
    def check(self):
        super().check()
        if time.time()>self.stage_deadline:raise StagePause()


def layer_name(name):return name.split('.')[2] if name.startswith('model.layers.') else 'embedding_readout_other'


def gram_summary(g):
    g=np.asarray(g,float);norm=np.sqrt(np.maximum(np.diag(g),0));den=norm[:,None]*norm[None,:];valid=den>1e-30
    cos=np.divide(g,den,out=np.zeros_like(g),where=valid).clip(-1,1)
    return dict(names=NAMES,gram=g.tolist(),norms=norm.tolist(),cosines=cos.tolist(),cosine_defined=valid.tolist())


def grams(vec):
    total=np.zeros((4,4));layers={}
    for name in vec['history']:
        group=layer_name(name);a=layers.setdefault(group,np.zeros((4,4)))
        for i,k in enumerate(NAMES):
            for j in range(i,4):
                v=ng.dot(vec[k][name],vec[NAMES[j]][name]);total[i,j]+=v;a[i,j]+=v
                if j!=i:total[j,i]+=v;a[j,i]+=v
    return dict(global_geometry=gram_summary(total),layers={k:gram_summary(v) for k,v in layers.items()})


def temporal_decomposition(previous,current):
    """All six Gram-entry changes decomposed exactly into vector changes."""
    total={};layers={}
    for i,u in enumerate(NAMES):
        for v in NAMES[i+1:]:
            key=u+'__'+v;terms=np.zeros(4);bylayer={}
            for name in current[u]:
                old_u,old_v=previous[u][name],previous[v][name];du=current[u][name]-old_u;dv=current[v][name]-old_v
                vals=np.array([ng.dot(du,old_v),ng.dot(old_u,dv),ng.dot(du,dv),ng.dot(current[u][name],current[v][name])-ng.dot(old_u,old_v)])
                terms+=vals;group=layer_name(name);bylayer.setdefault(group,np.zeros(4));bylayer[group]+=vals
            pack=lambda x:dict(first_vector_change=float(x[0]),second_vector_change=float(x[1]),joint_change=float(x[2]),actual_dot_change=float(x[3]),closure_error=float(x[:3].sum()-x[3]))
            total[key]=pack(terms)
            for group,x in bylayer.items():layers.setdefault(group,{})[key]=pack(x)
    # Distinguish actual history direction from changes of the stored momentum.
    accounting={}
    for task in ['gA','gB']:
        sums=np.zeros(4)
        for name in current[task]:
            m0=previous['raw_m'][name];dm=current['raw_m'][name]-m0;k0=previous['history_scale'][name];dk=current['history_scale'][name]-k0;g=current[task][name]
            sums+=np.array([ng.dot(g,k0*dm),ng.dot(g,dk*m0),ng.dot(g,dk*dm),ng.dot(g,current['history'][name]-previous['history'][name])])
        accounting[task]=dict(momentum_change=float(sums[0]),effective_scale_change=float(sums[1]),joint_change=float(sums[2]),actual_history_change_projection=float(sums[3]),rounding_residual=float(sums[3]-sums[:3].sum()))
    vector_changes={}
    for name in NAMES:
        blocks={};global_values=np.zeros(4)
        for parameter,x in current[name].items():
            y=previous[name][parameter];d=x-y
            values=np.array([ng.dot(y,y),ng.dot(x,x),ng.dot(x,y),ng.dot(d,d)])
            group=layer_name(parameter);blocks.setdefault(group,np.zeros(4));blocks[group]+=values;global_values+=values
        def describe(values):
            old,new,cross,change=values;den=math.sqrt(max(old*new,0))
            return dict(previous_norm=math.sqrt(max(old,0)),current_norm=math.sqrt(max(new,0)),change_norm=math.sqrt(max(change,0)),
                relative_change=math.sqrt(max(change,0))/max(math.sqrt(max(old,0)),1e-30),previous_current_cosine=float(np.clip(cross/den,-1,1)) if den>1e-30 else None)
        vector_changes[name]=dict(global_change=describe(global_values),layers={k:describe(v) for k,v in blocks.items()})
    return dict(vector_changes=vector_changes,global_changes=total,layer_changes=layers,history_memory_scale_accounting=accounting,
        convention='Current minus previous pre-update vectors in the same native parameter coordinates. Effective scale includes updated second moment, bias correction and learning rate; these are algebraic, not causal, decompositions.')


def capture_step(e,data,seed,step):
    before={n:q.detach().cpu().clone() for n,q in e.params.items()}
    train=e.gradient(p.minibatches(data,'B',seed,step,e.c));history,current=ng.channels(e);raw={};scales={};names={id(v):k for k,v in e.params.items()}
    for group in e.opt.param_groups:
        b1,b2=group['betas']
        for q in group['params']:
            n=names[id(q)];state=e.opt.state[q];t=int(state['step'].item())+1 if 'step' in state else 1
            m=state['exp_avg'] if 'exp_avg' in state else torch.zeros_like(q);v=state['exp_avg_sq'] if 'exp_avg_sq' in state else torch.zeros_like(q)
            den=(b2*v+(1-b2)*q.grad.square()).sqrt()/math.sqrt(1-b2**t)+group['eps']
            raw[n]=m.detach().cpu().clone();scales[n]=(-group['lr']*b1/(1-b1**t)/den).detach().cpu()
    e.opt.step();e.opt.zero_grad(set_to_none=True);after={n:q.detach().cpu().clone() for n,q in e.params.items()};delta={n:after[n]-before[n] for n in before}
    return before,after,delta,dict(history=history,current=current,raw_m=raw,history_scale=scales),dict(raw_gradient_norm=train['raw_norm'],clip=train['clip'])


def population_gradient(e,rows,batch,ctl=None):
    e.opt.zero_grad(set_to_none=True);loss=0.;byentity=[]
    for start in range(0,len(rows),batch):
        if ctl:ctl.check()
        rr=rows[start:start+batch];values=e.loss_rows(rr);loss+=float(values.detach().sum())/len(rows)
        byentity.extend(dict(entity=r['entity'],q0=r['q'][0],loss=float(v)) for r,v in zip(rr,values.detach()))
        (values.sum()/len(rows)).backward()
    grad={n:q.grad.detach().cpu().clone() for n,q in e.params.items()};e.opt.zero_grad(set_to_none=True)
    return grad,dict(mean=loss,entities=byentity)


def measure_vectors(e,s,data,before,after,delta,vec,alpha,ctl,job):
    p.assign(e,before,after,alpha);loss={}
    for name,split in zip(['gA','gB'],s['gradient_splits']):
        ctl.pulse(phase='exact full-population gradient',job=job,alpha=alpha,split=split)
        vec[name],loss[split]=population_gradient(e,data[split],s['gradient_batch'],ctl)
    g=grams(vec);functional={};dnorm=math.sqrt(sum(ng.dot(d,d) for d in delta.values()))
    residual2=sum(ng.dot(delta[n]-vec['history'][n]-vec['current'][n],delta[n]-vec['history'][n]-vec['current'][n]) for n in delta)
    channel_audit=dict(actual_delta_norm=dnorm,rounding_residual_norm=math.sqrt(max(residual2,0)),relative_rounding_residual=math.sqrt(max(residual2,0))/max(dnorm,1e-30))
    for i,(name,split) in enumerate(zip(['gA','gB'],s['gradient_splits']),2):
        slope=sum(ng.dot(vec[name][n],delta[n]) for n in delta);gnorm=g['global_geometry']['norms'][i]
        functional[split]=dict(slope=slope,gradient_norm=gnorm,delta_norm=dnorm,cosine=slope/max(gnorm*dnorm,1e-30),cosine_defined=gnorm*dnorm>1e-30,
            history=g['global_geometry']['gram'][i][0],current=g['global_geometry']['gram'][i][1],
            roundoff=slope-g['global_geometry']['gram'][i][0]-g['global_geometry']['gram'][i][1])
    return dict(alpha=alpha,loss=loss,geometry=g,functional=functional,channel_audit=channel_audit)


def restore_before(e,s,data,seed,step,ctl):
    src=Path(s['source'])/f'seed{seed}';forks=[f for f in src.glob('fork*/fork.pt') if int(f.parent.name[4:])<=step]
    if forks:
        f=max(forks,key=lambda f:int(f.parent.name[4:]));z=p.load(f);start=int(z['step']);
        if start!=int(f.parent.name[4:]):raise ValueError('Fork label/state disagree.')
        e.restore(z['engine']);del z
    else:
        e.restore(p.load(src/'anchor.pt'));start=1
        if e.c.transition=='reset':e.reset_optimizer()
    for t in range(start,step):
        ctl.pulse(phase='replaying original prefix',seed=seed,step=t)
        e.gradient(p.minibatches(data,'B',seed,t,e.c));e.opt.step();e.opt.zero_grad(set_to_none=True)


def trajectory(e,s,con,ctl):
    while True:
        remaining=False
        for seed in s['seeds']:
            existing=con.execute('SELECT MAX(step),COUNT(*) FROM trajectory WHERE seed=?',(seed,)).fetchone();last=existing[0] or 0
            if existing[1]!=last:raise ValueError('Non-contiguous saved trajectory.')
            if last>=s['end_step']:continue
            remaining=True;ctl.check();data=p.make_data(e.tok,e.c,seed)
            if p.digest(data)!=read(Path(s['source'])/f'seed{seed}/data.json')['hash']:raise ValueError('Source data hash mismatch.')
            previous=None
            restore_before(e,s,data,seed,max(1,last),ctl)
            if last:
                before,after,delta,previous,_=capture_step(e,data,seed,last)
                try:measure_vectors(e,s,data,before,after,delta,previous,0.,ctl,f'seed{seed}_update{last:03d}')
                finally:p.assign(e,before,after,1.)
                del before,after,delta
            for step in range(last+1,min(last+s['chunk_steps'],s['end_step'])+1):
                ctl.check();start=time.time();before,after,delta,vec,training=capture_step(e,data,seed,step)
                try:
                    measured=measure_vectors(e,s,data,before,after,delta,vec,0.,ctl,f'seed{seed}_update{step:03d}')
                    changes=temporal_decomposition(previous,vec) if previous else None
                    p.assign(e,before,after,1.)
                    ctl.pulse(phase='actual post-update losses',seed=seed,step=step)
                    losses=ng.evaluate(e,data,s['gradient_batch']);consequence={}
                    for split in s['gradient_splits']:
                        change=losses[split]['mean']-measured['loss'][split]['mean'];consequence[split]=dict(actual_loss_change=change,linear=measured['functional'][split]['slope'],finite_remainder=change-measured['functional'][split]['slope'])
                    rec=dict(seed=seed,step=step,pre=measured,temporal_changes=changes,post_losses=losses,functional_consequence=consequence,training=training,seconds=time.time()-start)
                    con.execute('INSERT INTO trajectory VALUES (?,?,?)',(seed,step,json.dumps(rec,allow_nan=False)));con.commit();previous=vec
                finally:p.assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True)
                del before,after,delta;gc.collect()
            report(s['output']);ctl.storage()
        if not remaining:break


def path_event(e,s,job,con,ctl):
    jid=p.job_id(job);folder=Path(s['output'])/jid;folder.mkdir(exist_ok=True)
    if read(folder/'PATH_DONE.json'):return
    data,before,after,dirs=p.prepare(e,s,job,ctl)
    # p.prepare returns GPU directions; keep the four-vector dot products on CPU to bound VRAM.
    delta={n:after[n]-before[n] for n in before};fixed={k:{n:v.cpu() for n,v in dirs[k].items()} for k in ['history','current']};del dirs
    try:
        def point(alpha):
            row=con.execute('SELECT record FROM path WHERE job=? AND alpha=?',(jid,float(alpha))).fetchone()
            if row:return json.loads(row[0])
            vec=dict(fixed);rec=measure_vectors(e,s,data,before,after,delta,vec,float(alpha),ctl,jid)
            con.execute('INSERT INTO path VALUES (?,?,?)',(jid,float(alpha),json.dumps(rec,allow_nan=False)));con.commit();return rec
        results={}
        for split in s['gradient_splits']:
            initial={float(a):None for a in np.linspace(0,1,17)}
            def scalar(alpha):
                r=point(alpha);f=r['functional'][split]
                base=point(0.)['functional'][split];dn=f['gradient_norm']-base['gradient_norm'];dc=f['cosine']-base['cosine'];D=base['delta_norm']
                return dict(loss=r['loss'][split]['mean'],projections=dict(delta=f['slope'],history=f['history'],current=f['current'],roundoff=f['roundoff'],
                    magnitude=D*dn*base['cosine'],orientation=D*base['gradient_norm']*dc,joint=D*dn*dc))
            initial={a:scalar(a) for a in initial}
            results[split]=refine.adaptive_integral(scalar,initial,s['integration_atol'],s['integration_rtol'],s['max_path_points'],s['max_path_depth'],convergence_keys=['delta','history','current','roundoff','magnitude','orientation','joint'])
        # Recompute decompositions on each loss's accepted leaf grid, never average unequal populations.
        for split,r in results.items():
            f0=point(0.)['functional'][split];D=f0['delta_norm'];n0=f0['gradient_norm'];c0=f0['cosine'];terms=np.zeros(3)
            for leaf in r['leaves']:
                a,b=leaf['a'],leaf['b'];xx=np.linspace(a,b,5);values=[]
                for alpha in xx:
                    f=point(float(alpha))['functional'][split];dn=f['gradient_norm']-n0;dc=f['cosine']-c0
                    values.append([D*dn*c0,D*n0*dc,D*dn*dc])
                v=np.asarray(values);terms+=(b-a)/12*(v[0]+4*v[1]+2*v[2]+4*v[3]+v[4])
            r['orientation_magnitude_accounting']=dict(magnitude_change=float(terms[0]),orientation_change=float(terms[1]),interaction=float(terms[2]),
                integrated_nonlinear_sum=float(terms.sum()),endpoint_remainder=r['finite_remainder'],valid_cover=r['complete_cover'],
                convention='Expansion around alpha=0 of D*norm(g)*cos(g,delta); reference- and path-dependent algebraic accounting, not causal effects. Cosine is defined as zero if its denominator vanishes.')
        audit=[]
        for alpha in [0.,.5,1.]:
            exact=point(alpha)
            for width in s.get('slope_fd_widths',[s['fd_epsilon'],s['fd_epsilon']/2]):
                a=max(0.,alpha-width);b=min(1.,alpha+width);losses={}
                for x in [a,b]:
                    ctl.pulse(phase='full-population slope finite-difference audit',job=jid,alpha=x)
                    p.assign(e,before,after,x)
                    with torch.no_grad():
                        losses[x]={sp:sum(float(e.loss_rows(data[sp][i:i+s['gradient_batch']]).sum()) for i in range(0,len(data[sp]),s['gradient_batch']))/len(data[sp]) for sp in s['gradient_splits']}
                for sp in s['gradient_splits']:
                    fd=(losses[b][sp]-losses[a][sp])/(b-a);slope=exact['functional'][sp]['slope']
                    tol=s['fd_atol']+s['fd_rtol']*max(abs(fd),abs(slope))
                    audit.append(dict(alpha=alpha,split=sp,width=width,finite_difference=fd,analytic=slope,error=abs(fd-slope),passed=abs(fd-slope)<=tol,stencil='central' if 0<alpha<1 else 'one-sided'))
        atomic_json(dict(checks=audit,note='Finite-width differences of the same full-population loss. Endpoints use one-sided differences; curvature and FP32 rounding can cause failures.'),folder/'slope_audits.json')
        for sp,r in results.items():
            if 'slope_fd_widths' not in s:r['slope_audits_passed']=all(a['passed'] for a in audit if a['split']==sp)
            else:
                flags=[]
                for alpha in [0.,.5,1.]:
                    aa=[a for a in audit if a['split']==sp and a['alpha']==alpha]
                    flags.append(any(x['passed'] and y['passed'] and abs(x['finite_difference']-y['finite_difference'])<=s['fd_atol']+s['fd_rtol']*max(abs(x['finite_difference']),abs(y['finite_difference'])) for x,y in zip(aa,aa[1:])))
                r['slope_audits_passed']=all(flags)
        atomic_json(results,folder/'path_accounting.json');atomic_json(dict(done=True),folder/'PATH_DONE.json')
    finally:p.assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True);del before,after,delta,fixed;gc.collect()


def curvature_measure(e,rows,directions,batch,ctl,job,alpha,split):
    """Population-matched Hessian contractions. All entities, no monitor substitution."""
    params=tuple(e.params.values());names=list(e.params);acc={k:0. for k in ['hHh','hHc','cHh','cHc','dHd']};lossvalue=0.;layer_acc={}
    for start in range(0,len(rows),batch):
        ctl.pulse(phase='full-population Hessian-vector products',job=job,alpha=alpha,split=split,batch_start=start)
        loss=e.loss_rows(rows[start:start+batch]).sum()/len(rows);lossvalue+=float(loss.detach());g=torch.autograd.grad(loss,params,create_graph=True)
        for label in ['history','current','delta']:
            contraction=sum(torch.sum(v*directions[label][n],dtype=torch.float64) for n,v in zip(names,g))
            hv=torch.autograd.grad(contraction,params,retain_graph=label!='delta',allow_unused=True)
            targets={'history':[('hHh','history'),('cHh','current')],'current':[('hHc','history'),('cHc','current')],'delta':[('dHd','delta')]}[label]
            for key,target in targets:
                for n,v in zip(names,hv):
                    if v is None:continue
                    value=float(torch.sum(v*directions[target][n],dtype=torch.float64));acc[key]+=value
                    group=layer_name(n);layer_acc.setdefault(group,{k:0. for k in ['hHh','hHc','cHh','cHc','dHd']});layer_acc[group][key]+=value
            del contraction,hv
        del loss,g
    acc['layer_contributions']=layer_acc;acc['layer_convention']='Outer contraction restricted to this parameter layer, with the HVP using the full-model direction. Includes cross-layer Hessian couplings; not a within-layer Hessian.'
    acc['loss']=lossvalue;acc['entities']=[r['entity'] for r in rows];return acc


def curvature_event(e,s,job,con,ctl):
    jid=p.job_id(job);folder=Path(s['output'])/jid
    if read(folder/'CURVATURE_DONE.json'):return
    nodes=[(float(a),json.loads(r)) for a,r in con.execute('SELECT alpha,record FROM path WHERE job=? ORDER BY alpha',(jid,))]
    if len(nodes)<2:return
    scored=[]
    for (a,x),(b,y) in zip(nodes,nodes[1:]):
        score=max(abs(y['functional'][sp]['slope']-x['functional'][sp]['slope'])/(b-a) for sp in s['gradient_splits']);scored.append((score,(a+b)/2))
    selected=[]
    for score,a in sorted(scored,reverse=True):
        if all(abs(a-b)>.01 for b in selected):selected.append(a)
        if len(selected)>=s['curvature_points_per_event']:break
    atomic_json(dict(alphas=selected,selection='Largest sampled absolute secant change of task directional derivative; retrospective, population-matched.'),folder/'curvature_points.json')
    data,before,after,dirs=p.prepare(e,s,job,ctl)
    try:
        for alpha in selected:
            p.assign(e,before,after,alpha)
            for split in s['gradient_splits']:
                if con.execute('SELECT 1 FROM curvature WHERE job=? AND alpha=? AND split=?',(jid,alpha,split)).fetchone():continue
                r=curvature_measure(e,data[split],dirs,s['curvature_batch'],ctl,jid,alpha,split)
                tol=s['curvature_symmetry_atol']+s['curvature_symmetry_rtol']*max(abs(r['hHc']),abs(r['cHh']));r['mixed_symmetry_error']=abs(r['hHc']-r['cHh']);r['mixed_symmetry_passed']=r['mixed_symmetry_error']<=tol
                # An independent finite difference of full-population directional gradients.
                slopes=[];width=s['fd_epsilon'];aa=max(0.,alpha-width);bb=min(1.,alpha+width)
                for x in [aa,bb]:
                    ctl.pulse(phase='full-population curvature finite-difference audit',job=jid,alpha=x,split=split)
                    p.assign(e,before,after,x);g,_=population_gradient(e,data[split],s['gradient_batch'],ctl)
                    slopes.append(sum(ng.dot(g[n],after[n]-before[n]) for n in g));del g
                fd=(slopes[1]-slopes[0])/(bb-aa);tol=s['curvature_symmetry_atol']+s['fd_rtol']*max(abs(fd),abs(r['dHd']))
                r['directional_curvature_fd']=dict(value=fd,autograd=r['dHd'],width=width,error=abs(fd-r['dHd']),passed=abs(fd-r['dHd'])<=tol)
                p.assign(e,before,after,alpha)
                r['note']='Same full entity/prompt population as the associated loss; local Hessian contractions, not full spectra. Symmetry check is an internal audit, not independent validation.'
                con.execute('INSERT INTO curvature VALUES (?,?,?,?)',(jid,alpha,split,json.dumps(r,allow_nan=False)));con.commit()
        atomic_json(dict(done=True),folder/'CURVATURE_DONE.json')
    finally:p.assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True);del before,after,dirs;gc.collect()


def report(output):
    import fcntl
    out=Path(output)
    if not out.exists():return
    with (out/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return _report(output)


def _report(output):
    import csv
    out=Path(output);s=read(out/'settings.json')
    if not s or not (out/'measurements.sqlite').exists():return
    con=connect(out)
    try:
        records=[json.loads(r[0]) for r in con.execute('SELECT record FROM trajectory ORDER BY seed,step')]
        lines=[f'FOUR-VECTOR NATURAL DYNAMICS — {len(records)}/{len(s["seeds"])*s["end_step"]} updates measured'];table=[]
        for r in records:
            row=dict(seed=r['seed'],step=r['step']);g=r['pre']['geometry']['global_geometry']
            for i,u in enumerate(NAMES):
                row[u+'_norm']=g['norms'][i]
                for j,v in enumerate(NAMES[i+1:],i+1):row[u+'__'+v+'_dot']=g['gram'][i][j];row[u+'__'+v+'_cos']=g['cosines'][i][j]
            for sp,f in r['functional_consequence'].items():
                for k,v in f.items():row[sp+'_'+k]=v
            table.append(row)
        if table:
            with (out/'trajectory_summary.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
        for seed in s['seeds']:
            rr=[r for r in records if r['seed']==seed]
            if rr:
                r=rr[-1];lines.append(f'Seed {seed}: through update {r["step"]}; '+ '; '.join(f'{sp} dL {v["actual_loss_change"]:+.6g}, linear {v["linear"]:+.6g}, remainder {v["finite_remainder"]:+.6g}' for sp,v in r['functional_consequence'].items()))
        for j in s['diagnostic_jobs']:
            jid=p.job_id(j);account=read(out/jid/'path_accounting.json',{})
            for sp,r in account.items():
                a=r['orientation_magnitude_accounting'];lines.append(f'{jid} {sp}: dL {r["loss_change"]:+.6g}; remainder {r["finite_remainder"]:+.6g}; closure {r["closure_error"]:+.3g}; quadrature_passed={r["quadrature_passed"]}, capped={r["cap_reached"]}; orientation {a["orientation_change"]:+.6g}, magnitude {a["magnitude_change"]:+.6g}, joint {a["interaction"]:+.6g}')
        for jid,alpha,sp,raw in con.execute('SELECT job,alpha,split,record FROM curvature ORDER BY job,alpha,split'):
            r=json.loads(raw);lines.append(f'{jid} {sp} alpha={alpha:.6f}: dHd {r["dHd"]:+.6g}; hHh {r["hHh"]:+.6g}; hHc {r["hHc"]:+.6g}; cHc {r["cHc"]:+.6g}; symmetry={r["mixed_symmetry_passed"]}; curvature FD={r["directional_curvature_fd"]["passed"]}')
        for j in s['diagnostic_jobs']:
            audit=read(out/p.job_id(j)/'slope_audits.json',{}).get('checks',[])
            if audit:lines.append(f'{p.job_id(j)} slope FD checks: {sum(a["passed"] for a in audit)}/{len(audit)} passed (including one-sided endpoint tests).')
        lines+=['Gradients use the complete fixed task populations, without clipping or sketches. h/c include the original training-gradient clipping.',
            'Path quadrature certifies numerical checks on population means only, not each entity. Capped/failed checks remain unresolved.',
            'Layer quantities use disjoint native parameter blocks, not activation covariance spectra.',
            'Loss consequences and decompositions are measured; they do not by themselves establish a causal reference-frame mechanism.']
        (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        if records:plot_trajectory(out,s,records)
        plot_paths(out,s,con)
    finally:con.close()


def plot_trajectory(out,s,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(s['seeds']),3,figsize=(15,3.5*len(s['seeds'])),squeeze=False)
    for row,seed in enumerate(s['seeds']):
        rr=[r for r in records if r['seed']==seed];xx=[r['step'] for r in rr]
        for sp in s['gradient_splits']:axes[row,0].plot(xx,[r['functional_consequence'][sp]['actual_loss_change'] for r in rr],label=sp)
        for i,j,label in [(0,2,'h · gA'),(1,2,'c · gA'),(0,3,'h · gB'),(1,3,'c · gB')]:
            axes[row,1].plot(xx,[r['pre']['geometry']['global_geometry']['gram'][i][j] for r in rr],label=label)
        for i,j,label in [(0,1,'h, c'),(0,2,'h, gA'),(1,2,'c, gA'),(2,3,'gA, gB')]:
            axes[row,2].plot(xx,[r['pre']['geometry']['global_geometry']['cosines'][i][j] if r['pre']['geometry']['global_geometry']['cosine_defined'][i][j] else np.nan for r in rr],label=label)
        for ax,title in zip(axes[row],['Actual loss change','Pre-update task projections','Pre-update cosines']):
            ax.set_title(f'Seed {seed}: {title}');ax.set_xlabel('Natural update');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=140);plt.close(fig)


def plot_paths(out,s,con):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jobs=[p.job_id(j) for j in s['diagnostic_jobs'] if con.execute('SELECT 1 FROM path WHERE job=?',(p.job_id(j),)).fetchone()]
    if not jobs:return
    fig,axes=plt.subplots(len(jobs),3,figsize=(15,3.5*len(jobs)),squeeze=False)
    for i,jid in enumerate(jobs):
        rr=[json.loads(r[0]) for r in con.execute('SELECT record FROM path WHERE job=? ORDER BY alpha',(jid,))];xx=[r['alpha'] for r in rr]
        for sp in s['gradient_splits']:
            axes[i,0].plot(xx,[r['loss'][sp]['mean']-rr[0]['loss'][sp]['mean'] for r in rr],label=sp)
            axes[i,1].plot(xx,[r['functional'][sp]['slope'] for r in rr],label=sp)
            axes[i,2].plot(xx,[r['functional'][sp]['cosine'] for r in rr],label=sp)
        for ax,title in zip(axes[i],['Loss change from first saved point','Exact directional derivative','cos(task gradient, actual update)']):
            ax.set_title(jid+'\n'+title);ax.set_xlabel('Fraction of actual displacement');ax.axhline(0,color='gray',lw=.6);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'finite_paths.png',dpi=140);plt.close(fig)


def run(s):
    import fcntl
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        con=None;ctl=Control(s);start=time.time();duration=ctl.deadline-start
        try:
            c=validate(s)
            source=Path(s['source']);files=[source/'config.json']
            for seed in sorted(set(s['seeds'])|{j['seed'] for j in s['diagnostic_jobs']}):
                folder=source/f'seed{seed}';files.extend([folder/'anchor.pt',folder/'data.json']);files.extend(sorted(folder.glob('fork*/fork.pt')))
            manifest={str(f.relative_to(source)):dict(size=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in files}
            old=read(out/'source_file_metadata.json')
            if old and old!=manifest:raise ValueError('Original source file metadata changed. Use a new output directory to avoid mixing reconstructions.')
            atomic_json(manifest,out/'source_file_metadata.json')
            ctl.pulse(phase='loading original OLMo');e=p.Engine(c);con=connect(out)
            atomic_json(dict(torch=torch.__version__,python=sys.version,cuda=torch.version.cuda,device=str(c.device),parameter_bytes=e.pb,
                gpu=torch.cuda.get_device_name(c.device) if str(c.device).startswith('cuda') else None),out/'hardware.json')
            ctl.stage_deadline=start+duration*s['trajectory_budget_fraction']
            try:trajectory(e,s,con,ctl)
            except StagePause:pass
            ctl.stage_deadline=start+duration*(s['trajectory_budget_fraction']+s['path_budget_fraction'])
            try:
                for j in s['diagnostic_jobs']:ctl.check();path_event(e,s,j,con,ctl);ctl.storage()
            except StagePause:pass
            ctl.stage_deadline=ctl.deadline
            for j in s['diagnostic_jobs']:
                if read(out/p.job_id(j)/'PATH_DONE.json'):ctl.check();curvature_event(e,s,j,con,ctl);ctl.storage()
            n=con.execute('SELECT COUNT(*) FROM trajectory').fetchone()[0]
            complete=n==len(s['seeds'])*s['end_step'] and all(read(out/p.job_id(j)/'CURVATURE_DONE.json') for j in s['diagnostic_jobs'])
            final='complete' if complete else 'paused';message='All configured measurements finished; inspect numerical flags.' if complete else 'Session stage budgets reached. Launch again to resume saved work.'
        except (ng.Pause,StagePause) as exc:final='paused';message=str(exc) or 'Stage budget reached.'
        except Exception:final='failed';message=traceback.format_exc();(out/'error.txt').write_text(message)
        finally:
            if con:con.close()
        try:report(out)
        except Exception:(out/'report_error.txt').write_text(traceback.format_exc())
        atomic_json(dict(status=final,message=message,last_progress=time.time()),out/'status.json')
        try:export(out)
        except Exception:(out/'export_error.txt').write_text(traceback.format_exc())


def signature(s):
    runtime={'hours','reserve_minutes','minimum_free_gb','max_output_gb','output','trajectory_budget_fraction','path_budget_fraction','chunk_steps'}
    return hashlib.sha256(json.dumps({k:v for k,v in s.items() if k not in runtime},sort_keys=True).encode()).hexdigest()


def launch(s):
    import fcntl
    validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if ng.alive(out):print('Already running. Refresh status.');return str(out)
        old=read(out/'settings.json')
        if old and signature(old)!=signature(s):
            out=out.with_name(out.name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);out.mkdir();s['output']=str(out)
            print('Scientific settings changed. Preserved previous results; new run:',out)
        atomic_json(s,out/'settings.json');(out/'STOP').unlink(missing_ok=True)
        for file in Path(__file__).parent.glob('*.py'):
            dest=out/'code'/file.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest)
        atomic_json(dict(source_config=read(Path(s['source'])/'config.json'),code={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}),out/'provenance.json')
        atomic_json(dict(status='launching',last_progress=time.time()),out/'status.json')
        with (out/'worker.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,str(out/'code/four_vector_study.py'),'--worker',str(out/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            if ng.alive(out) or proc.poll() is not None:break
            time.sleep(.1)
        print('Started/resumed:',out);return str(out)


def status(output):
    out=Path(output);s=read(out/'settings.json',{});st=read(out/'status.json',{'status':'not_started'});st['alive']=ng.alive(out)
    st['seconds_since_worker_progress']=round(max(0,time.time()-st.get('last_progress',time.time())),1)
    st['updates_measured']=0;st['updates_target']=len(s.get('seeds',[]))*s.get('end_step',0)
    if (out/'measurements.sqlite').exists():
        with sqlite3.connect(f'file:{out / "measurements.sqlite"}?mode=ro',uri=True) as con:
            st['updates_measured']=con.execute('SELECT COUNT(*) FROM trajectory').fetchone()[0]
            st['path_points_saved']=con.execute('SELECT COUNT(*) FROM path').fetchone()[0]
            st['curvature_populations_saved']=con.execute('SELECT COUNT(*) FROM curvature').fetchone()[0]
    st['paths_finished']=sum(bool(read(out/p.job_id(j)/'PATH_DONE.json')) for j in s.get('diagnostic_jobs',[]))
    if out.exists():st['free_gib']=round(shutil.disk_usage(out).free/1024**3,1)
    return st


def refresh(output):
    st=status(output);print(json.dumps(st,indent=2));return st


def stop(output):ng.stop(output)


def restart(s):
    if ng.alive(s['output']):
        stop(s['output']);print('Stop requested. Refresh until alive=False, then run restart again.');return
    old=Path(s['output']);s['output']=str(old.with_name(old.name+'_fresh_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]));return launch(s)


def show(output):
    report(output);out=Path(output);f=out/'REPORT.txt';print(f.read_text() if f.exists() else 'No completed records yet.')
    from IPython.display import display,Image
    for name in ['trajectory.png','finite_paths.png']:
        if (out/name).exists():display(Image(filename=str(out/name)))


def export(output):
    import fcntl
    out=Path(output)
    with (out/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return _export(output)


def _export(output):
    import zipfile
    out=Path(output);dest=out/'four_vector_share.zip';tmp=out/'four_vector_share.tmp';snapshot=out/'export_snapshot.sqlite'
    if (out/'measurements.sqlite').exists():
        with sqlite3.connect(out/'measurements.sqlite') as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for f in out.rglob('*'):
            if f.is_file() and f.suffix in {'.json','.txt','.csv','.png','.py'}:z.write(f,f.relative_to(out))
        if snapshot.exists():z.write(snapshot,'measurements.sqlite')
        z.writestr('EXPORT_SCOPE.txt','All saved four-vector scalar, layer, entity-loss, path and curvature records, settings and code. No raw gradient tensors or model checkpoints are produced.\n')
    os.replace(tmp,dest);snapshot.unlink(missing_ok=True);return str(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');args=parser.parse_args()
    if args.worker:run(read(args.worker))
    else:parser.print_help()
