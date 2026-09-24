import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/natural_components_mpl')
os.environ.setdefault('USE_TF','0')
import argparse,fcntl,gc,json,random,time,traceback,sys,platform,importlib.metadata
from pathlib import Path
import torch
from association_core import Config,minibatches
from study_math import StudyEngine,weights,assign,evaluate,channels,norm,dot
from components import component_gradients,projections,complete_directions,digest,layer
from storage import Control,Paused,read,write,db_open,put,get
from tasks import dataset
from path_analysis import path

def evaluations(e,data,s,ctl):
    out={}
    for sp in ['A_valid','B_valid','A_test','B_test']:
        ctl.pulse(phase='natural endpoint evaluation',split=sp)
        out[sp]=evaluate(e,data[sp],s['eval_batch'],ctl)
    return out

def gate(initial,anchor,s):
    gain=initial['A_valid']['mean']-anchor['A_valid']['mean'];acc=anchor['A_valid']['restricted_accuracy']
    return dict(passed=gain>=s['acquisition_min_gain'] and acc>=s['acquisition_min_accuracy'],loss_gain=gain,restricted_accuracy=acc)

def sample_plan(records,s,seed):
    """Uniform draw independent of outcomes; enrichment uses validation only."""
    rng=random.Random(94117+seed);ts=list(range(1,s['b_steps']+1))
    uniform=sorted(rng.sample(ts,min(s['uniform_updates'],len(ts))));reasons={t:['uniform'] for t in uniform}
    pools={'large_A':[],'ordinary':[],'both_worsen':[]}
    for r in records:
        a=r['post']['A_valid']['mean']-r['pre']['A_valid']['mean'];b=r['post']['B_valid']['mean']-r['pre']['B_valid']['mean']
        if a>s['event_threshold']:pools['large_A'].append(r['step'])
        if a<=s['ordinary_threshold']:pools['ordinary'].append(r['step'])
        if a>s['event_threshold'] and b>s['event_threshold']:pools['both_worsen'].append(r['step'])
    chosen={}
    for kind,vals in pools.items():
        if s.get('selection_large_only') and kind!='large_A':continue
        sel=sorted(rng.sample(vals,min(s['enriched_per_stratum'],len(vals))));chosen[kind]=sel
        for t in sel:reasons.setdefault(t,[]).append(kind)
    return dict(uniform=uniform,pool_sizes={k:len(v) for k,v in pools.items()},enriched=chosen,selected=[dict(step=t,reasons=reasons[t]) for t in sorted(reasons)],selection='Uniform independent of outcomes; enriched pools use A_valid/B_valid only; no test-based selection.')

def recorded_update(e,data,s,ctl,seed,t,pre,previous_geometry=None):
    before=weights(e);audit=e.gradient(minibatches(data,'B',seed,t,e.c))
    train={n:p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for n,p in e.params.items()}
    h,c=channels(e);e.opt.step();after=weights(e);dirs=complete_directions(before,after,h,c);del h,c
    e.opt.zero_grad(set_to_none=True)
    assign(e,before)
    try:
        ctl.pulse(phase='exact pre-update confusion/leakage gradients',step=t)
        gs,loss=component_gradients(e,data['A_valid'],s['derivative_batch'],ctl);geo=projections(gs,dirs,train)
        gram_names=['history','current','confusion_gradient','leakage_gradient','training_gradient']
        vectors=[dirs['h'],dirs['c'],gs['confusion'],gs['leakage'],train]
        gram=[[dot(x,y) for y in vectors] for x in vectors];del gs,vectors,train
    finally:assign(e,after)
    post=evaluations(e,data,s,ctl)
    for k,key in [('total','mean'),('confusion','confusion'),('leakage','leakage')]:
        observed=post['A_valid'][key]-pre['A_valid'][key]
        geo[k]['observed_change']=observed;geo[k]['finite_remainder']=observed-geo[k]['slopes']['d']
        geo[k]['gradient_loss_endpoint_check']=loss[k]-pre['A_valid'][key]
        if previous_geometry:
            geo[k]['recent_projection_changes']={v:geo[k]['slopes'][v]-previous_geometry[k]['slopes'][v] for v in dirs}
            for name,g in geo[k]['layers'].items():g['recent_projection_change']=g['slopes']['d']-previous_geometry[k]['layers'][name]['slopes']['d']
    out=dict(step=t,pre=pre,post=post,geometry=geo,gram_names=gram_names,gram=gram,channel_norms={k:norm(v) for k,v in dirs.items()},clipping={k:audit[k] for k in ['raw_norm','clip']},endpoint_hash=digest(after))
    del before,after,dirs
    return out

def case(s,m,seed,variant,con,ctl,job):
    if get(con,job,'case_done'):
        for name in ['active_checkpoint.pt','active_anchor.pt']:
            f=Path(s['output'])/name
            if f.exists():
                z=torch.load(f,map_location='cpu',weights_only=True)
                if z['job']==job:f.unlink()
        return
    out=Path(s['output']);ck=out/'active_checkpoint.pt';anchorfile=out/'active_anchor.pt'
    cfg=Config(model=m['repo'],revision=m['revision'],device=s['device'],threads=s['threads'],smoke=s['smoke'],lr=s['learning_rates'][m['name']],beta1=s['beta1'],beta2=s['beta2'],eps=s['eps'],clip=s['clip'],microbatch=s['microbatch'],accumulation=s['accumulation'],max_length=s['max_length'])
    ctl.pulse(job=job,phase='load pinned FP32 model');e=StudyEngine(cfg)
    try:
        f=out/'data'/(job.replace('/','_')+'.json')
        if f.exists():saved=read(f);data=saved['data'];meta=saved['meta']
        else:data,meta=dataset(e,s,seed,'confirmation',variant);write(f,dict(data=data,meta=meta))
        put(con,job,0,'spec',dict(model=m,seed=seed,variant=variant,data=meta,optimizer='Adam',lr=cfg.lr))
        if ck.exists():
            z=torch.load(ck,map_location='cpu',weights_only=True)
            if z['job']!=job:raise RuntimeError('Active checkpoint belongs to '+z['job'])
            e.restore(z['engine']);phase=z['phase'];step=z['step'];del z
        else:
            phase='A';step=0
            initial=evaluations(e,data,s,ctl);put(con,job,0,'initial',initial)
            ctl.checkpoint(dict(job=job,phase=phase,step=step,engine=e.pack()),ck)
        if phase=='A':
            for t in range(step+1,s['a_steps']+1):
                ctl.check();ctl.pulse(phase='task A acquisition',step=t,total_steps=s['a_steps'])
                e.gradient(minibatches(data,'A',seed,t,e.c));e.opt.step()
                if t%s['checkpoint_every']==0 or t==s['a_steps']:ctl.checkpoint(dict(job=job,phase='A',step=t,engine=e.pack()),ck)
            a=evaluations(e,data,s,ctl);g=gate(get(con,job,'initial')[0],a,s)
            put(con,job,0,'anchor',dict(evaluations=a,acquisition=g,weight_hash=digest(weights(e))))
            ctl.checkpoint(dict(job=job,engine=e.pack()),anchorfile)
            ctl.checkpoint(dict(job=job,phase='B',step=0,engine=e.pack()),ck);phase='B';step=0
        if phase=='B':
            con.execute("DELETE FROM records WHERE job=? AND kind='step' AND step>?",(job,step));con.commit()
            rr=get(con,job,'step');previous=rr[-1]['post'] if rr else get(con,job,'anchor')[0]['evaluations'];previous_geo=rr[-1]['geometry'] if rr else None
            for t in range(step+1,s['b_steps']+1):
                ctl.check();ctl.pulse(phase='natural update',step=t,total_steps=s['b_steps'])
                r=recorded_update(e,data,s,ctl,seed,t,previous,previous_geo);put(con,job,t,'step',r);previous=r['post'];previous_geo=r['geometry']
                if t%s['checkpoint_every']==0 or t==s['b_steps']:ctl.checkpoint(dict(job=job,phase='B',step=t,engine=e.pack()),ck)
            put(con,job,0,'natural_done',dict(final=previous,steps=s['b_steps']))
            plan=sample_plan(get(con,job,'step'),s,seed);put(con,job,0,'selection',plan)
            z=torch.load(anchorfile,map_location='cpu',weights_only=True)
            if z['job']!=job:raise RuntimeError('Wrong replay anchor')
            e.restore(z['engine']);del z
            ctl.checkpoint(dict(job=job,phase='replay',step=0,engine=e.pack()),ck);phase='replay';step=0
        if phase=='replay' and s.get('defer_paths',False):
            put(con,job,0,'case_done',dict(complete=True,scope='natural trajectory; paths deferred'))
            ck.unlink(missing_ok=True);anchorfile.unlink(missing_ok=True)
            return
        if phase=='replay':
            plan=get(con,job,'selection')[0];selected={r['step']:r['reasons'] for r in plan['selected']};original={r['step']:r for r in get(con,job,'step')}
            stop=max(selected,default=0)
            for t in range(step+1,stop+1):
                ctl.check();ctl.pulse(phase='deterministic replay',step=t,total_steps=stop)
                if t in selected:ctl.checkpoint(dict(job=job,phase='replay',step=t-1,engine=e.pack()),ck)
                before=weights(e) if t in selected else None
                e.gradient(minibatches(data,'B',seed,t,e.c));ch=channels(e) if before is not None else None;e.opt.step();e.opt.zero_grad(set_to_none=True)
                after=weights(e);sha=digest(after)
                if sha!=original[t]['endpoint_hash']:raise ArithmeticError(f'Replay diverged at update {t}; cached paths cannot be reused')
                if before is not None:
                    dirs=complete_directions(before,after,*ch);del ch
                    for sp in ['A_valid','A_test']:
                        pop=data[sp][:s['path_population']];pj=f'{job}/update{t:03d}/{sp}'
                        put(con,pj,0,'selection',dict(reasons=selected[t],source='validation-only enrichment or uniform draw'))
                        path(e,before,after,dirs,pop,s,ctl,con,pj,sp)
                    del dirs,before
                del after
                if t%s['checkpoint_every']==0 or t in selected or t==stop:ctl.checkpoint(dict(job=job,phase='replay',step=t,engine=e.pack()),ck)
            put(con,job,0,'case_done',dict(complete=True))
            ck.unlink(missing_ok=True);anchorfile.unlink(missing_ok=True)
    finally:
        del e;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def queue(s):
    # Paired answer conditions adjacent. Only one case's two checkpoints retained.
    return [(m,seed,v) for seed in s['seeds'] for m in s['models'] for v in s['answer_conditions']]

def run(s):
    from study import validate
    from reporting import report
    validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True);ctl=Control(s);con=db_open(out)
    try:
        env=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},cuda=torch.version.cuda)
        if not s['smoke']:env['packages']['datasets']=importlib.metadata.version('datasets')
        old=read(out/'environment.json')
        if old and old!=env:raise RuntimeError('Environment changed; use original environment to resume')
        write(out/'environment.json',env);q=queue(s);write(out/'queue.json',[dict(model=m,seed=seed,condition=v) for m,seed,v in q])
        for m,seed,v in q:
            job=f'natural/{m["name"]}/seed{seed}/{v}'
            ctl.pulse(job=job,phase='start case',completed_jobs=con.execute("SELECT COUNT(*) FROM records WHERE kind='case_done'").fetchone()[0],total_jobs=len(q))
            case(s,m,seed,v,con,ctl,job);report(con,out,s)
        report(con,out,s)
        lim=read(out/'summary.json')['limitations']
        write(out/'status.json',dict(status='complete_with_limitations' if lim else 'complete',last_progress=time.time(),completed_jobs=len(q),total_jobs=len(q),limitations=lim))
    except Paused as ex:write(out/'status.json',dict(status='paused',message=str(ex),last_progress=time.time(),**ctl.fields))
    except Exception:write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields))
    finally:
        try:report(con,out,s)
        finally:con.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--settings',required=True);a=ap.parse_args();s=read(a.settings);p=Path(s['output']);p.mkdir(parents=True,exist_ok=True)
    with open(p/'worker.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:sys.exit(0)
        run(s)
