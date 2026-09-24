"""Paired first-moment interventions, continued training, and frozen forecasts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/history_geometry_mpl')
import argparse,copy,fcntl,gc,hashlib,importlib.metadata,json,platform,sys,time,traceback
from pathlib import Path
import torch
from association_core import Config,minibatches
from study_math import StudyEngine,weights,assign,difference,grad,dot,norm,evaluate
from optimizers import make_optimizer,native_channels,reset_history
from diagnostics import geometry,match,audit_match,state_audit,layer_name
from tasks import dataset
from storage import Control,Paused,read,write,db_open,put,get
from finite_paths import measure
from study import validate
MODES=['preserve','reset','reset_blocks']

def configuration(s,m,lr):
    return Config(model=m['repo'],revision=m['revision'],device=s['device'],threads=s['threads'],smoke=s['smoke'],lr=lr,beta1=s['momentum'],beta2=s['beta2'],eps=s['eps'],clip=s['clip'],microbatch=s['microbatch'],accumulation=s['accumulation'],max_length=s['max_length'])
def evaluations(e,data,s,ctl):
    result={}
    for sp in ['A_valid','B_valid','A_test','B_test']:
        ctl.pulse(phase='evaluate fixed population',split=sp);result[sp]=evaluate(e,data[sp],s['eval_batch'],ctl)
    return result

def fingerprint(pack):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            a=x.detach().cpu().contiguous();h.update(str((tuple(a.shape),str(a.dtype))).encode());h.update(a.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(pack);return h.hexdigest()

def one_step(e,data,s,ctl,opt,seed,t,pre,mode='preserve',target=None,do_path=False,con=None,path_job=None):
    ctl.check();ctl.pulse(phase='old-task validation gradient',step=t)
    before=weights(e);gA=grad(e,data['A_valid'],s['eval_batch'],ctl)
    batches=minibatches(data,'B',seed,t,e.c)
    ctl.pulse(phase='training minibatch gradient',step=t);gaudit=e.gradient(batches)
    train={n:(p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')) for n,p in e.params.items()}
    ch=native_channels(e,opt);e.opt.step();norm_audit=None
    if mode=='reset_blocks':
        if target is None:raise ValueError('Missing preserved target displacement')
        candidate=difference(weights(e),before);d,norm_audit=match(candidate,target,True)
        assign(e,{n:before[n]+d[n] for n in before});norm_audit=audit_match(e,before,norm_audit,True,s['norm_match_rtol'])
        if ch is not None:ch=tuple({n:v*norm_audit[layer_name(n)]['scale'] for n,v in channel.items()} for channel in ch)
    after=weights(e);delta=difference(after,before)
    # Only gA is needed every step; avoid an unused full B population backward pass.
    gg=geometry(gA,{},train,delta,ch);gg.pop('population_interference',None)
    post=evaluations(e,data,s,ctl);gg['nonlinear_remainder_A_valid']=post['A_valid']['mean']-pre['A_valid']['mean']-gg['projection']
    # Names expose the four vectors used here: old-task gradient, current batch gradient,
    # history displacement, current displacement. They are not the older gA/gB Gram table.
    if ch is not None:
        h,c=ch;hh=dot(h,h);cc=dot(c,c);hc=dot(h,c);th=dot(train,h);tc=dot(train,c)
        at=-gg['training_interference'];ah=gg['history_projection'];ac=gg['current_projection']
        gram=[[gg['gA_norm']**2,at,ah,ac],[at,dot(train,train),th,tc],[ah,th,hh,hc],[ac,tc,hc,cc]]
        gg['four_vectors']={'names':['A_valid_gradient','clipped_B_batch_gradient','history_displacement','current_displacement'],'gram':gram}
    r=dict(step=t,pre=pre,post=post,geometry=gg,norm_audit=norm_audit,clipping={'raw_norm':gaudit['raw_norm'],'factor':gaudit['clip']})
    if do_path:
        ctl.pulse(phase='audit first branch displacement',condition=mode)
        r['path']=measure(e,before,after,data['A_valid'],s,ctl,con,path_job,ch)
    return r

def branch(e,start,data,s,ctl,con,parent,opt,seed,t,mode,target,pre,anchor_sha,do_path):
    job=f'{parent}/fork{t:03d}/{mode}';ck=Path(s['output'])/'branch_checkpoint.pt'
    if get(con,job,'branch_done'):
        if ck.exists():
            z=torch.load(ck,map_location='cpu',weights_only=True)
            if z['job']==job:ck.unlink()
        return
    ctl.pulse(job=job,phase='start paired branch',condition=mode)
    put(con,job,0,'branch_spec',dict(parent=parent,fork=t,condition=mode,optimizer=opt,anchor_sha256=anchor_sha,
        minibatch_seed=seed,steps=s['branch_steps'],same_initial_weights=True,first_step_only_intervention=True,
        subsequent_steps='Native optimizer, no repeated reset or norm matching',pre=pre))
    if ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=True)
        if state['job']!=job:raise RuntimeError('Branch checkpoint belongs to '+state['job'])
        e.restore(state['engine']);offset=state['offset']
        con.execute("DELETE FROM records WHERE job=? AND kind='branch_step' AND step>?",(job,offset));con.commit()
        previous=get(con,job,'branch_step')[-1]['post'] if offset else pre
    else:
        e.restore(start)
        if fingerprint(e.pack())!=anchor_sha:raise ArithmeticError('Branch did not restore the identical weights and optimizer')
        con.execute("DELETE FROM records WHERE job=? AND kind='branch_step'",(job,));con.commit()
        reset_count=0
        if mode!='preserve':reset_count=reset_history(e)
        sa=state_audit(start,e.pack())
        if not sa['other_state_preserved']:raise ArithmeticError('Non-history optimizer state changed')
        put(con,job,0,'state_audit',dict(**sa,reset_tensor_count=reset_count,identical_start_sha256=anchor_sha))
        offset=0;previous=pre
        # Offset0 checkpoint includes the intervention. Resume never resets it twice.
        ctl.checkpoint(dict(job=job,offset=0,engine=e.pack()),ck)
    for k in range(offset+1,s['branch_steps']+1):
        ctl.check();ctl.pulse(job=job,phase='continued paired branch',condition=mode,branch_offset=k,total_branch_steps=s['branch_steps'])
        r=one_step(e,data,s,ctl,opt,seed,t+k-1,previous,mode if k==1 else 'preserve',target if k==1 else None,
            do_path and k==1,con,job+'/path')
        r['branch_offset']=k;put(con,job,k,'branch_step',r);previous=r['post']
        if k%s['checkpoint_every']==0 or k==s['branch_steps']:ctl.checkpoint(dict(job=job,offset=k,engine=e.pack()),ck)
    put(con,job,s['branch_steps'],'branch_done',dict(complete=True,final=previous,steps=s['branch_steps'],endpoint_sha256=fingerprint(e.pack())))
    ck.unlink(missing_ok=True)
    ctl.fields['completed_branches']=con.execute("SELECT count(*) FROM records WHERE kind='branch_done'").fetchone()[0]

def forks(e,data,s,ctl,con,parent,opt,seed,t,pre):
    start=e.pack();sha=fingerprint(start);batches=minibatches(data,'B',seed,t,e.c)
    # Reference update computed from the preserved native state at the same LR/batch.
    e.gradient(batches);e.opt.step();target=difference(weights(e),{n:start['model'][n] for n in e.params});e.restore(start)
    do_path=seed in s['path_seeds'] and t in s['path_fork_steps']
    try:
        for mode in MODES:branch(e,start,data,s,ctl,con,parent,opt,seed,t,mode,target,pre,sha,do_path)
    finally:e.restore(start)

def acquisition(initial,anchor,s):
    gain=initial['A_valid']['mean']-anchor['A_valid']['mean'];acc=anchor['A_valid']['restricted_accuracy']
    return dict(passed=gain>=s['acquisition_min_gain'] and acc>=s['acquisition_min_accuracy'],loss_gain=gain,restricted_accuracy=acc,
        min_gain=s['acquisition_min_gain'],min_accuracy=s['acquisition_min_accuracy'])

def case(s,m,opt,seed,con,ctl,job):
    if get(con,job,'natural_done'):
        ck=Path(s['output'])/'natural_checkpoint.pt'
        if ck.exists():
            z=torch.load(ck,map_location='cpu',weights_only=True)
            if z['job']==job:ck.unlink()
        return
    out=Path(s['output']);ck=out/'natural_checkpoint.pt';lr=s['learning_rates'][m['name']][opt]
    ctl.pulse(job=job,phase='loading pinned checkpoint',optimizer=opt,seed=seed)
    e=StudyEngine(configuration(s,m,lr));e.opt=make_optimizer(e,opt,lr,s)
    try:
        path=out/'data'/f"{m['name']}_seed{seed}.json"
        if path.exists():saved=read(path);data=saved['data'];meta=saved['meta']
        else:data,meta=dataset(e,s,seed,'confirmation','standard');write(path,dict(data=data,meta=meta))
        put(con,job,0,'spec',dict(model=m,optimizer=opt,lr=lr,seed=seed,data=meta,a_steps=s['a_steps'],b_steps=s['b_steps'],branch_steps=s['branch_steps']))
        if ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=True)
            if state['job']!=job:raise RuntimeError('Natural checkpoint belongs to '+state['job'])
            e.restore(state['engine']);phase=state['phase'];step=state['step']
            if phase=='B':con.execute("DELETE FROM records WHERE job=? AND kind='step' AND step>?",(job,step))
            else:con.execute("DELETE FROM records WHERE job=? AND kind='acquisition' AND step>?",(job,step))
            con.commit()
        else:
            initial=evaluations(e,data,s,ctl);put(con,job,0,'initial',initial);phase='A';step=0
            ctl.checkpoint(dict(job=job,phase=phase,step=step,engine=e.pack()),ck)
        if phase=='A':
            for t in range(step+1,s['a_steps']+1):
                ctl.check();ctl.pulse(job=job,phase='task A acquisition',step=t,total_steps=s['a_steps'])
                e.gradient(minibatches(data,'A',seed,t,e.c));e.opt.step()
                if t%32==0 or t==s['a_steps']:put(con,job,t,'acquisition',{'A_valid':evaluate(e,data['A_valid'],s['eval_batch'],ctl)})
                if t%s['checkpoint_every']==0 or t==s['a_steps']:ctl.checkpoint(dict(job=job,phase='A',step=t,engine=e.pack()),ck)
            anchor=evaluations(e,data,s,ctl);gate=acquisition(get(con,job,'initial')[0],anchor,s)
            put(con,job,0,'anchor',dict(evaluations=anchor,acquisition=gate))
            ctl.checkpoint(dict(job=job,phase='B',step=0,engine=e.pack()),ck);step=0
        anchor=get(con,job,'anchor')[0]
        previous=anchor['evaluations'] if step==0 else get(con,job,'step')[-1]['post']
        for t in range(step+1,s['b_steps']+1):
            ctl.check();ctl.pulse(job=job,phase='natural trajectory',step=t,total_steps=s['b_steps'])
            if t in s['fork_steps']:
                # Durable exact pre-intervention anchor, needed across budget pauses.
                ctl.checkpoint(dict(job=job,phase='B',step=t-1,engine=e.pack()),ck)
                forks(e,data,s,ctl,con,job,opt,seed,t,previous)
                from reporting import report
                report(con,out,False)
            ctl.pulse(job=job,phase='natural measured update',step=t)
            r=one_step(e,data,s,ctl,opt,seed,t,previous)
            # Preserved branch's first step must reproduce the natural first step exactly.
            if t in s['fork_steps']:
                p=get(con,f'{job}/fork{t:03d}/preserve','branch_step')[0]
                err=max(abs(r['post'][sp]['mean']-p['post'][sp]['mean']) for sp in r['post'])
                r['preserve_replay_audit']=dict(max_loss_difference=err,passed=err<=1e-10)
                if err>1e-10:raise ArithmeticError('Preserved branch failed deterministic natural-step replay')
            put(con,job,t,'step',r);previous=r['post']
            if t%s['checkpoint_every']==0 or t==s['b_steps']:ctl.checkpoint(dict(job=job,phase='B',step=t,engine=e.pack()),ck)
        put(con,job,s['b_steps'],'natural_done',dict(complete=True,final=previous,acquisition=anchor['acquisition']))
        ck.unlink(missing_ok=True)
    finally:
        del e;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def queue(s):
    q=[]
    # Both-model Adam evidence first; memory without adaptive preconditioning last.
    for m in s['models']:
        for seed in s['seeds']:q.append((m,'adam',seed))
    if s['include_smollm_momentum']:
        m=s['models'][0]
        if 'momentum_sgd' not in s['learning_rates'][m['name']]:raise ValueError('Missing momentum LR for first/small model')
        for seed in s['seeds']:q.append((m,'momentum_sgd',seed))
    return q

def run(s):
    from reporting import report
    validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True);ctl=Control(s);con=db_open(out)
    try:
        env=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},cuda=torch.version.cuda)
        if not s['smoke']:env['packages']['datasets']=importlib.metadata.version('datasets')
        old=read(out/'environment.json')
        if old and old!=env:raise RuntimeError('Environment changed; resume using the same interpreter/environment')
        write(out/'environment.json',env);write(out/'settings.json',s)
        q=queue(s);write(out/'queue.json',[dict(model=m,optimizer=o,seed=seed) for m,o,seed in q])
        write(out/'protocol.json',dict(fresh_seeds=s['seeds'],LR_source='Frozen prior optimizer-generality development selections; no fresh-test LR selection',
            branch_timing='Immediately before fixed updates; reset once; native continued training',predictors='Frozen artifact bundled before new run; old test labels now development labels only',
            acquisition_failed='Retained, flagged, excluded from primary prediction and acquired-task retention summaries',
            fixed_population='SST-2 A / AG News B; exact full validation gradients; full-vocabulary held-out evaluation'))
        for m,opt,seed in q:
            job=f'natural/{m["name"]}/seed{seed}/{opt}'
            done=con.execute("SELECT count(*) FROM records WHERE kind='natural_done'").fetchone()[0]
            bd=con.execute("SELECT count(*) FROM records WHERE kind='branch_done'").fetchone()[0]
            ctl.check();ctl.pulse(job=job,phase='starting trajectory',completed_jobs=done,total_jobs=len(q),completed_branches=bd,total_branches=len(q)*len(s['fork_steps'])*3)
            case(s,m,opt,seed,con,ctl,job)
            ctl.pulse(phase='frozen prediction and paired summaries');report(con,out,True)
        failed_gates=sum(not json.loads(r[0])['acquisition']['passed'] for r in con.execute("SELECT record FROM records WHERE kind='anchor'"))
        paths=[json.loads(r[0]) for r in con.execute("SELECT record FROM records WHERE kind='path_done'")]
        unresolved=sum(not all(p[k] for k in ['closure_pass','curvature_integral_pass','derivative_pass']) for p in paths)
        write(out/'status.json',dict(status='complete_with_limitations' if failed_gates or unresolved else 'complete',last_progress=time.time(),completed_jobs=len(q),total_jobs=len(q),completed_branches=len(q)*len(s['fork_steps'])*3,acquisition_failures=failed_gates,unresolved_paths=unresolved))
    except Paused as exc:write(out/'status.json',dict(status='paused',message=str(exc),last_progress=time.time(),**ctl.fields))
    except Exception:write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields))
    finally:
        try:report(con,out,False)
        finally:con.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--settings',required=True);a=parser.parse_args();s=read(a.settings);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'worker.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:sys.exit(0)
        run(s)
