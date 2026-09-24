import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('MPLCONFIGDIR','/tmp/history_dynamics_mpl')
import argparse,fcntl,gc,hashlib,importlib.metadata,json,platform,sqlite3,sys,time,traceback
from pathlib import Path
import torch
from association_core import Config,minibatches
from study_math import StudyEngine,weights,evaluate
from components import component_gradients,digest
from dynamics import Ages,capture,collect
from cohorts import Cohorts
from storage import Control,Paused,db_open,get,put,write,read

def source_active(root):
    f=Path(root)/'worker.lock'
    if not f.exists():return False
    with open(f,'rb') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);return False
        except BlockingIOError:return True

class ReplayControl(Control):
    def check(self):
        super().check()
        if getattr(self,'computing',False) and source_active(self.s['source']):raise Paused('Original source worker became active; replay paused to avoid GPU contention.')

def snapshot(s,ctl):
    out=Path(s['output']);src=Path(s['source']);dest=out/'source_snapshot.sqlite'
    if dest.exists():return
    while source_active(src):
        ctl.check();ctl.pulse(phase='waiting for original worker to become idle');time.sleep(5)
    ctl.check()
    if not (src/'results.sqlite').exists():raise FileNotFoundError('Source results.sqlite not found')
    a=sqlite3.connect((src/'results.sqlite').as_uri()+'?mode=ro',uri=True);b=sqlite3.connect(str(dest)+'.tmp')
    try:
        a.backup(b);b.execute('PRAGMA journal_mode=DELETE')
    finally:a.close();b.close()
    with sqlite3.connect(str(dest)+'.tmp') as c:jobs=[r[0] for r in c.execute("SELECT job FROM records WHERE kind='natural_done' ORDER BY job")]
    if not jobs:raise RuntimeError('No completed natural trajectories in source; no replay started')
    settings=read(src/'settings.json');env=read(src/'environment.json')
    if not settings or settings['version']!='natural-components-1':raise ValueError('Expected natural_components source run')
    write(out/'source_settings.json',settings);write(out/'source_environment.json',env)
    write(out/'source_code_hashes.json',read(src/'code_hashes.json',{}))
    for job in jobs:
        f=src/'data'/(job.replace('/','_')+'.json');raw=f.read_bytes();dst=out/'data'/f.name;dst.parent.mkdir(exist_ok=True);dst.write_bytes(raw)
    write(out/'snapshot_manifest.json',dict(jobs=jobs,source=str(src),source_settings_sha256=hashlib.sha256((src/'settings.json').read_bytes()).hexdigest(),data_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'data').iterdir()},note='Frozen completed trajectories; later source completion is not silently added. New output directory for a later snapshot.'))
    os.replace(str(dest)+'.tmp',dest)

def check_algebra(row,s):
    failures=[]
    for r in row['geometry']:
        for key in ['cross_time_closure','channel_change_closure','magnitude_orientation_closure']:
            value=r.get(key)
            if value is not None and abs(value)>s['closure_atol']+s['closure_rtol']*max(abs(r['projection']),abs(r.get('previous_projection',0))):failures.append(dict(population=r['population'],component=r['component'],channel=r['channel'],layer=r['layer'],check=key,error=value))
    ar=row['age_audit'];ar['relative_moment_residual']=ar['moment_residual_norm']/max(ar['moment_norm'],1e-30);ar['relative_history_residual']=ar['history_residual_norm']/max(ar['history_norm'],1e-30)
    ar['passed']=ar['relative_moment_residual']<=s['age_residual_rtol'] and ar['relative_history_residual']<=s['age_residual_rtol']
    row['algebra_audit']=dict(passed=not failures,failures=failures)
    if failures:raise ArithmeticError('Cross-time algebra failed; see recorded diagnostic')

def case(s,source,con,ctl,job):
    if get(con,job,'done'):return
    out=Path(s['output']);ss=read(out/'source_settings.json');spec=get(source,job,'spec')[0];anchor=get(source,job,'anchor')[0];orig={r['step']:r for r in get(source,job,'step')};seed=spec['seed'];m=spec['model']
    if len(orig)!=ss['b_steps']:raise ValueError('Source trajectory is incomplete')
    cfg=Config(model=m['repo'],revision=m['revision'],device=s['device'],threads=ss['threads'],smoke=ss['smoke'],lr=spec['lr'],beta1=ss['beta1'],beta2=ss['beta2'],eps=ss['eps'],clip=ss['clip'],microbatch=ss['microbatch'],accumulation=ss['accumulation'],max_length=ss['max_length'])
    ctl.pulse(job=job,phase='load pinned model for replay');e=StudyEngine(cfg)
    data=read(out/'data'/(job.replace('/','_')+'.json'))['data']
    pop={k:data[k][:s['test_population']] if k=='A_test' else data[k] for k in s['populations']}
    af=out/'active_A_anchor.pt';ck=out/'acquisition_checkpoint.pt'
    put(con,job,0,'spec',dict(source=spec,acquired=anchor['acquisition']['passed'],populations={k:[r['example_id'] for r in v] for k,v in pop.items()},population_sizes={k:len(v) for k,v in pop.items()}))
    try:
        if af.exists():
            z=torch.load(af,map_location='cpu',weights_only=True)
            if z['job']!=job:raise RuntimeError('Replay anchor belongs to another case')
            e.restore(z['engine']);del z
        else:
            start=0
            if ck.exists():
                z=torch.load(ck,map_location='cpu',weights_only=True)
                if z['job']!=job:raise RuntimeError('Acquisition checkpoint belongs to another case')
                e.restore(z['engine']);start=z['step'];del z
            for t in range(start+1,ss['a_steps']+1):
                ctl.check();ctl.pulse(phase='reconstruct task A optimizer state',step=t,total_steps=ss['a_steps'])
                e.gradient(minibatches(data,'A',seed,t,e.c));e.opt.step()
                if t%s['checkpoint_every']==0 or t==ss['a_steps']:ctl.checkpoint(dict(job=job,step=t,engine=e.pack()),ck)
            if digest(weights(e))!=anchor['weight_hash']:raise ArithmeticError('Task-A anchor hash mismatch; no attribution will be reported as replay')
            ctl.checkpoint(dict(job=job,engine=e.pack()),af);ck.unlink(missing_ok=True)
        if digest(weights(e))!=anchor['weight_hash']:raise ArithmeticError('Restored anchor mismatch')
        moment={n:e.opt.state[p]['exp_avg'].detach().cpu() for n,p in e.params.items()};ages=Ages(moment,ss['beta1']);del moment
        existing=get(con,job,'update');done_steps=[r['step'] for r in existing];completed=max(done_steps,default=0)
        if done_steps!=list(range(1,completed+1)):raise RuntimeError('Non-contiguous diagnostic records; do not silently bridge a gap')
        previous=None;cohorts=Cohorts(s.get('cohort_origins',[1,8,32,64]),s.get('cohort_max_age',32))
        for t in range(1,ss['b_steps']+1):
            ctl.check();ctl.pulse(phase='rebuild age state' if t<completed else 'measure cross-time geometry',step=t,total_steps=ss['b_steps'])
            need=t>=max(completed,1);grads={};losses={}
            if need:
                for sp,rows in pop.items():
                    ctl.pulse(phase='exact component gradients before update',split=sp,step=t)
                    grads[sp],losses[sp]=component_gradients(e,rows,s['derivative_batch'],ctl)
            e.gradient(minibatches(data,'B',seed,t,e.c));cap=capture(e)
            if t>completed:
                row=collect(e,grads,losses,cap,ages,previous,s['norm_floor'],ctl);row.update(step=t,job=job)
                row['cohorts']=cohorts.measure(t,grads,cap,s['norm_floor'],ctl)
            e.opt.step();e.opt.zero_grad(set_to_none=True)
            actual=digest(weights(e))
            if actual!=orig[t]['endpoint_hash']:raise ArithmeticError(f'Exact endpoint replay failed at B update {t}; stop, do not loosen this check')
            if t>completed:
                row['replay_audit']=dict(passed=True,endpoint_hash=actual)
                row['outcomes']={}
                for sp,rows in pop.items():
                    post=evaluate(e,rows,s['derivative_batch'],ctl)
                    row['outcomes'][sp]=dict(post={k:post[k] for k in ['mean','confusion','leakage','full_accuracy','restricted_accuracy','answer_mass']},change={k:post['mean' if k=='total' else k]-losses[sp][k] for k in ['total','confusion','leakage']})
                row['source_full_population_changes']={sp:{k:orig[t]['post'][sp]['mean' if k=='total' else k]-orig[t]['pre'][sp]['mean' if k=='total' else k] for k in ['total','confusion','leakage']} for sp in ['A_valid','A_test']}
                check_algebra(row,s);put(con,job,t,'update',row)
            ages.advance({n:v['training'] for n,v in cap['state'].items()})
            cohorts.record(t,cap)
            if need:previous=dict(grads=grads,cap=cap)
            if t>completed and t%16==0:
                from reporting import report
                report(con,out,s)
        put(con,job,0,'done',dict(complete=True,updates=ss['b_steps'],acquired=anchor['acquisition']['passed']))
        af.unlink(missing_ok=True)
    finally:
        del e;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def run(s):
    from reporting import report
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True);ctl=ReplayControl(s);con=db_open(out);source=None
    try:
        snapshot(s,ctl)
        source=sqlite3.connect((out/'source_snapshot.sqlite').as_uri()+'?mode=ro&immutable=1',uri=True)
        env=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},cuda=torch.version.cuda)
        source_env=read(out/'source_environment.json');required=source_env.get('packages',{})
        for k in ['torch','transformers','numpy']:
            if required.get(k)!=env['packages'][k]:raise RuntimeError('Source/runtime package mismatch for '+k+'; use the original environment')
        write(out/'environment.json',env);jobs=read(out/'snapshot_manifest.json')['jobs'];write(out/'queue.json',jobs)
        for job in jobs:
            if get(con,job,'done'):
                for f in [out/'active_A_anchor.pt',out/'acquisition_checkpoint.pt']:
                    if f.exists():
                        z=torch.load(f,map_location='cpu',weights_only=True)
                        if z['job']==job:f.unlink()
                continue
            while source_active(s['source']):
                ctl.computing=False;ctl.check();ctl.pulse(phase='waiting for original worker to become idle');time.sleep(5)
            ctl.computing=True;ctl.pulse(job=job,phase='start replay',completed_jobs=con.execute("SELECT COUNT(*) FROM records WHERE kind='done'").fetchone()[0],total_jobs=len(jobs))
            case(s,source,con,ctl,job);report(con,out,s)
        report(con,out,s);summary=read(out/'summary.json');limited=summary['age_audits_pass']<summary['updates'];write(out/'status.json',dict(status='complete_with_limitations' if limited else 'complete',scope='frozen source snapshot',completed_jobs=len(jobs),total_jobs=len(jobs),last_progress=time.time()))
    except Paused as ex:write(out/'status.json',dict(status='paused',message=str(ex),last_progress=time.time(),**ctl.fields))
    except Exception:write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields))
    finally:
        try:report(con,out,s)
        finally:
            con.close()
            if source is not None:source.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--settings',required=True);a=ap.parse_args();s=read(a.settings)
    with open(Path(s['output'])/'worker.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:sys.exit(0)
        run(s)
