"""Deferred, fixed-rule finite-path audits of already saved natural trajectories."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('USE_TF','0')
import argparse,fcntl,gc,time,traceback,importlib.metadata,sys,platform
from pathlib import Path
import torch
from storage import Control,Paused,read,write,db_open,get,put
from association_core import Config,minibatches
from study_math import StudyEngine,weights,channels
from components import digest,complete_directions
from path_analysis import path
from run_study import queue
from reporting import report

def run(s):
    out=Path(s['output']);ctl=Control(s);con=db_open(out)
    try:
        env=dict(python=sys.version,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},cuda=torch.version.cuda)
        if not s['smoke']:env['packages']['datasets']=importlib.metadata.version('datasets')
        if env!=read(out/'environment.json'):raise RuntimeError('Environment changed since natural trajectories; restore the recorded environment')
        q=queue(s)
        for m,seed,variant in q:
            job=f'natural/{m["name"]}/seed{seed}/{variant}'
            if get(con,job,'deferred_paths_done'):continue
            if not get(con,job,'natural_done'):raise RuntimeError('Finish natural trajectories first')
            ctl.pulse(job=job,phase='reconstruct A anchor for selected paths')
            cfg=Config(model=m['repo'],revision=m['revision'],device=s['device'],threads=s['threads'],smoke=s['smoke'],lr=s['learning_rates'][m['name']],beta1=s['beta1'],beta2=s['beta2'],eps=s['eps'],clip=s['clip'],microbatch=s['microbatch'],accumulation=s['accumulation'],max_length=s['max_length'])
            e=StudyEngine(cfg);data=read(out/'data'/(job.replace('/','_')+'.json'))['data'];ck=out/'path_acquisition.pt'
            start=0
            if ck.exists():
                z=torch.load(ck,map_location='cpu',weights_only=True)
                if z['job']!=job:raise RuntimeError('Path checkpoint job mismatch')
                e.restore(z['engine']);start=z['step'];del z
            for t in range(start+1,s['a_steps']+1):
                ctl.check();ctl.pulse(phase='reconstruct A anchor for selected paths',step=t,total_steps=s['a_steps'])
                e.gradient(minibatches(data,'A',seed,t,cfg));e.opt.step()
                if t%s['checkpoint_every']==0 or t==s['a_steps']:ctl.checkpoint(dict(job=job,step=t,engine=e.pack()),ck)
            if digest(weights(e))!=get(con,job,'anchor')[0]['weight_hash']:raise ArithmeticError('Path A-anchor replay mismatch')
            selected={r['step']:r['reasons'] for r in get(con,job,'selection')[0]['selected']};orig={r['step']:r for r in get(con,job,'step')}
            for t in range(1,max(selected,default=0)+1):
                ctl.check();ctl.pulse(phase='replay B for paths',step=t,total_steps=max(selected))
                before=weights(e) if t in selected else None
                e.gradient(minibatches(data,'B',seed,t,cfg));ch=channels(e) if before is not None else None;e.opt.step();e.opt.zero_grad(set_to_none=True)
                after=weights(e)
                if digest(after)!=orig[t]['endpoint_hash']:raise ArithmeticError('Path B-endpoint replay mismatch')
                if before is not None:
                    dirs=complete_directions(before,after,*ch);del ch
                    for sp in ['A_valid','A_test']:
                        pj=f'{job}/update{t:03d}/{sp}';put(con,pj,0,'selection',dict(reasons=selected[t],source='fixed uniform draw / validation-only large-event draw'))
                        path(e,before,after,dirs,data[sp][:s['path_population']],s,ctl,con,pj,sp)
                    del dirs,before
                del after
            put(con,job,0,'deferred_paths_done',dict(complete=True));ck.unlink(missing_ok=True)
            del e;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
            report(con,out,s)
        write(out/'status.json',dict(status='complete',scope='selected path jobs; inspect individual audit failures',last_progress=time.time()))
    except Paused as ex:write(out/'status.json',dict(status='paused',message=str(ex),last_progress=time.time(),**ctl.fields))
    except Exception:write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields))
    finally:report(con,out,s);con.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--settings',required=True);s=read(ap.parse_args().settings)
    with open(Path(s['output'])/'worker.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(s)
