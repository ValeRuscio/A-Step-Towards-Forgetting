"""Priority scheduler. Scientific settings are immutable; operational budgets can resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/optimizer_generality_mpl')
import argparse,copy,fcntl,gc,hashlib,importlib.metadata,json,math,platform,sys,time,traceback
from pathlib import Path
import torch
import numpy as np
from association_core import Config,minibatches
from study_math import StudyEngine,evaluate,grad,weights,difference,dot,norm
from optimizers import make_optimizer,native_channels
from diagnostics import geometry,history_probes,path_probe
from tasks import dataset
from storage import Control,Paused,read,write,db_open,put,get
from timing_analysis import run_sources,analyze_groups,load_sqlite

def cfg(s,m,lr):return Config(model=m['repo'],revision=m['revision'],device=s['device'],threads=s['threads'],smoke=s['smoke'],lr=lr,beta1=s['momentum'],beta2=s['beta2'],eps=s['eps'],clip=s['clip'],microbatch=s['microbatch'],accumulation=s['accumulation'],max_length=s['max_length'])
def all_eval(e,data,s,ctl,development=False):
 out={}
 for sp in ['A_valid','B_valid']+([] if development else ['A_test','B_test']):
  ctl.pulse(phase='evaluate fixed population',split=sp);out[sp]=evaluate(e,data[sp],s['eval_batch'],ctl)
 return out

def outcome_valid(initial,anchor,s):
 gain=initial['A_valid']['mean']-anchor['A_valid']['mean'];acc=anchor['A_valid']['restricted_accuracy']
 return dict(passed=gain>=s['acquisition_min_gain'] and acc>=s['acquisition_min_accuracy'],loss_gain=gain,restricted_accuracy=acc,min_gain=s['acquisition_min_gain'],min_accuracy=s['acquisition_min_accuracy'])

def case(s,m,opt,lr,seed,role,variant,con,ctl,job):
 root=Path(s['output']);ck=root/'active_checkpoint.pt';development=role=='development';ast=s['calibration_a_steps'] if development else s['a_steps'];bst=s['calibration_b_steps'] if development else s['appendix_b_steps'] if job.startswith('appendix/') else s['b_steps']
 if get(con,job,'done'):return
 ctl.check();ctl.pulse(job=job,phase='loading pinned model',optimizer=opt,lr=lr)
 e=StudyEngine(cfg(s,m,lr));e.opt=make_optimizer(e,opt,lr,s)
 try:
  datap=root/'data'/(hashlib.sha256(f"{m['name']}/{seed}/{role}/{variant}".encode()).hexdigest()[:20]+'.json')
  if datap.exists():saved=read(datap);data=saved['data'];meta=saved['meta']
  else:
   ctl.pulse(phase='prepare paired tokenized examples');data,meta=dataset(e,s,seed,role,variant);write(datap,dict(data=data,meta=meta))
  spec=dict(model=m,optimizer=opt,lr=lr,seed=seed,role=role,variant=variant,a_steps=ast,b_steps=bst,data=meta,initial_state='pinned pretrained weights; fresh native optimizer; preserved A history into B')
  put(con,job,0,'spec',spec)
  if ck.exists():
   state=torch.load(ck,map_location='cpu',weights_only=True)
   if state['job']!=job:raise RuntimeError('An unfinished checkpoint belongs to another job: '+state['job'])
   e.restore(state['engine']);phase=state['phase'];step=state['step']
   if phase=='A':con.execute("DELETE FROM records WHERE job=? AND ((kind='acquisition' AND step>?) OR kind IN ('anchor','step','probe','path','done'))",(job,step))
   else:con.execute("DELETE FROM records WHERE job=? AND kind IN ('step','probe','path','done') AND step>?",(job,step))
   con.commit()
  else:
   # Only the one active checkpoint is retained. Initial records are reproducibly replayable.
   con.execute("DELETE FROM records WHERE job=? AND kind NOT IN ('spec')",(job,));con.commit();phase='A';step=0
   initial=all_eval(e,data,s,ctl,development);put(con,job,0,'initial',initial)
   ctl.checkpoint(dict(job=job,phase=phase,step=step,engine=e.pack()),ck)
  if phase=='A':
   for t in range(step+1,ast+1):
    ctl.check();ctl.pulse(phase='task A acquisition',step=t,total_steps=ast)
    e.gradient(minibatches(data,'A',seed,t,e.c));e.opt.step()
    if t%s['calibration_eval_every']==0 or t==ast:put(con,job,t,'acquisition',{'A_valid':evaluate(e,data['A_valid'],s['eval_batch'],ctl)})
    if t%s['checkpoint_every']==0 or t==ast:ctl.checkpoint(dict(job=job,phase='A',step=t,engine=e.pack()),ck)
   anchor=all_eval(e,data,s,ctl,development);initial=get(con,job,'initial')[0];put(con,job,0,'anchor',dict(evaluations=anchor,acquisition=outcome_valid(initial,anchor,s)))
   ctl.checkpoint(dict(job=job,phase='B',step=0,engine=e.pack()),ck);step=0
  anchor=get(con,job,'anchor')[0];previous=anchor['evaluations'] if step==0 else get(con,job,'step')[-1]['post'] if get(con,job,'step') else all_eval(e,data,s,ctl,development)
  for t in range(step+1,bst+1):
   ctl.check();ctl.pulse(phase='task B update',step=t,total_steps=bst)
   measured=not development
   # Calibration has sparse evaluations and deliberately never computes a test outcome.
   if measured:
    pre=previous;ctl.pulse(phase='old-task validation gradient',step=t);gA=grad(e,data['A_valid'],s['eval_batch'],ctl)
    ctl.pulse(phase='new-task validation gradient');gB=grad(e,data['B_valid'],s['eval_batch'],ctl);before=weights(e)
    probe=t in s['probe_steps'];start=e.pack() if probe else None
   batches=minibatches(data,'B',seed,t,e.c);ctl.pulse(phase='training minibatch gradient');gaudit=e.gradient(batches)
   if measured:
    train={n:(p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')) for n,p in e.params.items()};ch=native_channels(e,opt)
   e.opt.step()
   if measured:
    after=weights(e);delta=difference(after,before);gg=geometry(gA,gB,train,delta,ch);post=all_eval(e,data,s,ctl)
    gg['nonlinear_remainder_A_valid']=post['A_valid']['mean']-pre['A_valid']['mean']-gg['projection']
    put(con,job,t,'step',dict(step=t,pre=pre,post=post,geometry=gg,clipping={'raw_gradient_norm':gaudit['raw_norm'],'factor':gaudit['clip']},optimizer_audit=getattr(e.opt,'last_audit',None)))
    if probe:
     end=e.pack()
     if opt not in ['sgd','adam_beta1_0']:
      probes=history_probes(e,start,end,batches,data,s,ctl)
      put(con,job,t,'probe',dict(step=t,population='A_test/B_test',preserved_post=post,pre=pre,branches=probes,continuation=False))
     if seed in s['path_seeds'] or job.startswith('appendix/'):
      # Fixed audit steps, not selected after looking at test damage.
      result=path_probe(e,before,after,ch,data['A_valid'],s,ctl);put(con,job,t,'path',result)
     del end,start
    previous=post;del before,after,delta,gA,gB,train,ch
   elif t%s['calibration_eval_every']==0 or t==bst:
    vals=all_eval(e,data,s,ctl,True);put(con,job,t,'step',dict(step=t,pre=previous,post=vals,geometry=None));previous=vals
   if t%s['checkpoint_every']==0 or t==bst:ctl.checkpoint(dict(job=job,phase='B',step=t,engine=e.pack()),ck)
  rows=get(con,job,'step');final=rows[-1]['post'];score=float(np.mean([r['post']['B_valid']['mean']+s['calibration_retention_weight']*max(0.,r['post']['A_valid']['mean']-anchor['evaluations']['A_valid']['mean']) for r in rows]))
  put(con,job,bst,'done',dict(complete=True,acquisition=anchor['acquisition'],calibration_score=score if development else None,final=final,steps=bst));ck.unlink(missing_ok=True)
 finally:
  del e;gc.collect()
  if torch.cuda.is_available():torch.cuda.empty_cache()

def choose(con,model,opt,s,prefix='calibration'):
 name=f'{prefix}/{model}/{opt}';old=get(con,name,'selection')
 if old:return old[0]
 rows=[]
 for i,lr in enumerate(s['lr_grid'][opt]):
  job=f'{name}/candidate{i}';done=get(con,job,'done');failed=get(con,job,'failed')
  if done:
   x=done[0];rows.append(dict(lr=lr,score=x['calibration_score'],eligible=x['acquisition']['passed'],acquisition=x['acquisition'],job=job))
  elif failed:rows.append(dict(lr=lr,score=None,eligible=False,failed=failed[0],job=job))
  else:raise RuntimeError('Calibration selection before candidates finish')
 eligible=[r for r in rows if r['eligible']];best=min(eligible,key=lambda r:(r['score'],r['lr'])) if eligible else None
 selection=dict(model=model,optimizer=opt,lr=best['lr'] if best else None,candidates=rows,rule='Minimize mean(B_valid NLL + positive A_valid excess above A anchor), subject to A acquisition gate; tie -> smaller LR. No test values read.',valid=best is not None)
 put(con,name,0,'selection',selection);return selection

def resolve_assets(s,out):
 old=read(out/'resolved_assets.json')
 if old:return old
 resolved=copy.deepcopy(s['models'])
 if not s['smoke']:
  from huggingface_hub import HfApi
  api=HfApi()
  for m in resolved:m['revision']=api.model_info(m['repo'],revision=m['revision']).sha
 write(out/'resolved_assets.json',resolved);return resolved

def run(s):
 from reporting import report
 out=Path(s['output']);out.mkdir(parents=True,exist_ok=True);ctl=Control(s);con=db_open(out)
 try:
  versions={}
  for package in ['torch','transformers','numpy','datasets','huggingface-hub']:
   try:versions[package]=importlib.metadata.version(package)
   except importlib.metadata.PackageNotFoundError:
    if package=='datasets' and s['smoke']:versions[package]='not installed (smoke only)'
    else:raise
  env=dict(python=sys.version,platform=platform.platform(),packages=versions,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',cuda=torch.version.cuda)
  old=read(out/'environment.json')
  if old and old!=env:raise RuntimeError('Environment changed on resume; use the recorded environment or restart')
  write(out/'environment.json',env);ctl.pulse(phase='timing and persistence analysis')
  if not (out/'timing_existing/REPORT.txt').exists():run_sources(s,out/'timing_existing')
  ctl.check();models=resolve_assets(s,out);queue=[]
  for m in models:
   for opt in s['optimizers']:
    for i,lr in enumerate(s['lr_grid'][opt]):queue.append(dict(job=f"calibration/{m['name']}/{opt}/candidate{i}",model=m,opt=opt,lr=lr,seed=s['calibration_seed'],role='development',variant='standard'))
   # This model's calibration and confirmation finish before moving to the next model.
   for seed in s['seeds']:
    for opt in s['optimizers']:queue.append(dict(job=f"confirmation/{m['name']}/seed{seed}/{opt}/standard",model=m,opt=opt,lr=None,seed=seed,role='confirmation',variant='standard'))
  if s['label_controls']:
   for m in models:
    if m['name'] not in s['label_models']:continue
    for seed in s['seeds']:
     for variant in ['counterbalanced_shared','counterbalanced_disjoint']:queue.append(dict(job=f"confirmation/{m['name']}/seed{seed}/{s['label_optimizer']}/{variant}",model=m,opt=s['label_optimizer'],lr=None,seed=seed,role='confirmation',variant=variant))
  if s['appendix']:
   m=models[0]
   for opt in s['appendix_optimizers']:
    for i,lr in enumerate(s['lr_grid'][opt]):queue.append(dict(job=f"appendix_calibration/{m['name']}/{opt}/candidate{i}",model=m,opt=opt,lr=lr,seed=s['calibration_seed'],role='development',variant='standard'))
    queue.append(dict(job=f"appendix/{m['name']}/seed{s['appendix_seed']}/{opt}/standard",model=m,opt=opt,lr=None,seed=s['appendix_seed'],role='confirmation',variant='standard'))
  write(out/'queue.json',queue);write(out/'protocol.json',dict(test_selection=False,development_pool='text hash bucket<2/10',confirmation_pool='text hash bucket>=2/10',optimizer_boundary='native history from task A retained into task B',calibration='model-specific, optimizer-specific development selection; learning rates frozen before each confirmation phase',appendix='single-seed preliminary matrix-hybrid variants, not official tuned optimizer comparison',reduced_precision=False,weight_decay=0))
  for index,q in enumerate(queue):
   if get(con,q['job'],'done') or get(con,q['job'],'failed') or get(con,q['job'],'skipped'):continue
   ctl.check();ctl.pulse(phase='starting queue unit',job=q['job'],completed_jobs=index,total_jobs=len(queue))
   lr=q['lr']
   if lr is None:
    sel=choose(con,q['model']['name'],q['opt'],s,'appendix_calibration' if q['job'].startswith('appendix/') else 'calibration');lr=sel['lr']
    if lr is None:
     put(con,q['job'],0,'skipped',{'reason':'No development LR passed the old-task acquisition gate; do not interpret absence of learning as retention.'});continue
   try:case(s,q['model'],q['opt'],lr,q['seed'],q['role'],q['variant'],con,ctl,q['job'])
   except Paused:raise
   except Exception:
    if q['role']=='development' or q['job'].startswith('appendix/'):
     put(con,q['job'],0,'failed',dict(error=traceback.format_exc(),note='Numerical/dependency failure retained; no silent optimizer fallback.'));(out/'active_checkpoint.pt').unlink(missing_ok=True)
    else:raise
   report(s,con)
  # Metrics exclude development and appendix fits; unacquired confirmation trajectories are labelled in summaries.
  ctl.pulse(phase='final timing and uncertainty analysis',completed_jobs=len(queue),total_jobs=len(queue))
  groups=load_sqlite(out/'results.sqlite')
  excluded=[]
  for job in list(groups):
   done=get(con,job,'done')
   if not done or not done[0]['acquisition']['passed']:excluded.append(job);del groups[job]
  write(out/'timing_excluded_acquisition.json',excluded)
  analyze_groups(groups,out/'timing_new',reps=s['bootstrap_replicates'])
  from study_prediction import analyze as frozen_analyze
  frozen=read(Path(__file__).with_name('frozen_baselines.json'))
  fixed_metrics=[]
  for job,records in groups.items():fixed_metrics.extend(dict(job=job,**row) for row in frozen_analyze(records,frozen))
  write(out/'original_frozen_prediction_metrics.json',fixed_metrics)
  report(s,con)
  from reporting import plot
  plot(s,con)
  limitations=con.execute("SELECT count(*) FROM records WHERE kind IN ('skipped','failed')").fetchone()[0]
  invalid=sum(not x['acquisition']['passed'] for (raw,) in con.execute("SELECT record FROM records WHERE kind='done' AND job LIKE 'confirmation/%'") for x in [json.loads(raw)])
  write(out/'status.json',dict(status='complete_with_limitations' if limitations or invalid else 'complete',last_progress=time.time(),completed_jobs=len(queue),total_jobs=len(queue),failed_or_skipped=limitations,confirmation_acquisition_failures=invalid,message='Inspect acquisition gates and individual numerical audits; completed does not mean every audit passed.'))
 except Paused as exc:write(out/'status.json',dict(status='paused',message=str(exc),last_progress=time.time(),**ctl.fields))
 except Exception:write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields))
 finally:
  try:report(s,con)
  finally:con.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--settings',required=True);args=p.parse_args();s=read(args.settings);out=Path(s['output'])
 with open(out/'worker.lock','a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:sys.exit(0)
  run(s)
