"""Timestamp-aligned unfitted forecasts. No target fitting or threshold optimization."""
import json,math,gzip,sqlite3,hashlib,zipfile,tempfile
from pathlib import Path
import numpy as np
from storage import write
from predict_forgetting import scores

def small_record(r):
 x={'step':r['step'],'pre':{},'post':{},'geometry':{}}
 for when in ['pre','post']:
  for sp in ['A_valid','B_valid','A_test','B_test']:
   if sp in r.get(when,{}):x[when][sp]={'mean':r[when][sp]['mean']}
 g=r.get('geometry') or {}
 for k in ['projection','update_cosine','update_norm','gA_norm','population_interference','training_interference']:
  if k in g:x['geometry'][k]=g[k]
 return x

def load_sqlite(path):
 uri='file:'+str(Path(path).resolve())+'?mode=ro'
 # Use normal read-only mode for live databases, so committed WAL rows are visible.
 with sqlite3.connect(uri,uri=True) as con:
  groups={}
  for job,raw in con.execute("SELECT job,record FROM records WHERE kind='step' ORDER BY job,step"):
   if not (job.endswith('/natural') or job.startswith('confirmation/')):continue
   r=json.loads(raw)
   if 'A_test' not in r.get('post',{}):continue
   groups.setdefault(job,[]).append(small_record(r))
 return groups

def load_source(path):
 p=Path(path)
 if p.suffix=='.gz':return json.loads(gzip.decompress(p.read_bytes()))
 if p.suffix=='.zip':
  with zipfile.ZipFile(p) as z:
   names=[n for n in z.namelist() if Path(n).name=='results.sqlite']
   if len(names)!=1:raise ValueError('Need exactly one results.sqlite in share ZIP')
   with tempfile.TemporaryDirectory() as d:
    target=Path(d)/'results.sqlite'
    with z.open(names[0]) as src,open(target,'wb') as dst:
     import shutil;shutil.copyfileobj(src,dst)
    return load_sqlite(target)
 return load_sqlite(p/'results.sqlite' if p.is_dir() else p)

def forecast_rows(records,horizon,threshold):
 by={r['step']:r for r in records};out=[]
 for decision in sorted(by):
  target=decision+horizon
  if decision<10 or target not in by or decision-1 not in by:continue
  prev=by[decision-1];g=prev.get('geometry') or {}
  if not all(k in g for k in ['projection','update_cosine','update_norm','gA_norm']):continue
  # At theta_(decision-1): current validation loss is known, no gradient of the upcoming batch is used.
  deltas=[]
  for k in range(max(min(by),decision-8),decision):
   if k in by:deltas.append(by[k]['post']['A_valid']['mean']-by[k]['pre']['A_valid']['mean'])
  if not deltas:continue
  ew=deltas[0]
  for value in deltas[1:]:ew=.5*value+.5*ew
  s=dict(previous_projection=g['projection'],previous_cosine=g['update_cosine'],previous_update_norm=g['update_norm'],previous_gradient_norm=g['gA_norm'],latest_validation_increase=deltas[-1],mean_validation_increase4=float(np.mean(deltas[-4:])),mean_validation_increase8=float(np.mean(deltas)),ewma_validation_increase=ew,current_validation_loss=by[decision]['pre']['A_valid']['mean'])
  for key in ['population_interference','training_interference']:
   if key in g:s['previous_'+key]=g[key]
  delta=by[target]['post']['A_test']['mean']-by[target]['pre']['A_test']['mean']
  out.append(dict(decision_step=decision,target_step=target,gradient_measurement_step=decision-1,last_observed_update=decision-1,delta=delta,y=int(delta>threshold),scores=s))
 return out

def metric(y,v):
 r=scores([{'y':int(x)} for x in y],v);r.pop('brier',None);return r

def block_interval(rows,a,b,block,reps,seed):
 if not rows or a not in rows[0]['scores'] or b not in rows[0]['scores']:return None
 y=np.asarray([x['y'] for x in rows]);av=np.asarray([x['scores'][a] for x in rows]);bv=np.asarray([x['scores'][b] for x in rows]);rng=np.random.default_rng(seed);ds=[];n=len(rows)
 for _ in range(reps):
  starts=rng.integers(0,n,size=math.ceil(n/block));idx=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
  ma=metric(y[idx],av[idx])['auroc'];mb=metric(y[idx],bv[idx])['auroc']
  if ma is not None and mb is not None:ds.append(ma-mb)
 return dict(block_length=block,resamples=reps,usable=len(ds),interval_95=np.quantile(ds,[.025,.975]).tolist() if ds else None,note='Within-trajectory circular-block bootstrap; descriptive, not independent-seed inference or a stationarity guarantee.')

def analyze_groups(groups,out,horizons=(0,1,3,7),thresholds=(.02,.05,.1),reps=300):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);metrics=[];uncertainty=[];predictions=[];negative=[]
 for job,records in sorted(groups.items()):
  for horizon in horizons:
   for threshold in thresholds:
    rows=forecast_rows(records,horizon,threshold)
    if not rows:continue
    common=dict(job=job,horizon_after_decision=horizon,threshold=threshold)
    for name in rows[0]['scores']:
     y=[r['y'] for r in rows];v=[r['scores'][name] for r in rows]
     metrics.append(dict(**common,predictor=name,period='all',**metric(y,v)))
     for i,indices in enumerate(np.array_split(np.arange(len(rows)),3)):
      if len(indices):metrics.append(dict(**common,predictor=name,period=['early','middle','late'][i],**metric(np.asarray(y)[indices],np.asarray(v)[indices])))
    if horizon==0 and threshold==.05:
     for baseline in ['latest_validation_increase','mean_validation_increase4','mean_validation_increase8','ewma_validation_increase']:
      for block in [8,16]:uncertainty.append(dict(**common,contrast='previous_projection minus '+baseline,**block_interval(rows,'previous_projection',baseline,block,reps,9021)))
     # A descriptive lag placebo, not a randomization p-value for nonstationary series.
     v=np.array([r['scores']['previous_projection'] for r in rows]);y=[r['y'] for r in rows]
     for shift in [16,32,48]:
      if len(v)>2*shift:negative.append(dict(**common,shift=shift,**metric(y,np.roll(v,shift)),note='Circular temporal shift; not a causal/randomization test.'))
    predictions.extend(dict(**common,**r) for r in rows)
 write(out/'metrics.json',metrics);write(out/'paired_block_intervals.json',uncertainty);write(out/'shift_controls.json',negative);write(out/'prediction_records.json',predictions)
 lines=['TIMING AND PERSISTENCE — unfitted scores, no target-seed tuning','Decision before update d: use geometry from completed update d-1, latest validation change from d-1, and validation loss at theta_(d-1).','Horizon0 predicts update d; horizon1 predicts d+1; horizon3 predicts d+3; horizon7 predicts d+7. No current/future test loss is a feature.','Raw score orientation is fixed (larger means predicted harm); no sign flipping for below-chance results.','CIs are within-trajectory block-bootstrap diagnostics; they do not substitute for independent seeds.']
 for r in metrics:
  if r['period']=='all' and r['threshold']==.05 and r['horizon_after_decision']==0:lines.append(f"{r['job']} {r['predictor']}: AUROC={r['auroc']}; n={r['n']}; events={r['positives']}")
 (out/'REPORT.txt').write_text('\n'.join(lines));return metrics

def run_sources(s,out):
 groups={};provenance=[]
 paths=[Path(__file__).with_name('existing_natural_records.json.gz')]+[Path(x) for x in s.get('existing_sources',[])]
 for p in paths:
  if not p.exists():raise FileNotFoundError(f'Configured existing source missing: {p}')
  loaded=load_source(p);provenance.append(dict(path=str(p),jobs=list(loaded)))
  for job,rows in loaded.items():
   if job in groups and groups[job]!=rows:raise ValueError('Conflicting duplicate natural trajectory: '+job)
   groups[job]=rows
 write(Path(out)/'sources.json',provenance);return analyze_groups(groups,out,reps=s['bootstrap_replicates'])
