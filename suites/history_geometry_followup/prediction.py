"""Frozen old-run fits; strict preceding-update features and new-seed evaluation."""
import json, math, gzip, hashlib
from pathlib import Path
import numpy as np
from predict_forgetting import fit, predict, scores
from storage import read,write,get
HERE=Path(__file__).resolve().parent
THRESHOLDS=(.02,.05,.10)

def examples(records, threshold=.05, horizon=0):
    by={r['step']:r for r in records};out=[]
    for d in sorted(by):
        # Decision immediately before d; all features end with completed d-1.
        hist=[by[t] for t in range(d-8,d) if t in by]
        if len(hist)!=8 or d+horizon not in by:continue
        last=hist[-1];g=last['geometry'];x={}
        for domain in ['A','B']:
            changes=np.array([r['post'][domain+'_valid']['mean']-r['pre'][domain+'_valid']['mean'] for r in hist])
            x[domain+'_loss']=last['post'][domain+'_valid']['mean']
            for lag in [1,2,4,8]:
                x[f'{domain}_mean{lag}']=float(changes[-lag:].mean())
                x[f'{domain}_std{lag}']=float(changes[-lag:].std())
            x[domain+'_max8']=float(changes.max());x[domain+'_min8']=float(changes.min())
            w=.7**np.arange(7,-1,-1);x[domain+'_ewma']=float(changes@w/w.sum())
        x['time']=math.log1p(d)
        for key in ['projection','history_projection','current_projection','training_interference']:
            x[key]=math.asinh(g[key])
        x['cosine']=g['update_cosine'];x['norm_update']=math.log1p(g['update_norm']);x['norm_gA']=math.log1p(g['gA_norm'])
        x['projection_change']=math.asinh(g['projection']-hist[-2]['geometry']['projection'])
        target=by[d+horizon];delta=target['post']['A_test']['mean']-target['pre']['A_test']['mean']
        # Retrospective labels for onset/continuation; never inserted into x.
        previous_test=[r['post']['A_test']['mean']-r['pre']['A_test']['mean'] for r in hist[-4:]]
        out.append(dict(x=x,y=int(delta>threshold),delta=delta,decision=d,feature_step=d-1,target_step=d+horizon,
                        quiet_before_decision=all(v<=threshold for v in previous_test),
                        raw_projection=g['projection'],raw_loss_change=x['A_mean1']))
    return out

def columns(x):
    loss=[k for k in x if k.startswith(('A_','B_')) or k=='time']
    geometry=['projection','cosine','norm_update','norm_gA','history_projection','current_projection','training_interference','projection_change']
    return dict(loss_history=loss,geometry=geometry,combined=loss+geometry,
                loss_plus_projection=loss+['projection'],loss_plus_channels=loss+['history_projection','current_projection'])

def freeze(source_path,dest):
    raw=Path(source_path).read_bytes();source=json.loads(gzip.decompress(raw))
    result=dict(version=1,source_sha256=hashlib.sha256(raw).hexdigest(),source_note=source['note'],
                protocol='Fixed ridge logistic C=.1, train-only standardization; same optimizer class and source pool for all feature sets. Old inspected data are development only. No new-seed fitting.',fits={})
    groups=source['groups']
    for scope in ['pooled','smollm2_135m','pythia410m','adam','momentum_sgd']:
        selected=[g for g in groups if scope=='pooled' or scope in g['job'].split('/')]
        result['fits'][scope]={}
        for h in [0,1,3,7]:
            for threshold in THRESHOLDS:
                rows=sum([examples(g['records'],threshold,h) for g in selected],[])
                if not rows:continue
                result['fits'][scope][f'{h}/{threshold}']={name:fit(rows,cols) for name,cols in columns(rows[0]['x']).items()}
    write(dest,result);return result

def metric(rows,p,probability=True):
    if not rows:return dict(n=0,positives=0,auroc=None,average_precision=None)
    out=scores(rows,p)
    if not probability:out.pop('brier',None);return out
    y=np.array([r['y'] for r in rows]);p=np.asarray(p);pc=np.clip(p,1e-12,1-1e-12)
    out['log_loss']=float(-(y*np.log(pc)+(1-y)*np.log1p(-pc)).mean())
    out['calibration_bins']=[]
    for lo in np.arange(0,1,.1):
        mask=(p>=lo)&(p<(lo+.1) if lo<.9 else p<=1)
        if mask.any():out['calibration_bins'].append(dict(lower=float(lo),n=int(mask.sum()),mean_probability=float(p[mask].mean()),event_rate=float(y[mask].mean())))
    return out

def analyze(con,out):
    out=Path(out);frozen=read(HERE/'frozen_predictors.json');metrics=[];saved=[];excluded=[]
    jobs=[x[0] for x in con.execute("SELECT DISTINCT job FROM records WHERE kind='natural_done'")]
    for job in jobs:
        gate=get(con,job,'anchor')[0]['acquisition']
        if not gate['passed']:excluded.append(dict(job=job,gate=gate));continue
        records=get(con,job,'step');model=job.split('/')[1];optimizer=job.split('/')[3]
        # Pooled is primary; source-specific fits distinguish transfer tests explicitly.
        scopes=['pooled',model,'pythia410m' if model=='smollm2_135m' else 'smollm2_135m',optimizer,
                'momentum_sgd' if optimizer=='adam' else 'adam']
        for h in [0,1,3,7]:
            for th in THRESHOLDS:
                rows=examples(records,th,h)
                if not rows:continue
                subsets={'all':list(range(len(rows))),
                    'quiet_before_decision':[i for i,r in enumerate(rows) if r['quiet_before_decision']],
                    'recent_harm_before_decision':[i for i,r in enumerate(rows) if not r['quiet_before_decision']]}
                for scope in dict.fromkeys(x for x in scopes if x in frozen['fits']):
                    models=frozen['fits'][scope][f'{h}/{th}'];preds={k:predict(m,rows) for k,m in models.items()}
                    if scope=='pooled':preds.update(raw_projection=np.array([r['raw_projection'] for r in rows]),latest_validation_increase=np.array([r['raw_loss_change'] for r in rows]))
                    for name,p in preds.items():
                        for subset,ids in subsets.items():
                            metrics.append(dict(job=job,source_scope=scope,horizon=h,threshold=th,predictor=name,subset=subset,
                                **metric([rows[i] for i in ids],p[ids],name not in ['raw_projection','latest_validation_increase'])))
                    if scope=='pooled':
                        for i,r in enumerate(rows):saved.append(dict(job=job,horizon=h,threshold=th,**{k:v for k,v in r.items() if k!='x'},predictions={k:float(v[i]) for k,v in preds.items()}))
    paired=[]
    index={(r['job'],r['source_scope'],r['horizon'],r['threshold'],r['subset'],r['predictor']):r for r in metrics}
    for r in metrics:
        if r['predictor']!='combined':continue
        b=index[tuple(r[k] for k in ['job','source_scope','horizon','threshold','subset'])+('loss_history',)]
        paired.append({**{k:r[k] for k in ['job','source_scope','horizon','threshold','subset','n','positives']},
            **{k+'_combined_minus_loss':r.get(k)-b.get(k) if r.get(k) is not None and b.get(k) is not None else None for k in ['auroc','average_precision','brier','log_loss']}})
    write(out/'prediction_metrics.json',metrics);write(out/'paired_prediction_differences.json',paired);write(out/'prediction_records.json',saved);write(out/'prediction_exclusions.json',excluded)
    return metrics
