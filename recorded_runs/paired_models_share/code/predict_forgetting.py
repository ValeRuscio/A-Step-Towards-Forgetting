"""Frozen, grouped prediction tests. No post-update values are used as features.
Exploratory OLMo leave-one-seed-out; frozen OLMo fits transfer to new models.
"""
from pathlib import Path
import sqlite3,json,math,hashlib
import numpy as np

PROTOCOL=dict(version='prediction-2',threshold=.05,lead_steps=[0,1],regularization_C=.1,
    primary='lead=1: preceding pre-update measurement predicts next update A_test increase > 0.05 KL',
    secondary='lead=0: current-gradient-informed pre-update diagnostic, not advance warning',
    fitting='Only original OLMo trajectories. Leave one complete seed out for exploratory evaluation. New models never enter fitting.',
    caveat='Protocol chosen after inspecting OLMo; OLMo evaluation is exploratory, transfer results are prospective only if frozen before inspecting new outcomes.')

def read_records(db):
    with sqlite3.connect(f'file:{Path(db).resolve()}?mode=ro',uri=True) as c:
        return [json.loads(r[0]) for r in c.execute('SELECT record FROM trajectory ORDER BY seed,step')]

def features(r):
    g=r['pre']['geometry']['global_geometry'];pre=r['pre']['loss'];v={'log_step':math.log1p(r['step']),'A_loss':pre['A_test']['mean'],'B_loss':pre['B_test']['mean']}
    for i,n in enumerate(['h','c','gA','gB']):v['norm_'+n]=math.log1p(g['norms'][i])
    for i in range(4):
        for j in range(i+1,4):
            v[f'cos_{i}_{j}']=g['cosines'][i][j] if g['cosine_defined'][i][j] else 0.
            v[f'dot_{i}_{j}']=math.asinh(g['gram'][i][j])
    v['local_A_projection']=math.asinh(g['gram'][2][0]+g['gram'][2][1])
    ch=r['temporal_changes']
    for name in ['history','current','gA','gB']:
        x=ch['vector_changes'][name]['global_change'] if ch else {}
        v['change_angle_'+name]=x.get('previous_current_cosine') or 0.
        v['change_size_'+name]=math.log1p(min(x.get('relative_change',0.),1e12))
    for key in ['history__gA','current__gA','history__gB','current__gB']:
        x=ch['global_changes'][key] if ch else {}
        for term in ['first_vector_change','second_vector_change','joint_change']:v['change_'+key+'_'+term]=math.asinh(x.get(term,0.))
    layers=r['pre']['geometry']['layers'];ids=sorted(int(k) for k in layers if k.isdigit())
    for bucket in range(3):
        selected=np.array_split(ids,3)[bucket];a=sum((np.asarray(layers[str(i)]['gram']) for i in selected),np.zeros((4,4)))
        v[f'layerthird_{bucket}_Aprojection']=math.asinh(a[2,0]+a[2,1])
    return v

def dataset(records,lead):
    by={(r['seed'],r['step']):r for r in records};rows=[]
    for target in records:
        src=by.get((target['seed'],target['step']-lead))
        if src is None or src['temporal_changes'] is None:continue
        x=features(src);previous=by.get((src['seed'],src['step']-1))
        for sp in ['A_test','B_test']:x['last_change_'+sp]=src['pre']['loss'][sp]['mean']-previous['pre']['loss'][sp]['mean'] if previous else 0.
        rows.append(dict(seed=target['seed'],step=target['step'],feature_step=src['step'],x=x,
            delta=target['functional_consequence']['A_test']['actual_loss_change'],y=int(target['functional_consequence']['A_test']['actual_loss_change']>PROTOCOL['threshold'])))
    return rows

def columns(rows):
    keys=sorted(rows[0]['x']);base=['log_step','A_loss','B_loss','last_change_A_test','last_change_B_test']
    return {'loss_only':base,'local_projection':base+['local_A_projection'],
            'static_geometry':[k for k in keys if not k.startswith('change_')], 'geometry_plus_changes':keys}

def fit(rows,cols):
    x=np.asarray([[r['x'][k] for k in cols] for r in rows]);
    if not np.isfinite(x).all():raise FloatingPointError('Nonfinite predictor feature')
    y=np.asarray([r['y'] for r in rows]);mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.
    if len(set(y))<2:return dict(columns=cols,mean=mu.tolist(),scale=sd.tolist(),constant=float(y.mean()))
    design=np.column_stack([(x-mu)/sd,np.ones(len(x))]);w=np.zeros(design.shape[1]);pen=np.full(len(w),1/PROTOCOL['regularization_C']);pen[-1]=0.
    objective=lambda w:float(np.logaddexp(0,np.einsum('ij,j->i',design,w)).sum()-np.sum(y*np.einsum('ij,j->i',design,w))+.5*np.sum(pen*w*w))
    converged=False
    for iteration in range(100):
        z=np.einsum('ij,j->i',design,w);prob=1/(1+np.exp(-np.clip(z,-700,700)));grad=np.einsum('ij,i->j',design,prob-y)+pen*w
        hess=np.einsum('ni,n,nj->ij',design,prob*(1-prob),design)+np.diag(pen+1e-10)
        delta=np.linalg.solve(hess,grad);step=1.;old=objective(w)
        for _ in range(30):
            if objective(w-step*delta)<=old:break
            step*=.5
        w-=step*delta
        if np.max(np.abs(grad))<1e-7:converged=True;break
    if not converged:raise ArithmeticError('Frozen logistic fit did not converge; refusing unvalidated predictions')
    return dict(columns=cols,mean=mu.tolist(),scale=sd.tolist(),coef=w[:-1].tolist(),intercept=float(w[-1]),iterations=iteration+1,
        objective='Sum binary log loss + 0.5/C times squared coefficients; intercept unpenalized. Damped Newton solver.')


def predict(model,rows):
    if 'constant' in model:return np.full(len(rows),model['constant'])
    x=np.asarray([[r['x'][k] for k in model['columns']] for r in rows]);z=np.einsum('ij,j->i',(x-np.asarray(model['mean']))/np.asarray(model['scale']),np.asarray(model['coef']))+model['intercept']
    return 1/(1+np.exp(-np.clip(z,-700,700)))

def scores(rows,prob):
    y=np.array([r['y'] for r in rows]);prob=np.asarray(prob);two=len(set(y))==2
    if two:
        pos=prob[y==1];neg=prob[y==0];auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
        order=np.argsort(-prob,kind='stable');yy=y[order];pp=prob[order];tp=0;seen=0;ap=0.;i=0
        while i<len(y):
            j=i+1
            while j<len(y) and pp[j]==pp[i]:j+=1
            added=int(yy[i:j].sum());tp+=added;seen+=j-i;ap+=(added/y.sum())*(tp/seen);i=j
    else:auc=None;ap=None
    return dict(n=len(rows),positives=int(y.sum()),prevalence=float(y.mean()),auroc=auc,average_precision=float(ap) if ap is not None else None,brier=float(np.mean((prob-y)**2)))


def freeze(original_db,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);dest=out/'frozen_predictors.json'
    if dest.exists():return json.loads(dest.read_text())
    records=read_records(original_db);result={'protocol':PROTOCOL,'training_db_sha256':hashlib.sha256(Path(original_db).read_bytes()).hexdigest(),'training_records_sha256':hashlib.sha256(json.dumps(records,sort_keys=True,allow_nan=False).encode()).hexdigest(),'models':{},'exploratory_heldout_seed':[]}
    for lead in PROTOCOL['lead_steps']:
        rows=dataset(records,lead)
        if not rows:raise ValueError('No usable original trajectories')
        result['models'][str(lead)]={}
        for name,cols in columns(rows).items():
            result['models'][str(lead)][name]=fit(rows,cols)
            for seed in sorted({r['seed'] for r in rows}):
                train=[r for r in rows if r['seed']!=seed];test=[r for r in rows if r['seed']==seed]
                if not train:continue
                model=fit(train,cols);result['exploratory_heldout_seed'].append(dict(lead=lead,model=name,test_seed=seed,**scores(test,predict(model,test))))
    dest.write_text(json.dumps(result,indent=2,allow_nan=False));return result

def evaluate(db,frozen,out,label):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);records=read_records(db);results=[];predictions=[]
    for lead in PROTOCOL['lead_steps']:
        rows=dataset(records,lead)
        if not rows:continue
        for name,model in frozen['models'][str(lead)].items():
            probs=predict(model,rows)
            for seed in sorted({r['seed'] for r in rows}):
                idx=[i for i,r in enumerate(rows) if r['seed']==seed];results.append(dict(architecture=label,lead=lead,predictor=name,seed=seed,**scores([rows[i] for i in idx],probs[idx])))
            predictions.extend(dict(architecture=label,lead=lead,predictor=name,seed=r['seed'],step=r['step'],feature_step=r['feature_step'],actual_delta=r['delta'],target=r['y'],probability=float(prob)) for r,prob in zip(rows,probs))
    (out/'prediction_metrics.json').write_text(json.dumps(results,indent=2,allow_nan=False));(out/'predictions.json').write_text(json.dumps(predictions,allow_nan=False))
    return results
