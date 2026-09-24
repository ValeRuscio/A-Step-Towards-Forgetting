"""Source-only baseline fits; timestamp-aligned forecasts; raw scores never get Brier."""
import json,math,sqlite3
from pathlib import Path
import numpy as np
from predict_forgetting import fit,predict,scores

def basic(record):
    g=record.get('geometry') or {}; a=record['pre']['A_valid']['mean'];b=record['pre']['B_valid']['mean']
    return dict(log_step=math.log1p(record['step']),A_loss=a,B_loss=b,projection=math.asinh(g.get('projection',0.)),norm_update=math.log1p(g.get('update_norm',0.)),norm_gA=math.log1p(g.get('gA_norm',0.)),cosine=g.get('update_cosine',0.),population_interference=math.asinh(g.get('population_interference',0.)),training_interference=math.asinh(g.get('training_interference',g.get('population_interference',0.))))

def add_history(x,history):
    for domain in ['A','B']:
        vals=np.array([r['pre'][domain+'_valid']['mean'] for r in history],float)
        dif=np.diff(vals)
        for lag in [1,2,4,8]:
            x[f'{domain}_lag{lag}']=float(vals[-1]-vals[-1-lag]) if len(vals)>lag else 0.
            d=dif[-lag:];x[f'{domain}_mean{lag}']=float(d.mean()) if len(d) else 0.;x[f'{domain}_std{lag}']=float(d.std()) if len(d) else 0.
        x[domain+'_max8']=float(dif[-8:].max()) if len(dif) else 0.
    return x

def features(history):return add_history(basic(history[-1]),history)
def examples(records,lead,threshold):
    records=sorted(records,key=lambda r:r['step']);by={r['step']:r for r in records};out=[]
    for target in records:
        t=target['step']-lead
        if t not in by or t<9:continue
        hist=[r for r in records if r['step']<=t];x=features(hist)
        delta=target['post']['A_test']['mean']-target['pre']['A_test']['mean']
        out.append(dict(x=x,y=int(delta>threshold),delta=delta,step=target['step'],feature_step=t,seed=target.get('seed',0)))
    return out

def columns(row):
    loss=[k for k in row['x'] if k.startswith(('A_','B_')) or k=='log_step']
    return {'loss_history':loss,'norm_only':['norm_update','norm_gA'],'cosine_only':['cosine'],'norm_and_cosine':['norm_update','norm_gA','cosine'],'projection_only':['projection'],'history_plus_projection':loss+['projection'],'old_new_population':['population_interference'],'history_plus_interference':loss+['population_interference']}

def freeze(source,dest,thresholds=(.02,.05,.1),alarm_quantile=.8):
    """source: list of complete original OLMo trajectories in new compact schema."""
    artifact={'source':'Original OLMo seeds 1/2/3; retrospective development source, no target fitting','lead':{},'thresholds':list(thresholds),'alarm_quantile':alarm_quantile,'measurement_shift':'Historical predictors use A_test/B_test gradients; online replication uses A_valid/B_valid. Report this shift explicitly.'}
    for lead in [0,1]:
        artifact['lead'][str(lead)]={}
        for threshold in thresholds:
            rows=sum([examples(records,lead,threshold) for records in source],[])
            if not rows:raise ValueError('Need >=10 consecutive source measurements')
            models={k:fit(rows,v) for k,v in columns(rows[0]).items()}
            cut={k:float(np.quantile(predict(v,rows),alarm_quantile)) for k,v in models.items()}
            artifact['lead'][str(lead)][str(threshold)]={'models':models,'alarm_cutoffs':cut,'projection_raw_cutoff':float(np.quantile([math.sinh(r['x']['projection']) for r in rows],alarm_quantile))}
    Path(dest).write_text(json.dumps(artifact,indent=2,allow_nan=False));return artifact

def score_policy(frozen,history,kind,threshold=.05):
    # At pre-update t, history ends at pre-update t-1: no current-gradient peek.
    if len(history)<9:return False,None
    f=frozen['lead']['1'][str(threshold)];x=features(history)
    if kind=='projection':
        value=math.sinh(x['projection']);return value>f['projection_raw_cutoff'],value
    model=f['models']['loss_history'];value=float(predict(model,[{'x':x}])[0]);return value>f['alarm_cutoffs']['loss_history'],value

def analyze(records,frozen):
    out=[]
    for lead in [0,1]:
        for threshold in frozen['thresholds']:
            rows=examples(records,lead,threshold)
            if not rows:continue
            for name,model in frozen['lead'][str(lead)][str(threshold)]['models'].items():
                out.append(dict(lead=lead,threshold=threshold,predictor=name,**scores(rows,predict(model,rows))))
            for name,key in [('raw_projection','projection'),('raw_population_interference','population_interference'),('raw_training_interference','training_interference'),('raw_cosine','cosine'),('raw_update_norm','norm_update')]:
                if name=='raw_training_interference' and any('training_interference' not in (r.get('geometry') or {}) for r in records):continue
                metric=scores(rows,[r['x'][key] for r in rows]);metric.pop('brier')
                out.append(dict(lead=lead,threshold=threshold,predictor=name,**metric))
    return out

def convert_original(path,immutable=False):
    with sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro'+('&immutable=1' if immutable else ''),uri=True) as con:
        records=[json.loads(x[0]) for x in con.execute('select record from trajectory order by seed,step')]
    result={}
    for r in records:
        g=r['pre']['geometry']['global_geometry'];gram=np.array(g['gram']);dn=math.sqrt(max(gram[0,0]+gram[1,1]+2*gram[0,1],0));an=float(g['norms'][2]);pr=float(gram[2,0]+gram[2,1]);pre={};post={}
        for domain in ['A','B']:
            loss=r['pre']['loss'][domain+'_test']['mean'];delta=r['functional_consequence'][domain+'_test']['actual_loss_change']
            pre[domain+'_valid']={'mean':loss};pre[domain+'_test']={'mean':loss};post[domain+'_test']={'mean':loss+delta}
        rr=dict(seed=r['seed'],step=r['step'],pre=pre,post=post,geometry=dict(projection=pr,update_norm=dn,gA_norm=an,update_cosine=pr/(an*dn) if an*dn else 0.,population_interference=-float(gram[2,3])))
        result.setdefault(r['seed'],[]).append(rr)
    return list(result.values())

def decompose_original(path,immutable=False):
    """Use recorded -log(answer mass), never infer missing pre-update support."""
    with sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro'+('&immutable=1' if immutable else ''),uri=True) as con:
        records=[json.loads(x[0]) for x in con.execute('select record from trajectory order by seed,step')]
    by={(r['seed'],r['step']):r for r in records};out=[]
    for r in records:
        previous=by.get((r['seed'],r['step']-1))
        for split,table in r.get('post_losses',{}).items():
            old={x['entity']:x for x in previous.get('post_losses',{}).get(split,{}).get('rows',[])} if previous else {}
            for x in table.get('rows',[]):
                if 'support' not in x:continue
                before=old.get(x['entity']);leak=float(x['support']);row=dict(seed=r['seed'],step=r['step'],split=split,entity=x['entity'],loss=x['loss'],leakage=leak,confusion=x['loss']-leak)
                if before and 'support' in before:
                    row.update(loss_change=x['loss']-before['loss'],leakage_change=leak-before['support'],confusion_change=(x['loss']-leak)-(before['loss']-before['support']))
                else:row['change_unavailable']='No immediately preceding saved answer-mass measurement'
                out.append(row)
    return out
