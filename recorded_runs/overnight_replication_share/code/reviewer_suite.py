"""Resumable reviewer experiments. Run through Run_Reviewer_Experiments.ipynb."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLCONFIGDIR','/tmp/reviewer_suite_mpl')
import argparse,copy,fcntl,hashlib,json,math,random,shutil,sqlite3,subprocess,sys,time,traceback,uuid,zipfile,platform,importlib.metadata
from pathlib import Path
import numpy as np
import torch
from association_core import Config,minibatches,digest
from study_math import StudyEngine,weights,assign,difference,plus,dot,norm,grad,evaluate,channels,geometry,reset,fit_frame,activation_probe,directional_hessian
from study_tasks import dataset,pin_assets
from study_prediction import score_policy,analyze
VERSION='reviewer-suite-1'
HERE=Path(__file__).resolve().parent

def read(p,default=None):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default

def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,p)

def defaults(output='runs/reviewer_suite_v1'):
    return dict(version=VERSION,output=str(Path(output).resolve()),python=sys.executable,device='cuda:0',threads=8,
        models=[{'name':'olmo1b','repo':'allenai/OLMo-1B-hf','revision':'aee7752d9c08ee4775e9b0091426d8410e8f6a89'}, {'name':'pythia410m','repo':'EleutherAI/pythia-410m','revision':'main'}],
        seeds=[11,12,13],tasks=['registry','text'],lr=2e-5,beta1=.9,beta2=.99,clip=1.,a_steps=600,b_steps=128,microbatch=2,accumulation=2,eval_batch=4,entities=32,max_length=256,
        text_train=1024,text_valid=32,text_test=128,checkpoint_every=8,hours=11.5,minimum_free_gib=40.,max_output_gib=180.,
        experiments=['prediction','optimizer','matched','frame'],lr_factors=[.25,1.,2.],optimizer_modes=['preserve','reset_m','reset_both'],matched_modes=['preserve','reset_m','reset_v','reset_both','fresh'],matched_steps=16,
        alarm_threshold=.05,alarm_lr_factor=.25,alarm_budget=16,frame_layers=[0,1],frame_fixed_steps=[20],frame_event_threshold=.05,frame_event_limit=2,frame_controls=4,frame_calibration=16,frame_test=16,
        path_points=17,path_max_points=65,path_fd_eps=[.01,.005,.0025],path_rtol=.05,path_atol=2e-4,topology=True,smoke=False)

def smoke_settings(output):
    s=defaults(output);s.update(models=[dict(name='tiny',repo='tiny-gpt_neox',revision='smoke')],device='cpu',threads=1,seeds=[11],tasks=['registry'],smoke=True,a_steps=2,b_steps=10,entities=4,microbatch=2,accumulation=1,eval_batch=2,checkpoint_every=2,lr=.001,hours=.2,minimum_free_gib=0,max_output_gib=1,frame_layers=[0],frame_fixed_steps=[2],frame_event_limit=0,frame_calibration=2,frame_test=2,frame_controls=1,path_points=5,path_max_points=9,lr_factors=[.5,1.],matched_steps=2,alarm_budget=2,topology=True)
    return s

def science(s):return {k:v for k,v in s.items() if k not in ['hours','minimum_free_gib','max_output_gib','python','output','device','threads','checkpoint_every']}
def signature(s):return hashlib.sha256(json.dumps(science(s),sort_keys=True).encode()).hexdigest()

def prepare(s):
    s=copy.deepcopy(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    old=read(out/'settings.json')
    if old:
        # main was resolved once. Never silently move the checkpoint on resume.
        for m in s['models']:
            previous=next((x for x in old['models'] if x['name']==m['name'] and x['repo']==m['repo']),None)
            if previous and m['revision']=='main':m['revision']=previous['revision']
        if 'datasets' in old and 'datasets' not in s:s['datasets']=old['datasets']
        if signature(old)!=signature(s):raise ValueError('Scientific settings changed. Use restart(SETTINGS) to create a separate run; existing results are preserved.')
    elif not s['smoke']:s=pin_assets(s)
    if s['path_points']<3 or s['path_points']%2!=1 or s['path_max_points']<2*s['path_points']-1:raise ValueError('path_points must be odd and >=3')
    if s['alarm_threshold'] not in [.02,.05,.1]:raise ValueError('Use an existing frozen event threshold')
    if min(s['a_steps'],s['b_steps'],s['microbatch'],s['accumulation'],s['eval_batch'],s['checkpoint_every'])<1:raise ValueError('Positive counts required')
    if any(x<1 or x>s['b_steps'] for x in s['frame_fixed_steps']):raise ValueError('Frame steps outside trajectory')
    if s['frame_calibration']+s['frame_test']>s['entities'] and 'registry' in s['tasks']:raise ValueError('Frame entity partitions overlap')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    previous=read(out/'code_hashes.json')
    if previous and previous!=hashes:raise ValueError('Code changed during run; start a new directory to keep provenance unambiguous')
    write(out/'code_hashes.json',hashes);write(out/'settings.json',s)
    frozen=HERE/'frozen_baselines.json'
    if not frozen.exists():raise FileNotFoundError('Missing bundled frozen_baselines.json')
    if not (out/'frozen_baselines.json').exists():shutil.copy2(frozen,out/'frozen_baselines.json')
    elif (out/'frozen_baselines.json').read_bytes()!=frozen.read_bytes():raise ValueError('Frozen predictor changed')
    return s

def alive(out):
    out=Path(out)
    if not out.exists():return False
    with open(out/'worker.lock','a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return False
        except BlockingIOError:return True

def status(settings):
    out=Path(settings['output'] if isinstance(settings,dict) else settings);v=read(out/'status.json',{'status':'not_started'});v['alive']=alive(out)
    if v.get('last_progress'):v['seconds_since_progress']=round(time.time()-v['last_progress'],1)
    if not v['alive'] and v.get('status') in ['running','launching'] and v.get('seconds_since_progress',0)>30:
        v['recorded_status']=v['status'];v['status']='not_running';v['hint']='No worker owns the lock. Inspect worker.log; launch again to resume.'
    if out.exists():v['free_gib']=round(shutil.disk_usage(out).free/2**30,1)
    return v

def launch(s):
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'launch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if alive(out):return status(s)
        old=read(out/'status.json',{})
        if old.get('status')=='launching' and time.time()-old.get('last_progress',0)<30:return status(s)
        s=prepare(s);(out/'STOP').unlink(missing_ok=True)
        if old.get('status')=='complete':return status(s)
        write(out/'status.json',dict(status='launching',last_progress=time.time()))
        with open(out/'worker.log','a') as log:
            p=subprocess.Popen([s['python'],str(HERE/'reviewer_suite.py'),'--worker',str(out/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=HERE)
        return dict(status='launching',pid=p.pid,output=str(out))

def stop(s):
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True);(out/'STOP').touch();return 'Stop requested. The last atomic checkpoint is retained; unfinished units replay on resume.'

def restart(s):
    s=copy.deepcopy(s);original=s['output'];s['output']=s['output']+'_restart_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]
    if alive(original):raise RuntimeError('Stop the existing run and wait for alive=False before restarting')
    launch(s);return s

class Paused(Exception):pass
class Control:
    def __init__(self,s):self.s=s;self.out=Path(s['output']);self.deadline=time.time()+s['hours']*3600;self.fields={};self.last_storage=0
    def pulse(self,**kw):
        if kw.get('phase')!=self.fields.get('phase'):
            for key in ['alpha','layer','condition','corner','split','epsilon','target_step']:
                if key not in kw:self.fields.pop(key,None)
        if 'job' in kw and kw['job']!=self.fields.get('job'):
            self.fields.pop('step',None);self.fields.pop('actions',None)
        self.fields.update(kw);write(self.out/'status.json',dict(status='running',last_progress=time.time(),**self.fields))
    def check(self):
        if (self.out/'STOP').exists():raise Paused('Stop requested')
        if time.time()>self.deadline:raise Paused('Session budget reached; launch again to resume')
        if shutil.disk_usage(self.out).free<self.s['minimum_free_gib']*2**30:raise Paused('Disk reserve reached')
    def checkpoint(self,state,path):
        self.check();size=sum(v.numel()*v.element_size() for v in state['engine']['model'].values())*3
        if shutil.disk_usage(self.out).free < self.s['minimum_free_gib']*2**30+size*1.15:raise Paused('Insufficient space for atomic checkpoint')
        if time.time()-self.last_storage>30:
            self.last_storage=time.time();used=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
            if used+size>self.s['max_output_gib']*2**30:raise Paused('Output storage cap reached; increase max_output_gib to resume')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)

def db_open(out):
    con=sqlite3.connect(Path(out)/'results.sqlite',timeout=60);con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS records (job TEXT, step INTEGER, kind TEXT, record TEXT, PRIMARY KEY(job,step,kind))');con.commit();return con

def put(con,job,step,kind,value):
    con.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(job,step,kind,json.dumps(value,allow_nan=False)));con.commit()
def get(con,job,kind):return [json.loads(x[0]) for x in con.execute('SELECT record FROM records WHERE job=? AND kind=? ORDER BY step',(job,kind))]

def conditions(s):
    result=[dict(name='natural',mode='preserve',lr=1.,steps=s['b_steps'])]
    if 'prediction' in s['experiments']:
        result += [dict(name=n,mode='preserve',lr=1.,steps=s['b_steps'],policy=n) for n in ['projection','loss_history','random_projection','random_loss_history']]
    if 'optimizer' in s['experiments']:
        for mode in s['optimizer_modes']:
            for rate in s['lr_factors']:
                if mode=='preserve' and rate==1:continue
                result.append(dict(name=f'{mode}_lr{rate}',mode=mode,lr=rate,steps=s['b_steps']))
    if 'matched' in s['experiments']:
        result += [dict(name='matched_'+mode,mode=mode,lr=1.,steps=min(s['matched_steps'],s['b_steps']),matched=True) for mode in s['matched_modes']]
    return result

def all_eval(e,data,s,ctl):return {split:evaluate(e,data[split],s['eval_batch'],ctl) for split in ['A_valid','B_valid','A_test','B_test']}

def path_audit(e,before,after,h,c,data,s,ctl):
    delta=difference(after,before);result={};grid=np.linspace(0,1,s['path_points'])
    try:
        for split in ['A_test','B_test']:
            points=[]
            grid=np.linspace(0,1,s['path_points'])
            for alpha in grid:
                ctl.check();assign(e,{n:v+float(alpha)*delta[n] for n,v in before.items()});g=grad(e,data[split],s['eval_batch'],ctl)
                points.append(dict(alpha=float(alpha),loss=evaluate(e,data[split],s['eval_batch'],ctl)['mean'],slope=dot(g,delta),history=dot(g,h),current=dot(g,c)))
                ctl.pulse(phase='finite path audit',split=split,alpha=float(alpha))
            def simpson(key):
                y=np.array([p[key] for p in points]);return float((y[0]+y[-1]+4*y[1:-1:2].sum()+2*y[2:-1:2].sum())/(3*(len(y)-1)))
            observed=points[-1]['loss']-points[0]['loss'];tol=s['path_atol']+s['path_rtol']*abs(observed);local_error=None
            while 2*len(points)-1<=s['path_max_points']:
                cache={p['alpha']:p for p in points};grid=np.linspace(0,1,2*len(points)-1)
                for alpha in grid:
                    if float(alpha) in cache:continue
                    ctl.check();assign(e,{n:v+float(alpha)*delta[n] for n,v in before.items()});g=grad(e,data[split],s['eval_batch'],ctl)
                    cache[float(alpha)]=dict(alpha=float(alpha),loss=evaluate(e,data[split],s['eval_batch'],ctl)['mean'],slope=dot(g,delta),history=dot(g,h),current=dot(g,c))
                    ctl.pulse(phase='refining finite path',split=split,alpha=float(alpha))
                points=[cache[float(a)] for a in grid];y=np.array([p['slope'] for p in points]);dx=1/(len(points)-1);local_error=0.
                for j in range(0,len(y)-1,4):
                    coarse=2*dx/3*(y[j]+4*y[j+2]+y[j+4]);fine=dx/3*(y[j]+4*y[j+1]+2*y[j+2]+4*y[j+3]+y[j+4]);local_error+=abs(fine-coarse)/15
                if abs(simpson('slope')-observed)<=tol and local_error<=tol:break
            local_error=float(local_error) if local_error is not None else None
            integral=simpson('slope')
            # Curvature at maximum adjacent slope change; central difference slopes at two spacings.
            j=int(np.argmax(np.abs(np.diff([p['slope'] for p in points]))));alpha=float((grid[j]+grid[j+1])/2)
            curvature=[]
            for eps in s['path_fd_eps']:
                at=[];slopes=[]
                for a in [alpha-eps,alpha,alpha+eps]:
                    assign(e,{n:v+a*delta[n] for n,v in before.items()});at.append(evaluate(e,data[split],s['eval_batch'],ctl)['mean']);slopes.append(dot(grad(e,data[split],s['eval_batch'],ctl),delta))
                fd=(at[2]-at[0])/(2*eps);curvature.append(dict(epsilon=eps,autograd_slope=slopes[1],fd_slope=fd,slope_pass=abs(fd-slopes[1])<=s['path_atol']+s['path_rtol']*max(abs(fd),abs(slopes[1])),directional_curvature=(slopes[2]-slopes[0])/(2*eps),loss_second_difference=(at[2]-2*at[1]+at[0])/eps**2))
            stable=abs(curvature[-1]['directional_curvature']-curvature[-2]['directional_curvature'])<=s['path_atol']+s['path_rtol']*max(abs(curvature[-1]['directional_curvature']),abs(curvature[-2]['directional_curvature'])) if len(curvature)>1 else False
            assign(e,{n:v+alpha*delta[n] for n,v in before.items()})
            try:
                hv=directional_hessian(e,data[split],delta,s['eval_batch'],ctl)
                hv_audit=dict(value=hv,passed=stable and abs(hv-curvature[-1]['directional_curvature'])<=s['path_atol']+s['path_rtol']*max(abs(hv),abs(curvature[-1]['directional_curvature'])))
            except (RuntimeError,NotImplementedError) as exc:hv_audit=dict(value=None,passed=False,error=str(exc))
            result[split]=dict(hessian_audit=hv_audit,points=points,observed=observed,integral=integral,closure=integral-observed,closure_pass=abs(integral-observed)<=tol,quadrature_error_estimate=local_error,quadrature_pass=local_error is not None and local_error<=tol,resolved=abs(integral-observed)<=tol and local_error is not None and local_error<=tol and all(x['slope_pass'] for x in curvature),history_integral=simpson('history'),current_integral=simpson('current'),curvature_alpha=alpha,curvature=curvature,curvature_stable=stable,note='Exact directional Hessian checked against finite differences; bounded nested quadrature remains unresolved when closure fails. No unresolved path is reported as validated.')
    finally:assign(e,after)
    return result

def frames(e,before,after,h,c,data,s,ctl,job,step,con):
    residual=difference(difference(after,before),plus(h,c));allids=sorted(set(r['entity'] for r in data['A_valid']));rng=random.Random(8041);rng.shuffle(allids)
    if s.get('_task')=='registry':
        cal=set(allids[:s['frame_calibration']]);test=set(allids[s['frame_calibration']:s['frame_calibration']+s['frame_test']]);calrows=[r for sp in ['A_valid','B_valid'] for r in data[sp] if r['entity'] in cal]
        held={sp:[r for r in data[sp] if r['entity'] in test] for sp in ['A_test','B_test']}
    else:
        calrows=sum([data[sp][:s['frame_calibration']] for sp in ['A_valid','B_valid']],[]);held={sp:data[sp][:s['frame_test']] for sp in ['A_test','B_test']}
    if any(not v for v in held.values()):raise ValueError('No held-out frame entities')
    try:
        for layer in s['frame_layers']:
            if layer>=e.model.config.num_hidden_layers:continue
            ctl.pulse(phase='fit validation reference frame',layer=layer)
            assign(e,before);_,y=activation_probe(e,calrows,layer,ctl=ctl)
            assign(e,after);_,x=activation_probe(e,calrows,layer,ctl=ctl);frame=fit_frame(x,y)
            controls=['unpatched','sham','rotation','inverse','mean','scale']
            for i in range(s['frame_controls']):
                a=np.random.default_rng(6001+i).normal(size=frame['basis'].shape);frame['orthogonal_'+str(i)]=np.linalg.qr(a,mode='reduced')[0];controls+=['orthogonal_'+str(i),'additive_'+str(i)]
            assign(e,before);references={sp:activation_probe(e,prompts,layer,ctl=ctl)[1] for sp,prompts in held.items()}
            # Fixed fit and control maps across all four channel corners.
            for condition in controls:
                if con.execute('SELECT 1 FROM records WHERE job=? AND step=? AND kind=?',(job,step,f'frame_layer{layer}_{condition}')).fetchone():continue
                rows={}
                for u,v in [(0,0),(1,0),(0,1),(1,1)]:
                    assign(e,{n:before[n]+u*h[n]+v*c[n]+u*v*residual[n] for n in before})
                    for split,prompts in held.items():
                        ctl.pulse(phase='held-out frame factorial',layer=layer,condition=condition,corner=f'{u}{v}',split=split)
                        metrics,acts=activation_probe(e,prompts,layer,frame,condition,6001+int(condition.rsplit('_',1)[-1]) if '_' in condition and condition.rsplit('_',1)[-1].isdigit() else 6001,ctl)
                        metrics['mse_to_pre']=float(np.mean((acts-references[split])**2))
                        metrics['activation_centered_singular_values']=np.linalg.svd(acts-acts.mean(0),compute_uv=False).tolist()
                        if s['topology'] and condition in ['unpatched','rotation']:
                            from geometry_math import persistent_homology
                            d=np.linalg.norm(acts[:,None,:]-acts[None,:,:],axis=-1);ref=references[split];dref=np.linalg.norm(ref[:,None,:]-ref[None,:,:],axis=-1);scale=float(np.median(dref[np.triu_indices(len(dref),1)]));metrics['ph_scale']=scale;metrics['ph']=persistent_homology(d/max(scale,1e-12),np.linspace(0,3,31).tolist())
                        rows[split+f'_{u}{v}']=metrics
                put(con,job,step,f'frame_layer{layer}_{condition}',dict(layer=layer,condition=condition,rank=frame['basis'].shape[1],calibration_mse_before=float(np.mean((x-y)**2)),calibration_mse_after=float(np.mean((x+(x@frame['basis'])@(frame['rot']-np.eye(frame['rot'].shape[0]))@frame['basis'].T-y)**2)),corners=rows))
    finally:assign(e,after)

def trajectory(e,data,s,seed,condition,anchor,root,con,ctl,frozen,case):
    job=case+'/'+condition['name'];done=get(con,job,'done')
    if done:return
    checkpoint=root/'rolling.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint.exists() else None
    if saved and saved.get('job')==job:
        e.restore(saved['engine']);start=saved['step']+1;events=saved.get('frame_events',0)
        con.execute("DELETE FROM records WHERE job=? AND step>? AND kind IN ('step','done')",(job,saved['step']));con.commit()
    else:
        e.restore(anchor);reset(e,condition['mode']);start=1;events=0
        con.execute('DELETE FROM records WHERE job=?',(job,));con.commit()
        ctl.checkpoint(dict(engine=e.pack(),job=job,step=0,frame_events=0),checkpoint)
    history=get(con,job,'step');random_steps=set()
    policy=condition.get('policy','')
    if policy.startswith('random_'):
        source=condition.get('random_source',policy[len('random_'):]);reference=get(con,case+'/'+source,'step')
        if len(reference)!=s['b_steps']:raise RuntimeError('Matched random schedule requires completed alarm-policy trajectory')
        count=sum(bool(r['action']) for r in reference);rng=random.Random(condition.get('schedule_seed',seed*1009+(1 if source=='projection' else 2)));random_steps=set(rng.sample(list(range(10,s['b_steps']+1)),count))
        put(con,job,0,'random_schedule',dict(steps=sorted(random_steps),matched_to=condition.get('matched_policies',[source]),count=count,schedule_seed=condition.get('schedule_seed',seed*1009+(1 if source=='projection' else 2)),independent_of_test_losses=True))
    actions=sum(bool(r['action']) for r in history)
    for step in range(start,condition['steps']+1):
        ctl.check();ctl.pulse(job=job,step=step,phase='pre-update evaluation')
        pre=all_eval(e,data,s,ctl);action=False;alarm_score=None
        if policy in ['projection','loss_history']:
            action,alarm_score=score_policy(frozen,history,policy,s['alarm_threshold']);action=bool(action and actions<s['alarm_budget'])
        elif policy.startswith('random_'):action=step in random_steps
        rate=s['lr']*condition['lr']*(s['alarm_lr_factor'] if action else 1.)
        for pg in e.opt.param_groups:pg['lr']=rate
        before=weights(e);ctl.pulse(phase='exact validation gradients')
        measure=condition['name']=='natural' or policy=='projection' or (condition.get('matched') and step==1)
        ga=grad(e,data['A_valid'],s['eval_batch'],ctl) if measure else None
        gb=grad(e,data['B_valid'],s['eval_batch'],ctl) if condition['name']=='natural' else None
        batches=minibatches(data,'B',seed,step,e.c);train=e.gradient(batches)
        train_g={n:p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu') for n,p in e.params.items()} if condition['name']=='natural' else {}
        h,c=channels(e);e.opt.step();after=weights(e);delta=difference(after,before);match=None
        if condition.get('matched') and step==1:
            branch_state=e.pack();e.restore(anchor);e.gradient(batches);base_before=weights(e);e.opt.step();base_delta=difference(weights(e),base_before);target=norm(base_delta);actual=norm(delta)
            if actual<=1e-30:raise ArithmeticError('Cannot norm-match zero displacement')
            factor=target/actual;e.restore(branch_state)
            with torch.no_grad():
                for n,p in e.params.items():p.copy_((before[n]+factor*delta[n]).to(p.device))
            h={n:factor*v for n,v in h.items()};c={n:factor*v for n,v in c.items()};after=weights(e);delta=difference(after,before)
            match=dict(target_norm=target,actual_norm=norm(delta),factor=factor,relative_error=abs(norm(delta)-target)/max(target,1e-30),state_policy='Reset/preserved moments advance normally; only the first displacement is scaled. Clock preserved except fresh.')
            if match['relative_error']>1e-3:raise ArithmeticError('Actual FP32 norm-matching audit failed')
            del branch_state,base_before,base_delta
        if condition['name']=='natural':geom=geometry(ga,gb,h,c,delta,train_g)
        elif measure:
            projection=dot(ga,delta);dn=norm(delta);an=norm(ga)
            geom=dict(projection=projection,update_norm=dn,gA_norm=an,update_cosine=projection/max(an*dn,1e-30),history_projection=dot(ga,h),current_projection=dot(ga,c),measurement='A validation projection only')
        else:geom=None
        del ga,gb,train_g
        post=all_eval(e,data,s,ctl);actions+=int(action)
        record=dict(seed=seed,step=step,pre=pre,post=post,geometry=geom,action=action,alarm_score=alarm_score,lr=rate,training=dict(raw_norm=train['raw_norm'],clip=train['clip']),matched=match,initial_A_projection=geom['projection'] if geom else None,A_remainder=post['A_valid']['mean']-pre['A_valid']['mean']-geom['projection'] if geom else None)
        put(con,job,step,'step',record);history.append(record)
        eligible=condition['name']=='natural' and 'frame' in s['experiments']
        ids=sorted({row['entity'] for row in data['A_valid']})
        if s.get('_task')=='registry':random.Random(8041).shuffle(ids)
        else:ids=[row['entity'] for row in data['A_valid']]
        selection_ids=set(ids[:s['frame_calibration']])
        selection_delta=float(np.mean([row['loss'] for row in post['A_valid']['rows'] if row['entity'] in selection_ids])-np.mean([row['loss'] for row in pre['A_valid']['rows'] if row['entity'] in selection_ids]))
        event=selection_delta>s['frame_event_threshold']
        selected=eligible and (step in s['frame_fixed_steps'] or (event and events<s['frame_event_limit']))
        if selected:
            if event and step not in s['frame_fixed_steps']:events+=1
            put(con,job,step,'selection',dict(rule='fixed_step' if step in s['frame_fixed_steps'] else 'first_validation_loss_events',validation_delta=selection_delta,selection_entities=sorted(selection_ids),test_not_used=True))
            if not con.execute('SELECT 1 FROM records WHERE job=? AND step=? AND kind=?',(job,step,'path')).fetchone():put(con,job,step,'path',path_audit(e,before,after,h,c,data,s,ctl))
            frames(e,before,after,h,c,data,s,ctl,job,step,con)
        if step%s['checkpoint_every']==0 or step==condition['steps']:
            ctl.pulse(phase='atomic continuation checkpoint');ctl.checkpoint(dict(engine=e.pack(),job=job,step=step,frame_events=events),checkpoint)
        ctl.pulse(phase='update complete',actions=actions)
        del before,after,delta,h,c
    put(con,job,condition['steps'],'done',dict(complete=True,updates=condition['steps'],actions=actions))
    # Snapshot is no longer needed: next condition starts from the shared A anchor.
    checkpoint.unlink(missing_ok=True)

def run(s):
    out=Path(s['output']);ctl=Control(s);con=db_open(out);frozen=read(out/'frozen_baselines.json')
    versions={}
    for pkg in ['torch','transformers','numpy','datasets','huggingface-hub']:
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:versions[pkg]=None
    environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),architecture=platform.machine(),packages=versions,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    previous=read(out/'environment.json')
    if previous and previous!=environment:
        write(out/'status.json',dict(status='failed',message='Environment changed since this run started; use a new directory.',last_progress=time.time()));con.close();raise RuntimeError('Environment changed since this run started')
    write(out/'environment.json',environment)
    jobs=conditions(s);total=len(s['models'])*len(s['seeds'])*len(s['tasks'])*len(jobs);ctl.fields['jobs_total']=total
    try:
        for item in s['models']:
            for task in s['tasks']:
                for seed in s['seeds']:
                    case=f"{item['name']}/{task}/seed{seed}";root=out/'checkpoints'/item['name']/task/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
                    if all(get(con,case+'/'+j['name'],'done') for j in jobs):continue
                    ctl.check();ctl.pulse(job=case,phase='loading pinned model')
                    cfg=Config(model=item['repo'],revision=item['revision'],device=s['device'],threads=s['threads'],smoke=s['smoke'],lr=s['lr'],beta1=s['beta1'],beta2=s['beta2'],clip=s['clip'],a_steps=s['a_steps'],b_steps=s['b_steps'],microbatch=s['microbatch'],accumulation=s['accumulation'],entities=s['entities'],max_length=s['max_length'])
                    e=StudyEngine(cfg);local=dict(s,_task=task)
                    datapath=out/'data'/item['name']/task/f'seed{seed}.json';data=read(datapath)
                    if data is None:
                        ctl.pulse(phase='preparing disjoint task data');data=dataset(e,s,seed,task);write(datapath,data)
                    put(con,case,0,'data',dict(sha256=digest(data),splits={k:len(v) for k,v in data.items()},precision='FP32 parameters/forward, FP64 loss/reductions',optimizer='torch.optim.Adam; no decay; moments retained from task A'))
                    anchorpath=root/'anchor.pt'
                    if anchorpath.exists():anchor=torch.load(anchorpath,map_location='cpu',weights_only=True)['engine']
                    else:
                        acquisition=root/'acquisition.pt';saved=torch.load(acquisition,map_location='cpu',weights_only=True) if acquisition.exists() else None
                        start=1
                        if saved:e.restore(saved['engine']);start=saved['step']+1
                        elif not get(con,case,'initial'):put(con,case,0,'initial',all_eval(e,data,s,ctl))
                        for t in range(start,s['a_steps']+1):
                            ctl.check();e.gradient(minibatches(data,'A',seed,t,cfg));e.opt.step();ctl.pulse(phase='task A acquisition',step=t)
                            if t%s['checkpoint_every']==0 or t==s['a_steps']:ctl.checkpoint(dict(engine=e.pack(),step=t),acquisition)
                        anchor=e.pack();ctl.checkpoint(dict(engine=anchor),anchorpath);acquisition.unlink(missing_ok=True)
                        put(con,case,0,'anchor',all_eval(e,data,s,ctl))
                    if not get(con,case,'anchor'):
                        e.restore(anchor);put(con,case,0,'anchor',all_eval(e,data,s,ctl))
                    for condition in jobs:
                        trajectory(e,data,local,seed,condition,anchor,root,con,ctl,frozen,case)
                        completed=con.execute("SELECT count(*) FROM records WHERE kind='done'").fetchone()[0];ctl.pulse(completed_jobs=completed,phase='condition complete')
                        report(s)
                    # Completed case can be reproduced from pinned model, saved data and batch seeds.
                    # One anchor per case is kept for follow-up analyses; no checkpoint-per-update storage.
                    del e,anchor
                    if torch.cuda.is_available():torch.cuda.empty_cache()
        write(out/'status.json',dict(status='complete',last_progress=time.time(),completed_jobs=total,jobs_total=total))
    except Paused as exc:write(out/'status.json',dict(status='paused',message=str(exc),last_progress=time.time(),**ctl.fields))
    except Exception:
        write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields));raise
    finally:
        con.close();report(s)

def report(s):
    out=Path(s['output']);db=out/'results.sqlite'
    if not db.exists():return []
    con=db_open(out);con.execute('BEGIN');frozen=read(out/'frozen_baselines.json');summaries=[];prediction=[];frame_summary=[];paired=[]
    jobs=[x[0] for x in con.execute("SELECT DISTINCT job FROM records WHERE kind='step'")]
    for job in jobs:
        records=get(con,job,'step');finished=bool(get(con,job,'done'));first=records[0];last=records[-1]
        result=dict(job=job,complete=finished,updates=len(records),actions=sum(r['action'] for r in records))
        case=job.rsplit('/',1)[0];initial=get(con,case,'initial');anchor=get(con,case,'anchor')
        if initial and anchor:
            result['A_acquisition']=dict(loss_before=initial[0]['A_test']['mean'],loss_after=anchor[0]['A_test']['mean'],accuracy_before=initial[0]['A_test']['restricted_accuracy'],accuracy_after=anchor[0]['A_test']['restricted_accuracy'])
        for domain in ['A','B']:
            split=domain+'_test';values=[r['post'][split]['mean'] for r in records];base=first['pre'][split]['mean']
            result[domain]=dict(initial=base,final=values[-1],mean=float(np.mean(values)),peak=max(values),mean_positive_excess=float(np.maximum(np.array(values)-base,0).mean()),final_confusion=last['post'][split]['confusion'],final_leakage=last['post'][split]['leakage'],accuracy=last['post'][split]['restricted_accuracy'],full_accuracy=last['post'][split]['full_accuracy'])
        summaries.append(result)
        if job.endswith('/natural'):prediction.extend(dict(job=job,**r) for r in analyze(records,frozen))
    for job in jobs:
        case=job.rsplit('/',1)[0];baseline=get(con,case+'/natural','step')
        if not baseline or not get(con,case+'/natural','done'):continue
        target=baseline[-1]['post']['B_valid']['mean'];records=get(con,job,'step')
        crossing=next((r for r in records if r['post']['B_valid']['mean']<=target),None)
        paired.append(dict(job=job,B_validation_target=target,first_reach_step=crossing['step'] if crossing else None,A_test_at_reach=crossing['post']['A_test']['mean'] if crossing else None,note='First saved endpoint reaching natural final B-validation loss; no interpolation or extrapolation. Missing means target not reached.'))
    groups={}
    for job,step,raw in con.execute("SELECT job,step,record FROM records WHERE kind LIKE 'frame_layer%'"):
        r=json.loads(raw);groups.setdefault((job,step,r['layer']),{})[r['condition']]=r
    for (job,step,layer),conditions_ in groups.items():
        if 'unpatched' not in conditions_:continue
        for condition,r in conditions_.items():
            for sp in ['A_test','B_test']:
                def means(rr):return {k:rr['corners'][sp+'_'+k]['mean'] for k in ['00','10','01','11']}
                a=means(r);b=means(conditions_['unpatched']);interaction=lambda z:z['11']-z['10']-z['01']+z['00']
                endpoint=a['11']-b['11'];base=a['00']-b['00'];components={}
                for component in ['confusion','leakage','restricted_accuracy','full_accuracy']:
                    aa={k:float(np.mean([x[component] for x in r['corners'][sp+'_'+k]['rows']])) for k in ['00','10','01','11']}
                    bb={k:float(np.mean([x[component] for x in conditions_['unpatched']['corners'][sp+'_'+k]['rows']])) for k in ['00','10','01','11']}
                    components[component]=dict(endpoint_change=aa['11']-bb['11'],baseline_change=aa['00']-bb['00'],interaction_change=interaction(aa)-interaction(bb))
                frame_summary.append(dict(job=job,step=step,layer=layer,condition=condition,split=sp,components=components,endpoint_change=endpoint,baseline_change=base,forgetting_change=endpoint-base,interaction_change=interaction(a)-interaction(b),unpatched_event=b['11']-b['00'],patched_event=a['11']-a['00']))
    con.close();write(out/'summary.json',summaries);write(out/'prediction_metrics.json',prediction);write(out/'frame_summary.json',frame_summary);write(out/'acquisition_matched_retention.json',paired)
    lines=['REVIEWER EXPERIMENT SUITE — test outcomes never trigger interventions',f'{sum(r["complete"] for r in summaries)} completed conditions; {len(summaries)} started']
    for r in summaries:lines.append(f"{r['job']}: {r['updates']} updates; A {r['A']['initial']:.5f} -> {r['A']['final']:.5f}; B {r['B']['initial']:.5f} -> {r['B']['final']:.5f}; actions={r['actions']}; complete={r['complete']}")
    (out/'REPORT.txt').write_text('\n'.join(lines));return summaries

def show_results(s):
    report(s);out=Path(s['output']);print((out/'REPORT.txt').read_text() if (out/'REPORT.txt').exists() else 'No completed updates yet.')
    import matplotlib.pyplot as plt
    con=db_open(out);jobs=[x[0] for x in con.execute("SELECT DISTINCT job FROM records WHERE kind='step'")]
    for case in sorted({j.rsplit('/',1)[0] for j in jobs}):
        fig,ax=plt.subplots(1,2,figsize=(11,3.5))
        for j in jobs:
            if j.rsplit('/',1)[0]!=case:continue
            rows=get(con,j,'step')
            for a,domain in zip(ax,['A','B']):a.plot([r['step'] for r in rows],[r['post'][domain+'_test']['mean'] for r in rows],label=j.rsplit('/',1)[-1]);a.set(xlabel='B update',ylabel=domain+' held-out loss')
        ax[1].legend(fontsize=6);fig.suptitle(case);fig.tight_layout();fig.savefig(out/(case.replace('/','_')+'.png'),dpi=160);plt.show()
    con.close()

def export(s):
    out=Path(s['output']);report(s);dest=out/'reviewer_results_share.zip';snap=out/'share_snapshot.sqlite'
    with sqlite3.connect(out/'results.sqlite') as src,sqlite3.connect(snap) as target:src.backup(target)
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(snap,'results.sqlite')
        for p in out.iterdir():
            if p.suffix in ['.json','.txt','.png']:z.write(p,p.name)
        for p in HERE.glob('*.py'):z.write(p,'code/'+p.name)
    snap.unlink();return str(dest)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--smoke');args=parser.parse_args()
    if args.smoke:s=prepare(smoke_settings(args.smoke))
    elif args.worker:s=read(args.worker)
    else:parser.error('Use notebook or --smoke OUTPUT')
    out=Path(s['output'])
    with open(out/'worker.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:sys.exit(0)
        if s.get('overnight_version'):
            from overnight_engine import run as overnight_run
            overnight_run(s)
        else:run(s)
