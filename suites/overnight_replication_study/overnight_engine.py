"""Priority-ordered replication, random controls, event branches, deferred audits."""
import gc,hashlib,importlib.metadata,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import reviewer_suite as r
from association_core import Config,minibatches,digest
from study_math import StudyEngine,weights,difference,channels
from study_tasks import dataset
from event_controls import select_events,replay,branch,curvature


def core_conditions(s):
    return [dict(name=name,mode=mode,lr=rate,steps=s['b_steps'],**({'policy':policy} if policy else {})) for name,mode,rate,policy in [
        ('natural','preserve',1.,None),('projection','preserve',1.,'projection'),('loss_history','preserve',1.,'loss_history'),
        ('preserve_lr0.25','preserve',.25,None),('reset_m_lr1.0','reset_m',1.,None),('reset_m_lr0.25','reset_m',.25,None)]]


def cases(s):
    return [(item,task,seed,f"{item['name']}/{task}/seed{seed}") for item in s['models'] for task in s['tasks'] for seed in s['seeds']]


def random_conditions(s,con,case,seed):
    budgets={}
    for policy in ['projection','loss_history']:
        if not r.get(con,case+'/'+policy,'done'):raise RuntimeError('Random controls require completed policies')
        rows=r.get(con,case+'/'+policy,'step');count=sum(x['action'] for x in rows);budgets.setdefault(count,[]).append(policy)
    result=[]
    for count,policies in sorted(budgets.items()):
        for rep in range(s['random_replicates']):
            schedule_seed=int(hashlib.sha256(f'overnight-random-v1/{seed}/{count}/{rep}'.encode()).hexdigest()[:12],16)
            result.append(dict(name=f'random_budget{count}_rep{rep+1}',mode='preserve',lr=1.,steps=s['b_steps'],policy='random_'+policies[0],random_source=policies[0],matched_policies=policies,replicate=rep+1,schedule_seed=schedule_seed))
    return result


def load_case(s,item,task,seed,case,con,ctl):
    root=Path(s['output'])/'checkpoints'/item['name']/task/f'seed{seed}';root.mkdir(parents=True,exist_ok=True)
    ctl.pulse(job=case,phase='loading pinned model')
    cfg=Config(model=item['repo'],revision=item['revision'],device=s['device'],threads=s['threads'],smoke=s['smoke'],lr=s['lr'],beta1=s['beta1'],beta2=s['beta2'],clip=s['clip'],a_steps=s['a_steps'],b_steps=s['b_steps'],microbatch=s['microbatch'],accumulation=s['accumulation'],entities=s['entities'],max_length=s['max_length'])
    e=StudyEngine(cfg);datafile=Path(s['output'])/'data'/item['name']/task/f'seed{seed}.json';data=r.read(datafile)
    if data is None:
        ctl.pulse(phase='loading pinned task data');data=dataset(e,s,seed,task);r.write(datafile,data)
    r.put(con,case,0,'data',dict(sha256=digest(data),splits={k:len(v) for k,v in data.items()},precision='FP32 model; FP64 loss/reductions'))
    anchorfile=root/'anchor.pt'
    if anchorfile.exists():anchor=torch.load(anchorfile,map_location='cpu',weights_only=True)['engine']
    else:
        ck=root/'acquisition.pt';saved=torch.load(ck,map_location='cpu',weights_only=True) if ck.exists() else None;t0=1
        if saved:e.restore(saved['engine']);t0=saved['step']+1
        elif not r.get(con,case,'initial'):r.put(con,case,0,'initial',r.all_eval(e,data,s,ctl))
        for t in range(t0,s['a_steps']+1):
            ctl.check();e.gradient(minibatches(data,'A',seed,t,cfg));e.opt.step();ctl.pulse(phase='task A acquisition',step=t)
            if t%s['checkpoint_every']==0 or t==s['a_steps']:ctl.checkpoint(dict(engine=e.pack(),step=t),ck)
        anchor=e.pack();ctl.checkpoint(dict(engine=anchor),anchorfile);ck.unlink(missing_ok=True)
    if not r.get(con,case,'anchor'):
        e.restore(anchor);r.put(con,case,0,'anchor',r.all_eval(e,data,s,ctl))
    return e,data,anchor,root


def validation(s):
    if s.get('overnight_version')!='replication-1':raise ValueError('Use overnight_study.defaults()')
    if s['random_replicates']<1 or s['event_horizon']<1 or s['event_checkpoint_every']<1:raise ValueError('Positive replicate/horizon/checkpoint counts required')
    if not set(s['event_modes'])<= {'preserve','reset_m','reset_m_global','reset_m_layer'}:raise ValueError('Unknown event mode')
    if 'preserve' not in s['event_modes']:raise ValueError('Event study needs a preserved-state control')
    if any(x<1 or x>s['b_steps'] for x in s['event_fixed_steps']):raise ValueError('Fixed event step outside trajectory')
    if not s['smoke'] and (set(s['seeds']) & {1,2,3,11}):raise ValueError('Replication defaults use fresh seeds 12/13; do not relabel the inspected seeds as independent replication')


def run(s):
    out=Path(s['output']);ctl=r.Control(s);con=r.db_open(out);frozen=r.read(out/'frozen_baselines.json')
    try:
        validation(s);versions={}
        for pkg in ['torch','transformers','numpy','datasets','huggingface-hub']:
            try:versions[pkg]=importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:versions[pkg]=None
        env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
        old=r.read(out/'environment.json')
        if old and old!=env:raise RuntimeError('Environment changed; start a new directory')
        r.write(out/'environment.json',env)
        definitions=cases(s);primary=core_conditions(s);ctl.fields.update(core_jobs_total=len(definitions)*len(primary))
        # Phase 1 completes both seeds before spending budget on additional controls.
        for phase in ['core','random','events','historical_curvature','deferred_frames']:
            if phase=='historical_curvature':
                if s.get('curvature_reaudit_source') and not r.get(con,'historical_curvature','reaudit_done'):
                    ctl.pulse(stage=phase,job='seed11_update005_B_test',phase='starting historical curvature audit');reaudit_history(s,con,ctl)
                continue
            if phase=='deferred_frames' and not s['deferred_frames']:continue
            for item,task,seed,case in definitions:
                ctl.check();ctl.pulse(stage=phase,job=case,phase='checking saved units')
                local=dict(s,_task=task,experiments=[])
                jobs=primary if phase=='core' else random_conditions(s,con,case,seed) if phase=='random' else []
                if jobs and all(r.get(con,case+'/'+j['name'],'done') for j in jobs):continue
                natural=r.get(con,case+'/natural','step') if phase in ['events','deferred_frames'] else []
                selections=select_events(natural,local) if natural else []
                if phase=='events' and all(r.get(con,f"{case}/event{x['step']:03d}/{mode}",'event_done') for x in selections for mode in s['event_modes']):continue
                if phase=='deferred_frames' and all(r.get(con,f"{case}/frame{x['step']:03d}",'deferred_done') for x in selections):continue
                e,data,anchor,root=load_case(s,item,task,seed,case,con,ctl)
                if jobs:
                    for condition in jobs:
                        r.trajectory(e,data,local,seed,condition,anchor,root,con,ctl,frozen,case)
                        ctl.pulse(completed_trajectories=con.execute("SELECT count(*) FROM records WHERE kind='done'").fetchone()[0],phase='trajectory complete');report(s)
                else:
                    for selection in selections:
                        step=selection['step'];r.put(con,case,step,'event_selection',selection)
                        if phase=='events' and all(r.get(con,f'{case}/event{step:03d}/{m}','event_done') for m in s['event_modes']):continue
                        if phase=='deferred_frames' and r.get(con,f'{case}/frame{step:03d}','deferred_done'):continue
                        start,replay_audit=replay(e,anchor,data,local,seed,step,natural,ctl);r.put(con,case,step,'replay_audit',replay_audit)
                        if phase=='events':
                            for mode in s['event_modes']:
                                branch(e,data,local,seed,case,selection,mode,start,root,con,ctl)
                                ctl.pulse(completed_event_branches=con.execute("SELECT count(*) FROM records WHERE kind='event_done'").fetchone()[0],phase='event branch complete');report(s)
                        else:
                            before=weights(e);e.gradient(minibatches(data,'B',seed,step,e.c));h,c=channels(e);e.opt.step();after=weights(e)
                            pathjob=case+'/natural'
                            if not con.execute("SELECT 1 FROM records WHERE job=? AND step=? AND kind='path'",(pathjob,step)).fetchone():r.put(con,pathjob,step,'path',r.path_audit(e,before,after,h,c,data,local,ctl))
                            r.frames(e,before,after,h,c,data,local,ctl,pathjob,step,con);r.put(con,f'{case}/frame{step:03d}',step,'deferred_done',dict(complete=True))
                            del before,after,h,c
                        del start
                del e,data,anchor;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
        report(s);r.write(out/'status.json',dict(status='complete',last_progress=time.time(),completed_trajectories=con.execute("SELECT count(*) FROM records WHERE kind='done'").fetchone()[0],completed_event_branches=con.execute("SELECT count(*) FROM records WHERE kind='event_done'").fetchone()[0],message='All configured stages completed. Inspect individual audit results; completion does not imply every audit passed.'))
    except r.Paused as exc:r.write(out/'status.json',dict(status='paused',message=str(exc),last_progress=time.time(),**ctl.fields))
    except Exception:
        r.write(out/'status.json',dict(status='failed',message=traceback.format_exc(),last_progress=time.time(),**ctl.fields));raise
    finally:con.close();report(s)


def reaudit_history(s,con,ctl):
    """Reads the completed seed-11 run; never writes to it or trains its continuation."""
    import sqlite3
    source=Path(s['curvature_reaudit_source']).expanduser().resolve();old=r.read(source/'settings.json')
    if not old:raise FileNotFoundError('Historical curvature source needs the extracted full run with settings, data and anchor, not just the share ZIP')
    if r.read(source/'status.json',{}).get('status')!='complete':raise ValueError('Historical source must be stopped and complete')
    item=old['models'][0];seed=s['curvature_reaudit_seed'];step=s['curvature_reaudit_step'];case=f"{item['name']}/text/seed{seed}"
    for key in ['lr','beta1','beta2','clip','microbatch','accumulation','max_length']:
        if old[key]!=s[key]:raise ValueError('Historical replay settings differ: '+key)
    e=StudyEngine(Config(model=item['repo'],revision=item['revision'],device=s['device'],threads=s['threads'],smoke=old['smoke'],lr=old['lr'],beta1=old['beta1'],beta2=old['beta2'],clip=old['clip'],microbatch=old['microbatch'],accumulation=old['accumulation'],max_length=old['max_length']))
    data=r.read(source/'data'/item['name']/'text'/f'seed{seed}.json');anchor=torch.load(source/'checkpoints'/item['name']/'text'/f'seed{seed}'/'anchor.pt',map_location='cpu',weights_only=True)['engine']
    with sqlite3.connect('file:'+str(source/'results.sqlite')+'?mode=ro',uri=True) as previous:
        records=r.get(previous,case+'/natural','step')
        manifest=r.get(previous,case,'data')
        if not manifest or manifest[0]['sha256']!=digest(data):raise ValueError('Historical data digest mismatch')
    start,audit=replay(e,anchor,data,s,seed,step,records,ctl);before=weights(e);e.gradient(minibatches(data,'B',seed,step,e.c));e.opt.step();delta=difference(weights(e),before)
    for alpha in s['curvature_alphas']:
        key='alpha_'+str(alpha)
        if r.get(con,'historical_curvature',key):continue
        ctl.check();value=curvature(e,data,dict(s,curvature_alphas=[alpha]),before,delta,ctl)
        r.put(con,'historical_curvature',step,key,dict(source=str(source),seed=seed,step=step,split='B_test',replay=audit,measurements=value))
    r.put(con,'historical_curvature',step,'reaudit_done',dict(complete=True,note='Completion is not a passing curvature audit.'))
    del e,anchor,start,before,delta;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def report(s):
    out=Path(s['output'])
    if not (out/'results.sqlite').exists():return {}
    summaries=r.report(s);con=r.db_open(out);con.execute('BEGIN');events=[];comparisons=[];targets=[]
    for job, in con.execute("SELECT DISTINCT job FROM records WHERE kind='event_step'"):
        rows=r.get(con,job,'event_step');first=rows[0];last=rows[-1]
        events.append(dict(job=job,complete=bool(r.get(con,job,'event_done')),first_A_change=first['post']['A_test']['mean']-first['pre']['A_test']['mean'],first_B_change=first['post']['B_test']['mean']-first['pre']['B_test']['mean'],final_A=last['post']['A_test']['mean'],final_B=last['post']['B_test']['mean'],A_mean_positive_excess=float(np.mean([max(x['post']['A_test']['mean']-first['pre']['A_test']['mean'],0) for x in rows])),norm_audit=first['norm_audit'],first_geometry=first['geometry']))
    by={x['job']:x for x in summaries}
    for item,task,seed,case in cases(s):
        for policy in ['projection','loss_history']:
            p=by.get(case+'/'+policy)
            if not p or not p['complete']:continue
            controls=[]
            for name,x in by.items():
                if not name.startswith(case+'/random_budget') or not x['complete']:continue
                schedule=r.get(con,name,'random_schedule')
                if schedule and policy in schedule[0]['matched_to']:controls.append(x)
            metrics={}
            for domain,key in [('A','final'),('B','final'),('A','mean_positive_excess'),('A','full_accuracy')]:
                values=[x[domain][key] for x in controls];value=p[domain][key]
                metrics[domain+'_'+key]=dict(policy=value,random_values=values,random_mean=float(np.mean(values)) if values else None,policy_minus_random_mean=value-float(np.mean(values)) if values else None)
            comparisons.append(dict(case=case,policy=policy,random_schedules=len(controls),metrics=metrics,note='Within-seed random schedule variability is not independent-model replication; no p-values.'))
        for name,x in by.items():
            if not name.startswith(case+'/'):continue
            rows=r.get(con,name,'step')
            for target in s.get('acquisition_targets',[.75,1.,1.5,2.]):
                for window in [1,4]:
                    found=next((i for i in range(window-1,len(rows)) if all(rows[j]['post']['B_valid']['mean']<=target for j in range(i-window+1,i+1))),None)
                    targets.append(dict(job=name,B_validation_target=target,consecutive_endpoints=window,first_reach_step=rows[found]['step'] if found is not None else None,A_test_at_reach=rows[found]['post']['A_test']['mean'] if found is not None else None))
    audit_rows=[dict(job=j,step=t,kind=k,record=json.loads(raw)) for j,t,k,raw in con.execute("SELECT job,step,kind,record FROM records WHERE job='historical_curvature'")]
    con.close();r.write(out/'event_history_summary.json',events);r.write(out/'policy_random_comparison.json',comparisons);r.write(out/'fixed_acquisition_targets.json',targets);r.write(out/'curvature_reaudit.json',audit_rows)
    lines=['OVERNIGHT REPLICATION — fresh seeds, frozen policies',f"Completed trajectories: {sum(x['complete'] for x in summaries)}; completed event branches: {sum(x['complete'] for x in events)}"]
    for x in comparisons:
        a=x['metrics']['A_final'];b=x['metrics']['B_final'];lines.append(f"{x['case']} {x['policy']}: {x['random_schedules']} matched random schedules; A={a['policy']:.6f}, random mean={a['random_mean']}; B={b['policy']:.6f}, random mean={b['random_mean']}")
    for x in events:lines.append(f"{x['job']}: immediate A {x['first_A_change']:+.6f}, B {x['first_B_change']:+.6f}; complete={x['complete']}")
    (out/'OVERNIGHT_REPORT.txt').write_text('\n'.join(lines));return dict(trajectories=summaries,events=events,random_comparisons=comparisons)
