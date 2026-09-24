"""Paired event-local history interventions with actual displacement audits."""
import hashlib,json,math,random,re
from pathlib import Path
import numpy as np
import torch
import reviewer_suite as r
from study_math import weights,assign,difference,norm,dot,grad,channels,reset,evaluate,directional_hessian
from association_core import minibatches


def group(name):
    m=re.match(r'(?:model|gpt_neox)\.layers\.(\d+)\.',name)
    return 'layer_'+m.group(1) if m else 'embedding_readout_other'


def match_delta(candidate,target,mode,rtol=1e-3):
    """Scale candidate displacement globally or by entire transformer block."""
    groups={}
    for name in candidate:groups.setdefault(group(name) if mode=='layer' else 'global',[]).append(name)
    result={};audit={}
    for key,names in groups.items():
        cn=norm({n:candidate[n] for n in names});tn=norm({n:target[n] for n in names})
        if cn==0 and tn>0:raise ArithmeticError('Cannot norm-match zero candidate in '+key)
        factor=tn/cn if cn else 0.
        if not math.isfinite(factor):raise ArithmeticError('Nonfinite norm factor')
        for n in names:result[n]=factor*candidate[n]
        audit[key]=dict(target=tn,before=cn,factor=factor)
    return result,audit


def audit_actual(actual,audit,mode,rtol):
    for key,info in audit.items():
        v={n:x for n,x in actual.items() if mode=='global' or group(n)==key};value=norm(v)
        info.update(actual=value,absolute_error=abs(value-info['target']),relative_error=abs(value-info['target'])/max(info['target'],1e-30))
        info['passed']=abs(value-info['target'])<=1e-10+rtol*info['target']
    if not all(x['passed'] for x in audit.values()):raise ArithmeticError('Actual FP32 displacement norm audit failed: '+json.dumps(audit))
    return audit


def select_events(records,s):
    """First qualifying calibration-validation events plus predetermined fixed times."""
    selected=[];hits=0
    for rec in sorted(records,key=lambda x:x['step']):
        pre=rec['pre']['A_valid']['rows'];post=rec['post']['A_valid']['rows']
        if s.get('_task')=='registry':
            ids=sorted({x['entity'] for x in pre});random.Random(8041).shuffle(ids)
        else:ids=[x['entity'] for x in pre]
        use=set(ids[:s['frame_calibration']]);a={x['entity']:x['loss'] for x in pre};b={x['entity']:x['loss'] for x in post}
        delta=float(np.mean([b[i]-a[i] for i in use]));fixed=rec['step'] in s['event_fixed_steps']
        event=delta>s['event_threshold'] and hits<s['event_count']
        if fixed or event:
            selected.append(dict(step=rec['step'],rule='fixed_step' if fixed else 'first_calibration_validation_increase',validation_delta=delta,entities=sorted(use),test_outcomes_used=False))
            if event and not fixed:hits+=1
    return selected


def replay(e,anchor,data,s,seed,step,records,ctl):
    e.restore(anchor);ctl.pulse(phase='replaying natural prefix',step=step)
    for t in range(1,step):
        ctl.check();e.gradient(minibatches(data,'B',seed,t,e.c));e.opt.step()
        if t%8==0:ctl.pulse(phase='replaying natural prefix',step=t,target_step=step)
    expected=next(x for x in records if x['step']==step)['pre'];observed=r.all_eval(e,data,s,ctl);errors={}
    for split in expected:
        old={x['entity']:x['loss'] for x in expected[split]['rows']};new={x['entity']:x['loss'] for x in observed[split]['rows']}
        if set(old)!=set(new):raise ValueError('Replay identities do not match')
        diffs=[abs(new[k]-old[k]) for k in old];ok=all(abs(new[k]-old[k])<=s['replay_atol']+s['replay_rtol']*abs(old[k]) for k in old)
        errors[split]=dict(max_error=max(diffs),passed=ok)
    if not all(x['passed'] for x in errors.values()):raise ArithmeticError('Natural replay failed; no event claim made: '+json.dumps(errors))
    return e.pack(),errors


def state_audit(e,start):
    original=start['optimizer']['state'];current=e.opt.state_dict()['state'];max_v=0.;clocks=True;mzero=True
    for key,st in original.items():
        cur=current[key]
        if 'step' in st:clocks=clocks and float(st['step'])==float(cur['step'])
        if 'exp_avg_sq' in st:max_v=max(max_v,float((st['exp_avg_sq'].double()-cur['exp_avg_sq'].detach().cpu().double()).abs().max()))
        if 'exp_avg' in cur:mzero=mzero and bool((cur['exp_avg']==0).all())
    if not clocks or max_v!=0:raise ArithmeticError('History intervention changed clock/second moment')
    return dict(clock_preserved=clocks,second_moment_max_difference=max_v,first_moment_is_zero=mzero)


def branch(e,data,s,seed,case,selection,mode,start,root,con,ctl):
    event=selection['step'];job=f'{case}/event{event:03d}/{mode}'
    if r.get(con,job,'event_done'):return
    savedpath=root/'event_rolling.pt';saved=torch.load(savedpath,map_location='cpu',weights_only=True) if savedpath.exists() else None
    horizon=min(s['event_horizon'],s['b_steps']-event+1)
    if saved and saved.get('job')==job:
        e.restore(saved['engine']);offset=saved['offset']+1
        con.execute("DELETE FROM records WHERE job=? AND kind='event_step' AND step>?",(job,saved['offset']));con.commit()
    else:
        e.restore(start);offset=0
        if mode!='preserve':reset(e,'reset_m')
        audit=state_audit(e,start)
        if mode!='preserve' and not audit['first_moment_is_zero']:raise ArithmeticError('Moment reset failed')
        r.put(con,job,0,'event_setup',dict(selection=selection,state_audit=audit,horizon=horizon,norm_matching_first_step_only=True,layer_groups='transformer blocks; embedding/readout/other as one group'))
        ctl.checkpoint(dict(engine=e.pack(),job=job,offset=-1),savedpath)
    for off in range(offset,horizon):
        ctl.check();t=event+off;ctl.pulse(job=job,phase='event-local history branch',step=t,condition=mode)
        pre=r.all_eval(e,data,s,ctl);before=weights(e);ga=grad(e,data['A_valid'],s['eval_batch'],ctl) if off==0 else None
        batches=minibatches(data,'B',seed,t,e.c);e.gradient(batches);h,c=channels(e);e.opt.step();after=weights(e);delta=difference(after,before);matching=None
        if off==0 and mode in ['reset_m_global','reset_m_layer']:
            branch_after=e.pack();e.restore(start);e.gradient(batches);e.opt.step();target=difference(weights(e),before)
            matchmode='global' if mode.endswith('global') else 'layer';scaled,matching=match_delta(delta,target,matchmode)
            e.restore(branch_after);assign(e,{n:before[n]+scaled[n] for n in before});after=weights(e);delta=difference(after,before)
            audit_actual(delta,matching,matchmode,s['norm_match_rtol'])
            for n in h:
                factor=matching['global' if matchmode=='global' else group(n)]['factor'];h[n]*=factor;c[n]*=factor
            del branch_after,target,scaled
        post=r.all_eval(e,data,s,ctl)
        geom=None
        if off==0:
            dn=norm(delta);gn=norm(ga);proj=dot(ga,delta)
            geom=dict(projection=proj,cosine=proj/max(dn*gn,1e-30),update_norm=dn,gradient_norm=gn,history_projection=dot(ga,h),current_projection=dot(ga,c),nonlinear_remainder=post['A_valid']['mean']-pre['A_valid']['mean']-proj,
                per_layer={key:dict(norm=norm({n:v for n,v in delta.items() if group(n)==key}),projection=dot({n:v for n,v in ga.items() if group(n)==key},delta)) for key in sorted({group(n) for n in delta})})
        r.put(con,job,off,'event_step',dict(seed=seed,event_step=event,offset=off,step=t,condition=mode,pre=pre,post=post,geometry=geom,norm_audit=matching))
        if (off+1)%s['event_checkpoint_every']==0 or off==horizon-1:ctl.checkpoint(dict(engine=e.pack(),job=job,offset=off),savedpath)
        del before,after,delta,h,c,ga
    r.put(con,job,horizon,'event_done',dict(complete=True,event=event,condition=mode,updates=horizon));savedpath.unlink(missing_ok=True)


def curvature(e,data,s,before,delta,ctl):
    """Read-only reevaluation; all widths and failures retained, no best-width selection."""
    result=[];rows=data['B_test'];epsilons=s['curvature_eps']
    try:
        for alpha in s['curvature_alphas']:
            ctl.pulse(phase='curvature re-audit',alpha=alpha,split='B_test')
            assign(e,{n:v+alpha*delta[n] for n,v in before.items()});hv=directional_hessian(e,rows,delta,s['eval_batch'],ctl);slope=dot(grad(e,rows,s['eval_batch'],ctl),delta);widths=[]
            for eps in epsilons:
                losses=[];slopes=[]
                for sign in [-1,1]:
                    assign(e,{n:v+(alpha+sign*eps)*delta[n] for n,v in before.items()});losses.append(evaluate(e,rows,s['eval_batch'],ctl)['mean']);slopes.append(dot(grad(e,rows,s['eval_batch'],ctl),delta))
                fd=(slopes[1]-slopes[0])/(2*eps);fs=(losses[1]-losses[0])/(2*eps)
                close=lambda a,b:abs(a-b)<=s['path_atol']+s['path_rtol']*max(abs(a),abs(b))
                widths.append(dict(epsilon=eps,curvature_fd=fd,curvature_agrees=close(fd,hv),slope_fd=fs,slope_agrees=close(fs,slope)))
                ctl.pulse(phase='curvature width complete',alpha=alpha,epsilon=eps)
            pairs=[dict(widths=[widths[i]['epsilon'],widths[i+1]['epsilon']],passed=widths[i]['curvature_agrees'] and widths[i+1]['curvature_agrees'] and widths[i]['slope_agrees'] and widths[i+1]['slope_agrees'] and close(widths[i]['curvature_fd'],widths[i+1]['curvature_fd'])) for i in range(len(widths)-1)]
            # Primary criterion is the last two widths, fixed before seeing this run.
            result.append(dict(alpha=alpha,autograd_hessian=hv,slope=slope,widths=widths,adjacent_pairs=pairs,primary_pass=pairs[-1]['passed']))
    finally:assign(e,before)
    return result
