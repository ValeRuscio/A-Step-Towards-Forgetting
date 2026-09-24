"""Actual-displacement measurements and explicitly separate counterfactual probes."""
import math,re,copy
import numpy as np
import torch
from study_math import weights,assign,difference,dot,norm,grad,evaluate,directional_hessian
from optimizers import native_channels,reset_history

def layer_name(n):
 m=re.search(r'(?:^|\.)layers\.(\d+)\.',n)
 return 'block_'+m.group(1) if m else 'embedding_readout_other'

def geometry(gA,gB,train,delta,ch):
 dn=norm(delta);an=norm(gA);projection=dot(gA,delta)
 result=dict(projection=projection,update_norm=dn,gA_norm=an,update_cosine=projection/(an*dn) if an*dn else 0.,training_interference=-dot(gA,train),population_interference=-dot(gA,gB),layers={})
 for n in delta:
  key=layer_name(n);v=result['layers'].setdefault(key,dict(projection=0.,squared_norm=0.))
  v['projection']+=dot({n:gA[n]},{n:delta[n]});v['squared_norm']+=dot({n:delta[n]},{n:delta[n]})
 if ch is not None:
  h,c=ch;result.update(history_projection=dot(gA,h),current_projection=dot(gA,c),channel_roundoff_norm=norm({n:delta[n]-h.get(n,0)-c.get(n,0) for n in delta}))
 else:result['channel_note']='No additive history/current attribution for this optimizer variant; only total actual displacement measured.'
 return result

def match(candidate,target,by_block):
 groups={}
 for n in target:groups.setdefault(layer_name(n) if by_block else 'global',[]).append(n)
 out={};audits={}
 for group,names in groups.items():
  a=norm({n:candidate[n] for n in names});b=norm({n:target[n] for n in names})
  if a==0 and b>0:raise ArithmeticError('Cannot norm-match zero candidate')
  scale=b/a if a else 0.
  for n in names:out[n]=candidate[n]*scale
  audits[group]=dict(target=b,scale=scale)
 return out,audits

def audit_match(e,before,audits,by_block,tol):
 actual=difference(weights(e),before)
 for key,v in audits.items():
  group={n:d for n,d in actual.items() if (layer_name(n) if by_block else 'global')==key};v['actual']=norm(group);v['relative_error']=abs(v['actual']-v['target'])/max(v['target'],1e-30);v['passed']=abs(v['actual']-v['target'])<=1e-10+tol*v['target']
 if not all(v['passed'] for v in audits.values()):raise ArithmeticError('Actual assigned norm audit failed')
 return audits

def state_audit(before,after):
 a=before['optimizer']['state'];b=after['optimizer']['state'];changed=[];maxother=0.
 for k,st in a.items():
  for name,val in st.items():
   if name in ['exp_avg','momentum_buffer']:changed.append(name);continue
   def compare(x,y):
    if isinstance(x,torch.Tensor):return float((x.double()-y.double()).abs().max()) if x.numel() else 0.
    if isinstance(x,dict):return max([compare(v,y[n]) for n,v in x.items()]+[0.])
    if isinstance(x,list):return max([compare(u,v) for u,v in zip(x,y)]+[0.])
    return 0. if x==y else float('inf')
   maxother=max(maxother,compare(val,b[k][name]))
 return dict(history_fields=sorted(set(changed)),max_other_state_difference=maxother,other_state_preserved=maxother==0.)

def history_probes(e,start,end,batches,data,s,ctl):
 before={n:start['model'][n] for n in e.params};actual=difference({n:end['model'][n] for n in e.params},before);out={}
 try:
  for mode in ['reset','reset_global','reset_blocks']:
   ctl.check();ctl.pulse(phase='matched optimizer-history probe',condition=mode)
   e.restore(start);count=reset_history(e);audit=state_audit(start,e.pack())
   if not audit['other_state_preserved']:raise ArithmeticError('Reset modified non-history state')
   e.gradient(batches);e.opt.step();candidate=difference(weights(e),before);ma=None
   if mode!='reset':
    d,ma=match(candidate,actual,mode=='reset_blocks');assign(e,{n:before[n]+d[n] for n in before});ma=audit_match(e,before,ma,mode=='reset_blocks',s['norm_match_rtol'])
   vals={sp:evaluate(e,data[sp],s['eval_batch'],ctl) for sp in ['A_valid','B_valid','A_test','B_test']}
   out[mode]=dict(history_tensors_reset=count,state_audit=audit,norm_audit=ma,post=vals,immediate_only=True)
 finally:e.restore(end)
 return out

def path_probe(e,before,after,ch,rows,s,ctl):
 delta=difference(after,before);points=[]
 try:
  for alpha in np.linspace(0,1,s['path_points']):
   ctl.pulse(phase='finite-update validation path',alpha=float(alpha));ctl.check();assign(e,{n:v+float(alpha)*delta[n] for n,v in before.items()})
   g=grad(e,rows,s['eval_batch'],ctl);r=dict(alpha=float(alpha),loss=evaluate(e,rows,s['eval_batch'],ctl)['mean'],slope=dot(g,delta))
   if ch is not None:r.update(history=dot(g,ch[0]),current=dot(g,ch[1]))
   points.append(r)
  def simpson(k):
   y=[p[k] for p in points];return (y[0]+y[-1]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2]))/(3*(len(y)-1))
  observed=points[-1]['loss']-points[0]['loss'];integral=simpson('slope');tol=s['audit_atol']+s['audit_rtol']*abs(observed)
  j=int(np.argmax(np.abs(np.diff([p['slope'] for p in points]))));alpha=(points[j]['alpha']+points[j+1]['alpha'])/2
  assign(e,{n:v+alpha*delta[n] for n,v in before.items()});hv=directional_hessian(e,rows,delta,s['eval_batch'],ctl);checks=[]
  for eps in s['audit_eps']:
   ls=[];sl=[]
   for a in [alpha-eps,alpha,alpha+eps]:
    ctl.pulse(phase='directional derivative audit',alpha=a,epsilon=eps);assign(e,{n:v+a*delta[n] for n,v in before.items()});ls.append(evaluate(e,rows,s['eval_batch'],ctl)['mean']);sl.append(dot(grad(e,rows,s['eval_batch'],ctl),delta))
   fd=(ls[2]-ls[0])/(2*eps);cur=(sl[2]-sl[0])/(2*eps)
   close=lambda a,b:abs(a-b)<=s['audit_atol']+s['audit_rtol']*max(abs(a),abs(b))
   checks.append(dict(epsilon=eps,slope_fd=fd,slope=sl[1],slope_pass=close(fd,sl[1]),curvature_fd=cur,curvature_pass=close(cur,hv)))
  stable=close(checks[-1]['curvature_fd'],checks[-2]['curvature_fd']);coarse=[x['slope'] for x in points[::2]]
  coarse_integral=(coarse[0]+coarse[-1]+4*sum(coarse[1:-1:2])+2*sum(coarse[2:-1:2]))/(3*(len(coarse)-1));qerr=abs(integral-coarse_integral)/15
  result=dict(population='A_valid',n=len(rows),points=points,observed=observed,initial_slope=points[0]['slope'],nonlinear_remainder=observed-points[0]['slope'],integral=integral,closure=integral-observed,closure_pass=abs(integral-observed)<=tol,quadrature_error_estimate=qerr,quadrature_pass=qerr<=tol,curvature_alpha=alpha,hessian=hv,audits=checks,hessian_pass=stable and all(x['curvature_pass'] and x['slope_pass'] for x in checks[-2:]))
  if ch is not None:result.update(history_integral=simpson('history'),current_integral=simpson('current'))
  return result
 finally:assign(e,after)
