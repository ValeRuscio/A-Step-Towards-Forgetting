"""Same captured native displacement, different forward computations.
HF quantized inference AD is not the original training STE; FP32 is a surrogate.
"""
from pathlib import Path
import contextlib,json,types,torch
import torch.nn.functional as F
from open_assets import atomic,HF_SHA
from open_observer import cpu,dot,norm,normalize_name

def explicit_masks(model):
 # The pinned HF release targets a newer mask-helper signature. For this
 # uncached, unpadded probe only, implement its documented exact boolean rule.
 def mask(window=False,**kw):
  if kw.get('past_key_values') is not None or kw.get('attention_mask') is not None:raise ValueError('Proxy supports unpadded, uncached diagnostic probes only')
  h=kw['inputs_embeds'];t=h.shape[1];q=torch.arange(t,device=h.device)[:,None];k=torch.arange(t,device=h.device)[None,:];ok=k<=q
  if window:ok=ok & ((k//32)>=(q//32)-(kw['config'].sliding_window//32-1))
  return ok[None,None,:,:]
 scope=model.forward.__func__.__globals__
 scope['create_causal_mask']=lambda **kw:mask(False,**kw)
 scope['create_sliding_window_causal_mask']=lambda **kw:mask(True,**kw)
 return mask

def audit_capture(path,s,pulse):
 from transformers import AutoConfig,AutoModelForCausalLM
 root=Path(s['output']);raw=torch.load(path,map_location='cpu',weights_only=True)
 state={k:{normalize_name(n):v for n,v in raw[k].items()} for k in ['before','delta','history','current']}
 cfg=AutoConfig.from_pretrained('Gensyn/open-1b-base',revision=HF_SHA,trust_remote_code=True);cfg.quantized_forward=True
 # Build once WITH all scale parameters. Toggling config before construction would drop them.
 model=AutoModelForCausalLM.from_config(cfg,trust_remote_code=True,code_revision=HF_SHA)
 model.load_state_dict(state['before'],strict=True);model.to(s['device'],dtype=torch.float32).eval();model.config.use_cache=False;explicit_masks(model)
 params=dict(model.named_parameters());b=state['before'];d=state['delta'];tokens=torch.tensor(json.loads((root/'probe.json').read_text())['tokens'],device=s['device']).unsqueeze(0)
 if set(params)!=set(b):raise ValueError('Native/HF parameter map not exact; no order-based optimizer mapping permitted')
 originals={m:m.forward for m in model.modules() if m.__class__.__name__=='Open1BLinear'}
 result=dict(step=raw['step'],native_replay_verified=json.loads((root/'replay_result.json').read_text()).get('match') is True,modes={},attention_mask='Explicit uncached/unpadded causal and 32-token-block-aligned window masks; math SDPA')
 def assign(alpha):
  with torch.no_grad():
   for n,p in params.items():p.copy_((b[n]+alpha*d[n]).to(p.device))
 def loss():
  from torch.nn.attention import sdpa_kernel,SDPBackend
  with sdpa_kernel(SDPBackend.MATH):out=model(input_ids=tokens[:,:-1],use_cache=False,return_dict=True).logits
  return F.cross_entropy(out.float().reshape(-1,out.shape[-1]),tokens[:,1:].reshape(-1))
 def evaluate(a,grad=False):
  assign(a)
  with torch.enable_grad() if grad else torch.no_grad():
   l=loss();gs=torch.autograd.grad(l,list(params.values()),allow_unused=True) if grad else None
  return float(l.detach()),{n:cpu(g) if g is not None else torch.zeros_like(b[n]) for (n,p),g in zip(params.items(),gs)} if grad else None
 for mode in ['hf_quantized_inference','fp32_unquantized_surrogate']:
  for m,forward in originals.items():
   m.forward=forward if mode=='hf_quantized_inference' else types.MethodType(lambda self,x:F.linear(x,self.weight),m)
  record=dict(note='HF inference round/clamp autograd, not original training STE or native quantized attention.' if mode=='hf_quantized_inference' else 'FP32 unquantized linear operations, same master parameters and native displacement; not a replay of training.',points=[],fd=[])
  for alpha in [0.,.25,.5,.75,1.]:
   pulse(phase='auditing '+mode,step=raw['step'],alpha=alpha);l,g=evaluate(alpha,True);record['points'].append(dict(alpha=alpha,loss=l,g_norm=norm(g),slope=dot(g,d),history=dot(g,state['history']),current=dot(g,state['current'])))
  slope=record['points'][2]['slope']
  for w in s['fd_widths']:
   lo,_=evaluate(.5-w);hi,_=evaluate(.5+w);fd=(hi-lo)/(2*w);tol=s['fd_atol']+s['fd_rtol']*max(abs(fd),abs(slope));record['fd'].append(dict(width=w,finite_difference=fd,autograd=slope,passed=abs(fd-slope)<=tol))
  fd=record['fd'];passed=any(fd[i]['passed'] and fd[i+1]['passed'] and abs(fd[i]['finite_difference']-fd[i+1]['finite_difference'])<=s['fd_atol']+s['fd_rtol']*max(abs(fd[i]['finite_difference']),abs(fd[i+1]['finite_difference'])) for i in range(len(fd)-1));record['derivative_validated']=passed
  if mode=='fp32_unquantized_surrogate' and passed:
   try:
    assign(.5);l=loss();gs=torch.autograd.grad(l,list(params.values()),create_graph=True,allow_unused=True)
    directional=sum((g*d[n].to(g.device)).sum() for (n,p),g in zip(params.items(),gs) if g is not None)
    hv=torch.autograd.grad(directional,list(params.values()),allow_unused=True);curv=sum(float((v.detach()*d[n].to(v.device)).double().sum()) for (n,p),v in zip(params.items(),hv) if v is not None)
    del gs,hv,directional,l
    checks=[]
    for w in s['fd_widths']:
     _,gm=evaluate(.5-w,True);_,gp=evaluate(.5+w,True);f=(dot(gp,d)-dot(gm,d))/(2*w);checks.append(dict(width=w,finite_difference=f,autograd=curv,passed=abs(f-curv)<=s['fd_atol']+s['fd_rtol']*max(abs(f),abs(curv))))
    record['curvature']=dict(value=curv,checks=checks,validated=sum(c['passed'] for c in checks)>=2)
   except Exception as exc:record['curvature']=dict(validated=False,error=repr(exc))
  else:record['curvature']=dict(validated=False,reason='Not attempted for quantized inference or failed first derivative gate')
  result['modes'][mode]=record
 atomic(result,root/'proxy'/f'step_{raw["step"]}.json')
 del model;torch.cuda.empty_cache() if torch.cuda.is_available() else None
 return result

def run(s,pulse):
 root=Path(s['output']);replay=root/'replay_result.json'
 if not replay.exists() or json.loads(replay.read_text()).get('match') is not True:raise ValueError('Requires a verified native replay; no final-HF-weights substitute')
 for p in sorted((root/'captures').glob('step_*.pt')):
  dest=root/'proxy'/(p.stem+'.json')
  if not dest.exists():audit_capture(p,s,pulse)
 return dict(status='complete',message='Pilot and surrogate audits finished; inspect validity flags, not just completion.')
