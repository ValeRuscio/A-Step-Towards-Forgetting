"""Observers around the ORIGINAL replay optimizer. Never substitute its updates.
Native gradients are explicitly surrogate algorithmic gradients until FD audited.
"""
from pathlib import Path
import contextlib,json,math,os,time
import torch
import torch.nn.functional as F
from open_assets import atomic

def cpu(t):
 if hasattr(t,'full_tensor'):t=t.full_tensor()
 return t.detach().to('cpu',dtype=torch.float32).clone()
def dot(a,b):return sum(float((a[n].double()*b[n].double()).sum()) for n in a)
def norm(a):return math.sqrt(max(0,dot(a,a)))
def normalize_name(n):return n.replace('_checkpoint_wrapped_module.','')
def channels(model,opt):
 names={id(p):n for n,p in model.named_parameters()};before={n:cpu(p) for n,p in model.named_parameters()};h={};c={};decay={};audit=[]
 for group in opt.param_groups:
  b1,b2=group['betas'];lr=float(group['lr']);eps=float(group['eps']);wd=float(group['weight_decay'])
  if group.get('amsgrad'):raise ValueError('Pilot does not support AMSGrad')
  for p in group['params']:
   n=names[id(p)];st=opt.state[p]
   if not all(k in st and st[k] is not None for k in ['step','exp_avg','exp_avg_sq']):raise ValueError('Native optimizer state absent/offloaded: '+n)
   t=int(st['step'].item());m=cpu(st['exp_avg']);v=cpu(st['exp_avg_sq'])
   if t<=0 or not torch.isfinite(m).all() or not torch.isfinite(v).all() or (v<0).any():raise ValueError('Invalid historical optimizer state: '+n)
   active=p.grad is not None;g=cpu(p.grad) if active else torch.zeros_like(m)
   den=((b2*v+(1-b2)*g.square())/(1-b2**(t+1))).sqrt()+eps
   fac=-lr/(1-b1**(t+1))
   h[n]=fac*b1*m/den if active else torch.zeros_like(m);c[n]=fac*(1-b1)*g/den if active else torch.zeros_like(m)
   decay[n]=-lr*wd*before[n] if active else torch.zeros_like(m)
   audit.append(dict(name=n,step=t,active=active,m_norm=float(m.norm()),v_norm=float(v.norm()),lr=lr,betas=[b1,b2],eps=eps,weight_decay=wd))
 if set(h)!=set(before):raise ValueError('Optimizer does not cover every model parameter')
 if not any(r['m_norm']>0 for r in audit):raise ValueError('Zero optimizer history; refusing fresh-optimizer substitute')
 return before,h,c,decay,audit

class Observer:
 def __init__(self,s,pulse):self.s=s;self.root=Path(s['output']);self.pulse=pulse;self.pending=None;self.records=[]
 def tick(self,ctx):
  self.pulse(phase='native replay microbatch',step=ctx['step']+1)
 def before(self,ctx):
  self.pulse(phase='auditing inherited optimizer state',step=ctx['step']+1)
  if ctx['offload_optimizer'] or ctx.get('offload_master') or ctx.get('offload_grads'):raise RuntimeError('Pilot observer requires resident master parameters and optimizer; replay offload was selected. Use a compatible larger-memory worker.')
  model=ctx['model'];opt=ctx['optimizer'];b,h,c,d,a=channels(model,opt)
  self.pending=dict(before=b,history=h,current=c,decay=d,audit=a,step=ctx['step']+1,skip=bool(ctx['skip']))
 def after_optimizer(self,ctx):
  self.pending['after_optimizer']={n:cpu(p) for n,p in ctx['model'].named_parameters()}
 def after(self,ctx):
  state=self.pending;b=state['before'];model=ctx['model'];end={n:cpu(p) for n,p in model.named_parameters()}
  delta={n:end[n]-b[n] for n in b};post={n:end[n]-state['after_optimizer'][n] for n in b}
  numerical={n:state['after_optimizer'][n]-b[n]-state['history'][n]-state['current'][n]-state['decay'][n] for n in b}
  if state['skip']:
   # Channels are counterfactual on skipped steps; never attribute them to the actual update.
   interpretation='Skipped optimizer step: h/c are unused hypothetical channels.'
  else:interpretation='Native AdamW numerator channels; actual displacement additionally includes decay, kernel rounding, scale refresh and clamps.'
  rec=dict(step=state['step'],skipped=state['skip'],optimizer_state_audit=state['audit'],norms={k:norm(v) for k,v in [('actual',delta),('history',state['history']),('current',state['current']),('decay',state['decay']),('post_optimizer_refresh_or_clamp',post),('optimizer_formula_residual',numerical)]},history_current_dot=dot(state['history'],state['current']),interpretation=interpretation,canonical_replay_verified=False)
  # A small fixed probe; it is not claimed to be disjoint from the original pretraining corpus.
  tokens=torch.tensor(self.s['probe_token_ids'],dtype=torch.long,device=ctx['dev']).unsqueeze(0)
  original_modes={m:m.training for m in list(model.modules())+list(ctx['_grad_source']().modules())};rng=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  def assign(alpha):
   with torch.no_grad():
    for n,p in model.named_parameters():p.copy_((b[n]+alpha*delta[n]).to(p.device))
   ctx['_sync_grad_model']()
  def loss_and_grad(alpha,want_grad):
   assign(alpha);source=ctx['_grad_source']();params=dict(source.named_parameters());source.eval()
   _autocast_ctx=ctx['_autocast_ctx']
   with torch.enable_grad() if want_grad else torch.no_grad():
    with _autocast_ctx(ctx['dev'].type=='cuda',mixed_precision=ctx['cfg'].run.mixed_precision):
     logits=source(tokens[:,:-1]).logits
     # This evaluation CE is deliberately separate from the native training CE+z-loss.
     loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),tokens[:,1:].reshape(-1))
    grad=torch.autograd.grad(loss,list(params.values()),allow_unused=True) if want_grad else None
   return float(loss.detach()),({n:cpu(g) if g is not None else torch.zeros_like(b[n]) for (n,p),g in zip(params.items(),grad)} if want_grad else None)
  try:
   rec['native_probe']=dict(label='Original quantized native operators + diagnostic CE; autograd is an algorithmic/surrogate derivative until audited.',points=[],derivative_checks=[])
   for alpha in [0.,.5,1.]:
    self.pulse(phase='native quantized probe',step=state['step'],alpha=alpha)
    loss,g=loss_and_grad(alpha,True);rec['native_probe']['points'].append(dict(alpha=alpha,loss=loss,gradient_norm=norm(g),actual_projection=dot(g,delta),history_projection=dot(g,state['history']),current_projection=dot(g,state['current'])))
   slope=rec['native_probe']['points'][1]['actual_projection']
   for width in self.s['fd_widths']:
    lo,_=loss_and_grad(.5-width,False);hi,_=loss_and_grad(.5+width,False);fd=(hi-lo)/(2*width);tol=self.s['fd_atol']+self.s['fd_rtol']*max(abs(fd),abs(slope))
    rec['native_probe']['derivative_checks'].append(dict(width=width,finite_difference=fd,autograd=slope,error=abs(fd-slope),passed=abs(fd-slope)<=tol))
   checks=rec['native_probe']['derivative_checks'];rec['native_probe']['derivative_validated']=any(checks[i]['passed'] and checks[i+1]['passed'] and abs(checks[i]['finite_difference']-checks[i+1]['finite_difference'])<=self.s['fd_atol']+self.s['fd_rtol']*max(abs(checks[i]['finite_difference']),abs(checks[i+1]['finite_difference'])) for i in range(len(checks)-1))
   rec['native_probe']['hessian_status']='Not attempted: surrogate/quantized native operators require derivative validation first; no smooth Hessian claim.'
  except Exception as exc:
   if type(exc).__name__=='Pause':raise
   rec['native_probe']=dict(derivative_validated=False,error=repr(exc),note='Unsupported derivatives are a pilot outcome, not a passed audit.')
  finally:
   with torch.no_grad():
    for n,p in model.named_parameters():p.copy_(end[n].to(p.device))
   ctx['_sync_grad_model']()
   for m,mode in original_modes.items():m.training=mode
   torch.random.set_rng_state(rng)
   if cuda is not None:torch.cuda.set_rng_state_all(cuda)
  dest=self.root/'captures'/f'step_{state["step"]}.pt';dest.parent.mkdir(exist_ok=True)
  self.pulse(phase='saving one-step analysis state',step=state['step'])
  torch.save(dict(before=b,delta=delta,history=state['history'],current=state['current'],step=state['step']),dest.with_suffix('.tmp'));os.replace(dest.with_suffix('.tmp'),dest)
  atomic(rec,self.root/'native'/f'step_{state["step"]}.json');self.records.append(rec);self.pending=None
