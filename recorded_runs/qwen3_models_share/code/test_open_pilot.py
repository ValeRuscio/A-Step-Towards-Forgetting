"""CPU mathematical/loading gates; does not claim validation of native RepOps kernels."""
import ast,contextlib,json,tempfile,types,sys
from pathlib import Path
import torch
from open_observer import channels,Observer
from open_pilot import instrument_source

def test_channels():
 torch.manual_seed(4);m=torch.nn.Linear(3,2);opt=torch.optim.AdamW(m.parameters(),lr=.01,betas=(.9,.95),weight_decay=.1,foreach=False)
 for _ in range(5):m(torch.ones(2,3)).square().mean().backward();opt.step();opt.zero_grad()
 m(torch.ones(2,3)).square().mean().backward();b,h,c,d,a=channels(m,opt);opt.step()
 for n,p in m.named_parameters():torch.testing.assert_close(p.detach().cpu()-b[n],h[n]+c[n]+d[n],atol=2e-7,rtol=2e-5)
 assert all(x['step']==5 for x in a)
 fresh=torch.optim.AdamW(m.parameters())
 try:channels(m,fresh)
 except ValueError:pass
 else:raise AssertionError('Fresh optimizer silently accepted')
 print('PASS native-channel algebra versus Torch AdamW; rejects missing history')

def test_hooks(source):
 s=Path(source).read_text();x=instrument_source(s);ast.parse(x)
 assert x.count('_pilot_observer.before(locals())')==1
 assert x.count('_pilot_observer.after_optimizer(locals())')==1
 assert x.count('_pilot_observer.after(locals())')==1
 assert x.count('_pilot_observer.tick(locals())')==1
 try:instrument_source(s.replace('_completed_step = step + 1','_completed_step = step + 2'))
 except ValueError:pass
 else:raise AssertionError('Unknown upstream layout accepted')
 print('PASS exact pinned upstream hook placement and source-change rejection')

def test_observer():
 # Mock only the native autocast import. Real native forward is not tested here.
 module=types.ModuleType('pretrain.cli.audit_replay');module._autocast_ctx=lambda *a,**kw:contextlib.nullcontext();sys.modules[module.__name__]=module
 class Tiny(torch.nn.Module):
  def __init__(self):super().__init__();self.emb=torch.nn.Embedding(16,4);self.head=torch.nn.Linear(4,16)
  def forward(self,ids):return types.SimpleNamespace(logits=self.head(self.emb(ids)))
 torch.manual_seed(3);m=Tiny();opt=torch.optim.AdamW(m.parameters(),lr=.01,foreach=False)
 ids=torch.tensor([[1,2,3,4,5,6,7,8]])
 for _ in range(3):m(ids).logits.square().mean().backward();opt.step();opt.zero_grad()
 m(ids).logits.square().mean().backward()
 with tempfile.TemporaryDirectory() as tmp:
  s=dict(output=tmp,probe_token_ids=ids[0].tolist(),fd_widths=[.02,.01,.005],fd_atol=.002,fd_rtol=.1)
  obs=Observer(s,lambda **kw:None);ctx=dict(model=m,optimizer=opt,step=3,skip=False,offload_optimizer=False,dev=torch.device('cpu'),cfg=types.SimpleNamespace(run=types.SimpleNamespace(mixed_precision=False)),_grad_source=lambda:m,_sync_grad_model=lambda:None,_autocast_ctx=lambda *a,**kw:contextlib.nullcontext())
  obs.before(ctx);opt.step();obs.after_optimizer(ctx)
  # Explicit non-optimizer change represents a scale refresh or clamp.
  with torch.no_grad():m.head.bias.add_(.001)
  expected={n:p.clone() for n,p in m.named_parameters()};grads={n:p.grad.clone() for n,p in m.named_parameters()};rng=torch.random.get_rng_state().clone()
  obs.after(ctx)
  for n,p in m.named_parameters():assert torch.equal(p,expected[n]) and torch.equal(p.grad,grads[n])
  assert torch.equal(rng,torch.random.get_rng_state()) and m.training
  r=obs.records[0];assert r['norms']['post_optimizer_refresh_or_clamp']>0
  assert r['native_probe']['derivative_validated'],r['native_probe']
  assert not r['canonical_replay_verified']
 print('PASS observer restores weights, gradients, RNG and mode; separates post-step changes')
if __name__=='__main__':
 test_channels();test_observer()
 if len(sys.argv)>1:test_hooks(sys.argv[1])
