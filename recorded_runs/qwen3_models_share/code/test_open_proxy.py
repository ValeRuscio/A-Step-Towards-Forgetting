"""Run against downloaded pinned HF modeling/configuration source, tiny dimensions."""
import importlib.util,json,sys,tempfile,types
from pathlib import Path
import torch
from unittest.mock import patch
from open_assets import atomic

def main(src):
 package=types.ModuleType('open_test_model');package.__path__=[str(Path(src).resolve())];sys.modules[package.__name__]=package
 if sys.version_info < (3,10):
  # Test-host-only annotation compatibility; no forward/gradient code is altered.
  name='open_test_model.configuration_open1b';mod=types.ModuleType(name);mod.__package__='open_test_model';sys.modules[name]=mod
  exec(compile('from __future__ import annotations\n'+(Path(src)/'configuration_open1b.py').read_text(),str(Path(src)/'configuration_open1b.py'),'exec'),mod.__dict__)
 from open_test_model.configuration_open1b import Open1BConfig
 from open_test_model.modeling_open1b import Open1BForCausalLM
 cfg=Open1BConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=8,intermediate_size=32,vocab_size=32,max_position_embeddings=128,sliding_window=64,swa_full_every=2,quantized_forward=True)
 torch.manual_seed(1);m=Open1BForCausalLM(cfg);b={n:p.detach().clone() for n,p in m.named_parameters()};d={n:torch.randn_like(p)*1e-4 for n,p in m.named_parameters()}
 from proxy_audit import audit_capture
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);atomic(dict(tokens=[1,2,3,4,5,6,7,8,9]),root/'probe.json');atomic(dict(match=True),root/'replay_result.json');capture=root/'capture.pt';torch.save(dict(before=b,delta=d,history={n:.7*x for n,x in d.items()},current={n:.3*x for n,x in d.items()},step=1),capture)
  s=dict(output=tmp,device='cpu',fd_widths=[.04,.02,.01],fd_atol=.002,fd_rtol=.1)
  with patch('transformers.AutoConfig.from_pretrained',return_value=cfg),patch('transformers.AutoModelForCausalLM.from_config',side_effect=lambda *a,**kw:Open1BForCausalLM(cfg)):
   r=audit_capture(capture,s,lambda **kw:None)
  assert set(r['modes'])=={'hf_quantized_inference','fp32_unquantized_surrogate'}
  assert r['modes']['fp32_unquantized_surrogate']['derivative_validated']
  assert r['modes']['fp32_unquantized_surrogate']['curvature']['validated']
  print('PASS pinned OPEN HF architecture in tiny dimensions: exact parameter mapping, two distinct forwards, FP32 derivatives and curvature')
if __name__=='__main__':main(sys.argv[1])
