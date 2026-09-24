"""One canonical OPEN-1B pretraining step; observer only, then surrogate audits."""
from pathlib import Path
import importlib.util,json,os,platform,sys,time,traceback,shutil
from open_assets import atomic,prepare,SOURCE_SHA,HF_SHA

def defaults(output):
 return dict(output=str(Path(output).resolve()),checkpoint_step=50300,steps=1,minimum_free_gib=60.,hours=11.5,device='cuda',fd_widths=[.02,.01,.005,.0025],fd_atol=.002,fd_rtol=.1,
  probe_text='A scientific measurement should distinguish the observed result from the assumptions used to interpret it. Repeated observations help estimate variability. A change in a model can affect predictions on previously learned material even when performance on new examples improves. Careful comparisons use the same evaluation examples before and after an update.',
  probe_token_ids=None,proxy_audits=True,version='open-history-pilot-1')

def readiness():
 result=dict(platform=platform.platform(),machine=platform.machine(),python=sys.executable,ready=False)
 try:
  import torch,repop
  from repop import ops
  if not hasattr(ops,'adamw_kernel_step'):raise ImportError('Repop AdamW operator unavailable')
  result.update(torch=torch.__version__,repop_file=repop.__file__,ready=True)
 except Exception as exc:result.update(reason=repr(exc),action='Use an isolated official Gensyn replay environment with matching RepOps/pretrain artifacts. Published Linux wheels document x86-64; GH200 hosts are usually aarch64 and may require a supported build or an x86-64 GPU worker. No fresh-optimizer or HF-gradient fallback is allowed.')
 return result

def instrument_source(source):
 anchors=[('                batch = next(iters[r])\n','                _pilot_observer.tick(locals())\n                batch = next(iters[r])\n'),('        _halted = False\n','        _pilot_observer.before(locals())\n        _halted = False\n'),('        _completed_step = step + 1\n','        _pilot_observer.after_optimizer(locals())\n        _completed_step = step + 1\n'),('        # ---- Canonical state hash (post-step, grads still live)', '        _pilot_observer.after(locals())\n\n        # ---- Canonical state hash (post-step, grads still live)')]
 for old,new in anchors:
  if source.count(old)!=1:raise ValueError('Pinned replay hook location changed; refusing ambiguous instrumentation')
  source=source.replace(old,new)
 return source

def run(s,pulse):
 root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
 if s['version']!='open-history-pilot-1' or not 1<=s['steps']<=3:raise ValueError('This pilot allows 1–3 actual pretraining steps only')
 ready=readiness();atomic(ready,root/'readiness.json')
 if not ready['ready']:return dict(status='dependency_blocked',message=ready['action'])
 if shutil.disk_usage(root).free<(s['minimum_free_gib']+80)*1024**3:raise RuntimeError('Need 80 GiB working headroom plus the configured reserve for checkpoint, data, and analysis capture')
 tree,ck,target=prepare(s,pulse);sys.path.insert(0,str(tree/'src'))
 from transformers import AutoTokenizer
 if s['probe_token_ids'] is None:
  tokenizer=AutoTokenizer.from_pretrained('Gensyn/open-1b-base',revision=HF_SHA,trust_remote_code=False)
  s=dict(s,probe_token_ids=tokenizer.encode(s['probe_text'],add_special_tokens=False)[:128])
 if len(s['probe_token_ids'])<8:raise ValueError('Probe needs at least 8 token IDs')
 atomic(dict(tokens=s['probe_token_ids'],text=s['probe_text'],tokenizer_revision=HF_SHA,note='Small fixed diagnostic text, not a validated retention benchmark and not certified absent from pretraining.'),root/'probe.json')
 native_path=tree/'src/pretrain/cli/audit_replay.py';patched=root/'instrumented_audit_replay.py';patched.write_text(instrument_source(native_path.read_text()))
 import pretrain.cli
 spec=importlib.util.spec_from_file_location('pretrain.cli.audit_replay',patched);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 from open_observer import Observer
 obs=Observer(s,pulse);module._pilot_observer=obs
 pulse(phase='canonical native replay',step=s['checkpoint_step'],target=s['checkpoint_step']+s['steps'])
 result=module.audit_replay(str(ck),device=s['device'],until_step=s['checkpoint_step']+s['steps'],expect_hash=target,gcs_root='gs://gensyn-open-1b/data/shards',fetch_dest=str(root/'assets/data'),fold_spill_dir=str(root/'assets/fold'),loss_log=str(root/'native_rank0_losses.json'))
 atomic(result,root/'replay_result.json')
 matched=result.get('match') is True
 for rec in obs.records:
  rec['canonical_replay_verified']=matched;atomic(rec,root/'native'/f'step_{rec["step"]}.json')
 if not matched:return dict(status='audit_failed',message='Native replay did not match the published target hash. Measurements are unverified; no derivative or historical-mechanism claims.')
 # Pilot validity is distinct from derivative validity.
 valid=bool(obs.records) and all(r.get('native_probe',{}).get('derivative_validated',False) for r in obs.records)
 atomic(dict(canonical_replay_verified=True,native_derivative_checks_passed=valid,full_study_authorized=False,source_commit=SOURCE_SHA,note='A successful state hash verifies the instrumented replay. FD agreement on a small probe does not validate all surrogate derivatives or establish forgetting.'),root/'pilot_gate.json')
 if s['proxy_audits']:
  # Separate process avoids sharing replay model memory, CUDA contexts, or native operators.
  return dict(status='native_complete',message='Native replay verified. Run proxy audit stage in the Qwen/Transformers environment.',native_derivatives_passed=valid)
 return dict(status='complete',message='Native pilot finished; inspect pilot_gate.json before any expansion.')
