"""Quiet notebook API. The queue can span multiple sessions without overwriting results."""
import copy,fcntl,json,os,subprocess,sys,time,uuid,zipfile,sqlite3,shutil
from pathlib import Path
from storage import read,write,alive,hashes
HERE=Path(__file__).resolve().parent

def defaults(output='runs/optimizer_generality_v1'):
 return dict(version='optimizer-generality-1',output=str(Path(output).resolve()),python=sys.executable,device='cuda:0',threads=8,hours=11.5,minimum_free_gib=40.,max_output_gib=80.,checkpoint_every=8,
  models=[dict(name='smollm2_135m',repo='HuggingFaceTB/SmolLM2-135M',revision='main'),dict(name='pythia410m',repo='EleutherAI/pythia-410m',revision='9879c9b5f8bea9051dcb0e68dff21493d67e9d4f')],
  datasets={'sst2':'8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb','ag_news':'eb185aade064a813bc0b7f42de02595523103ca4'},
  seeds=[21,22,23],calibration_seed=901,optimizers=['sgd','momentum_sgd','adam_beta1_0','adam'],
  lr_grid={'sgd':[.001,.01,.1],'momentum_sgd':[.0001,.001,.01],'adam_beta1_0':[.000005,.00002,.00008],'adam':[.000005,.00002,.00008],'shampoo_block':[.0001,.001,.01],'muon_fp32':[.005,.02,.08]},
  momentum=.9,beta2=.99,eps=1e-8,clip=1.,microbatch=2,accumulation=2,eval_batch=4,max_length=256,text_train=1024,text_valid=32,text_test=128,
  calibration_a_steps=300,calibration_b_steps=64,calibration_eval_every=8,a_steps=600,b_steps=128,
  acquisition_min_gain=.1,acquisition_min_accuracy=.6,calibration_retention_weight=1.,
  probe_steps=[8,32],path_seeds=[21],path_points=9,audit_eps=[.02,.01,.005],audit_atol=.0002,audit_rtol=.05,norm_match_rtol=.001,
  label_controls=True,label_models=['smollm2_135m'],label_optimizer='adam',
  appendix=True,appendix_optimizers=['shampoo_block','muon_fp32'],appendix_seed=31,appendix_b_steps=64,
  shampoo_block=128,shampoo_root_frequency=10,shampoo_damping=1e-4,muon_ns_steps=5,aux_lr_ratio={'shampoo_block':.02,'muon_fp32':.001},
  bootstrap_replicates=300,existing_sources=[],smoke=False)

def smoke_settings(output):
 s=defaults(output);s.update(models=[dict(name='tiny_llama',repo='tiny-llama',revision='smoke'),dict(name='tiny_neox',repo='tiny-gpt_neox',revision='smoke')],device='cpu',threads=1,seeds=[21],calibration_a_steps=2,calibration_b_steps=2,a_steps=2,b_steps=12,calibration_eval_every=1,microbatch=2,accumulation=1,eval_batch=2,text_train=8,text_valid=4,text_test=4,max_length=80,probe_steps=[2],path_points=5,minimum_free_gib=0,max_output_gib=2,hours=.5,checkpoint_every=1,acquisition_min_gain=-100.,acquisition_min_accuracy=0.,bootstrap_replicates=10,label_models=['tiny_llama'],appendix_b_steps=4,shampoo_block=8,shampoo_root_frequency=2,smoke=True)
 s['lr_grid']={k:[.001] for k in s['lr_grid']};return s

def validate(s):
 if s['version']!='optimizer-generality-1':raise ValueError('Use experiment.defaults()')
 for key in ['a_steps','b_steps','checkpoint_every','calibration_a_steps','calibration_b_steps','eval_batch','microbatch','accumulation','text_valid','text_train','text_test','calibration_eval_every','shampoo_block','shampoo_root_frequency']:
  if s[key]<1:raise ValueError('Positive '+key+' required')
 if s['path_points']<5 or (s['path_points']-1)%4:raise ValueError('Use path_points=5,9,17,... for nested Simpson audit')
 if len(s['audit_eps'])<2 or not all(x>0 for x in s['audit_eps']) or sorted(s['audit_eps'],reverse=True)!=s['audit_eps']:raise ValueError('Provide at least two decreasing positive audit widths')
 if any(x<1 or x>s['b_steps'] for x in s['probe_steps']):raise ValueError('Probe step outside main trajectory')
 if len(set(s['seeds']))!=len(s['seeds']) or s['calibration_seed'] in s['seeds'] or s['appendix_seed'] in s['seeds']:raise ValueError('Calibration, main and appendix seeds must be distinct')
 if not s['smoke'] and set(s['seeds']) & {1,2,3,11,12,13}:raise ValueError('Use fresh confirmation seeds')
 if not set(s['optimizers'])<={'sgd','momentum_sgd','adam','adam_beta1_0'}:raise ValueError('Unknown primary optimizer')
 if not set(s['appendix_optimizers'])<={'shampoo_block','muon_fp32'}:raise ValueError('Unknown appendix variant')
 for opt in s['optimizers']+s['appendix_optimizers']:
  if not s['lr_grid'][opt] or any(v<=0 for v in s['lr_grid'][opt]):raise ValueError('Positive learning rates required')
 if len({len(s['lr_grid'][opt]) for opt in s['optimizers']})>1:raise ValueError('Primary optimizers need equal candidate counts')
 if not 0<=s['momentum']<1 or not 0<s['beta2']<1:raise ValueError('Invalid momentum/beta2')
 if not s['models'] or len({m['name'] for m in s['models']})!=len(s['models']):raise ValueError('Unique model names required')
 if any(x not in {m['name'] for m in s['models']} for x in s['label_models']):raise ValueError('Unknown label-control model')

def science(s):return {k:v for k,v in s.items() if k not in ['hours','minimum_free_gib','max_output_gib','python','threads','checkpoint_every']}
def status(s):
 p=Path(s['output'] if isinstance(s,dict) else s);v=read(p/'status.json',{'status':'not_started'});v['alive']=alive(p)
 if 'last_progress' in v:v['seconds_since_worker_progress']=round(time.time()-v['last_progress'],1)
 if not v['alive'] and v.get('status') in ['running','launching'] and v.get('seconds_since_worker_progress',0)>30:v['recorded_status']=v['status'];v['status']='not_running'
 if p.exists():v['free_gib']=round(shutil.disk_usage(p).free/2**30,1)
 return v

def launch(s):
 validate(s);s=copy.deepcopy(s);p=Path(s['output']);p.mkdir(parents=True,exist_ok=True)
 with open(p/'launch.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if alive(p):return status(s)
  st=read(p/'status.json',{})
  if st.get('status')=='launching' and time.time()-st.get('last_progress',0)<30:return status(s)
  old=read(p/'settings.json')
  if old and science(old)!=science(s):raise ValueError('Scientific settings changed. restart(SETTINGS) creates a new directory; old results remain intact.')
  h=hashes(HERE);previous=read(p/'code_hashes.json')
  if previous and previous!=h:raise ValueError('Code/data configuration changed; restart in a new directory')
  write(p/'settings.json',s);write(p/'code_hashes.json',h)
  if st.get('status') in ['complete','complete_with_limitations']:return status(s)
  (p/'STOP').unlink(missing_ok=True);write(p/'status.json',dict(status='launching',last_progress=time.time()))
  with open(p/'worker.log','a') as log:child=subprocess.Popen([s['python'],str(HERE/'worker.py'),'--settings',str(p/'settings.json')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  return dict(status='launching',pid=child.pid,output=str(p))

def stop(s):
 p=Path(s['output']);p.mkdir(parents=True,exist_ok=True);(p/'STOP').touch();return 'Stop requested. Wait for alive=False; Launch/resume retains completed work.'
def restart(s):
 if alive(s['output']):raise RuntimeError('Stop the existing worker before starting a fresh run')
 n=copy.deepcopy(s);n['output']=s['output']+'_restart_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4];launch(n);return n

def analyze_existing(s):
 from timing_analysis import run_sources
 return run_sources(s,Path(s['output'])/'timing_existing')
def show_results(s):
 p=Path(s['output']);f=p/'REPORT.txt'
 print(f.read_text() if f.exists() else 'No completed report yet. Refresh status.')
 if (p/'overview.png').exists():
  from IPython.display import display,Image
  display(Image(filename=str(p/'overview.png')))
def export(s):
 p=Path(s['output']);dest=p/'optimizer_generality_share.zip';tmp=dest.with_suffix('.tmp')
 with tempfile_dir() as d:
  snapshot=Path(d)/'results.sqlite'
  if (p/'results.sqlite').exists():
   with sqlite3.connect('file:'+str((p/'results.sqlite').resolve())+'?mode=ro',uri=True) as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
   if snapshot.exists():z.write(snapshot,'results.sqlite')
   for f in p.rglob('*'):
    if not f.is_file() or f==tmp or f.suffix not in ['.json','.txt','.png','.pdf']:continue
    if 'data' in f.relative_to(p).parts:continue
    z.write(f,str(f.relative_to(p)))
   for f in HERE.iterdir():
    if f.is_file() and f.suffix in ['.py','.json','.gz','.md']:z.write(f,'code/'+f.name)
 os.replace(tmp,dest);return str(dest)
def tempfile_dir():
 import tempfile;return tempfile.TemporaryDirectory()
