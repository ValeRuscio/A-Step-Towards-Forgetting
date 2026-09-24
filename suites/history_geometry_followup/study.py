"""Quiet notebook API; idempotent launch, cooperative pause, immutable science."""
import copy,fcntl,os,subprocess,sys,time,uuid,sqlite3,zipfile,tempfile,shutil
from pathlib import Path
from storage import read,write,alive,hashes
HERE=Path(__file__).resolve().parent

def defaults(output='runs/history_geometry_followup_v1'):
    return dict(version='history-geometry-1',output=str(Path(output).resolve()),python=sys.executable,device='cuda:0',threads=8,hours=23.,minimum_free_gib=40.,max_output_gib=60.,checkpoint_every=4,
        models=[dict(name='smollm2_135m',repo='HuggingFaceTB/SmolLM2-135M',revision='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'),dict(name='pythia410m',repo='EleutherAI/pythia-410m',revision='9879c9b5f8bea9051dcb0e68dff21493d67e9d4f')],
        datasets={'sst2':'8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb','ag_news':'eb185aade064a813bc0b7f42de02595523103ca4'},
        seeds=[41,42,43],include_smollm_momentum=True,
        learning_rates={'smollm2_135m':{'adam':8e-5,'momentum_sgd':.01},'pythia410m':{'adam':5e-6}},
        a_steps=600,b_steps=128,fork_steps=[8,32,64],branch_steps=32,
        path_seeds=[41],path_fork_steps=[8,32],path_initial_points=9,path_max_points=65,hessian_initial_points=5,hessian_max_points=17,
        audit_eps=[.02,.01,.005],audit_atol=2e-4,audit_rtol=.05,norm_match_rtol=.001,
        acquisition_min_gain=.1,acquisition_min_accuracy=.6,
        momentum=.9,beta2=.99,eps=1e-8,clip=1.,microbatch=2,accumulation=2,eval_batch=4,max_length=256,text_train=1024,text_valid=32,text_test=128,smoke=False)

def smoke_settings(output):
    s=defaults(output);s.update(models=[dict(name='tiny_llama',repo='tiny-llama',revision='smoke')],learning_rates={'tiny_llama':{'adam':.001,'momentum_sgd':.01}},include_smollm_momentum=True,
        seeds=[41],a_steps=3,b_steps=12,fork_steps=[2],branch_steps=3,path_fork_steps=[2],device='cpu',threads=1,microbatch=2,accumulation=1,eval_batch=2,text_train=8,text_valid=4,text_test=4,max_length=80,
        path_initial_points=5,path_max_points=9,hessian_initial_points=5,hessian_max_points=5,checkpoint_every=1,minimum_free_gib=0,max_output_gib=2,hours=1,acquisition_min_gain=-100.,acquisition_min_accuracy=0.,smoke=True)
    return s

def validate(s):
    if s['version']!='history-geometry-1':raise ValueError('Use study.defaults()')
    for k in ['a_steps','b_steps','branch_steps','checkpoint_every','eval_batch','microbatch','accumulation','text_train','text_valid','text_test']:
        if not isinstance(s[k],int) or s[k]<1:raise ValueError('Positive integer required: '+k)
    for k in ['path_initial_points','path_max_points','hessian_initial_points','hessian_max_points']:
        n=s[k]-1
        if n<4 or n&(n-1):raise ValueError(k+' must be 5,9,17,33,...')
    for prefix in ['path','hessian']:
        if s[prefix+'_max_points']<s[prefix+'_initial_points']:raise ValueError('Max resolution below initial')
    if not s['fork_steps'] or len(set(s['fork_steps']))!=len(s['fork_steps']) or any(t<1 or t>s['b_steps'] for t in s['fork_steps']):raise ValueError('Unique forks inside natural trajectory required')
    if not set(s['path_fork_steps'])<=set(s['fork_steps']):raise ValueError('Path forks must be scheduled forks')
    if len(set(s['seeds']))!=len(s['seeds']) or not set(s['path_seeds'])<=set(s['seeds']):raise ValueError('Unique seeds and valid path seeds required')
    if not s['smoke'] and set(s['seeds'])&{1,2,3,11,12,13,21,22,23,31}:raise ValueError('Use fresh seeds, not previously inspected development seeds')
    if s['hours']<=0 or len(s['audit_eps'])<2 or sorted(s['audit_eps'],reverse=True)!=s['audit_eps'] or min(s['audit_eps'])<=0:raise ValueError('Invalid budget or derivative widths')
    for m in s['models']:
        if not s['smoke'] and (len(m['revision'])!=40 or any(c not in '0123456789abcdef' for c in m['revision'])):raise ValueError('Pinned model revision required')
        if s['learning_rates'][m['name']]['adam']<=0:raise ValueError('Positive learning rate required')

def science(s):return {k:v for k,v in s.items() if k not in ['python','hours','threads','minimum_free_gib','max_output_gib','checkpoint_every']}
def status(s):
    p=Path(s['output'] if isinstance(s,dict) else s);r=read(p/'status.json',{'status':'not_started'});r['alive']=alive(p)
    if 'last_progress' in r:r['seconds_since_worker_progress']=round(time.time()-r['last_progress'],1)
    if not r['alive'] and r['status'] in ['running','launching'] and r.get('seconds_since_worker_progress',0)>30:r['recorded_status']=r['status'];r['status']='not_running'
    if p.exists():r['free_gib']=round(shutil.disk_usage(p).free/2**30,1)
    return r

def launch(s):
    validate(s);s=copy.deepcopy(s);p=Path(s['output']);p.mkdir(parents=True,exist_ok=True)
    with open(p/'launch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if alive(p):return status(s)
        st=read(p/'status.json',{})
        if st.get('status')=='launching' and time.time()-st.get('last_progress',0)<30:return status(s)
        old=read(p/'settings.json')
        if old and science(old)!=science(s):raise ValueError('Scientific settings changed. Use restart(SETTINGS) for a new directory; existing results are preserved.')
        h=hashes(HERE);previous=read(p/'code_hashes.json')
        if previous and previous!=h:raise ValueError('Code or frozen predictors changed. Use a new output directory.')
        write(p/'settings.json',s);write(p/'code_hashes.json',h)
        if st.get('status') in ['complete','complete_with_limitations']:return status(s)
        (p/'STOP').unlink(missing_ok=True);write(p/'status.json',dict(status='launching',last_progress=time.time()))
        with open(p/'worker.log','a') as log:
            child=subprocess.Popen([s['python'],str(HERE/'run_study.py'),'--settings',str(p/'settings.json')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        return dict(status='launching',pid=child.pid,output=str(p))
def stop(s):
    p=Path(s['output']);p.mkdir(parents=True,exist_ok=True);(p/'STOP').touch();return 'Stop requested. Wait for alive=False, then Launch/resume.'
def restart(s):
    if alive(s['output']):raise RuntimeError('Stop the active worker first; wait until alive=False.')
    n=copy.deepcopy(s);n['output']+='_restart_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4];launch(n);return n

def show_results(s):
    p=Path(s['output']);f=p/'REPORT.txt';print(f.read_text() if f.exists() else 'No completed results yet. Use status().')
    if (p/'overview.png').exists():
        from IPython.display import display,Image
        display(Image(filename=str(p/'overview.png')))
def analyze(s):
    # Completed natural trajectories only; safe read-only snapshot while worker runs.
    from reporting import report
    with tempfile.TemporaryDirectory() as d:
        snap=Path(d)/'results.sqlite'
        with sqlite3.connect('file:'+str(Path(s['output'])/'results.sqlite')+'?mode=ro',uri=True) as src,sqlite3.connect(snap) as dst:src.backup(dst)
        out=Path(s['output'])/'manual_analysis';out.mkdir(exist_ok=True)
        with sqlite3.connect(snap) as con:report(con,out,True)
    return str(out)
def export(s):
    p=Path(s['output']);dest=p/'history_geometry_share.zip';tmp=dest.with_suffix('.tmp')
    with tempfile.TemporaryDirectory() as d:
        snapshot=Path(d)/'results.sqlite'
        if (p/'results.sqlite').exists():
            with sqlite3.connect('file:'+str(p/'results.sqlite')+'?mode=ro',uri=True) as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
            if snapshot.exists():z.write(snapshot,'results.sqlite')
            for f in p.rglob('*'):
                if f.is_file() and f.suffix in ['.json','.txt','.png','.pdf'] and 'data' not in f.relative_to(p).parts:z.write(f,str(f.relative_to(p)))
            for f in HERE.iterdir():
                if f.is_file() and f.suffix in ['.py','.json','.gz','.md','.txt']:z.write(f,'code/'+f.name)
    os.replace(tmp,dest);return str(dest)
