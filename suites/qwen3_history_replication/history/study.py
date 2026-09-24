"""Quiet notebook API; idempotent launch, cooperative pause, immutable science."""
import copy,fcntl,os,subprocess,sys,time,uuid,sqlite3,zipfile,tempfile,shutil
from pathlib import Path
from storage import read,write,alive,hashes
HERE=Path(__file__).resolve().parent

def defaults(source='/home/ubuntu/9/runs/natural_forgetting_components_v1',output='runs/history_dynamics_replay_v1'):
    return dict(version='history-dynamics-1',source=str(Path(source).resolve()),output=str(Path(output).resolve()),python=sys.executable,device='cuda:0',hours=23.,threads=8,minimum_free_gib=40.,max_output_gib=60.,checkpoint_every=32,
                populations=['A_valid','A_test'],test_population=32,derivative_batch=1,norm_floor=1e-12,closure_atol=1e-8,closure_rtol=1e-6,age_residual_rtol=1e-4,event_threshold=.05,event_window=4)

def validate(s):
    if s['version']!='history-dynamics-1':raise ValueError('Use study.defaults()')
    source=Path(s['source']).resolve();out=Path(s['output']).resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('Replay output must be separate from original source, not nested in it')
    if not (source/'settings.json').exists():raise FileNotFoundError('Source settings.json not found: '+str(source))
    if not set(s['populations'])<=set(['A_valid','A_test']) or not s['populations']:raise ValueError('Choose A_valid and/or A_test')
    ss=read(source/'settings.json')
    if not 1<=s['test_population']<=ss['text_test']:raise ValueError('Invalid fixed held-out population size')
    if s['hours']<=0 or s['derivative_batch']<1 or s['checkpoint_every']<1:raise ValueError('Invalid budget or batch size')
    if s['norm_floor']<=0 or s['age_residual_rtol']<=0:raise ValueError('Invalid tolerance')

def science(s):return {k:v for k,v in s.items() if k not in ['python','hours','threads','minimum_free_gib','max_output_gib','checkpoint_every']}
def status(s):
    p=Path(s['output'] if isinstance(s,dict) else s);r=read(p/'status.json',{'status':'not_started'});r['alive']=alive(p)
    if 'last_progress' in r:r['seconds_since_worker_progress']=round(time.time()-r['last_progress'],1)
    if not r['alive'] and r['status'] in ['running','launching'] and r.get('seconds_since_worker_progress',0)>30:r['recorded_status']=r['status'];r['status']='not_running'
    if p.exists():r['free_gib']=round(shutil.disk_usage(p).free/2**30,1)
    if (p/'results.sqlite').exists():
        try:
            with sqlite3.connect((p/'results.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as c:
                r['saved_units']=dict(c.execute('SELECT kind,COUNT(*) FROM records GROUP BY kind'))
        except sqlite3.Error:pass
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
        if previous and previous!=h:raise ValueError('Code changed. Use a new output directory.')
        write(p/'settings.json',s);write(p/'code_hashes.json',h)
        if st.get('status') in ['complete','complete_with_limitations']:return status(s)
        (p/'STOP').unlink(missing_ok=True);write(p/'status.json',dict(status='launching',last_progress=time.time()))
        with open(p/'worker.log','a') as log:
            child=subprocess.Popen([s['python'],str(HERE/'run_replay.py'),'--settings',str(p/'settings.json')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
        with sqlite3.connect(snap) as con:report(con,out,s)
    return str(out)
def export(s):
    p=Path(s['output']);dest=p/'history_dynamics_share.zip';tmp=dest.with_suffix('.tmp')
    if not (p/'results.sqlite').exists():raise RuntimeError('No results database in this folder. Check SETTINGS[\'output\']; export will not create a metadata-only ZIP.')
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
