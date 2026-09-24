"""Notebook API: no torch import, one resumable background pipeline."""
import copy,fcntl,hashlib,json,os,shutil,sqlite3,subprocess,sys,tempfile,time,uuid,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,p)
def defaults(output=None):
    return dict(version='qwen3-history-replication-1',output=str(Path(output or HERE/'runs/qwen3_history_v1').resolve()),python=str(HERE/'.venv/bin/python'),hours=23.5,device='cuda:0',threads=8,minimum_free_gib=50.,max_output_gib=100.,seeds=[61,62,63],learning_rate=2e-5,run_paths=True,smoke=False)
def alive(p):
    p=Path(p)
    if not p.exists():return False
    with open(p/'pipeline.lock','a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True

def code_hashes():
    files=list(HERE.glob('*.py'))+list((HERE/'natural').glob('*.py'))+list((HERE/'history').glob('*.py'))
    return {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
def signature(s):return {k:v for k,v in s.items() if k not in ['hours','threads','minimum_free_gib','max_output_gib','python']}
def status(s):
    p=Path(s['output']);r=read(p/'status.json',{'status':'not_started'});r['alive']=alive(p)
    stage=r.get('stage');sub='history' if stage=='history' else 'natural'
    st=read(p/sub/'status.json') if stage else None
    if st:
        r['worker']=st
        r['seconds_since_worker_progress']=round(time.time()-st.get('last_progress',time.time()),1)
    if p.exists():r['free_gib']=round(shutil.disk_usage(p).free/2**30,1)
    if r['status'] in ['running','launching'] and not r['alive'] and time.time()-r.get('last_progress',0)>30:r['status']='not_running'
    for sub in ['natural','history']:
        f=p/sub/'results.sqlite'
        if f.exists():
            with sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True) as c:r[sub+'_saved_units']=dict(c.execute('SELECT kind,COUNT(*) FROM records GROUP BY kind'))
    return r

def launch(s):
    s=copy.deepcopy(s);p=Path(s['output']);p.mkdir(parents=True,exist_ok=True)
    if not Path(s['python']).is_file():raise FileNotFoundError('Run setup_gh200.py first, or set SETTINGS["python"] to the installed environment.')
    if s['hours']<=0 or len(set(s['seeds']))!=len(s['seeds']) or not s['seeds']:raise ValueError('Positive hours and distinct seeds required')
    if not s['smoke'] and set(s['seeds'])&{1,2,3,11,12,13,21,22,23,31,41,42,43,51,52,53}:raise ValueError('Use fresh confirmation seeds')
    with open(p/'launch.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if alive(p):return status(s)
        oldstatus=read(p/'status.json',{})
        if oldstatus.get('status')=='launching' and time.time()-oldstatus.get('last_progress',0)<30:return status(s)
        old=read(p/'settings.json')
        if old and signature(old)!=signature(s):raise ValueError('Scientific settings differ. Use a new output directory; old results are preserved.')
        oldhash=read(p/'code_hashes.json');newhash=code_hashes()
        if oldhash and oldhash!=newhash:raise ValueError('Code changed: use the original package to resume, or a new output directory.')
        if oldstatus.get('status') in ['complete','complete_with_limitations']:return status(s)
        write(p/'settings.json',s);write(p/'code_hashes.json',newhash);(p/'STOP').unlink(missing_ok=True)
        write(p/'status.json',dict(status='launching',last_progress=time.time()))
        with open(p/'pipeline.log','a') as log:subprocess.Popen([s['python'],str(HERE/'pipeline.py'),'--settings',str(p/'settings.json')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return status(s)
def stop(s):
    p=Path(s['output']);p.mkdir(parents=True,exist_ok=True);(p/'STOP').touch()
    for sub in ['natural','history']:
        if (p/sub).exists():(p/sub/'STOP').touch()
    return 'Stop requested. Wait for alive=False. Launch/resume continues saved work.'
def show_results(s):
    p=Path(s['output'])
    for sub in ['natural','history']:
        f=p/sub/'REPORT.txt'
        if f.exists():print('\n'+sub.upper()+'\n'+f.read_text())
    from IPython.display import display,Image
    for sub in ['natural','history']:
        f=p/sub/'overview.png'
        if f.exists():display(Image(filename=str(f)))
def export(s):
    p=Path(s['output']);dest=p/'qwen3_history_share.zip'
    if not any((p/sub/'results.sqlite').exists() for sub in ['natural','history']):raise RuntimeError('No measurements yet')
    with open(p/'export.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory() as td,zipfile.ZipFile(str(dest)+'.tmp','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for sub in ['natural','history']:
                f=p/sub/'results.sqlite'
                if not f.exists():continue
                snap=Path(td)/(sub+'.sqlite')
                with sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True) as a,sqlite3.connect(snap) as b:a.backup(b)
                z.write(snap,sub+'/results.sqlite')
            for f in p.rglob('*'):
                if f.is_file() and f.suffix in ['.json','.txt','.png'] and 'data' not in f.relative_to(p).parts:z.write(f,str(f.relative_to(p)))
            for name in code_hashes():z.write(HERE/name,'code/'+name)
        os.replace(str(dest)+'.tmp',dest)
    return str(dest)
