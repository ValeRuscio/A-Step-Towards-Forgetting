"""Standard-library notebook controls; explicit interpreters, quiet background workers."""
from pathlib import Path
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback,zipfile,sqlite3
from open_assets import atomic
class Pause(Exception):pass
def alive(root):
 p=Path(root)/'worker.lock'
 if not p.exists():return False
 with p.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
  except BlockingIOError:return True

def status(root):
 root=Path(root);p=root/'status.json';r=json.loads(p.read_text()) if p.exists() else dict(status='not_started')
 r['alive']=alive(root)
 if 'last_progress' in r:r['seconds_since_progress']=round(time.time()-r['last_progress'],1)
 if root.exists():r['free_gib']=round(shutil.disk_usage(root).free/1024**3,1)
 r['cases']={}
 for db in root.glob('runs/*/seed*/measurements.sqlite'):
  with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as c:
   r['cases'][str(db.parent.relative_to(root))]=dict(updates=c.execute('select count(*) from trajectory').fetchone()[0],path_points=c.execute('select count(*) from path').fetchone()[0])
 if (root/'readiness.json').exists():r['readiness']=json.loads((root/'readiness.json').read_text())
 print(json.dumps(r,indent=2));return r

def stop(root):
 Path(root).mkdir(parents=True,exist_ok=True);(Path(root)/'STOP').touch();print('Stop requested. Wait for alive=False. Qwen resumes saved work; the OPEN pilot replays an unverified interval from its original checkpoint.')

def start_pilot(s,python,stage='native'):
 root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
 with (root/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if alive(root):print('Already running');return
  prior=root/'settings.json';runtime={'hours','minimum_free_gib'}
  if prior.exists():
   old=json.loads(prior.read_text())
   if {k:v for k,v in old.items() if k not in runtime}!={k:v for k,v in s.items() if k not in runtime}:raise ValueError('Scientific pilot settings changed. Choose a new OUTPUT_OPEN directory to preserve the existing interval.')
  if stage=='native' and (root/'replay_result.json').exists() and json.loads((root/'replay_result.json').read_text()).get('match') is True:print('Native replay already verified. Launch proxy audits next.');return
  atomic(s,prior);(root/'STOP').unlink(missing_ok=True);code=root/'code';code.mkdir(exist_ok=True)
  for p in Path(__file__).parent.glob('*.py'):
   if p.resolve()!=(code/p.name).resolve():shutil.copy2(p,code/p.name)
  atomic(dict(status='launching',stage=stage,last_progress=time.time()),root/'status.json')
  with (root/'worker.log').open('a') as log:
   subprocess.Popen([str(python),str(code/'controls.py'),'--worker',str(prior),'--stage',stage],stdout=log,stderr=log,start_new_session=True)
 print('Launched '+stage+' stage. Refresh for status.')

def start_qwen(s,python):
 root=Path(s['output']);root.mkdir(parents=True,exist_ok=True);request=root/'launch_request.json'
 if (root/'settings.json').exists():
  old=json.loads((root/'settings.json').read_text());runtime={'hours','reserve_minutes','minimum_free_gb','max_output_gb','audit_fraction','replication_fraction','output','chunk_steps'}
  if {k:v for k,v in old.items() if k not in runtime}!={k:v for k,v in s.items() if k not in runtime}:raise ValueError('Scientific Qwen settings changed. Choose a new OUTPUT_QWEN directory before launch.')
 atomic(s,request)
 code='import json,paired_suite; paired_suite.launch(json.load(open(__import__("sys").argv[1])))'
 r=subprocess.run([str(python),'-c',code,str(request)],cwd=Path(__file__).parent,text=True,capture_output=True)
 print(r.stdout)
 if r.returncode:raise RuntimeError(r.stderr[-5000:])

def export(root):
 root=Path(root);dest=root.parent/(root.name+'_share.zip')
 if alive(root):raise RuntimeError('Stop and wait for alive=False before exporting the native pilot.')
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
  for p in root.rglob('*'):
   if p.is_file() and 'assets' not in p.relative_to(root).parts and p.suffix in ['.json','.txt','.py','.png']:
    z.write(p,p.relative_to(root))
 print(dest);return str(dest)

def worker(s,stage):
 root=Path(s['output']);start=time.time();last=0
 with (root/'worker.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  def pulse(**kw):
   nonlocal last
   if (root/'STOP').exists():raise Pause('Stop requested; unfinished native interval must be replayed from its original checkpoint.')
   if time.time()-start>s['hours']*3600:raise Pause('Session budget reached; restart the unfinished interval or increase the session budget.')
   if shutil.disk_usage(root).free<s['minimum_free_gib']*1024**3:raise Pause('Disk reserve reached')
   if time.time()-last>1:
    atomic(dict(status='running',stage=stage,last_progress=time.time(),**kw),root/'status.json');last=time.time()
  try:
   pulse(phase='checking environment')
   if stage=='native':
    from open_pilot import run
   else:
    from proxy_audit import run
   result=run(s,pulse)
  except Pause as exc:result=dict(status='paused',message=str(exc))
  except Exception:result=dict(status='failed',message=traceback.format_exc())
  atomic(dict(result,last_progress=time.time(),stage=stage),root/'status.json')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',required=True);p.add_argument('--stage',choices=['native','proxy'],required=True);a=p.parse_args();worker(json.loads(Path(a.worker).read_text()),a.stage)
