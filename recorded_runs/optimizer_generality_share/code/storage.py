"""Atomic metadata, immutable configuration, one live worker and bounded checkpoints."""
import os,json,time,uuid,fcntl,sqlite3,shutil,hashlib
from pathlib import Path

def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));os.replace(tmp,p)
def alive(out):
 out=Path(out)
 if not out.exists():return False
 with open(out/'worker.lock','a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);return False
  except BlockingIOError:return True
class Paused(Exception):pass
class Control:
 def __init__(self,s):self.s=s;self.out=Path(s['output']);self.deadline=time.monotonic()+s['hours']*3600;self.fields={}
 def check(self):
  if (self.out/'STOP').exists():raise Paused('Stop requested; resume preserves completed units')
  if time.monotonic()>self.deadline:raise Paused('Session time budget reached; launch again to resume')
  if shutil.disk_usage(self.out).free<self.s['minimum_free_gib']*2**30:raise Paused('Disk reserve reached')
 def pulse(self,**kw):
  if 'job' in kw and kw['job']!=self.fields.get('job'):self.fields={k:v for k,v in self.fields.items() if k in ['completed_jobs','total_jobs']}
  if kw.get('phase')!=self.fields.get('phase'):
   for k in ['alpha','split','epsilon','condition']:self.fields.pop(k,None)
  self.fields.update(kw);write(self.out/'status.json',dict(status='running',last_progress=time.time(),**self.fields))
 def checkpoint(self,x,p):
  import torch
  self.check()
  def size(v):
   if isinstance(v,torch.Tensor):return v.numel()*v.element_size()
   if isinstance(v,dict):return sum(size(a) for a in v.values())
   if isinstance(v,(list,tuple)):return sum(size(a) for a in v)
   return 0
  required=size(x)*1.15
  if shutil.disk_usage(self.out).free<required+self.s['minimum_free_gib']*2**30:raise Paused('Insufficient space for atomic checkpoint')
  used=sum(a.stat().st_size for a in self.out.rglob('*') if a.is_file())
  if used+required>self.s['max_output_gib']*2**30:raise Paused('Output storage cap reached')
  p=Path(p);tmp=p.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,p)
def db_open(out):
 c=sqlite3.connect(Path(out)/'results.sqlite',timeout=60);c.execute('PRAGMA journal_mode=WAL');c.execute('CREATE TABLE IF NOT EXISTS records(job TEXT, step INTEGER, kind TEXT, record TEXT, PRIMARY KEY(job,step,kind))');c.commit();return c
def put(c,j,t,k,r):c.execute('INSERT OR REPLACE INTO records VALUES(?,?,?,?)',(j,t,k,json.dumps(r,allow_nan=False)));c.commit()
def get(c,j,k):return [json.loads(r[0]) for r in c.execute('SELECT record FROM records WHERE job=? AND kind=? ORDER BY step',(j,k))]
def hashes(root):return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(root).iterdir() if p.is_file() and p.suffix in ['.py','.json','.gz']}
