"""Pinned public artifacts. Never download the whole checkpoint trajectory."""
from pathlib import Path
import base64,hashlib,json,os,shutil,tarfile,urllib.request,urllib.parse
SOURCE_SHA='d7b7b674cd851f7a16815229f060658180cb772b'
HF_SHA='c76f21681ae429e3bc2d3978cd07796ccf62229a'
BUCKET='gensyn-open-1b'
def atomic(x,p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def get_json(url):
 with urllib.request.urlopen(url,timeout=120) as r:return json.load(r)
def download(url,p,md5=None,minimum_free_gib=30):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and (not md5 or digest(p,'md5')==md5):return
 tmp=p.with_suffix(p.suffix+'.part');h=hashlib.md5()
 with urllib.request.urlopen(url,timeout=120) as r,tmp.open('wb') as f:
  while True:
   b=r.read(8*1024**2)
   if not b:break
   if shutil.disk_usage(p.parent).free<minimum_free_gib*1024**3:raise RuntimeError('Disk reserve reached during download; partial file retained, retry restarts this file.')
   f.write(b);h.update(b)
 if md5 and base64.b64encode(h.digest()).decode()!=md5:raise ValueError('Download checksum mismatch: '+str(p))
 os.replace(tmp,p)
def digest(p,kind='sha256'):
 h=hashlib.new(kind)
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return base64.b64encode(h.digest()).decode() if kind=='md5' else h.hexdigest()
def objects(prefix):
 found=[];token=None
 while True:
  q=dict(prefix=prefix,maxResults=1000)
  if token:q['pageToken']=token
  d=get_json('https://storage.googleapis.com/storage/v1/b/'+BUCKET+'/o?'+urllib.parse.urlencode(q));found+=d.get('items',[]);token=d.get('nextPageToken')
  if not token:return found

def prepare(s,pulse=lambda **kw:None):
 root=Path(s['output']);assets=root/'assets';assets.mkdir(parents=True,exist_ok=True)
 tree=assets/'upstream';pin=assets/'source_manifest.json'
 if not tree.exists():
  tar=assets/'upstream.tar.gz';download('https://api.github.com/repos/gensyn-ai/open-transformers/tarball/'+SOURCE_SHA,tar)
  with tarfile.open(tar) as tf:
   members=tf.getmembers();top=members[0].name.split('/')[0]
   for m in members:
    if m.issym() or m.islnk() or '..' in Path(m.name).parts or m.name.startswith('/'):raise ValueError('Unsafe source archive member')
   tf.extractall(assets)
  (assets/top).rename(tree)
  atomic(dict(commit=SOURCE_SHA,tar_sha256=digest(tar)),pin)
 step=int(s['checkpoint_step']);prefix=f'ckpt/step_{step:09d}/';ck=assets/f'step_{step:09d}'
 manifest=assets/'checkpoint_manifest.json'
 if manifest.exists():items=json.loads(manifest.read_text())['objects']
 else:
  items=objects(prefix)
  if not items or not any(x['name'].endswith('/_COMPLETE') for x in items):raise ValueError('Checkpoint missing or incomplete')
  atomic(dict(prefix=prefix,objects=items),manifest)
 required=sum(int(x['size']) for x in items if not (ck/x['name'][len(prefix):]).exists())
 if shutil.disk_usage(assets).free-required<s['minimum_free_gib']*1024**3:raise RuntimeError('Insufficient space for selected checkpoint plus disk reserve')
 for i,x in enumerate(items):
  pulse(phase='fetching one native checkpoint',file=i,total=len(items))
  rel=x['name'][len(prefix):]
  if '..' in Path(rel).parts or Path(rel).is_absolute():raise ValueError('Unsafe checkpoint path')
  url=f'https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/'+urllib.parse.quote(x['name'],safe='')+'?alt=media&generation='+x['generation']
  download(url,ck/rel,x.get('md5Hash'),s['minimum_free_gib'])
 hashes=assets/'state_hashes.jsonl'
 if not hashes.exists():
  entries=objects('logs/state_hashes.jsonl');x=next(x for x in entries if x['name']=='logs/state_hashes.jsonl')
  download(f'https://storage.googleapis.com/{BUCKET}/logs/state_hashes.jsonl?generation='+x['generation'],hashes,x.get('md5Hash'),s['minimum_free_gib'])
 target=step+int(s['steps']);entries=[json.loads(line) for line in hashes.read_text().splitlines() if line.strip()];hits=[r for r in entries if r['step']==target]
 if len(hits)!=1:raise ValueError('No unique canonical target-step hash')
 atomic(dict(step=target,state_hash=hits[0]['state_hash'],source_sha256=digest(hashes)),assets/'target.json')
 return tree,ck,hits[0]['state_hash']
