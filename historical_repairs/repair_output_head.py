"""Narrow, audited compatibility repair for natural_forgetting_components v1.
Upload beside the notebook; run: %run repair_output_head.py
Then run the existing Launch/resume cell. Does not restart or launch training.
"""
import argparse,fcntl,hashlib,json,os,sqlite3,time
from pathlib import Path

ORIGINAL={
 'study_math.py':'577ff61caef3a7de931c2c2e4d324582b7b42fcf79ec7bdf4c221692d8565843',
 'multimodel_engine.py':'b927e1d521f2b60105cd5c636021e9d42d439de8846903def264dc7afaada8c8',
}
REPLACEMENTS={
 "self.model.embed_out if self.kind=='gpt_neox' else self.model.lm_head":"self.model.get_output_embeddings()",
 "e.model.embed_out if e.kind=='gpt_neox' else e.model.lm_head":"e.model.get_output_embeddings()",
}
def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(p,data):
 p=Path(p);tmp=p.with_name(p.name+'.headfix.tmp');tmp.write_bytes(data);os.replace(tmp,p)
def jswrite(p,x):atomic(p,json.dumps(x,indent=2).encode())
def hashes(root):return {p.name:sha(p.read_bytes()) for p in root.iterdir() if p.is_file() and p.suffix in ['.py','.json','.gz']}

def repair(project,output):
 project=Path(project).resolve();output=Path(output).resolve();manifest=output/'code_hashes.json'
 if not manifest.exists():raise RuntimeError('No saved run manifest at '+str(output)+'. Supply --output with the existing run folder.')
 with open(output/'launch.lock','a') as launch,open(output/'worker.lock','a') as worker:
  for f in [launch,worker]:
   try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('Worker/launcher is active. Stop it and wait for alive=False before repair.')
  saved=json.loads(manifest.read_text());current=hashes(project);selfname=Path(__file__).name
  for n in set(saved)|set(current):
   if n in ORIGINAL or n==selfname:continue
   if saved.get(n)!=current.get(n):raise RuntimeError('Unrelated code difference: '+n+'. Repair refused; results untouched.')
  changes={};oldbytes={}
  for n,expected in ORIGINAL.items():
   data=(project/n).read_bytes();text=data.decode()
   if sha(data)==expected:original=data
   else:
    # Recognize this exact patch only, including a partially completed previous repair.
    original=text
    for old,new in REPLACEMENTS.items():original=original.replace(new,old)
    original=original.encode()
    if sha(original)!=expected:raise RuntimeError('Unrecognized source version: '+n+'. Repair refused.')
   patched=original.decode()
   for old,new in REPLACEMENTS.items():patched=patched.replace(old,new)
   patched=patched.encode();compile(patched,str(project/n),'exec')
   if saved.get(n) not in [expected,sha(patched)]:raise RuntimeError('Saved manifest does not match supported version: '+n)
   oldbytes[n]=original;changes[n]=patched
  # This repair is for a run that failed before Pythia acquisition, as in the reported traceback.
  db=output/'results.sqlite'
  if db.exists():
   with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
    count=con.execute("SELECT COUNT(*) FROM records WHERE job LIKE 'natural/pythia410m/%' AND kind IN ('anchor','step','path_done')").fetchone()[0]
    completed=con.execute("SELECT COUNT(*) FROM records WHERE kind='case_done'").fetchone()[0]
   if count:raise RuntimeError('Pythia already has measurements; this narrowly scoped migration refuses to mix implementations.')
  else:completed=0
  audit=output/'compatibility_repairs'/'output_head_api_v1';audit.mkdir(parents=True,exist_ok=True)
  if not (audit/'original_code_hashes.json').exists():jswrite(audit/'original_code_hashes.json',saved)
  for n,data in oldbytes.items():
   backup=audit/(n+'.original')
   if not backup.exists():atomic(backup,data)
  for n,data in changes.items():atomic(project/n,data)
  updated=hashes(project)
  jswrite(audit/'repair.json',dict(repair='output-head public accessor',timestamp=time.time(),reason='GPTNeoX output projection attribute differs between Transformers versions',before={n:sha(x) for n,x in oldbytes.items()},after={n:sha(x) for n,x in changes.items()},completed_cases_preserved=completed,settings_unchanged=True,results_database_unchanged=True,numerical_intent='Same output projection; use get_output_embeddings() instead of a fixed attribute name.'))
  jswrite(manifest,updated)
  print('Compatibility repair applied. Completed cases preserved:',completed)
  print('Existing run:',output)
  print('Run the notebook Launch/resume cell. Do NOT restart the experiment.')
 return str(output/'settings.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',default=str(Path(__file__).resolve().parent));p.add_argument('--output');a=p.parse_args()
 REPAIRED_SETTINGS=repair(a.project,a.output or str(Path(a.project)/'runs'/'natural_forgetting_components_v1'))
