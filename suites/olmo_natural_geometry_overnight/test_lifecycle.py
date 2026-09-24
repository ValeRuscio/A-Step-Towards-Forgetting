import sys,time,tempfile,json,sqlite3
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.resolve()))
import natural_geometry as n
import torch
from association_core import Config,Engine,make_data,digest
with tempfile.TemporaryDirectory() as td:
    root=Path(td);src=root/'source';src.mkdir();(src/'seed1').mkdir()
    c=Config(smoke=True,device='cpu',threads=1,entities=8,max_length=96,microbatch=2,accumulation=1,b_steps=128,seeds=[1])
    e=Engine(c);torch.save(e.pack(),src/'seed1/anchor.pt');n.atomic_json(c.__dict__,src/'config.json');n.atomic_json(dict(hash=digest(make_data(e.tok,c,1))),src/'seed1/data.json')
    s=n.defaults(src,root/'run');s.update(device='cpu',threads=1,hours=1,monitor_entities=2,topology_entities=4,end_step=128,geometry_every=2,curvature_every=0,minimum_free_gb=0,topology_nulls=0,matrix_nulls=0,matrix_sample_rows=8,matrix_sample_cols=16)
    p=n.launch(s);assert n.alive(p);assert n.launch(s)==p
    deadline=time.time()+45
    while not n.records(p) and time.time()<deadline:time.sleep(.1)
    assert n.records(p)
    n.stop(p)
    while n.alive(p) and time.time()<deadline:time.sleep(.1)
    assert not n.alive(p);st=n.status(p);assert st['status']=='paused',st
    checkpoint=torch.load(Path(p)/'seed1/resume.pt',weights_only=True);assert checkpoint['step']>=1
    n.launch(s);time.sleep(.1);n.stop(p)
    deadline=time.time()+45
    while n.alive(p) and time.time()<deadline:time.sleep(.1)
    assert not n.alive(p)
    s['end_step']=16;new=n.launch(s);assert new!=p and Path(p).exists();n.stop(new)
    deadline=time.time()+45
    while n.alive(new) and time.time()<deadline:time.sleep(.1)
    assert not n.alive(new)
    print('PASS: detached launch, idempotent launch, cooperative stop, resume, safe settings changes.')
