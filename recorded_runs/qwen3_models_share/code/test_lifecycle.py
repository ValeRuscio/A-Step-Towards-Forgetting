"""Test detached launch, duplicate-start guard, stop/resume and changed settings."""
import time,tempfile
from pathlib import Path
import torch
import precise_forgetting as p
from association_core import Config,Engine,make_data,digest


def wait_stopped(out,seconds=60):
    deadline=time.time()+seconds
    while p.ng.alive(out) and time.time()<deadline:time.sleep(.1)
    assert not p.ng.alive(out),p.status(out)


def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);src=root/'source';src.mkdir();(src/'seed1').mkdir()
        c=Config(smoke=True,device='cpu',threads=1,entities=4,max_length=96,microbatch=2,accumulation=1,b_steps=8,seeds=[1])
        e=Engine(c);torch.save(e.pack(),src/'seed1/anchor.pt');p.atomic_json(c.__dict__,src/'config.json');p.atomic_json(dict(hash=digest(make_data(e.tok,c,1))),src/'seed1/data.json')
        s=p.defaults(src,root/'run');s.update(jobs=[dict(seed=1,step=1),dict(seed=1,step=2)],device='cpu',threads=1,hours=1,minimum_free_gb=0,topology_landmarks=2,path_levels=[3,5],fd_entities_per_split=1)
        out=p.launch(s);assert p.ng.alive(out);assert p.launch(s)==out;p.stop(out);wait_stopped(out)
        assert p.status(out)['status']=='paused',p.status(out)
        p.launch(s);wait_stopped(out)
        assert p.status(out)['status']=='complete',p.status(out)
        s['jobs']=[dict(seed=1,step=2)];new=p.launch(s);assert new!=out;p.stop(new);wait_stopped(new)
        assert Path(out).exists();print('PASS: detached start, duplicate launch, cooperative pause, resume, settings-change sibling.')


if __name__=='__main__':main()
