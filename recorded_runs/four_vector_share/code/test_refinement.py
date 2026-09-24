import json,math,tempfile,hashlib
from pathlib import Path
import numpy as np
import torch
import refine_paths as r
from association_core import Config,Engine,make_data,minibatches,digest


def record(a,width=.008):
    g=2/math.sqrt(math.pi)/width*math.exp(-((a-.53)/width)**2)
    return dict(loss=math.erf((a-.53)/width),projections=dict(delta=g,history=1.3*g,current=-.3*g,roundoff=0.))


def main():
    q=r.adaptive_integral(record,{float(a):record(a) for a in np.linspace(0,1,17)},1e-5,.005,257,12)
    assert q['quadrature_passed'],q
    assert abs(q['integral']-2)<1e-4,q['integral']
    np.testing.assert_allclose(sum(q['contributions'][k] for k in ['history','current','roundoff']),q['integral'],atol=1e-12)
    cap=r.adaptive_integral(record,{float(a):record(a) for a in np.linspace(0,1,17)},1e-12,1e-12,33,5)
    assert not cap['quadrature_passed'] and cap['cap_reached']
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);src=root/'source';src.mkdir();(src/'seed1').mkdir()
        c=Config(smoke=True,device='cpu',threads=1,entities=4,max_length=96,microbatch=2,accumulation=1,b_steps=8,seeds=[1],lr=.0002)
        e=Engine(c);data=make_data(e.tok,c,1)
        e.gradient(minibatches(data,'A',1,1,c));e.opt.step();torch.save(e.pack(),src/'seed1/anchor.pt')
        r.atomic_json(c.__dict__,src/'config.json');r.atomic_json(dict(hash=digest(data)),src/'seed1/data.json')
        old=r.p.defaults(src,root/'previous');old.update(jobs=[dict(seed=1,step=2)],device='cpu',threads=1,minimum_free_gb=0,topology_landmarks=2,path_levels=[3,5],fd_entities_per_split=1,hours=1)
        prev=Path(old['output']);prev.mkdir();r.atomic_json(old,prev/'settings.json');r.p.run(old)
        assert r.p.status(prev)['status']=='complete',r.p.status(prev)
        # Verify the previous database is read-only throughout the follow-up.
        original_hash=hashlib.sha256((prev/'measurements.sqlite').read_bytes()).hexdigest()
        s=r.defaults(prev,root/'new');s.update(jobs=[dict(seed=1,step=2)],device='cpu',threads=1,minimum_free_gb=0,hours=1,interior_geometry_points=1)
        out=Path(s['output']);out.mkdir();r.atomic_json(s,out/'settings.json');r.run(s)
        assert r.status(out)['status']=='complete',r.status(out)
        con=r.connect(out);rows=[json.loads(a[0]) for a in con.execute('SELECT record FROM refined')]
        assert len(rows)==16 and all(a['resolved'] for a in rows),rows
        assert con.execute('SELECT COUNT(*) FROM slice_geometry').fetchone()[0]>=2
        assert not list(out.glob('*/full_token_scratch'))
        con.close();r.run(s);assert r.status(out)['status']=='complete'
        assert hashlib.sha256((prev/'measurements.sqlite').read_bytes()).hexdigest()==original_hash
        assert (out/'path_refinement_share.zip').exists()
        print('PASS: sharp-path adaptive integration, explicit caps, channel-sum identity, matched FD audits, original-input immutability, interior geometry, resume and export.')


if __name__=='__main__':main()
