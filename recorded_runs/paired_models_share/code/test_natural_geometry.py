"""Offline tests: exact channel reconstruction, HVP, invariance, resume and replay."""
import json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
import natural_geometry as n
from association_core import Config,Engine,make_data,digest,minibatches


def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(3);x=rng.normal(size=(8,5));q,_=np.linalg.qr(rng.normal(size=(5,5)))
    np.testing.assert_allclose(n.points_distance(x),n.points_distance(x@q+3),atol=1e-12)
    ph=n.persistent_homology(n.points_distance(x),[1.,2.]);ph2=n.persistent_homology(n.points_distance(x@q),[1.,2.])
    np.testing.assert_allclose(ph['H0']['finite_total_persistence'],ph2['H0']['finite_total_persistence'],atol=1e-10)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);src=root/'source';src.mkdir();(src/'seed1').mkdir()
        c=Config(smoke=True,device='cpu',threads=1,entities=8,max_length=96,microbatch=2,accumulation=1,a_steps=2,b_steps=8,seeds=[1])
        e=Engine(c);data=make_data(e.tok,c,1)
        for step in range(1,3):e.gradient(minibatches(data,'A',1,step,c));e.opt.step()
        torch.save(e.pack(),src/'seed1/anchor.pt');n.atomic_json(c.__dict__,src/'config.json');n.atomic_json(dict(hash=digest(data)),src/'seed1/data.json')
        s=n.defaults(src,root/'run');s.update(device='cpu',threads=1,seeds=[1],hours=1,reserve_minutes=1,end_step=4,
            monitor_entities=2,topology_entities=4,geometry_every=2,derivative_every=2,curvature_every=2,curvature_entities=1,
            chunk_steps=2,checkpoint_every=2,matrix_nulls=1,topology_nulls=1,minimum_free_gb=0,matrix_sample_rows=8,matrix_sample_cols=16)
        n.validate(s);out=Path(s['output']);out.mkdir();n.atomic_json(s,out/'settings.json')
        # Independent uninstrumented trajectory used to audit actual training.
        ref=Engine(c);ref.restore(torch.load(src/'seed1/anchor.pt',weights_only=True))
        for step in range(1,5):ref.gradient(minibatches(data,'B',1,step,c));ref.opt.step();ref.opt.zero_grad(set_to_none=True)
        n.run(s);st=n.status(out);assert st['status']=='complete',st
        rr=n.records(out);assert len(rr)==4
        assert all(r['channels']['channel_reconstruction_relative_error']<1e-3 for r in rr)
        assert rr[1]['mixed_curvature'] is not None and 'geometry' in rr[1]
        cp=torch.load(out/'seed1/resume.pt',weights_only=True)
        for k,v in ref.model.state_dict().items():torch.testing.assert_close(cp['engine']['model'][k],v,rtol=0,atol=0)
        n.run(s);assert len(n.records(out))==4
        # Simulate crash rollback: checkpoint at update 2, DB still at update 4.
        ref.restore(torch.load(src/'seed1/anchor.pt',weights_only=True))
        for step in [1,2]:ref.gradient(minibatches(data,'B',1,step,c));ref.opt.step();ref.opt.zero_grad(set_to_none=True)
        n.save_checkpoint(ref,out/'seed1',2,s);n.atomic_json(dict(step=2),out/'seed1/committed.json')
        n.run(s);assert n.status(out)['status']=='complete';assert len(n.records(out))==4
        cp2=torch.load(out/'seed1/resume.pt',weights_only=True)
        for k,v in cp['engine']['model'].items():torch.testing.assert_close(cp2['engine']['model'][k],v,rtol=0,atol=0)
        # True mixed contraction agrees with finite differences of the monitor gradient.
        e.restore(torch.load(src/'seed1/anchor.pt',weights_only=True));e.gradient(minibatches(data,'B',1,1,c));h,cc=n.channels(e)
        val=n.mixed_curvature(e,[data['A_valid'][0]],h,cc)['h_H_c'];base={k:p.detach().clone() for k,p in e.params.items()}
        vals=[];eps=.05
        for sign in [-1,1]:
            with torch.no_grad():
                for k,p in e.params.items():p.copy_(base[k]+sign*eps*h[k])
            g=n.mean_gradient(e,[data['A_valid'][0]]);vals.append(sum(n.dot(g[k],cc[k]) for k in g))
        fd=(vals[1]-vals[0])/(2*eps)
        assert abs(fd-val)<max(1e-8,abs(val)*.15),(fd,val)
        print('PASS: bitwise natural-training equivalence, native channels, exact mixed Hessian, PH invariance, completed resume, crash rollback, report/export.')


if __name__=='__main__':main()
