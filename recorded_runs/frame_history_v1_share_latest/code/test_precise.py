"""Offline validation on tiny original OLMo; no downloads."""
import tempfile,json,copy
from pathlib import Path
import numpy as np
import torch
import precise_forgetting as p
from association_core import Config,Engine,make_data,digest,minibatches


def test_all():
    # Balanced crossed-design decomposition must exactly partition variance.
    x=np.array([[1.,2,3,4],[4.,2,0,1]])
    d=p.paired_components(x,[0,1,2,3]);np.testing.assert_allclose(d['variance_prompt']+d['variance_entity']+d['variance_interaction'],d['total_variance'])
    # Simpson integrates the derivative of alpha^4 exactly on a uniform 5-grid.
    a=np.linspace(0,1,5);np.testing.assert_allclose(p.simpson(4*a**3),1.,atol=1e-14)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);src=root/'source';src.mkdir();(src/'seed1').mkdir()
        c=Config(smoke=True,device='cpu',threads=1,entities=4,max_length=96,microbatch=2,accumulation=1,b_steps=8,seeds=[1],lr=.0002)
        e=Engine(c);data=make_data(e.tok,c,1)
        for t in [1,2]:e.gradient(minibatches(data,'A',1,t,c));e.opt.step()
        torch.save(e.pack(),src/'seed1/anchor.pt');p.atomic_json(c.__dict__,src/'config.json');p.atomic_json(dict(hash=digest(data)),src/'seed1/data.json')
        s=p.defaults(src,root/'run');s.update(jobs=[dict(seed=1,step=2)],device='cpu',threads=1,hours=1,minimum_free_gb=0,topology_landmarks=2,
            path_levels=[3,5],fd_entities_per_split=1,frame_rank=2,layers=[0,1])
        p.validate(s);out=Path(s['output']);out.mkdir();p.atomic_json(s,out/'settings.json');con=p.connect(out);ctl=p.Control(s)
        # Standard trajectory independent of diagnostics.
        ref=Engine(c);ref.restore(torch.load(src/'seed1/anchor.pt',weights_only=True))
        for t in [1,2]:ref.gradient(minibatches(data,'B',1,t,c));ref.opt.step();ref.opt.zero_grad(set_to_none=True)
        p.endpoints(e,s,s['jobs'][0],con,ctl)
        for n,v in ref.model.state_dict().items():torch.testing.assert_close(e.model.state_dict()[n],v,rtol=0,atol=0)
        opt=copy.deepcopy(e.opt.state_dict())
        p.path_level(e,s,s['jobs'][0],3,con,ctl);p.path_level(e,s,s['jobs'][0],5,con,ctl)
        for n,v in ref.model.state_dict().items():torch.testing.assert_close(e.model.state_dict()[n],v,rtol=0,atol=0)
        for i,st in opt['state'].items():
            for k,v in st.items():torch.testing.assert_close(e.opt.state_dict()['state'][i][k],v,rtol=0,atol=0)
        summary=p.read(out/'seed1_update002/path_summary.json');assert summary['grid_points']==5
        assert summary['fd_passes']==summary['fd_total'],summary
        for split,q in summary['populations'].items():
            assert q['n']==4
            for row in q['rows']:
                np.testing.assert_allclose(sum(row['path_contributions'].values()),row['integrated_slope'],rtol=1e-8,atol=1e-9)
                assert abs(row['integration_closure_error'])<1e-3,row
        # Independent population gradient equals mean of per-entity projections.
        data,before,after,dirs=p.prepare(e,s,s['jobs'][0],ctl);p.assign(e,before,after,0.)
        g=p.ng.mean_gradient(e,data['A_test'])
        exact=sum(p.ng.dot(g[n],dirs['delta'][n].cpu()) for n in g)
        stored=np.mean([r['projections']['delta'] for r in p.get_rows(con,'seed1_update002',0.,'A_test')])
        np.testing.assert_allclose(exact,stored,rtol=2e-4,atol=1e-7)
        p.assign(e,before,after,1.)
        # Stop mid-path always restores actual post-update weights; remove two cached interior rows.
        con.execute("DELETE FROM probes WHERE alpha=0.5 AND entity=0");con.commit()
        # Remove completion summary so cached-grid shortcut does not bypass the restoration test.
        (out/'seed1_update002/path_summary.json').unlink()
        class StopMid(p.Control):
            def pulse(self,**fields):
                if fields.get('phase')=='exact derivative along actual displacement':raise p.ng.Pause('Injected stop after intermediate assignment')
                return super().pulse(**fields)
        try:p.path_level(e,s,s['jobs'][0],5,con,StopMid(s))
        except p.ng.Pause:pass
        for n,v in ref.model.state_dict().items():torch.testing.assert_close(e.model.state_dict()[n],v,rtol=0,atol=0)
        # Resume incomplete saved probes; no duplicate records, all rows recovered.
        p.path_level(e,s,s['jobs'][0],5,con,ctl)
        assert con.execute('SELECT COUNT(*) FROM probes').fetchone()[0]==5*4*4
        p.report(out);p.export(out)
        assert (out/'matched_gradients.png').exists();assert (out/'within_update_paths.png').exists()
        assert not list((out/'seed1_update002/scratch').glob('*.npz'))
        for n,v in ref.model.state_dict().items():torch.testing.assert_close(e.model.state_dict()[n],v,rtol=0,atol=0)
        con.close()
        # Full scheduler idempotently resumes completed endpoint and path work.
        p.run(s);assert p.status(out)['status']=='complete',p.status(out)
        print('PASS: exact natural weights and optimizer; same-population gradient; per-entity quadrature/FD; channel integrals; paired variance; cached resume; plots/export.')


if __name__=='__main__':test_all()
