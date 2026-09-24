import tempfile,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from unittest.mock import patch
import frame_history as s
from association_core import Config,Engine,make_data,minibatches,save,dump,digest

def test():
    rng=np.random.default_rng(3);x=rng.normal(size=(30,8));q,_=np.linalg.qr(rng.normal(size=(8,8)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    y=x@q;frame=s.fit_frame(x,y);z=s.rotate(x,frame['basis'],frame['rot'])
    assert np.max(np.abs(z-y))<1e-10
    assert np.max(np.abs(s.distance(z)-s.distance(x)))<1e-10
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve();src=root/'source';src.mkdir();out=root/'out';out.mkdir()
        c=Config(model='tiny-olmo',revision='0'*40,smoke=True,device='cpu',threads=1,seeds=[1],entities=8,eval_entities=8,a_steps=3,b_steps=3,microbatch=2,accumulation=1,lr=.001)
        dump(asdict(c),src/'config.json');e=Engine(c);data=make_data(e.tok,c,1)
        for t in range(1,4):e.gradient(minibatches(data,'A',1,t,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
        save(e.pack(),src/'seed1/anchor.pt');dump({'hash':digest(data)},src/'seed1/data.json')
        cfg=s.defaults(src,out);cfg.update(device='cpu',threads=1,jobs=[dict(seed=1,step=2)],layers=[0],alphas=[.625,1.],random_controls=1,hours=1,reserve_minutes=0,minimum_free_gb=0,topology_points=4)
        dump(cfg,out/'settings.json')
        original=s.Control.check;calls=[0]
        def limited(self):
            calls[0]+=1
            if calls[0]>100:raise s.f.ng.Pause('test interruption')
            return original(self)
        with patch.object(s.Control,'check',limited):s.run(cfg)
        assert s.status(out)['status']=='paused'
        partial={p.name:p.read_bytes() for p in (out/'units').glob('*.json')};assert partial
        s.run(cfg);st=s.status(out);assert st['status']=='complete',st
        assert all((out/'units'/name).read_bytes()==content for name,content in partial.items())
        records=[json.loads(p.read_text()) for p in (out/'units').glob('*.json')]
        assert len(records)==2*4*2*8,len(records)
        index={(r['alpha'],r['corner'],r['split'],r['condition']):r for r in records}
        for r in records:
            raw=index[(r['alpha'],r['corner'],r['split'],'unpatched')]
            if r['condition']=='sham':assert r['rows']==raw['rows']
            if r['condition'] in ['rotation','inverse','orthogonal_0']:assert r['norm_preservation_max_error']<2e-5
            if r['condition']=='additive_0':
                rotation=index[(r['alpha'],r['corner'],r['split'],'rotation')]
                assert np.max(np.abs(np.array([a['patch_norm'] for a in r['rows']])-np.array([a['patch_norm'] for a in rotation['rows']])))<2e-5
        assert len(s.summarize(out))==2*2*8
        before={p.name:p.read_bytes() for p in (out/'units').glob('*.json')};s.run(cfg)
        assert all(p.read_bytes()==before[p.name] for p in (out/'units').glob('*.json'))
        s.show_results(out,plot=True)
        assert (out/'effects.png').exists()
        assert Path(s.export(out)).exists()
        s.stop(out);s.run(cfg);assert s.status(out)['status']=='paused'
        assert s.launch(cfg)==str(out)
        for attempt in range(200):
            st=s.status(out)
            if st['status']=='complete' and not st['alive']:break
            if st['status']=='failed':raise AssertionError(st)
            time.sleep(.1)
        else:raise AssertionError('Worker did not finish in test timeout')
        # Natural endpoint replay and correction hooks do not alter parameters or optimizer.
        e.restore(s.f.p.load(src/'seed1/anchor.pt'));e.gradient(minibatches(data,'B',1,1,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
        w0,w1,delta,vec,_=s.f.capture_step(e,data,1,2)
        residual={n:delta[n]-vec['history'][n]-vec['current'][n] for n in delta}
        s.set_corner(e,w0,vec,residual,1,1,1)
        assert max(float((p.detach().cpu()-w1[n]).abs().max()) for n,p in e.params.items())<1e-7
        weights={n:p.detach().clone() for n,p in e.params.items()}
        optimizer_before=e.pack()['optimizer']
        fr=dict(np.load(out/'maps/seed1_update002_layer0.npz'))
        s.probe(e,data['A_test'],0,fr,'rotation')
        assert all(torch.equal(weights[n],p) for n,p in e.params.items())
        optimizer_after=e.pack()['optimizer']
        for k,state in optimizer_before['state'].items():
            for field,value in state.items():assert torch.equal(value,optimizer_after['state'][k][field])
        print('PASS rotation fit/isometry, full tiny factorial, sham, norm and additive controls, summaries, completed resume/export, natural endpoint and weight preservation')
if __name__=='__main__':test()
