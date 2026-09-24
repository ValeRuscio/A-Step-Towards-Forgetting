"""CPU tiny-OLMo integration checks; no downloads or external checkpoints."""
import json, tempfile, math, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import four_vector_study as f
from association_core import Config,Engine,make_data,minibatches,save,dump,digest

def check():
    # Cancellation cannot make unresolved channel integrals pass.
    def oscillatory(a):
        h=math.sin(23*a)
        return dict(loss=a,projections=dict(delta=1.,history=h,current=1-h,roundoff=0.))
    initial={float(a):oscillatory(float(a)) for a in np.linspace(0,1,17)}
    r=f.refine.adaptive_integral(oscillatory,initial,1e-8,1e-6,33,8,convergence_keys=['delta','history','current'])
    assert r['cap_reached'] and not r['quadrature_passed'] and abs(r['closure_error'])<1e-12
    torch.manual_seed(13)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve();src=root/'source';src.mkdir();out=root/'output';out.mkdir()
        c=Config(smoke=True,device='cpu',threads=1,seeds=[1],entities=4,eval_entities=4,a_steps=2,b_steps=4,lr=.001,microbatch=2,accumulation=1)
        e=Engine(c);data=make_data(e.tok,c,1)
        for t in range(1,3):e.gradient(minibatches(data,'A',1,t,c));e.opt.step()
        anchor=e.pack();save(anchor,src/'seed1/anchor.pt');dump({'hash':digest(data)},src/'seed1/data.json');dump(asdict(c),src/'config.json')
        s=f.defaults(src,out);s.update(device='cpu',threads=1,end_step=3,chunk_steps=2,diagnostic_jobs=[{'seed':1,'step':2}],topology_landmarks=2,
            hours=1,reserve_minutes=0,minimum_free_gb=0,gradient_batch=2,max_path_points=33,max_path_depth=5,integration_atol=1e-4,integration_rtol=.01)
        f.validate(s);f.atomic_json(s,out/'settings.json');con=f.connect(out);ctl=f.Control(s)
        f.trajectory(e,s,con,ctl)
        measured=e.pack()
        e.restore(anchor)
        for t in range(1,4):e.gradient(minibatches(data,'B',1,t,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
        assert f.p.tensor_hash(measured['model'])==f.p.tensor_hash(e.pack()['model']), 'Diagnostics changed weights'
        for k,v in measured['optimizer']['state'].items():
            for n,x in v.items():
                y=e.pack()['optimizer']['state'][k][n]
                if torch.is_tensor(x):assert torch.equal(x,y), 'Diagnostics changed optimizer'
        records=[json.loads(r[0]) for r in con.execute('SELECT record FROM trajectory ORDER BY step')]
        assert len(records)==3 and records[0]['temporal_changes'] is None
        for r in records[1:]:
            for v in r['temporal_changes']['global_changes'].values():assert abs(v['closure_error'])<1e-5
        # Completed trajectory is idempotent; resume reconstructs previous vectors correctly.
        f.trajectory(e,s,con,ctl);assert con.execute('SELECT COUNT(*) FROM trajectory').fetchone()[0]==3
        con.execute('DELETE FROM trajectory WHERE step=3');con.commit();f.trajectory(e,s,con,ctl)
        rerun=json.loads(con.execute('SELECT record FROM trajectory WHERE step=3').fetchone()[0]);assert rerun['pre']==records[2]['pre']
        f.path_event(e,s,s['diagnostic_jobs'][0],con,ctl)
        acc=f.read(out/'seed1_update002/path_accounting.json')
        for sp,r in acc.items():
            a=r['orientation_magnitude_accounting'];assert abs(a['integrated_nonlinear_sum']-(r['integral']-r['initial_slope']))<1e-7
        f.curvature_event(e,s,s['diagnostic_jobs'][0],con,ctl)
        rows=con.execute('SELECT alpha,split,record FROM curvature').fetchall();assert len(rows)==2
        data,before,after,dirs=f.p.prepare(e,s,s['diagnostic_jobs'][0],ctl)
        for alpha,sp,record in rows:
            record=json.loads(record);vals=[];eps=.01
            for k in ['hHh','hHc','cHh','cHc','dHd']:assert math.isclose(sum(v[k] for v in record['layer_contributions'].values()),record[k],rel_tol=1e-8,abs_tol=1e-10)
            for a in [alpha-eps,alpha+eps]:
                f.p.assign(e,before,after,a);g,_=f.population_gradient(e,data[sp],2)
                vals.append(sum(f.ng.dot(g[n],dirs['delta'][n]) for n in g))
            fd=(vals[1]-vals[0])/(2*eps)
            assert math.isclose(fd,record['dHd'],rel_tol=.04,abs_tol=2e-5),(fd,record['dHd'])
            assert record['mixed_symmetry_passed']
        f.p.assign(e,before,after,1)
        assert f.p.tensor_hash(e.pack()['model'])==f.p.tensor_hash(after)
        f.report(out);zipfile=f.export(out);assert Path(zipfile).exists()
        assert not list(out.rglob('*.pt')) and not list(out.rglob('*.npz'))
        # Exercise staged orchestration on existing, completed work.
        con.close();f.run(s);assert f.status(out)['status']=='complete',f.status(out)
        # Detached launch, duplicate launch, cooperative stop and subsequent resume.
        launched=f.launch(s);assert Path(launched)==out
        again=f.launch(s);assert again==launched
        f.stop(out)
        deadline=time.time()+45
        while f.ng.alive(out) and time.time()<deadline:time.sleep(.1)
        assert not f.ng.alive(out), 'Worker did not stop cooperatively'
        f.launch(s)
        deadline=time.time()+45
        while f.ng.alive(out) and time.time()<deadline:time.sleep(.1)
        assert f.status(out)['status']=='complete',f.status(out)
        changed=dict(s,end_step=4)
        new=Path(f.launch(changed));assert new!=out and (out/'measurements.sqlite').exists()
        f.stop(new)
        deadline=time.time()+45
        while f.ng.alive(new) and time.time()<deadline:time.sleep(.1)
        assert not f.ng.alive(new)
        print('PASS: launch/duplicate launch/stop/resume and changed-settings preservation.')
        print('PASS: exact natural weights and optimizer, temporal identities, deterministic resume, path accounting, full-population HVP finite differences, reports/export, staged run, no tensor dumps.')

if __name__=='__main__':check()
