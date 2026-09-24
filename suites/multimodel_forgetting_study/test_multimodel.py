"""No-download CPU tests across tiny original OLMo, GPT-NeoX and Qwen3."""
import sys,json,tempfile,copy,time,math
from pathlib import Path
from dataclasses import asdict
import torch,numpy as np
import paper_suite as suite
import four_vector_study as f
import predict_forgetting as prediction
from multimodel_engine import MultiEngine,install
from association_core import Config,make_data,minibatches,collate,save,dump


def test():
    install()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve()
        for kind in ['olmo','gpt_neox','qwen3']:
            src=root/kind/'source';src.mkdir(parents=True);out=root/kind/'run';out.mkdir()
            c=Config(model='tiny-'+kind,revision='0'*40,smoke=True,device='cpu',threads=1,entities=4,seeds=[1],a_steps=2,b_steps=4,microbatch=2,accumulation=1,lr=.001)
            e=MultiEngine(c);data=make_data(e.tok,c,1);inputs,q,labels=collate(data['A_test'],0,'cpu')
            with torch.no_grad():
                lp=e.model(**inputs).logits[:,-1,:].double().log_softmax(-1);reference=(q*(q.log()-lp[:,labels])).sum(-1)
                assert torch.allclose(e.loss_rows(data['A_test']),reference,atol=1e-10)
            assert len({p.data_ptr() for p in e.params.values()})==len(e.params)
            for t in range(1,3):e.gradient(minibatches(data,'A',1,t,c));e.opt.step()
            anchor=e.pack();save(anchor,src/'seed1/anchor.pt');dump(asdict(c),src/'config.json');dump({'hash':f.p.digest(data)},src/'seed1/data.json')
            s=f.defaults(src,out);s.update(device='cpu',threads=1,end_step=3,chunk_steps=2,diagnostic_jobs=[dict(seed=1,step=2)],topology_landmarks=2,minimum_free_gb=0,hours=1,reserve_minutes=0,max_path_points=33)
            f.atomic_json(s,out/'settings.json');con=f.connect(out);ctl=f.Control(s);f.trajectory(e,s,con,ctl);measured=e.pack()
            e.restore(anchor)
            for t in range(1,4):e.gradient(minibatches(data,'B',1,t,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
            assert f.p.tensor_hash(measured['model'])==f.p.tensor_hash(e.pack()['model']),kind
            for key,v in measured['optimizer']['state'].items():
                for k,x in v.items():
                    if torch.is_tensor(x):assert torch.equal(x,e.pack()['optimizer']['state'][key][k])
            f.path_event(e,s,s['diagnostic_jobs'][0],con,ctl);f.curvature_event(e,s,s['diagnostic_jobs'][0],con,ctl)
            for raw, in con.execute('SELECT record FROM curvature'):
                r=json.loads(raw);assert r['mixed_symmetry_passed'];assert r['directional_curvature_fd']['passed'],(kind,r)
            # Feature extraction never reads consequences or post-update losses.
            row=json.loads(con.execute('SELECT record FROM trajectory WHERE step=2').fetchone()[0]);x=prediction.features(row)
            changed=copy.deepcopy(row);changed['functional_consequence']={'A_test':{'actual_loss_change':1e9}};changed['post_losses']={}
            assert prediction.features(changed)==x
            records=[json.loads(x[0]) for x in con.execute('SELECT record FROM trajectory ORDER BY step')]
            lead=prediction.dataset(records,1);assert all(r['feature_step']==r['step']-1 for r in lead)
            con.close();print('PASS',kind,'forward, all-parameter gradients, unchanged weights/optimizer, finite paths and HVPs, feature timing')
        # Logistic solver and metrics: perfect ordering and ties have known answers.
        rows=[dict(x={'x':float(i)},y=int(i>9)) for i in range(20)]
        model=prediction.fit(rows,['x']);prob=prediction.predict(model,rows);metrics=prediction.scores(rows,prob)
        assert metrics['auroc']==1 and abs(metrics['average_precision']-1)<1e-12
        tied=prediction.scores(rows,np.full(20,.5));assert tied['auroc']==.5 and tied['average_precision']==.5
        # Acquisition resumes at exactly the last completed optimizer step.
        base=root/'training_source';base.mkdir();c=Config(model='tiny-gpt_neox',revision='0'*40,smoke=True,device='cpu',threads=1,entities=4,seeds=[1],a_steps=3,b_steps=4,microbatch=2,accumulation=1)
        s=dict(output=str(root),device='cpu',threads=1,hours=1,reserve_minutes=0,minimum_free_gb=0,max_output_gb=10,checkpoint_every=50,acquisition_max_valid_kl=.25)
        ctl=suite.Control(s);e=MultiEngine(c);calls=0
        original=ctl.pulse
        def stop_at_three(**kw):
            if kw.get('phase')=='learning original A task' and kw.get('step')==3:raise f.StagePause()
            return original(**kw)
        ctl.pulse=stop_at_three
        try:suite.acquire(e,c,base,1,s,ctl)
        except f.StagePause:pass
        z=suite.load(base/'seed1/A_progress.pt');assert z['step']==2
        ctl.pulse=original;suite.acquire(e,c,base,1,s,ctl);actual=suite.load(base/'seed1/anchor.pt')
        reference=MultiEngine(c);data=make_data(reference.tok,c,1)
        for t in range(1,4):reference.gradient(minibatches(data,'A',1,t,c));reference.opt.step()
        assert f.p.tensor_hash(actual['model'])==f.p.tensor_hash(reference.pack()['model'])
        print('PASS acquisition interruption/resume and prediction solver/metrics')

if __name__=='__main__':test()
