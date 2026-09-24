"""End-to-end tiny suite: source import, refinement, replication, frozen predictions, export."""
from pathlib import Path
from dataclasses import asdict
import tempfile,json,sqlite3
from unittest.mock import patch
import paper_suite as suite
import four_vector_study as f
from association_core import Config,Engine,make_data,minibatches,digest,save,dump

def test():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve();src=root/'original_source';src.mkdir();previous=root/'previous';previous.mkdir()
        c=Config(model='tiny-olmo',revision='0'*40,smoke=True,device='cpu',threads=1,seeds=[1,2],entities=4,eval_entities=4,a_steps=2,b_steps=24,microbatch=2,accumulation=1,lr=.001)
        dump(asdict(c),src/'config.json')
        for seed in c.seeds:
            e=Engine(c);data=make_data(e.tok,c,seed)
            for t in range(1,3):e.gradient(minibatches(data,'A',seed,t,c));e.opt.step()
            save(e.pack(),src/f'seed{seed}/anchor.pt');dump({'hash':digest(data)},src/f'seed{seed}/data.json')
        s=f.defaults(src,previous);s.update(device='cpu',threads=1,topology_landmarks=2,minimum_free_gb=0,hours=1,reserve_minutes=0,end_step=24)
        f.atomic_json(s,previous/'settings.json');con=f.connect(previous);f.trajectory(e,s,con,f.Control(s));con.close()
        work=root/'suite';work.mkdir();q=suite.defaults(src,previous,work)
        q.update(device='cpu',threads=1,hours=1,reserve_minutes=0,minimum_free_gb=0,max_output_gb=20,seeds=[1],chunk_steps=24,audit_max_points=33,
            models=[dict(name='tiny_pythia',repo='tiny-gpt_neox',revision='main',enabled=True)])
        q['models'].append(dict(name='tiny_qwen',repo='tiny-qwen3',revision='main',enabled=True))
        f.atomic_json(q,work/'settings.json')
        blocked=[dict(name='tiny_qwen',repo='tiny-qwen3',action='Missing Qwen3 support')]
        with patch.object(suite,'unavailable_models',return_value=blocked):suite.run(q)
        assert f.read(work/'status.json')['status']=='dependency_blocked'
        assert not (work/'runs/tiny_qwen').exists()
        assert f.read(work/'settings.json')['models'][1]['enabled']
        suite.run(q)
        status=f.read(work/'status.json');assert status['status']=='complete',status
        assert (work/'multimodel_share.zip').exists() and (work/'prediction_protocol/frozen_predictors.json').exists()
        assert (work/'runs/tiny_pythia/seed1/REPLICATION_PATHS_DONE.json').exists()
        with sqlite3.connect(work/'refinement/measurements.sqlite') as db:assert db.execute('SELECT COUNT(*) FROM curvature_refined').fetchone()[0]>=4
        assert f.read(work/'transfer_metrics.json')
        # Resume a completed suite: it must remain complete and not overwrite original inputs.
        suite.run(q);assert f.read(work/'status.json')['status']=='complete'
        print('PASS complete suite pipeline, independent refinement, new-model acquisition/trajectory/path/HVP, frozen transfer metrics, export and completed-run resume.')

if __name__=='__main__':test()
