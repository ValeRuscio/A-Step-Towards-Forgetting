"""No-download integration: four tiny variants, frozen import, paired report, resume."""
from pathlib import Path
from dataclasses import asdict
import tempfile,json,sqlite3,time
import paired_suite as s
from association_core import Config,dump

def test():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp).resolve();src=root/'source';src.mkdir();prior=root/'prior';(prior/'prediction_protocol').mkdir(parents=True)
        c=Config(model='tiny-olmo',smoke=True,device='cpu',threads=1,seeds=[1],entities=4,eval_entities=4,a_steps=2,b_steps=8,microbatch=2,accumulation=1,lr=.001)
        dump(asdict(c),src/'config.json')
        names=['loss_only','local_projection','static_geometry','geometry_plus_changes']
        frozen={'protocol':s.pred.PROTOCOL,'models':{str(lead):{n:{'constant':.25} for n in names} for lead in [0,1]}}
        raw=json.dumps(frozen).encode();(prior/'prediction_protocol/frozen_predictors.json').write_bytes(raw)
        out=root/'run';out.mkdir();cfg=s.defaults(src,prior,out)
        cfg.update(device='cpu',threads=1,seeds=[1],path_seeds=[1],hours=1,reserve_minutes=0,minimum_free_gb=0,max_output_gb=2,chunk_steps=8)
        for item in cfg['models']:item['repo']='tiny-'+item['architecture']
        dump(cfg,out/'settings.json');s.validate(cfg);s.run(cfg)
        st=s.f.read(out/'status.json');assert st['status']=='complete',st
        assert (out/'prediction_protocol/frozen_predictors.json').read_bytes()==raw
        pair=s.f.read(out/'paired_comparison.json');assert len(pair['cases'])==1 and len(pair['comparisons'])==0
        for item in cfg['models']:
            d=out/'runs'/item['name']/'seed1';assert (d/'REPLICATION_PATHS_DONE.json').exists()
            with sqlite3.connect(d/'measurements.sqlite') as con:
                assert con.execute('select count(*) from trajectory').fetchone()[0]==8
                assert con.execute('select count(*) from path').fetchone()[0]>0
        assert (out/'qwen3_models_share.zip').exists()
        assert not (out/'refinement').exists()
        s.run(cfg);assert s.f.read(out/'status.json')['status']=='complete'
        assert (out/'prediction_protocol/frozen_predictors.json').read_bytes()==raw
        # Real detached launcher must clear a stop request and resume completed output.
        s.stop(out);s.launch(cfg)
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if not s.f.ng.alive(out) and s.f.read(out/'status.json').get('status')=='complete':break
            time.sleep(.2)
        else:
            s.stop(out)
            raise AssertionError('Detached resume did not finish within 60 seconds')
        assert not (out/'STOP').exists()
        # Altered frozen artifact must not silently replace the original.
        (prior/'prediction_protocol/frozen_predictors.json').write_bytes(raw+b' ')
        try:s.import_frozen(cfg)
        except ValueError:pass
        else:raise AssertionError('Changed frozen artifact accepted')
        print('PASS Qwen3 pipeline, fixed predictors, data matching, paths, acquisition/outcome report, export/resume and immutable predictor check')
if __name__=='__main__':test()
