import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import reviewer_suite as core
import overnight_study as study
import overnight_engine as engine
from event_controls import match_delta,audit_actual,select_events,branch

class NewTests(unittest.TestCase):
    def test_global_and_layer_match(self):
        a={'model.layers.0.x':torch.tensor([3.,4.]),'model.layers.1.x':torch.tensor([0.,2.]),'lm_head.weight':torch.tensor([1.,1.])}
        b={'model.layers.0.x':torch.tensor([6.,8.]),'model.layers.1.x':torch.tensor([0.,1.]),'lm_head.weight':torch.tensor([4.,3.])}
        for mode in ['global','layer']:
            delta,audit=match_delta(a,b,mode);audit_actual(delta,audit,mode,1e-5)
            self.assertTrue(all(v['passed'] for v in audit.values()))
    def test_zero_candidate_refused(self):
        with self.assertRaises(ArithmeticError):match_delta({'a':torch.zeros(1)},{'a':torch.ones(1)},'global')
    def test_selection_does_not_read_test(self):
        s=study.smoke_settings('/tmp/unused');s.update(event_fixed_steps=[],event_count=1,frame_calibration=2)
        records=[dict(step=1,pre={'A_valid':{'rows':[{'entity':0,'loss':1.},{'entity':1,'loss':1.}]}},post={'A_valid':{'rows':[{'entity':0,'loss':1.2},{'entity':1,'loss':1.2}]},'A_test':{'mean':999.}})]
        a=select_events(records,s);records[0]['post']['A_test']['mean']=-999.;self.assertEqual(a,select_events(records,s));self.assertEqual(len(a),1)
    def test_historical_audit_is_read_only(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';old=core.smoke_settings(source);old.update(tasks=['text'],experiments=[],a_steps=1,b_steps=5)
            old=core.prepare(old);core.run(old)
            tracked=list(source.rglob('*.pt'))+list((source/'data').rglob('*.json'))+[source/'settings.json',source/'results.sqlite']
            hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
            new=study.smoke_settings(Path(temp)/'new');new.update(curvature_reaudit_source=str(source),curvature_alphas=[.328125],curvature_eps=[.02,.01])
            new=core.prepare(new);con=core.db_open(new['output']);engine.reaudit_history(new,con,core.Control(new))
            rows=core.get(con,'historical_curvature','alpha_0.328125');self.assertEqual(len(rows),1)
            self.assertEqual(len(rows[0]['measurements'][0]['widths']),2)
            self.assertEqual(hashes,{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked});con.close()
    def test_priority_resume_and_paired_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            s=study.smoke_settings(temp);s.update(event_count=0,random_replicates=2,a_steps=1)
            s=core.prepare(s)
            frozen=core.read(Path(temp)/'frozen_baselines.json')
            for f in frozen['lead']['1'].values():f['projection_raw_cutoff']=-1e9;f['alarm_cutoffs']['loss_history']=-1.
            core.write(Path(temp)/'frozen_baselines.json',frozen)
            original=core.Control.checkpoint;paused=[False]
            def checkpoint(ctl,state,path):
                original(ctl,state,path)
                if '/event' in state.get('job','') and state.get('offset')==0 and not paused[0]:paused[0]=True;raise core.Paused('Test event interruption')
            core.Control.checkpoint=checkpoint
            try:engine.run(s)
            finally:core.Control.checkpoint=original
            self.assertEqual(core.status(s)['status'],'paused')
            con=core.db_open(temp)
            self.assertEqual(con.execute("SELECT count(*) FROM records WHERE kind='done'").fetchone()[0],16)
            con.close();engine.run(s);self.assertEqual(core.status(s)['status'],'complete')
            con=core.db_open(temp)
            self.assertEqual(con.execute("SELECT count(*) FROM records WHERE kind='event_done'").fetchone()[0],8)
            for seed in [12,13]:
                case=f'tiny/registry/seed{seed}';random=engine.random_conditions(s,con,case,seed)
                self.assertEqual(len(random),2)
                for cond in random:
                    rows=core.get(con,case+'/'+cond['name'],'step');self.assertEqual(sum(x['action'] for x in rows),2)
                base=next(x for x in core.get(con,case+'/natural','step') if x['step']==2)
                preserved=core.get(con,case+'/event002/preserve','event_step')[0]
                self.assertEqual(base['post'],preserved['post'])
                for mode in ['reset_m_global','reset_m_layer']:
                    row=core.get(con,case+'/event002/'+mode,'event_step')[0]
                    self.assertTrue(all(x['passed'] for x in row['norm_audit'].values()))
                setup=core.get(con,case+'/event002/reset_m','event_setup')[0]
                self.assertTrue(setup['state_audit']['clock_preserved']);self.assertEqual(setup['state_audit']['second_moment_max_difference'],0.)
            con.close()

if __name__=='__main__':unittest.main()
