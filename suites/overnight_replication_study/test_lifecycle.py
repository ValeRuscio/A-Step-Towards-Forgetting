"""Integration: pause/replay equivalence, forced alarms, matched random budgets."""
import copy,json,tempfile,unittest
from pathlib import Path
import reviewer_suite as r

class Lifecycle(unittest.TestCase):
    def test_pause_resume_and_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            s=r.smoke_settings(str(Path(temp)/'run'));s.update(experiments=['prediction'],a_steps=1,b_steps=11,frame_fixed_steps=[],checkpoint_every=2)
            s=r.prepare(s);out=Path(s['output']);f=r.read(out/'frozen_baselines.json')
            # Deliberately force an alarm in this integration test, not in release defaults.
            for x in f['lead']['1'].values():x['projection_raw_cutoff']=-1e9;x['alarm_cutoffs']['loss_history']=-1.
            r.write(out/'frozen_baselines.json',f)
            original=r.Control.checkpoint;count=[0]
            def pause_after_checkpoint(ctl,state,path):
                original(ctl,state,path)
                if state.get('job','').endswith('/natural') and state.get('step')==2:
                    count[0]+=1
                    if count[0]==1:raise r.Paused('test interruption')
            r.Control.checkpoint=pause_after_checkpoint
            try:r.run(s)
            finally:r.Control.checkpoint=original
            self.assertEqual(r.status(s)['status'],'paused');r.run(s);self.assertEqual(r.status(s)['status'],'complete')
            con=r.db_open(out)
            for name in ['projection','loss_history']:
                a=r.get(con,'tiny/registry/seed11/'+name,'step');b=r.get(con,'tiny/registry/seed11/random_'+name,'step')
                self.assertEqual(sum(x['action'] for x in a),2);self.assertEqual(sum(x['action'] for x in b),2)
            resumed=r.get(con,'tiny/registry/seed11/natural','step');con.close()
            s2=r.smoke_settings(str(Path(temp)/'control'));s2.update(experiments=[],a_steps=1,b_steps=11,frame_fixed_steps=[],checkpoint_every=2);s2=r.prepare(s2);r.run(s2)
            con=r.db_open(s2['output']);uninterrupted=r.get(con,'tiny/registry/seed11/natural','step');con.close()
            self.assertEqual(resumed,uninterrupted)
    def test_settings_changes_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            s=r.smoke_settings(temp);r.prepare(s);s['hours']=2;r.prepare(s)
            s['lr']*=2
            with self.assertRaises(ValueError):r.prepare(s)

if __name__=='__main__':unittest.main()
