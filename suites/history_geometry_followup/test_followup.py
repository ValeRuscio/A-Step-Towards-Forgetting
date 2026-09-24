"""CPU tests: no model/data downloads. Run: python -m unittest -v test_followup"""
import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
import numpy as np
from storage import read,Paused,Control,db_open,get
from study import smoke_settings
from run_study import run
from prediction import examples,metric
from finite_paths import simpson

class FollowupTests(unittest.TestCase):
    def test_background_launch_pause_resume_and_read_only_status(self):
        import time
        import study
        with tempfile.TemporaryDirectory() as tmp:
            s=smoke_settings(tmp);s['include_smollm_momentum']=False;s['hours']=1e-9
            first=study.launch(s);self.assertEqual(first['status'],'launching')
            second=study.launch(s);self.assertIn(second['status'],['launching','running','paused'])
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                st=study.status(s)
                if st['status']=='paused' and not st['alive']:break
                time.sleep(.1)
            self.assertEqual(st['status'],'paused',st)
            before=(Path(tmp)/'status.json').read_bytes();study.status(s);self.assertEqual(before,(Path(tmp)/'status.json').read_bytes())
            s['hours']=1;study.launch(s)
            while time.monotonic()<deadline:
                st=study.status(s)
                if st['status'] in ['complete','complete_with_limitations','failed'] and not st['alive']:break
                time.sleep(.1)
            self.assertIn(st['status'],['complete','complete_with_limitations'],st)
            archived=Path(study.export(s));self.assertTrue(archived.exists())
            import zipfile
            with zipfile.ZipFile(archived) as z:
                self.assertIsNone(z.testzip());self.assertIn('results.sqlite',z.namelist())
                self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))

    def test_prediction_no_current_or_future_peek(self):
        def record(t):
            return dict(step=t,pre={d:{'mean':t*.1} for d in ['A_valid','B_valid','A_test','B_test']},post={d:{'mean':t*.1+.01} for d in ['A_valid','B_valid','A_test','B_test']},
                geometry=dict(projection=.2,history_projection=.1,current_projection=.1,training_interference=.3,update_cosine=.2,update_norm=.3,gA_norm=.4))
        records=[record(t) for t in range(1,15)];a=examples(records)[0]
        modified=copy.deepcopy(records)
        for r in modified:
            if r['step']>=a['decision']:
                r['geometry']['projection']=99999;r['pre']['A_valid']['mean']=8888;r['post']['A_valid']['mean']=7777;r['post']['A_test']['mean']=999
        b=examples(modified)[0]
        self.assertEqual(a['x'],b['x']);self.assertNotEqual(a['y'],b['y']);self.assertEqual(a['feature_step'],a['decision']-1)
        # Past test data alter stratification, never features.
        for r in modified:
            if r['step']<a['decision']:r['post']['A_test']['mean']=1234
        self.assertEqual(a['x'],examples(modified)[0]['x'])

    def test_quadratic_identity(self):
        theta=np.array([.2,-.1]);d=np.array([.3,.4]);H=np.array([[3.,.2],[.2,2.]])
        loss=lambda x:.5*x@H@x
        linear=theta@H@d;observed=loss(theta+d)-loss(theta)
        weighted=[(1-a)*(d@H@d) for a in np.linspace(0,1,9)]
        self.assertAlmostEqual(observed,linear+simpson(weighted),places=14)
        self.assertAlmostEqual(observed,simpson((theta+a*d)@H@d for a in np.linspace(0,1,9)),places=14)

    def test_tied_score_and_empty_metrics(self):
        self.assertEqual(metric([],[])['n'],0)
        m=metric([{'y':0},{'y':1}],[.5,.5]);self.assertEqual(m['auroc'],.5);self.assertEqual(m['average_precision'],.5)

    def test_tiny_neox_and_momentum_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=smoke_settings(tmp);s['models']=[dict(name='tiny_neox',repo='tiny-gpt_neox',revision='smoke')]
            s['learning_rates']={'tiny_neox':{'adam':.001,'momentum_sgd':.01}}
            run(s);st=read(Path(tmp)/'status.json')
            self.assertIn(st['status'],['complete','complete_with_limitations'],st)
            self.assertEqual(st['completed_branches'],6)
            with sqlite3.connect(Path(tmp)/'results.sqlite') as c:
                paths=[json.loads(r[0]) for r in c.execute("SELECT record FROM records WHERE kind='path_done'")]
                self.assertEqual(len(paths),6)
                self.assertTrue(all(p['closure_pass'] and p['curvature_integral_pass'] and p['derivative_pass'] for p in paths))

    def test_branch_and_path_resume_exact(self):
        import run_study
        with tempfile.TemporaryDirectory() as tmp:
            baseline=smoke_settings(str(Path(tmp)/'baseline'));baseline['include_smollm_momentum']=False
            # Include a full nested path and both reset variants.
            run(baseline);self.assertIn(read(Path(baseline['output'])/'status.json')['status'],['complete','complete_with_limitations'])
            resumed=copy.deepcopy(baseline);resumed['output']=str(Path(tmp)/'resumed')
            original=run_study.Control
            class InterruptPath(Control):
                def pulse(self,**kw):
                    super().pulse(**kw)
                    if kw.get('phase')=='exact directional Hessian' and kw.get('alpha',0)>=.5:raise Paused('test interruption within path')
            run_study.Control=InterruptPath
            try:run(resumed)
            finally:run_study.Control=original
            self.assertEqual(read(Path(resumed['output'])/'status.json')['status'],'paused')
            self.assertTrue((Path(resumed['output'])/'branch_checkpoint.pt').exists())
            # Resume once, then interrupt after a later branch checkpoint.
            class InterruptBranch(Control):
                def checkpoint(self,x,p):
                    super().checkpoint(x,p)
                    if x.get('offset')==2 and '/reset/' not in x.get('job','') and x.get('job','').endswith('/reset_blocks'):raise Paused('test interruption after branch checkpoint')
            run_study.Control=InterruptBranch
            try:run(resumed)
            finally:run_study.Control=original
            self.assertEqual(read(Path(resumed['output'])/'status.json')['status'],'paused')
            run(resumed);self.assertIn(read(Path(resumed['output'])/'status.json')['status'],['complete','complete_with_limitations'])
            def contents(path):
                with sqlite3.connect(Path(path)/'results.sqlite') as c:return list(c.execute('SELECT job,step,kind,record FROM records ORDER BY job,step,kind'))
            self.assertEqual(contents(baseline['output']),contents(resumed['output']))
            with sqlite3.connect(Path(baseline['output'])/'results.sqlite') as c:
                first=[json.loads(r[0]) for r in c.execute("SELECT record FROM records WHERE kind='branch_step' AND step=1")]
                self.assertEqual(len(first),3)
                matched=[r for r in first if r['norm_audit']][0]
                self.assertTrue(all(v['passed'] for v in matched['norm_audit'].values()))
                audits=[json.loads(r[0]) for r in c.execute("SELECT record FROM records WHERE kind='state_audit'")]
                self.assertEqual(len({a['identical_start_sha256'] for a in audits}),1)
                self.assertTrue(all(a['other_state_preserved'] for a in audits))
                natural=[json.loads(r[0]) for r in c.execute("SELECT record FROM records WHERE kind='step' AND step=2")][0]
                self.assertTrue(natural['preserve_replay_audit']['passed'])

if __name__=='__main__':unittest.main()
