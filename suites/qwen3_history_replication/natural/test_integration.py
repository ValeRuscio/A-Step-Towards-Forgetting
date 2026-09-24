"""Offline CPU checks: tiny Llama and GPT-NeoX, intentional pause and exact replay."""
import json,sqlite3,tempfile,unittest
from pathlib import Path
import run_study
from study import smoke_settings,export,analyze
from storage import read,Control,Paused

class Integration(unittest.TestCase):
    def test_both_architectures_and_pause_resume(self):
        for arch in ['llama','gpt_neox']:
            with self.subTest(architecture=arch),tempfile.TemporaryDirectory() as root:
                s=smoke_settings(root);s['models']=[dict(name='tiny_'+arch,repo='tiny-'+arch,revision='smoke')];s['learning_rates']={'tiny_'+arch:.001}
                original=run_study.Control
                class PauseInPath(Control):
                    def pulse(self,**kw):
                        super().pulse(**kw)
                        if kw.get('phase')=='history/current Hessian bilinear forms':raise Paused('Intentional integration-test pause')
                try:
                    run_study.Control=PauseInPath;run_study.run(s)
                finally:run_study.Control=original
                self.assertEqual(read(Path(root)/'status.json')['status'],'paused')
                with sqlite3.connect(Path(root)/'results.sqlite') as c:
                    saved_job,saved_step,first=c.execute("SELECT job,step,record FROM records WHERE kind='path_point' ORDER BY job,step").fetchone()
                run_study.run(s)
                status=read(Path(root)/'status.json');self.assertIn(status['status'],['complete','complete_with_limitations'],status)
                summary=read(Path(root)/'summary.json');self.assertEqual(summary['completed_cases'],2);self.assertEqual(summary['completed_paths'],4)
                pairs=read(Path(root)/'answer_condition_pairs.json');self.assertTrue(pairs[0]['identical_A_weights']);self.assertTrue(pairs[0]['identical_example_selection'])
                with sqlite3.connect(Path(root)/'results.sqlite') as c:
                    second=c.execute("SELECT record FROM records WHERE kind='path_point' AND job=? AND step=?",(saved_job,saved_step)).fetchone()[0]
                self.assertEqual(first,second)
                self.assertFalse((Path(root)/'active_checkpoint.pt').exists());self.assertFalse((Path(root)/'active_anchor.pt').exists())
                self.assertTrue(Path(export(s)).exists())
                self.assertTrue((Path(analyze(s))/'REPORT.txt').exists())
                # Idempotent worker rerun must not create extra records.
                run_study.run(s);self.assertEqual(read(Path(root)/'summary.json'),summary)
    def test_empty_export_refused(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError):export(smoke_settings(root))

if __name__=='__main__':unittest.main()
