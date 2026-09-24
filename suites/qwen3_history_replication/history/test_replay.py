import json,sqlite3,tempfile,unittest,importlib.metadata,hashlib
from pathlib import Path
from association_core import Config,minibatches
from study_math import StudyEngine,evaluate,weights
from components import digest
from storage import db_open,put,write,read,Paused
import study,run_replay

def fixture(root,arch):
    root.mkdir();s=dict(version='natural-components-1',a_steps=2,b_steps=4,threads=1,smoke=True,beta1=.9,beta2=.99,eps=1e-8,clip=1.,microbatch=2,accumulation=1,max_length=32,text_test=2)
    model=dict(name='tiny_'+arch,repo='tiny-'+arch,revision='smoke');job='natural/'+model['name']+'/seed51/counterbalanced_shared'
    c=Config(model=model['repo'],revision='smoke',device='cpu',threads=1,smoke=True,lr=.001,beta1=.9,beta2=.99,eps=1e-8,clip=1.,microbatch=2,accumulation=1,max_length=32)
    e=StudyEngine(c);data={}
    for dom in ['A','B']:
        for split in ['train','valid','test']:
            data[dom+'_'+split]=[dict(ids=[1,3 if dom=='A' else 4,8+i,9],q=[1.,0.] if i%2==0 else [0.,1.],labels=[50,51],entity=i,example_id=f'{dom}/{split}/{i}',class_id=i%2) for i in range(2)]
    (root/'data').mkdir();write(root/'data'/(job.replace('/','_')+'.json'),dict(data=data,meta={}))
    write(root/'settings.json',s);write(root/'environment.json',dict(packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']}))
    con=db_open(root);put(con,job,0,'spec',dict(model=model,seed=51,variant='counterbalanced_shared',lr=.001))
    for t in range(1,3):e.gradient(minibatches(data,'A',51,t,c));e.opt.step()
    put(con,job,0,'anchor',dict(weight_hash=digest(weights(e)),acquisition=dict(passed=True)))
    for t in range(1,5):
        pre={sp:evaluate(e,data[sp],1) for sp in ['A_valid','A_test']};e.gradient(minibatches(data,'B',51,t,c));e.opt.step();post={sp:evaluate(e,data[sp],1) for sp in pre}
        put(con,job,t,'step',dict(step=t,pre=pre,post=post,endpoint_hash=digest(weights(e))))
    put(con,job,0,'natural_done',dict(complete=True));con.close();return job

class ReplayTests(unittest.TestCase):
    def test_exact_replay_pause_resume_and_export(self):
        for arch in ['llama','gpt_neox','qwen3']:
            with self.subTest(arch=arch),tempfile.TemporaryDirectory() as tmp:
                src=Path(tmp)/'source';out=Path(tmp)/'replay';job=fixture(src,arch);dbbefore=(src/'results.sqlite').read_bytes()
                s=study.defaults(src,out);s.update(device='cpu',test_population=2,minimum_free_gib=0,checkpoint_every=1)
                old=run_replay.ReplayControl
                class Pause(old):
                    def pulse(self,**kw):
                        super().pulse(**kw)
                        if kw.get('phase')=='measure cross-time geometry' and kw.get('step')==2:raise Paused('Intentional test interruption')
                try:run_replay.ReplayControl=Pause;run_replay.run(s)
                finally:run_replay.ReplayControl=old
                self.assertEqual(read(out/'status.json')['status'],'paused')
                with sqlite3.connect(out/'results.sqlite') as c:first=c.execute("SELECT record FROM records WHERE kind='update' AND step=1").fetchone()[0]
                run_replay.run(s);status=read(out/'status.json');self.assertEqual(status['status'],'complete',status)
                with sqlite3.connect(out/'results.sqlite') as c:
                    self.assertEqual(first,c.execute("SELECT record FROM records WHERE kind='update' AND step=1").fetchone()[0]);self.assertEqual(c.execute("SELECT COUNT(*) FROM records WHERE kind='update'").fetchone()[0],4)
                self.assertEqual((src/'results.sqlite').read_bytes(),dbbefore)
                self.assertTrue(Path(study.export(s)).exists());self.assertTrue((Path(study.analyze(s))/'REPORT.txt').exists())
                self.assertFalse((out/'active_A_anchor.pt').exists());run_replay.run(s)
                self.assertEqual(read(out/'summary.json')['updates'],4)

if __name__=='__main__':unittest.main()
