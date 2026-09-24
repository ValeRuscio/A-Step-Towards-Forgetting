import copy,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
import reviewer_suite as r
from association_core import Config,minibatches
from study_math import *
from study_prediction import features,examples,score_policy

class ScientificTests(unittest.TestCase):
    def setUp(self):
        self.s=r.smoke_settings('/tmp/unused');self.e=StudyEngine(Config(model='tiny-gpt_neox',smoke=True,device='cpu',threads=1,microbatch=2,accumulation=1,entities=4,max_length=128,lr=.001))
        from study_tasks import dataset
        self.data=dataset(self.e,self.s,11,'registry')
    def test_exact_one_hot_decomposition(self):
        rows=copy.deepcopy(self.data['A_test']);rows[0]['q']=[1.,0.]
        a,b,c=self.e.loss_rows(rows,True);self.assertTrue(torch.isfinite(a).all());self.assertLess(float((a-b-c).detach().abs().max()),1e-12)
        a.sum().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.e.params.values() if p.grad is not None))
    def test_four_class_one_hot(self):
        rows=copy.deepcopy(self.data['A_test'])
        for i,row in enumerate(rows):row['q']=[float(j==i%4) for j in range(4)];row['labels']=[60,61,62,63]
        result=evaluate(self.e,rows,2)
        self.assertTrue(math.isfinite(result['mean']))
        self.assertAlmostEqual(result['mean'],result['confusion']+result['leakage'],places=12)
    def test_text_split_preparation(self):
        import types,sys
        from unittest.mock import patch
        from study_tasks import dataset
        class Tok:
            def encode(self,text,add_special_tokens=False):
                if text in [' A',' B',' C',' D']:return [60+'ABCD'.index(text[-1])]
                return [1+ord(c)%58 for c in text]
        def loader(repo,revision):
            def rows(split,n):return [dict(sentence=repo+split+str(i),text=repo+split+str(i),label=i%(2 if 'sst2' in repo else 4)) for i in range(n)]
            return dict(train=rows('train',80),validation=rows('validation',30),test=rows('test',30))
        settings=dict(self.s,datasets={'sst2':'pinned','ag_news':'pinned'},text_train=12,text_valid=4,text_test=8,max_length=200)
        engine=types.SimpleNamespace(c=types.SimpleNamespace(smoke=False),tok=Tok())
        with patch.dict(sys.modules,{'datasets':types.SimpleNamespace(load_dataset=loader)}):data=dataset(engine,settings,11,'text')
        identities=[r['example_id'] for rows in data.values() for r in rows]
        self.assertEqual(len(identities),len(set(identities)))
        self.assertEqual(len(data['B_test'][0]['q']),4)
        self.assertTrue(all(sum(r['q'])==1 for rows in data.values() for r in rows))
    def test_native_adam_channels(self):
        for t in range(1,3):
            before=weights(self.e);self.e.gradient(minibatches(self.data,'B',11,t,self.e.c));h,c=channels(self.e);self.e.opt.step();delta=difference(weights(self.e),before)
            self.assertLess(norm(difference(delta,plus(h,c)))/norm(delta),2e-4)
    def test_anchor_moments_are_immutable(self):
        batch=minibatches(self.data,'B',11,1,self.e.c)
        self.e.gradient(batch);self.e.opt.step();anchor=self.e.pack();expected=copy.deepcopy(anchor)
        self.e.restore(anchor);self.e.gradient(batch);self.e.opt.step()
        for key,st in anchor['optimizer']['state'].items():
            for name,value in st.items():
                if torch.is_tensor(value):self.assertTrue(torch.equal(value,expected['optimizer']['state'][key][name]))
        self.e.restore(anchor);self.e.gradient(batch);self.e.opt.step();first=weights(self.e)
        self.e.restore(anchor);self.e.gradient(batch);self.e.opt.step();second=weights(self.e)
        self.assertEqual(norm(difference(first,second)),0.)
    def test_reset_clock(self):
        self.e.gradient(minibatches(self.data,'B',11,1,self.e.c));self.e.opt.step();clock=[float(st['step']) for st in self.e.opt.state.values()];v=[st['exp_avg_sq'].clone() for st in self.e.opt.state.values()]
        reset(self.e,'reset_m');self.assertEqual(clock,[float(st['step']) for st in self.e.opt.state.values()]);self.assertTrue(all(torch.equal(a,st['exp_avg_sq']) for a,st in zip(v,self.e.opt.state.values())))
    def test_rotation_isometry_and_holdout(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(6,16));q=np.linalg.qr(rng.normal(size=(16,16)))[0];f=fit_frame(x,x@q);z=rng.normal(size=(4,16));zz=z+(z@f['basis'])@(f['rot']-np.eye(len(f['rot'])))@f['basis'].T
        np.testing.assert_allclose(z@z.T,zz@zz.T,rtol=1e-10,atol=1e-10)
    def test_timestamps_no_future_features(self):
        records=[]
        for t in range(1,15):records.append(dict(step=t,seed=11,geometry=dict(projection=t*.01,update_norm=.1,gA_norm=2,update_cosine=.2,population_interference=.3),pre={x:{'mean':float(t)} for x in ['A_valid','B_valid','A_test','B_test']},post={'A_test':{'mean':t+.1}}))
        a=examples(records,1,.05);altered=copy.deepcopy(records);altered[-1]['post']['A_test']['mean']=1000;b=examples(altered,1,.05)
        self.assertEqual(a[-1]['x'],b[-1]['x']);self.assertEqual(a[-1]['feature_step'],a[-1]['step']-1)
    def test_hessian_matches_gradient_difference(self):
        rows=self.data['A_test'][:2];base=weights(self.e);g=grad(self.e,rows,2);d={n:v/max(norm(g),1e-30)*.001 for n,v in g.items()};hv=directional_hessian(self.e,rows,d,2)
        eps=.1;assign(self.e,{n:v+eps*d[n] for n,v in base.items()});a=dot(grad(self.e,rows,2),d);assign(self.e,{n:v-eps*d[n] for n,v in base.items()});b=dot(grad(self.e,rows,2),d)
        self.assertLess(abs(hv-(a-b)/(2*eps)),1e-6)

if __name__=='__main__':unittest.main()
