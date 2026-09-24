import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import experiment
from storage import Control,Paused,db_open,get,read,write
from optimizers import make_optimizer,native_channels,inv_fourth,orthogonalize,reset_history
from diagnostics import match,audit_match,state_audit
from worker import cfg,run,choose
from study_math import StudyEngine,weights,difference,norm
from tasks import label_mapping,partition,dataset
from timing_analysis import forecast_rows,block_interval,load_sqlite

class MathTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.s=experiment.smoke_settings(self.tmp.name);self.e=StudyEngine(cfg(self.s,self.s['models'][0],.001))
 def tearDown(self):self.tmp.cleanup()
 def test_native_channels_reproduce_actual_step(self):
  for name in ['sgd','momentum_sgd','adam_beta1_0','adam']:
   self.e.opt=make_optimizer(self.e,name,.001,self.s)
   for t in range(3):
    for p in self.e.params.values():p.grad=torch.ones_like(p)*(.03+t*.01)
    before=weights(self.e);h,c=native_channels(self.e,name);self.e.opt.step();d=difference(weights(self.e),before)
    residual={n:d[n]-h[n]-c[n] for n in d};self.assertLess(norm(residual)/max(norm(d),1e-30),2e-4)
 def test_matrix_optimizer_resume_exact(self):
  for name in ['shampoo_block','muon_fp32']:
   self.e.opt=make_optimizer(self.e,name,.001,self.s)
   for p in self.e.params.values():p.grad=torch.full_like(p,.03)
   self.e.opt.step();z=self.e.pack();saved=copy.deepcopy(z)
   for p in self.e.params.values():p.grad=torch.full_like(p,.02)
   self.e.opt.step();target=weights(self.e);self.e.restore(z)
   for p in self.e.params.values():p.grad=torch.full_like(p,.02)
   self.e.opt.step()
   self.assertTrue(all(torch.equal(v,target[n]) for n,v in weights(self.e).items()))
   self.assertTrue(all(torch.equal(z['model'][n],saved['model'][n]) for n in z['model']))
 def test_history_reset_preserves_other_state(self):
  for name in ['momentum_sgd','adam','shampoo_block','muon_fp32']:
   self.e.opt=make_optimizer(self.e,name,.001,self.s)
   for p in self.e.params.values():p.grad=torch.full_like(p,.03)
   self.e.opt.step();z=self.e.pack();self.assertGreater(reset_history(self.e),0);a=state_audit(z,self.e.pack());self.assertTrue(a['other_state_preserved'])
 def test_inverse_root_and_muon_zero(self):
  a=torch.diag(torch.tensor([1.,16.]));v,audit=inv_fourth(a,0.);self.assertTrue(torch.allclose(v,torch.diag(torch.tensor([1.,.5])),atol=1e-6));self.assertEqual(float(orthogonalize(torch.zeros(8,4)).norm()),0.)
 def test_norm_matching_assignment(self):
  before=weights(self.e);candidate={n:torch.full_like(v,.001) for n,v in before.items()};target={n:torch.full_like(v,.002) for n,v in before.items()}
  from study_math import assign
  for mode in [False,True]:
   d,a=match(candidate,target,mode);assign(self.e,{n:before[n]+v for n,v in d.items()});a=audit_match(self.e,before,a,mode,.001);self.assertTrue(all(x['passed'] for x in a.values()))
  with self.assertRaises(ArithmeticError):match({n:torch.zeros_like(v) for n,v in before.items()},target,False)
 def test_paired_codebooks(self):
  a=label_mapping(21,'counterbalanced_shared');b=label_mapping(21,'counterbalanced_disjoint');self.assertEqual(a['A'],b['A']);self.assertEqual(len(set(a['A'])&set(a['B'])),2);self.assertFalse(set(b['A'])&set(b['B']))
  d1,m1=dataset(self.e,self.s,21,'confirmation','counterbalanced_shared');d2,m2=dataset(self.e,self.s,21,'confirmation','counterbalanced_disjoint')
  for sp in d1:self.assertEqual([x['example_id'] for x in d1[sp]],[x['example_id'] for x in d2[sp]])
 def test_hash_partition_disjoint(self):
  d={str(i) for i in range(1000) if partition(str(i))=='development'};c={str(i) for i in range(1000) if partition(str(i))=='confirmation'};self.assertFalse(d&c);self.assertEqual(len(d|c),1000)

 def test_real_task_builder_with_mocked_corpora(self):
  import types,sys
  class Tokenizer:
   def encode(self,text,add_special_tokens=False):
    if len(text)==2 and text[0]==' ' and text[1] in 'ABCDEF':return [50+ord(text[1])-65]
    return [1+ord(c)%45 for c in text]
  def load(repo,revision):
   k=2 if 'sst2' in repo else 4;domain='a' if k==2 else 'b'
   return {split:[dict(sentence=f'{i} {domain} {split} sample',text=f'{i} {domain} {split} sample',label=i%k) for i in range(500)] for split in ['train','validation','test']}
  e=types.SimpleNamespace(tok=Tokenizer());s=copy.deepcopy(self.s);s.update(smoke=False,max_length=256)
  with patch.dict(sys.modules,{'datasets':types.SimpleNamespace(load_dataset=load)}):
   shared,ma=dataset(e,s,21,'confirmation','counterbalanced_shared');disjoint,mb=dataset(e,s,21,'confirmation','counterbalanced_disjoint');dev,md=dataset(e,s,901,'development')
  for split in shared:
   self.assertEqual([r['example_id'] for r in shared[split]],[r['example_id'] for r in disjoint[split]])
   if split.startswith('A_'):self.assertEqual(shared[split],disjoint[split])
  ids=lambda data:{r['example_id'] for rows in data.values() for r in rows}
  self.assertFalse(ids(shared)&ids(dev));self.assertEqual(len(ma['answer_token_overlap']),2);self.assertFalse(mb['answer_token_overlap'])

class TimingTests(unittest.TestCase):
 def records(self):
  return [dict(step=i,pre={sp:{'mean':float(i)} for sp in ['A_valid','B_valid','A_test','B_test']},post={sp:{'mean':float(i)+(.2 if i%2 else -.1)} for sp in ['A_valid','B_valid','A_test','B_test']},geometry=dict(projection=i*.01,update_cosine=.1,update_norm=.2,gA_norm=.3)) for i in range(1,33)]
 def test_no_future_features(self):
  rows=self.records();a=forecast_rows(rows,3,.05);x=next(r for r in a if r['decision_step']==12)
  for r in rows:
   if r['step']>=12:r['post']['A_valid']['mean']=1e6;r['geometry']['projection']=1e6
   if r['step']>12:r['pre']['A_valid']['mean']=1e6
   r['post']['A_test']['mean']+=1000
  y=next(r for r in forecast_rows(rows,3,.05) if r['decision_step']==12);self.assertEqual(x['scores'],y['scores']);self.assertNotEqual(x['delta'],y['delta']);self.assertEqual(x['last_observed_update'],11)
 def test_latest_change_uses_latest_completed(self):
  rows=self.records();rows[10]['post']['A_valid']['mean']=13.
  x=next(r for r in forecast_rows(rows,0,.05) if r['decision_step']==12);self.assertEqual(x['scores']['latest_validation_increase'],2.);self.assertEqual(x['gradient_measurement_step'],11)
 def test_block_bootstrap_paired_identity(self):
  rows=forecast_rows(self.records(),0,.05);r=block_interval(rows,'previous_projection','previous_projection',8,20,1);self.assertEqual(r['interval_95'],[0.,0.])

class IntegrationTests(unittest.TestCase):
 def small(self,p):
  s=experiment.smoke_settings(p);s.update(models=s['models'][:1],label_controls=False,label_models=['tiny_llama'],bootstrap_replicates=3);return s
 def test_calibration_ignores_test(self):
  with tempfile.TemporaryDirectory() as d:
   s=self.small(d);c=db_open(d)
   from storage import put
   s['lr_grid']['adam']=[.01,.02]
   for i in range(2):put(c,f'calibration/tiny/adam/candidate{i}',2,'done',dict(calibration_score=2-i,acquisition={'passed':True},final={'A_test':{'mean':1000 if i else 0}}))
   self.assertEqual(choose(c,'tiny','adam',s)['lr'],.02);c.close()
 def test_full_queue_and_exact_resume(self):
  with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
   sa=self.small(a);sb=self.small(b);run(sa);self.assertEqual(read(Path(a)/'status.json')['status'],'complete')
   original=Control.checkpoint;triggered=[]
   def pause(ctl,state,path):
    original(ctl,state,path)
    if state['job'].startswith('confirmation/') and '/momentum_sgd/' in state['job'] and state['phase']=='B' and state['step']==2 and not triggered:
     triggered.append(True);raise Paused('test interruption after checkpoint')
   with patch.object(Control,'checkpoint',pause):run(sb)
   self.assertEqual(read(Path(b)/'status.json')['status'],'paused');run(sb);self.assertEqual(read(Path(b)/'status.json')['status'],'complete')
   ca=db_open(a);cb=db_open(b)
   for kind in ['step','probe','path']:
    ra=ca.execute('SELECT job,step,record FROM records WHERE kind=? ORDER BY job,step',(kind,)).fetchall();rb=cb.execute('SELECT job,step,record FROM records WHERE kind=? ORDER BY job,step',(kind,)).fetchall();self.assertEqual(ra,rb)
   self.assertFalse((Path(b)/'active_checkpoint.pt').exists())
   for raw, in ca.execute("SELECT record FROM records WHERE kind='step' AND job LIKE 'calibration/%'"):self.assertNotIn('A_test',json.loads(raw)['post'])
   self.assertEqual(ca.execute("SELECT count(*) FROM records WHERE kind='done'").fetchone()[0],12)
   probes=json.loads((Path(a)/'history_probes.json').read_text());self.assertTrue(probes)
   self.assertTrue(all(x['state_audit']['other_state_preserved'] for x in probes));ca.close();cb.close()

if __name__=='__main__':unittest.main()
