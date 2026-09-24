import tempfile,unittest,copy,json,sqlite3
from pathlib import Path
import torch
from components import curvature,complete_directions,component_gradients,projections,digest
from path_analysis import path,simpson
from run_study import sample_plan
from storage import db_open,Control,get
from study import smoke_settings,validate

class Polynomial:
    def __init__(self):
        self.model=torch.nn.Module();self.model.register_parameter('w',torch.nn.Parameter(torch.tensor([.3,-.2],dtype=torch.float64)));self.params=dict(self.model.named_parameters())
    def loss_rows(self,rows,details=False):
        x,y=self.model.w
        a=(x*x+2*x*y+3*y*y+.1*x**4).repeat(len(rows));b=(2*x*x-x*y+4*y*y+.2*y**4).repeat(len(rows))
        return (a+b,a,b) if details else a+b

class Tests(unittest.TestCase):
    def test_quadratic_bilinear_forms_and_residual(self):
        e=Polynomial();d={'w':torch.tensor([.08,.11],dtype=torch.float64)};h={'w':torch.tensor([.03,-.02],dtype=torch.float64)};c={'w':torch.tensor([.04,.10],dtype=torch.float64)};r={'w':d['w']-h['w']-c['w']};ds=dict(d=d,h=h,c=c,r=r)
        v=curvature(e,[{}],ds,1)
        x,y=e.model.w.detach();H=torch.tensor([[2+1.2*x*x,2.],[2.,6.]],dtype=torch.float64)
        self.assertAlmostEqual(v['confusion']['hc'],float(h['w']@H@c['w']),12)
        self.assertAlmostEqual(v['confusion']['dd'],float(d['w']@H@d['w']),12)
        self.assertLess(abs(v['confusion']['direction_sum_residual']),1e-12)
    def test_audited_path_independent_integrals_cache_and_restore(self):
        with tempfile.TemporaryDirectory() as root:
            s=smoke_settings(root);s.update(audit_atol=1e-7,audit_rtol=.002,path_population=1,prompt_derivative_count=1)
            ctl=Control(s);con=db_open(root);e=Polynomial();before={'w':e.model.w.detach().clone()};after={'w':before['w']+torch.tensor([.08,.11],dtype=torch.float64)}
            dirs=complete_directions(before,after,{'w':torch.tensor([.03,-.02],dtype=torch.float64)},{'w':torch.tensor([.04,.10],dtype=torch.float64)})
            rows=[dict(example_id='fixed',class_id=0)]
            r=path(e,before,after,dirs,rows,s,ctl,con,'test','A_test')
            self.assertTrue(r['fully_audited'])
            for v in r['components'].values():self.assertLess(abs(v['hessian_closure']),1e-10)
            self.assertTrue(torch.equal(e.model.w,after['w']))
            self.assertEqual(path(e,before,after,dirs,rows,s,ctl,con,'test','A_test'),r)
            bad={'w':after['w']+.001}
            with self.assertRaises(ArithmeticError):path(e,before,bad,dirs,rows,s,ctl,con,'test','A_test')
            con.close()
    def test_fp64_recheck_is_separate_and_restores_native_dtype(self):
        from path_analysis import floating_recheck
        with tempfile.TemporaryDirectory() as root:
            s=smoke_settings(root);e=Polynomial();e.model.float();e.params=dict(e.model.named_parameters())
            before={'w':e.model.w.detach().clone()};after={'w':before['w']+.01}
            dirs=complete_directions(before,after,{'w':torch.tensor([.004,.002])},{'w':torch.tensor([.006,.008])})
            r=floating_recheck(e,before,after,dirs,[dict(example_id='x',class_id=0)],s,Control(s),.5)
            self.assertEqual(r['parameter_dtype'],'float64')
            self.assertEqual(e.model.w.dtype,torch.float32)
            self.assertTrue(torch.equal(e.model.w,after['w']))

    def test_selection_does_not_use_test_loss(self):
        s=smoke_settings('/tmp/unused');s.update(b_steps=16,uniform_updates=4,enriched_per_stratum=2)
        rr=[dict(step=t,pre={sp:{'mean':0.} for sp in ['A_valid','B_valid','A_test']},post={'A_valid':{'mean':.1 if t%2 else -.02},'B_valid':{'mean':.1 if t%3 else -.01},'A_test':{'mean':1e9}}) for t in range(1,17)]
        p=sample_plan(rr,s,51);z=copy.deepcopy(rr)
        for r in z:r['post']['A_test']['mean']=-1e12
        self.assertEqual(p,sample_plan(z,s,51))
        for r in z:r['post']['A_valid']['mean']=0.
        self.assertEqual(p['uniform'],sample_plan(z,s,51)['uniform'])
    def test_gradient_loss_and_projection_additivity(self):
        e=Polynomial();g,l=component_gradients(e,[{},{}],1);d={'d':{'w':torch.tensor([.1,.2],dtype=torch.float64)}};p=projections(g,d)
        self.assertAlmostEqual(l['total'],l['confusion']+l['leakage'],14)
        self.assertAlmostEqual(p['total']['slopes']['d'],p['confusion']['slopes']['d']+p['leakage']['slopes']['d'],14)

if __name__=='__main__':unittest.main()
