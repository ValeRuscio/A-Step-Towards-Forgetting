import unittest,torch
from types import SimpleNamespace
from dynamics import Ages,capture,collect

class Tests(unittest.TestCase):
    def engine(self):
        m=torch.nn.Linear(3,2,bias=False).double();return SimpleNamespace(model=m,params=dict(m.named_parameters()),opt=torch.optim.Adam(m.parameters(),lr=.01,betas=(.9,.99),eps=1e-8,foreach=False))
    def test_age_bins_against_explicit_history_and_native_adam(self):
        torch.manual_seed(7);e=self.engine();p=next(e.model.parameters())
        for i in range(2):p.grad=torch.randn_like(p);e.opt.step()
        initial=e.opt.state[p]['exp_avg'].clone();ages=Ages({'weight':initial},.9);history=[]
        for t in range(1,27):
            p.grad=torch.randn_like(p);cap=capture(e);v=ages.bins('weight')
            self.assertTrue(torch.allclose(v['task_A'],initial*.9**(t-1),atol=1e-14))
            for name,lo,hi in [('B_age_1_4',1,4),('B_age_5_16',5,16),('B_age_17plus',17,1000)]:
                explicit=sum((.1*.9**(age-1)*g for age,g in enumerate(reversed(history),1) if lo<=age<=hi),torch.zeros_like(p))
                self.assertTrue(torch.allclose(v[name],explicit,atol=1e-14),name)
            self.assertTrue(torch.allclose(sum(v.values()),e.opt.state[p]['exp_avg'],atol=1e-14))
            g={'weight':p.grad.detach().clone()};e.opt.step();ages.advance(g);history.append(g['weight'])
    def test_cross_time_and_scaling_identities(self):
        torch.manual_seed(11);e=self.engine();p=next(e.model.parameters());p.grad=torch.randn_like(p);e.opt.step();ages=Ages({'weight':e.opt.state[p]['exp_avg']},.9);prev=None
        for t in range(3):
            grads={'A_valid':{k:{'weight':torch.randn_like(p)} for k in ['confusion','leakage']}}
            p.grad=torch.randn_like(p);cap=capture(e);row=collect(e,grads,{},cap,ages,prev)
            if prev:
                for r in row['geometry']:
                    for k in ['cross_time_closure','channel_change_closure','magnitude_orientation_closure']:self.assertLess(abs(r[k]),1e-12,k)
            for r in row['age_contributions']:
                ref=next(v for v in row['geometry'] if v['population']==r['population'] and v['component']==r['component'] and v['layer']==r['layer'] and v['channel']=='h')
                self.assertAlmostEqual(sum(r['projections'].values()),ref['projection'],12)
            e.opt.step();ages.advance({n:s['training'] for n,s in cap['state'].items()});prev=dict(grads=grads,cap=cap)
    def test_zero_norm_not_a_fabricated_angle(self):
        from dynamics import cosine
        self.assertIsNone(cosine(0,0,1,1e-12))

if __name__=='__main__':unittest.main()
