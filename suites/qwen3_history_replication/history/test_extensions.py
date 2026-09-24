import unittest,torch
from types import SimpleNamespace
from dynamics import Ages,collect
from cohorts import Cohorts

class Extensions(unittest.TestCase):
    def test_cohort_decay_frozen_denominator_and_ages(self):
        c=Cohorts([1],3);g0=torch.tensor([2.,-1.]);inv0=torch.tensor([1.,2.],dtype=torch.float64)
        def cap(inv):return {'state':{'weight':dict(training=g0,inv=inv,clock=-.1,beta=.9)}}
        c.record(1,cap(inv0));grads={'A_test':{'confusion':{'weight':torch.tensor([1.,3.])},'leakage':{'weight':torch.tensor([2.,1.])}}}
        for t in [2,3,4]:
            inv=torch.tensor([3.,1.],dtype=torch.float64);rr=c.measure(t,grads,cap(inv))
            for r in rr:
                g=grads['A_test'][r['component']]['weight'] if r['component']!='total' else grads['A_test']['confusion']['weight']+grads['A_test']['leakage']['weight']
                u=-.1*.1*.9**(t-1)*g0.double()*(inv if r['denominator']=='current_denominator' else inv0)
                self.assertAlmostEqual(r['projection'],float((g*u).sum()),places=8)
            c.record(t,cap(inv))
        self.assertFalse(c.saved)
    def test_age_norms_and_projection_reconstruction(self):
        ages=Ages({'weight':torch.tensor([.2,-.4])},.9)
        ages.advance({'weight':torch.tensor([.5,1.])})
        m=sum(ages.bins('weight').values()).float();inv=torch.tensor([2.,3.],dtype=torch.float64);g=torch.tensor([.7,-.3])
        h=(-.1*inv*.9*m).float();cc=(-.1*inv*.1*g).float()
        cap={'h':{'weight':h},'c':{'weight':cc},'state':{'weight':dict(moment=m,training=g,inv=inv,clock=-.1,beta=.9)}}
        gg={'A_test':{'confusion':{'weight':torch.tensor([1.,2.])},'leakage':{'weight':torch.tensor([-1.,.5])}}}
        r=collect(SimpleNamespace(params={'weight':None}),gg,{},cap,ages,None)
        for a in r['age_contributions']:
            target=next(x for x in r['geometry'] if x['population']==a['population'] and x['component']==a['component'] and x['layer']==a['layer'] and x['channel']=='h')
            self.assertAlmostEqual(sum(a['projections'].values()),target['projection'],places=12)
            for k,co in a['cosines'].items():
                if co is not None:self.assertAlmostEqual(a['projections'][k],co*a['norms'][k]*target['gradient_norm'],places=12)

if __name__=='__main__':unittest.main()
