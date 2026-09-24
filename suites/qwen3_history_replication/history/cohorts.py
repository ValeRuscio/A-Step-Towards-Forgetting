"""Fixed B-gradient cohorts; no outcome selection, no optimizer deletion."""
import math
from dynamics import inner,cosine
from components import layer

class Cohorts:
    def __init__(self,origins=(1,8,32,64),max_age=32):
        self.origins=set(origins);self.max_age=max_age;self.saved={}
    def record(self,t,cap):
        if t in self.origins:
            # CPU tensors are immutable; keep references, not a full cap/optimizer copy.
            self.saved[t]={n:{'g':v['training'],'inv':v['inv']} for n,v in cap['state'].items()}
        self.saved={k:v for k,v in self.saved.items() if t-k<self.max_age}
    def measure(self,t,grads,cap,floor=1e-12,ctl=None):
        output=[]
        for origin,saved in self.saved.items():
            age=t-origin
            if not 1<=age<=self.max_age:continue
            sums={}
            for n,v in saved.items():
                if ctl:ctl.check();ctl.pulse(phase='CPU fixed-cohort contractions',origin_step=origin,age=age,parameter=n)
                st=cap['state'][n];beta=st['beta'];g0=v['g'].double()
                # Contribution of gradient g_origin to HISTORY at update t:
                # -lr/(1-beta^clock) * D_t * (1-beta)*beta^(t-origin)*g_origin.
                coeff=st['clock']*(1-beta)*beta**age
                dirs={'current_denominator':coeff*st['inv']*g0,
                      'origin_denominator':coeff*v['inv']*g0}
                for pop,gg in grads.items():
                    for comp in ['confusion','leakage','total']:
                        g=gg[comp][n] if comp!='total' else gg['confusion'][n]+gg['leakage'][n]
                        gsq=inner(g,g)
                        for mode,u in dirs.items():
                            pr,usq=inner(g,u),inner(u,u)
                            for group in [layer(n),'GLOBAL']:
                                r=sums.setdefault((pop,comp,mode,group),dict(projection=0.,gradient_squared=0.,channel_squared=0.))
                                r['projection']+=pr;r['gradient_squared']+=gsq;r['channel_squared']+=usq
            for (pop,comp,mode,group),v in sums.items():
                output.append(dict(origin_step=origin,age=age,step=t,population=pop,component=comp,denominator=mode,layer=group,**v,channel_norm=math.sqrt(v['channel_squared']),gradient_norm=math.sqrt(v['gradient_squared']),cosine=cosine(v['projection'],v['gradient_squared'],v['channel_squared'],floor)))
        return output
