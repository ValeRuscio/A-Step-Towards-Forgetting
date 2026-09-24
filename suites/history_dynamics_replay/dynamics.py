"""Exact algebraic cross-time decompositions; age bins with explicit native roundoff."""
from collections import deque
import math
import torch
from components import layer
from study_math import channels

class Ages:
    def __init__(self,anchor,beta):
        self.beta=beta;self.a={n:v.double().clone() for n,v in anchor.items()}
        self.old={n:torch.zeros_like(v) for n,v in self.a.items()};self.recent=deque()
    def bins(self,n):
        b=self.beta;z=torch.zeros_like(self.a[n]);young=z.clone();middle=z.clone()
        for age,g in enumerate(reversed(self.recent),1):
            if age<=4:young.add_(g[n].double(),alpha=(1-b)*b**(age-1))
            else:middle.add_(g[n].double(),alpha=(1-b)*b**(age-1))
        return dict(task_A=self.a[n],B_age_1_4=young,B_age_5_16=middle,B_age_17plus=self.old[n])
    def advance(self,g):
        b=self.beta
        expired=self.recent.popleft() if len(self.recent)==16 else None
        for n in self.a:
            self.a[n].mul_(b);self.old[n].mul_(b)
            if expired is not None:self.old[n].add_(expired[n].double(),alpha=(1-b)*b**16)
        self.recent.append(g)

def capture(e):
    """Native implementation channels plus double-valued numerator/scaling factors."""
    h,c=channels(e);state={};ps={id(p):n for n,p in e.params.items()}
    for pg in e.opt.param_groups:
        if pg.get('weight_decay',0) or pg.get('amsgrad',False):raise ValueError('Only native Adam without decay/AMSGrad')
        b,b2=pg['betas']
        for p in pg['params']:
            if p.grad is None:raise ValueError('Missing training gradient; age/clock semantics require active parameters')
            n=ps[id(p)];st=e.opt.state[p];t=int(st['step'])+1
            m=st['exp_avg'].detach().cpu().clone();g=p.grad.detach().cpu().clone()
            den=(b2*st['exp_avg_sq']+(1-b2)*p.grad.square()).sqrt()/math.sqrt(1-b2**t)+pg['eps']
            inv=1./den.detach().cpu().double();clock=-pg['lr']/(1-b**t)
            state[n]=dict(moment=m,training=g,inv=inv,clock=clock,beta=b,t=t)
    return dict(h=h,c=c,state=state)

def inner(a,b):return float((a.double()*b.double()).sum())
def cosine(dot,aa,bb,floor):
    if aa<=floor**2 or bb<=floor**2:return None
    return max(-1.,min(1.,dot/math.sqrt(aa*bb)))

def collect(e,grads,losses,cap,ages,previous,floor=1e-12):
    """One row per population/component/channel/layer + global. Parameter streaming.
    Component gradients may be FP64; native h/c are the exact recorded channel convention.
    """
    rows={};age_rows={};reconstruction_sq=0.;moment_sq=0.;history_residual_sq=0.;history_sq=0.
    for n in e.params:
        st=cap['state'][n];b=st['beta'];bins=ages.bins(n);m=st['moment'].double();summed=sum(bins.values());mr=m-summed
        reconstruction_sq+=inner(mr,mr);moment_sq+=inner(m,m)
        scale=st['clock']*st['inv'];age_h={k:scale*b*v for k,v in bins.items()}
        age_h['roundoff']=cap['h'][n].double()-sum(age_h.values())
        history_residual_sq+=inner(age_h['roundoff'],age_h['roundoff']);history_sq+=inner(cap['h'][n],cap['h'][n])
        for pop,gg in grads.items():
            for comp in ['confusion','leakage','total']:
                g=gg[comp][n] if comp!='total' else gg['confusion'][n]+gg['leakage'][n]
                gp=None
                if previous:gp=previous['grads'][pop][comp][n] if comp!='total' else previous['grads'][pop]['confusion'][n]+previous['grads'][pop]['leakage'][n]
                for group in [layer(n),'GLOBAL']:
                    key=(pop,comp,group);ar=age_rows.setdefault(key,{k:0. for k in age_h})
                    for k,v in age_h.items():ar[k]+=inner(g,v)
                for ch in ['h','c']:
                    u=cap[ch][n].double();numerator=(b*m if ch=='h' else (1-b)*st['training'].double());ideal=scale*numerator
                    vals=dict(projection=inner(g,u),g_squared=inner(g,g),channel_squared=inner(u,u),implementation_residual_projection=inner(g,u-ideal))
                    if previous:
                        old=previous['cap']['state'][n];up=previous['cap'][ch][n].double();np=(old['beta']*old['moment'].double() if ch=='h' else (1-old['beta'])*old['training'].double());sp=old['clock']*old['inv'];ip=sp*np
                        ga=(g+gp)/2;ua=(u+up)/2;du=u-up;dg=g-gp
                        # Product difference: scale_avg * delta_n + n_avg * delta_scale.
                        dn=scale.add(sp)*.5*(numerator-np)
                        na=(numerator+np)*.5
                        denom=na*((st['clock']+old['clock'])*.5)*(st['inv']-old['inv'])
                        clock=na*((st['inv']+old['inv'])*.5)*(st['clock']-old['clock'])
                        residual=(u-ideal)-(up-ip)
                        vals.update(previous_projection=inner(gp,up),previous_g_squared=inner(gp,gp),previous_channel_squared=inner(up,up),g_cross_time=inner(g,gp),channel_cross_time=inner(u,up),gradient_change=inner(dg,ua),channel_change=inner(ga,du),numerator_change=inner(ga,dn),adaptive_scaling_change=inner(ga,denom),clock_change=inner(ga,clock),roundoff_change=inner(ga,residual))
                    for group in [layer(n),'GLOBAL']:
                        row=rows.setdefault((pop,comp,ch,group),{k:0. for k in vals})
                        for k,v in vals.items():row[k]+=v
    output=[]
    for (pop,comp,ch,group),v in rows.items():
        v.update(population=pop,component=comp,channel=ch,layer=group)
        v['gradient_norm']=math.sqrt(v['g_squared']);v['channel_norm']=math.sqrt(v['channel_squared']);v['cosine']=cosine(v['projection'],v['g_squared'],v['channel_squared'],floor)
        if previous:
            pn=math.sqrt(v['previous_g_squared']);un=math.sqrt(v['previous_channel_squared']);pc=cosine(v['previous_projection'],v['previous_g_squared'],v['previous_channel_squared'],floor)
            v['projection_change']=v['projection']-v['previous_projection'];v['cross_time_closure']=v['projection_change']-v['gradient_change']-v['channel_change']
            v['channel_change_closure']=v['channel_change']-sum(v[k] for k in ['numerator_change','adaptive_scaling_change','clock_change','roundoff_change'])
            v['gradient_time_cosine']=cosine(v['g_cross_time'],v['g_squared'],v['previous_g_squared'],floor);v['channel_time_cosine']=cosine(v['channel_cross_time'],v['channel_squared'],v['previous_channel_squared'],floor)
            if pc is not None and v['cosine'] is not None:
                # Exact hierarchical symmetric split of P=(||g|| ||u||)*cos(theta).
                q=v['gradient_norm']*v['channel_norm'];qp=pn*un;ca=(v['cosine']+pc)/2
                v['orientation_effect']=(q+qp)/2*(v['cosine']-pc)
                v['gradient_magnitude_effect']=ca*(v['channel_norm']+un)/2*(v['gradient_norm']-pn)
                v['channel_magnitude_effect']=ca*(v['gradient_norm']+pn)/2*(v['channel_norm']-un)
                v['magnitude_orientation_closure']=v['projection_change']-sum(v[k] for k in ['orientation_effect','gradient_magnitude_effect','channel_magnitude_effect'])
            else:
                for k in ['orientation_effect','gradient_magnitude_effect','channel_magnitude_effect','magnitude_orientation_closure']:v[k]=None
        output.append(v)
    ar=[dict(population=p,component=c,layer=l,projections=v) for (p,c,l),v in age_rows.items()]
    return dict(geometry=output,age_contributions=ar,component_losses=losses,age_audit=dict(moment_residual_norm=math.sqrt(reconstruction_sq),moment_norm=math.sqrt(moment_sq),history_residual_norm=math.sqrt(history_residual_sq),history_norm=math.sqrt(history_sq)))
