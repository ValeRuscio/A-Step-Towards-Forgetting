"""Named, auditable optimizer variants. Zero weight decay throughout this study.
Appendix variants are not benchmarks of tuned official distributed implementations.
"""
import math,torch

def matrix_parameter(name,p):
    return p.ndim==2 and ('.layers.' in name) and not any(s in name for s in ['embed','lm_head','embed_out'])

def inv_fourth(a,damping):
    # Relative trace damping plus an absolute floor; all eigenvalue work is FP64.
    x=(a.double()+a.double().T)*.5
    scale=max(float(x.diagonal().mean()),1e-12);ridge=damping*scale+1e-12
    vals,vec=torch.linalg.eigh(x);vals=vals.clamp_min(0)+ridge
    z=(vec*vals.pow(-.25))@vec.T
    if not torch.isfinite(z).all():raise FloatingPointError('Shampoo inverse root is not finite')
    return z.to(a.dtype),dict(min_eigenvalue=float(vals.min()),max_eigenvalue=float(vals.max()),ridge=ridge)

def orthogonalize(g,steps=5):
    # Muon quintic Newton-Schulz approximation, FP32 (not BF16), no exact-SVD claim.
    x=g.float();transposed=x.shape[0]>x.shape[1]
    if transposed:x=x.T
    x=x/(x.norm()+1e-7)
    for _ in range(steps):
        a=x@x.T;b=-4.7750*a+2.0315*(a@a);x=3.4445*x+b@x
    return x.T if transposed else x

class MatrixHybrid(torch.optim.Optimizer):
    def __init__(self,named,kind,lr,s):
        groups=[];names={};self.variant=kind;self.last_audit={}
        for ismatrix in [True,False]:
            pairs=[(n,p) for n,p in named if matrix_parameter(n,p)==ismatrix]
            if pairs:
                groups.append(dict(params=[p for n,p in pairs],matrix=ismatrix,names=[n for n,p in pairs],lr=lr if ismatrix else lr*s['aux_lr_ratio'][kind]))
        defaults=dict(momentum=s['momentum'],beta2=s['beta2'],eps=s['eps'],block=s['shampoo_block'],root_frequency=s['shampoo_root_frequency'],damping=s['shampoo_damping'],ns_steps=s['muon_ns_steps'])
        super().__init__(groups,defaults)
    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise NotImplementedError('No closures')
        audit=dict(variant=self.variant,root_refreshes=0,max_preconditioner_condition=0.)
        for pg in self.param_groups:
            for p in pg['params']:
                if p.grad is None:continue
                g=p.grad;st=self.state[p];st['step']=int(st.get('step',0))+1;t=st['step'];mu=pg['momentum']
                if not pg['matrix']:
                    if 'exp_avg' not in st:st['exp_avg']=torch.zeros_like(p);st['exp_avg_sq']=torch.zeros_like(p)
                    st['exp_avg'].lerp_(g,1-mu);st['exp_avg_sq'].mul_(pg['beta2']).addcmul_(g,g,value=1-pg['beta2'])
                    d=st['exp_avg']/(1-mu**t)/(st['exp_avg_sq'].sqrt()/math.sqrt(1-pg['beta2']**t)+pg['eps'])
                elif self.variant=='muon_fp32':
                    if 'momentum_buffer' not in st:st['momentum_buffer']=torch.zeros_like(p)
                    st['momentum_buffer'].lerp_(g,1-mu)
                    mixed=g.lerp(st['momentum_buffer'],mu) # Nesterov EMA convention
                    d=orthogonalize(mixed,pg['ns_steps'])*math.sqrt(max(1.,p.shape[0]/p.shape[1]))
                else:
                    bs=pg['block'];pre=torch.empty_like(g)
                    if 'blocks' not in st:
                        st['blocks']=[]
                        for i in range(0,p.shape[0],bs):
                            for j in range(0,p.shape[1],bs):
                                nr=min(bs,p.shape[0]-i);nc=min(bs,p.shape[1]-j)
                                st['blocks'].append(dict(i=i,j=j,left=torch.zeros(nr,nr,device=p.device),right=torch.zeros(nc,nc,device=p.device),li=torch.eye(nr,device=p.device),ri=torch.eye(nc,device=p.device)))
                    for b in st['blocks']:
                        i,j=b['i'],b['j'];nr,nc=b['left'].shape[0],b['right'].shape[0];gg=g[i:i+nr,j:j+nc]
                        b['left'].add_(gg@gg.T);b['right'].add_(gg.T@gg)
                        if t==1 or t%pg['root_frequency']==0:
                            b['li'],la=inv_fourth(b['left'],pg['damping']);b['ri'],ra=inv_fourth(b['right'],pg['damping'])
                            audit['root_refreshes']+=2;audit['max_preconditioner_condition']=max(audit['max_preconditioner_condition'],la['max_eigenvalue']/la['min_eigenvalue'],ra['max_eigenvalue']/ra['min_eigenvalue'])
                        pre[i:i+nr,j:j+nc]=b['li']@gg@b['ri']
                    # Whole-matrix SGD-norm grafting before momentum, explicitly part of variant.
                    pn=float(pre.norm());gn=float(g.norm())
                    if pn>0:pre.mul_(gn/pn)
                    elif gn>0:raise FloatingPointError('Zero Shampoo direction with nonzero gradient')
                    if 'momentum_buffer' not in st:st['momentum_buffer']=torch.zeros_like(p)
                    st['momentum_buffer'].mul_(mu).add_(pre);d=st['momentum_buffer']
                if not torch.isfinite(d).all():raise FloatingPointError('Nonfinite optimizer direction')
                p.add_(d,alpha=-pg['lr'])
        self.last_audit=audit

def make_optimizer(e,name,lr,s):
    if name=='sgd':return torch.optim.SGD(e.model.parameters(),lr=lr,momentum=0,weight_decay=0,foreach=False)
    if name=='momentum_sgd':return torch.optim.SGD(e.model.parameters(),lr=lr,momentum=s['momentum'],dampening=0,nesterov=False,weight_decay=0,foreach=False)
    if name in ['adam','adam_beta1_0']:
        return torch.optim.Adam(e.model.parameters(),lr=lr,betas=(s['momentum'] if name=='adam' else 0.,s['beta2']),eps=s['eps'],weight_decay=0,foreach=False)
    if name in ['shampoo_block','muon_fp32']:return MatrixHybrid(list(e.params.items()),name,lr,s)
    raise ValueError(name)

def native_channels(e,name):
    """Channels sum to the proposed step, up to floating-point rounding. No Muon split."""
    if name in ['adam','adam_beta1_0']:
        from study_math import channels
        return channels(e)
    if name not in ['sgd','momentum_sgd']:return None
    h={};c={};names={id(p):n for n,p in e.params.items()}
    for pg in e.opt.param_groups:
        for p in pg['params']:
            if p.grad is None:continue
            n=names[id(p)];old=e.opt.state[p].get('momentum_buffer',torch.zeros_like(p))
            h[n]=(-pg['lr']*pg.get('momentum',0)*old).detach().cpu().clone();c[n]=(-pg['lr']*p.grad).detach().cpu().clone()
    return h,c

def reset_history(e):
    changed=0
    for st in e.opt.state.values():
        for k in ['momentum_buffer','exp_avg']:
            if k in st:st[k].zero_();changed+=1
    return changed
