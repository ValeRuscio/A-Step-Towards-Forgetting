"""Exact FP32 Adam displacements; FP64 reductions; no persistent gradient arrays."""
import math,copy
import numpy as np
import torch
from association_core import collate
from multimodel_engine import MultiEngine

class StudyEngine(MultiEngine):
    def restore(self,z):
        # load_state_dict may retain CPU tensor references. Branches must never
        # mutate the shared anchor's moments, even in CPU tests.
        self.opt.zero_grad(set_to_none=True)
        self.model.load_state_dict(z['model'])
        self.opt.load_state_dict(copy.deepcopy(z['optimizer']))

    def loss_rows(self,rows,details=False):
        inputs,q,labels=collate(rows,self.tok.pad_token_id,self.c.device)
        backbone=self.model.gpt_neox if self.kind=='gpt_neox' else self.model.model
        head=self.model.embed_out if self.kind=='gpt_neox' else self.model.lm_head
        logits=head(backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:])
        lp=logits.double().log_softmax(-1)
        # xlogy(0,0)=0: exact one-hot targets, without smoothing or NaNs.
        loss=(torch.special.xlogy(q,q)-q*lp[:,labels]).sum(-1)
        mass=-lp[:,labels].logsumexp(-1);conditional=loss-mass
        if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite loss')
        return (loss,conditional,mass) if details else loss

def dot(a,b):return sum(float((v.double()*b[n].double()).sum()) for n,v in a.items() if n in b)
def norm(a):return math.sqrt(max(dot(a,a),0.))
def plus(a,b):return {n:v+b[n] for n,v in a.items()}
def weights(e):return {n:p.detach().cpu().clone() for n,p in e.params.items()}
def assign(e,x):
    with torch.no_grad():
        for n,p in e.params.items():p.copy_(x[n].to(p.device))
def difference(a,b):return {n:a[n]-b[n] for n in a}
def grad(e,rows,batch,ctl=None):
    e.opt.zero_grad(set_to_none=True)
    for i in range(0,len(rows),batch):
        if ctl:ctl.check()
        (e.loss_rows(rows[i:i+batch]).sum()/len(rows)).backward()
        if ctl:ctl.pulse(completed_prompts=min(i+batch,len(rows)),population_size=len(rows))
    g={n:(p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu')) for n,p in e.params.items()}
    e.opt.zero_grad(set_to_none=True);return g

def evaluate(e,rows,batch=4,ctl=None):
    result=[]
    with torch.no_grad():
        for i in range(0,len(rows),batch):
            if ctl:ctl.check()
            block=rows[i:i+batch];inputs,q,labels=collate(block,e.tok.pad_token_id,e.c.device)
            backbone=e.model.gpt_neox if e.kind=='gpt_neox' else e.model.model
            head=e.model.embed_out if e.kind=='gpt_neox' else e.model.lm_head
            logits=head(backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]).double()
            lp=logits.log_softmax(-1);loss=(torch.special.xlogy(q,q)-q*lp[:,labels]).sum(-1)
            leakage=-lp[:,labels].logsumexp(-1);target=q.argmax(-1);chosen=logits[:,labels].argmax(-1)
            for j,r in enumerate(block):
                result.append(dict(entity=r['entity'],loss=float(loss[j]),confusion=float(loss[j]-leakage[j]),leakage=float(leakage[j]),answer_mass=float((-leakage[j]).exp()),restricted_accuracy=float(chosen[j]==target[j]),full_accuracy=float(logits[j].argmax()==labels[target[j]])))
    return dict(mean=float(np.mean([r['loss'] for r in result])),rows=result,**{k:float(np.mean([r[k] for r in result])) for k in ['confusion','leakage','answer_mass','restricted_accuracy','full_accuracy']})

def channels(e):
    names={id(p):n for n,p in e.params.items()};h={};c={}
    for pg in e.opt.param_groups:
        if pg.get('weight_decay',0) or pg.get('amsgrad',False):raise ValueError('Only Adam without decay/AMSGrad')
        b1,b2=pg['betas']
        for p in pg['params']:
            if p.grad is None:continue
            st=e.opt.state[p];t=int(st['step'])+1 if 'step' in st else 1
            m=st.get('exp_avg',torch.zeros_like(p));v=st.get('exp_avg_sq',torch.zeros_like(p))
            den=(b2*v+(1-b2)*p.grad.square()).sqrt()/math.sqrt(1-b2**t)+pg['eps'];fac=-pg['lr']/(1-b1**t)
            h[names[id(p)]]=(fac*b1*m/den).detach().cpu();c[names[id(p)]]=(fac*(1-b1)*p.grad/den).detach().cpu()
    return h,c

def reset(e,mode):
    if mode=='preserve':return
    if mode=='fresh':e.reset_optimizer();return
    keys={'reset_m':['exp_avg'],'reset_v':['exp_avg_sq'],'reset_both':['exp_avg','exp_avg_sq']}[mode]
    for st in e.opt.state.values():
        for key in keys:
            if key in st:st[key].zero_()

def geometry(gA,gB,h,c,delta,train_g):
    vec=[h,c,gA,gB]
    def block(v):
        g=[[dot(x,y) for y in v] for x in v];ns=[math.sqrt(max(g[i][i],0)) for i in range(4)]
        return dict(gram=g,norms=ns,cosines=[[g[i][j]/(ns[i]*ns[j]) if ns[i]*ns[j]>0 else None for j in range(4)] for i in range(4)])
    layers={}
    for n in h:
        parts=n.split('.');key=parts[2] if len(parts)>3 and parts[1]=='layers' else 'other'
        layers.setdefault(key,[]).append(n)
    dn=norm(delta);an=norm(gA);s=dot(gA,delta)
    return dict(global_geometry=block(vec),layers={k:block([{n:x[n] for n in names} for x in vec]) for k,names in layers.items()},projection=s,update_norm=dn,gA_norm=an,update_cosine=s/(dn*an) if dn*an else 0.,population_interference=-dot(gA,gB),training_interference=-dot(gA,train_g),roundoff_norm=norm(difference(delta,plus(h,c))))

# Proper Procrustes map in the joint row span; identity outside the fitted span.
def fit_frame(x,y):
    _,sv,vt=np.linalg.svd(np.concatenate([x,y]).astype(float),full_matrices=False)
    rank=int(np.sum(sv>max(sv[0],1e-30)*1e-10))
    if rank==0:raise ValueError('Zero-rank frame')
    basis=vt[:rank].T;a=np.einsum('ij,jk->ik',x,basis);b=np.einsum('ij,jk->ik',y,basis);u,_,v=np.linalg.svd(np.einsum('ji,jk->ik',a,b));sg=np.ones(rank)
    if np.linalg.det(np.einsum('ij,jk->ik',u,v))<0:sg[-1]=-1
    return dict(basis=basis,rot=np.einsum('ij,jk->ik',u*sg,v),shift=y.mean(0)-x.mean(0),scale=float(np.linalg.norm(y)/max(np.linalg.norm(x),1e-30)))

def activation_probe(e,rows,layer,frame=None,condition='unpatched',seed=0,ctl=None):
    module=(e.model.gpt_neox if e.kind=='gpt_neox' else e.model.model).layers[layer];acts=[];records=[]
    for row in rows:
        def hook(mod,args,out):
            z=out[0] if isinstance(out,tuple) else out;x=z[:,-1,:];y=x
            if frame is not None and condition not in ['unpatched','sham']:
                b=torch.as_tensor(frame['basis'],device=x.device,dtype=x.dtype);r=torch.as_tensor(frame['rot'],device=x.device,dtype=x.dtype);eye=torch.eye(len(r),device=x.device)
                correction=(x@b)@(r-eye)@b.T
                if condition=='rotation':y=x+correction
                elif condition=='inverse':y=x+(x@b)@(r.T-eye)@b.T
                elif condition=='mean':y=x+torch.as_tensor(frame['shift'],device=x.device,dtype=x.dtype)
                elif condition=='scale':y=x*frame['scale']
                elif condition.startswith('orthogonal'):
                    bb=torch.as_tensor(frame[condition],device=x.device,dtype=x.dtype);y=x+(x@bb)@(r-eye)@bb.T
                elif condition.startswith('additive'):
                    rng=np.random.default_rng(seed+int(row['entity'])*1009);a=torch.tensor(rng.normal(size=x.shape),device=x.device,dtype=x.dtype);y=x+a/a.norm(dim=-1,keepdim=True).clamp_min(1e-30)*correction.norm(dim=-1,keepdim=True)
                else:raise ValueError(condition)
            acts.append(y.detach().cpu().double().numpy()[0]);zz=z.clone();zz[:,-1,:]=y
            return (zz,)+out[1:] if isinstance(out,tuple) else zz
        handle=module.register_forward_hook(hook)
        try:records.extend(evaluate(e,[row],1,ctl)['rows'])
        finally:handle.remove()
    return dict(mean=float(np.mean([r['loss'] for r in records])),rows=records),np.asarray(acts)

def directional_hessian(e,rows,direction,batch,ctl=None):
    """Exact autograd d^T H d, streamed over prompts; may fail on unsupported kernels."""
    params=list(e.params.values());directions=[direction[n].to(p.device) for n,p in e.params.items()];total=0.
    for i in range(0,len(rows),batch):
        if ctl:ctl.check()
        loss=e.loss_rows(rows[i:i+batch]).sum()/len(rows)
        g=torch.autograd.grad(loss,params,create_graph=True,allow_unused=True)
        slope=sum((v*d).sum() for v,d in zip(g,directions) if v is not None)
        hv=torch.autograd.grad(slope,params,allow_unused=True)
        total+=sum(float((v.double()*d.double()).sum()) for v,d in zip(hv,directions) if v is not None)
        del g,hv,slope,loss
        if ctl:ctl.pulse(completed_prompts=min(i+batch,len(rows)),population_size=len(rows))
    return total
