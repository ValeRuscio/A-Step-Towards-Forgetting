"""Full-population gradients and channel bilinear Hessian forms, no surrogate Hessians."""
import math,re,hashlib
import torch
from study_math import dot,norm
COMPONENTS=('total','confusion','leakage')

def layer(n):
    m=re.search(r'(?:^|\.)layers\.(\d+)\.',n)
    return 'block_'+m.group(1) if m else 'embedding_readout_other'

def digest(x):
    h=hashlib.sha256()
    for n,v in sorted(x.items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def directions(e,before,after):
    from study_math import channels
    # Call BEFORE optimizer.step, then complete with actual endpoint separately.
    return channels(e)

def complete_directions(before,after,h,c):
    # Double subtraction retains the exact displacement between FP32 stored endpoints.
    d={n:after[n].double()-before[n].double() for n in before}
    h={n:h.get(n,torch.zeros_like(v)).double() for n,v in before.items()}
    c={n:c.get(n,torch.zeros_like(v)).double() for n,v in before.items()}
    r={n:d[n]-h[n]-c[n] for n in d}
    return dict(d=d,h=h,c=c,r=r)

def tensor_dot(gs,vs):
    return sum((g.double()*v.double()).sum() for g,v in zip(gs,vs) if g is not None)

def component_gradients(e,rows,batch,ctl=None):
    """Two exact population backward passes; total is their algebraic sum."""
    ps=list(e.params.values());out={};losses={}
    for k,idx in [('confusion',1),('leakage',2)]:
        acc={n:torch.zeros_like(p,device='cpu',dtype=torch.float64) for n,p in e.params.items()};loss_sum=0.
        for i in range(0,len(rows),batch):
            if ctl:ctl.check()
            vals=e.loss_rows(rows[i:i+batch],True)[idx];loss=vals.sum()/len(rows)
            g=torch.autograd.grad(loss,ps,allow_unused=True)
            for (n,p),v in zip(e.params.items(),g):
                if v is not None:acc[n].add_(v.detach().cpu().double())
            loss_sum+=float(loss.detach());del g,vals,loss
            if ctl:ctl.pulse(completed_prompts=min(i+batch,len(rows)),population_size=len(rows))
        out[k]=acc;losses[k]=loss_sum
    losses['total']=losses['confusion']+losses['leakage']
    return out,losses

def projections(grads,dirs,training=None,with_layers=True):
    out={}
    for k,g in grads.items():
        n=norm(g);p={v:dot(g,d) for v,d in dirs.items()}
        row=dict(gradient_norm=n,slopes=p,cosines={v:p[v]/(n*norm(d)) if n*norm(d)>0 else None for v,d in dirs.items()})
        if training is not None:row['training_dot']=dot(g,training)
        if with_layers:
            groups={}
            for name in g:groups.setdefault(layer(name),[]).append(name)
            row['layers']={}
            for key,names in groups.items():
                gg={n:g[n] for n in names};row['layers'][key]=dict(gradient_norm=norm(gg),slopes={v:dot(gg,d) for v,d in dirs.items()})
        out[k]=row
    # Compute total projections and norm without storing a third parameter-sized vector.
    a,b=grads['confusion'],grads['leakage'];gn=math.sqrt(max(dot(a,a)+dot(b,b)+2*dot(a,b),0.))
    p={v:out['confusion']['slopes'][v]+out['leakage']['slopes'][v] for v in dirs}
    out['total']=dict(gradient_norm=gn,slopes=p,cosines={v:p[v]/(gn*norm(d)) if gn*norm(d)>0 else None for v,d in dirs.items()})
    if training is not None:out['total']['training_dot']=dot(a,training)+dot(b,training)
    if with_layers:
        out['total']['layers']={key:{'slopes':{v:out['confusion']['layers'][key]['slopes'][v]+out['leakage']['layers'][key]['slopes'][v] for v in dirs}} for key in out['confusion']['layers']}
    return out

def curvature(e,rows,dirs,batch,ctl=None):
    """Three HVPs per component. Includes residual cross terms and mixed symmetry.
    No Hessian array, top-eigenvalue proxy, or subtraction of observed losses.
    """
    names=list(e.params);ps=list(e.params.values())
    vs={k:[v[n].to(device=p.device,dtype=p.dtype) for n,p in e.params.items()] for k,v in dirs.items()}
    result={}
    for component,idx in [('confusion',1),('leakage',2)]:
        sums={k:0. for k in ['dd','hh','hc','ch','cc','hr','cr','rr','direction_sum_residual']}
        for i in range(0,len(rows),batch):
            if ctl:ctl.check()
            loss=e.loss_rows(rows[i:i+batch],True)[idx].sum()/len(rows)
            gs=torch.autograd.grad(loss,ps,create_graph=True,allow_unused=True)
            local={}
            for pos,k in enumerate(['d','h','c']):
                slope=tensor_dot(gs,vs[k]);hv=torch.autograd.grad(slope,ps,retain_graph=pos<2,allow_unused=True)
                local[k]={j:float(tensor_dot(hv,vs[j]).detach()) for j in ['d','h','c','r']}
                del hv,slope
            vals=dict(dd=local['d']['d'],hh=local['h']['h'],hc=local['h']['c'],ch=local['c']['h'],cc=local['c']['c'],hr=local['h']['r'],cr=local['c']['r'],rr=local['d']['r']-local['h']['r']-local['c']['r'])
            vals['direction_sum_residual']=vals['dd']-(vals['hh']+vals['hc']+vals['ch']+vals['cc']+2*vals['hr']+2*vals['cr']+vals['rr'])
            for k,v in vals.items():sums[k]+=v
            del loss,gs,local
            if ctl:ctl.pulse(completed_prompts=min(i+batch,len(rows)),population_size=len(rows))
        result[component]=sums
    result['total']={k:result['confusion'][k]+result['leakage'][k] for k in result['confusion']}
    return result
