"""Finite-update, answer-bias and restricted loud/quiet-direction experiments.
Self-contained runner for the original OLMo association checkpoint format.
Run --help; use the companion notebook for detached launch/manual refresh.
The copied association core below preserves the source experiment semantics.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/forgetting_mechanisms_mpl")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Original association core: config.py
from dataclasses import dataclass,asdict
from pathlib import Path
import json,os,time,hashlib
import torch

@dataclass
class Config:
    model: str='allenai/OLMo-1B-hf'
    revision: str='main'
    device: str='cuda:0'
    seeds: tuple=(1,2,3)
    lr: float=2e-5
    beta1: float=.9
    beta2: float=.99
    eps: float=1e-8
    clip: float=1.
    a_steps: int=600
    b_steps: int=128
    eval_every: int=16
    forks: tuple=(16,48,96)
    horizon: int=32
    washout: int=32
    period: int=8
    ages: tuple=(4,5,6,7)
    microbatch: int=2
    accumulation: int=2
    entities: int=32
    max_length: int=96
    eval_entities: int=16
    transition: str='keep'
    hours: float=8.
    threads: int=8
    attn: str='sdpa'
    smoke: bool=False
    def validate(self):
        if not self.seeds or len(set(self.seeds))!=len(self.seeds):raise ValueError('Provide distinct replicate seeds')
        if not self.ages or self.period<=0 or self.eval_every<=0 or self.max_length<=0 or self.washout<0:raise ValueError('Invalid age, schedule or length setting')
        if len(set(self.ages))!=len(self.ages) or len(set(self.forks))!=len(self.forks):raise ValueError('Duplicate ages or forks')
        if self.transition not in ['keep','reset']:raise ValueError('transition must be keep or reset')
        if min(self.lr,self.clip,self.eps,self.a_steps,self.b_steps,self.horizon,self.microbatch,self.accumulation,self.entities,self.hours,self.threads)<=0:raise ValueError('Positive settings required')
        if not 0<self.beta1<1 or not 0<self.beta2<1:raise ValueError('Invalid Adam betas')
        if not self.forks or any(t<=max(self.ages) or t>self.b_steps or t%self.period for t in self.forks):raise ValueError('Forks must be valid intervention steps')
        if not 1<=min(self.ages)<=max(self.ages)<self.period:raise ValueError('Cohorts must be strictly historical and disjoint')
        if not 1<=self.eval_entities<=self.entities:raise ValueError('Invalid evaluation size')
        return self

def dump(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def read(p):return json.loads(Path(p).read_text())
def save(x,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');torch.save(x,q);q.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
class Budget:
    def __init__(self,hours):self.end=time.monotonic()+hours*3600
    def check(self):
        if time.monotonic()>self.end:raise TimeoutError('Budget exhausted; completed work and current trajectory are resumable')
def source_hash():
    root=Path(__file__).parent
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}

# Original association core: data.py
"""Exact two-token conditional relationships in a full-vocabulary language model."""
import random,hashlib,json
import torch

class ToyTokenizer:
    pad_token_id=0
    def encode(self,s,add_special_tokens=False):return [1+ord(c)%58 for c in s]

def make_data(tokenizer,c,seed):
    if c.smoke:labels=[61,62]
    else:
        labels=None
        for pair in [(' red',' blue'),(' yes',' no'),(' A',' B')]:
            ids=[tokenizer.encode(s,add_special_tokens=False) for s in pair]
            if all(len(i)==1 for i in ids) and ids[0]!=ids[1]:labels=[i[0] for i in ids];break
        if labels is None:raise ValueError('Cannot find two distinct single-token answer labels')
    rng=random.Random(160001+seed);result={}
    # Same entity in each split, different prompt form. This tests learned relations on new contexts,
    # not generalization to entities whose arbitrary label was never taught.
    templates={'train':['Registry {d}. Object {e}. Category:', 'In registry {d}, the category of object {e} is:'],
               'valid':['Registry {d} lookup. Category for object {e}:'],
               'test':['Object {e}, registry {d}. Its category is:']}
    for domain in ['A','B']:
        probs=[.9 if rng.randrange(2) else .1 for _ in range(c.entities)]
        for split,ts in templates.items():
            rows=[]
            for e in range(c.entities):
                for fmt in ts:
                    prompt=fmt.format(d=domain,e=f'{e:03d}');ids=tokenizer.encode(prompt,add_special_tokens=False)
                    if len(ids)>c.max_length:raise ValueError(f'Prompt has {len(ids)} tokens, exceeds max_length={c.max_length}; no silent truncation')
                    rows.append(dict(ids=ids,q=[probs[e],1-probs[e]],labels=labels,entity=e,prompt=prompt))
            result[domain+'_'+split]=rows
    sets=[set(tuple(r['ids']) for r in rows) for rows in result.values()]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):raise ValueError('Tokenized complete prefixes overlap across splits')
    return result

def minibatches(data,domain,seed,t,c):
    rng=random.Random(7000001+seed*100003+t*101);rows=data[domain+'_train']
    return [[rows[rng.randrange(len(rows))] for _ in range(c.microbatch)] for _ in range(c.accumulation)]
def collate(rows,pad,device):
    n=max(len(r['ids']) for r in rows);ids=[];mask=[]
    for r in rows:
        k=n-len(r['ids']);ids.append([pad]*k+r['ids']);mask.append([0]*k+[1]*len(r['ids']))
    mask=torch.tensor(mask,device=device);pos=(mask.cumsum(-1)-1).clamp_min(0)
    return dict(input_ids=torch.tensor(ids,device=device),attention_mask=mask,position_ids=pos),torch.tensor([r['q'] for r in rows],device=device,dtype=torch.float64),torch.tensor(rows[0]['labels'],device=device)
def digest(data):return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()

# Original association core: engine.py
"""Full FP32 OLMo/Adam training and audited tokenwise gradient factors."""
from pathlib import Path
import gc,inspect,math,time
import torch

class Engine:
    def __init__(self,c):
        self.c=c;torch.set_num_threads(c.threads);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        if c.device.startswith('cuda'):
            if not torch.cuda.is_available():raise RuntimeError('CUDA not available')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if hasattr(torch.backends.cuda,'enable_flash_sdp'):torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends.cuda,'enable_mem_efficient_sdp'):torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        from transformers import AutoModelForCausalLM,AutoTokenizer,OlmoConfig,OlmoForCausalLM
        if c.smoke:
            self.tok=ToyTokenizer();cfg=OlmoConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,pad_token_id=0,eos_token_id=63)
            cfg._attn_implementation='eager';self.model=OlmoForCausalLM(cfg)
        else:
            self.tok=AutoTokenizer.from_pretrained(c.model,revision=c.revision,trust_remote_code=False)
            # torch_dtype works across Transformers 4.x and 5.x (new versions may warn).
            self.model=AutoModelForCausalLM.from_pretrained(c.model,revision=c.revision,torch_dtype=torch.float32,attn_implementation=c.attn,trust_remote_code=False)
            if self.model.config.model_type!='olmo':raise ValueError('This tested implementation expects original OLMo, not OLMo2/3')
        if self.tok.pad_token_id is None:self.tok.pad_token_id=self.tok.eos_token_id
        if self.tok.pad_token_id is None:raise ValueError('Tokenizer needs a padding token')
        self.model.to(c.device,dtype=torch.float32).eval();self.model.config.use_cache=False
        if hasattr(self.model,'gradient_checkpointing_disable'):self.model.gradient_checkpointing_disable()
        self.params=dict(self.model.named_parameters());self.mods=dict(self.model.named_modules())
        n=self.model.config.num_hidden_layers;layers=sorted(set([0,n//2,n-1]))
        self.names=[f'model.layers.{l}.{suffix}' for l in layers for suffix in ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']]
        for name in self.names:
            m=self.mods.get(name)
            if not isinstance(m,torch.nn.Linear) or m.bias is not None:raise ValueError(f'Expected a bias-free Linear at {name}')
        self.selected={n:self.mods[n].weight for n in self.names}
        self.opt=torch.optim.Adam(self.model.parameters(),lr=c.lr,betas=(c.beta1,c.beta2),eps=c.eps,foreach=False)
        self.pb=sum(p.numel()*p.element_size() for p in self.model.parameters())
    def loss_rows(self,rows,details=False):
        inputs,q,labels=collate(rows,self.tok.pad_token_id,self.c.device)
        # Same OLMo forward computation, only final-position head logits are materialized.
        h=self.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
        logits=self.model.lm_head(h);lp=logits.double().log_softmax(-1)
        qq=q.double();loss=(qq*(qq.log()-lp[:,labels])).sum(-1)
        if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite exact conditional KL')
        mass=-lp[:,labels].logsumexp(-1);relation=loss-mass
        return (loss,relation,mass) if details else loss
    @torch.no_grad()
    def evaluate(self,rows):
        vals=[]
        for i in range(0,len(rows),self.c.microbatch):vals.extend(self.loss_rows(rows[i:i+self.c.microbatch]).cpu().tolist())
        return vals
    @torch.no_grad()
    def evaluate_details(self,rows):
        out=[[],[],[]]
        for i in range(0,len(rows),self.c.microbatch):
            vals=self.loss_rows(rows[i:i+self.c.microbatch],True)
            for a,b in zip(out,vals):a.extend(b.cpu().tolist())
        return out
    def gradient(self,batches,capture=False):
        self.opt.zero_grad(set_to_none=True);records={n:[] for n in self.names};handles=[]
        def hook(name):
            def forward(mod,inputs,output):
                h=inputs[0].detach().reshape(-1,inputs[0].shape[-1]).float().cpu().clone();slot={'h':h};records[name].append(slot)
                def backward(g):slot['e']=g.detach().reshape(-1,g.shape[-1]).float().cpu().clone()
                output.register_hook(backward)
            return forward
        if capture:handles=[self.mods[n].register_forward_hook(hook(n)) for n in self.names]
        try:
            for rows in batches:(self.loss_rows(rows).mean()/len(batches)).backward()
        finally:
            for h in handles:h.remove()
        norm=math.sqrt(sum(float(p.grad.double().square().sum()) for p in self.params.values() if p.grad is not None))
        if not math.isfinite(norm):raise FloatingPointError('Nonfinite global gradient norm; optimizer step refused')
        clip=min(1.,self.c.clip/max(norm,1e-30));audit={}
        if capture:
            for name,p in self.selected.items():
                if len(records[name])!=len(batches) or any('e' not in x for x in records[name]):raise RuntimeError('Repeated/missing hook calls; checkpointing and dropout must be disabled')
                # Matrix-sized reconstruction streamed one module at a time on GPU.
                recon=sum((v['e'].to(p.device).T@v['h'].to(p.device) for v in records[name]),torch.zeros_like(p))
                err=float((recon-p.grad).double().norm()/p.grad.double().norm().clamp_min(1e-20));audit[name]=err
                if not math.isfinite(err) or err>2e-4:raise ArithmeticError(f'Outer-product audit failed: {name} relative error={err}')
        for p in self.params.values():
            if p.grad is not None:p.grad.mul_(clip)
        return dict(factors=records,clip=clip,raw_norm=norm,audit=audit)
    def pack(self):
        def cpu(x):
            if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
            if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
            if isinstance(x,list):return [cpu(v) for v in x]
            return x
        return dict(model=cpu(self.model.state_dict()),optimizer=cpu(self.opt.state_dict()))
    def restore(self,z):
        self.opt.zero_grad(set_to_none=True);self.model.load_state_dict(z['model']);self.opt.load_state_dict(z['optimizer'])
    def reset_optimizer(self):self.opt=torch.optim.Adam(self.model.parameters(),lr=self.c.lr,betas=(self.c.beta1,self.c.beta2),eps=self.c.eps,foreach=False)
    def moments(self):
        return {n:{k:self.opt.state[p][k].detach().clone() for k in ['exp_avg','exp_avg_sq','step']} for n,p in self.selected.items()}
    @torch.no_grad()
    def nominal_step(self):
        before={n:p.detach().clone() for n,p in self.selected.items()};self.opt.step()
        return before,{n:p.detach()-before[n] for n,p in self.selected.items()}

# New experiment functions ----------------------------------------------------
import argparse, contextlib, csv, json, platform, shutil, signal, subprocess, sys
import threading, traceback, uuid, zipfile
import numpy as np


def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def chunks(x, size=1048576):
    flat = x.reshape(-1)
    for start in range(0, flat.numel(), size): yield flat[start:start + size]


def vdot(a, b):
    return sum(float(torch.dot(x.double(), y.double()))
               for k in a for x, y in zip(chunks(a[k]), chunks(b[k])))


def vnorm(a): return math.sqrt(max(0., vdot(a, a)))


def scaled(a, scale): return {k: v * float(scale) for k, v in a.items()}


def add_vectors(a, b): return {k: a[k] + b[k] for k in a}


@torch.no_grad()
def inject(e, base, direction=None, scale=1.):
    """Always reset from base; report FP32 realization error, including tied weights once."""
    err = size = 0.
    for name, p in e.params.items():
        target = base[name] if direction is None else base[name] + scale * direction[name]
        p.copy_(target.to(p.device))
        if direction is not None:
            actual = p.detach().cpu() - base[name]
            intended = scale * direction[name]
            for x, y in zip(chunks(actual), chunks(intended)):
                err += float((x.double() - y.double()).square().sum())
                size += float(y.double().square().sum())
    return math.sqrt(err / max(size, 1e-30))


@torch.no_grad()
def predictions(e, rows):
    result = []
    for i in range(0, len(rows), e.c.microbatch):
        batch = rows[i:i+e.c.microbatch]
        inputs, q, labels = collate(batch, e.tok.pad_token_id, e.c.device)
        h = e.model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[:, -1, :]
        lp = e.model.lm_head(h).double().log_softmax(-1)[:, labels]
        full = (q * (q.log() - lp)).sum(-1)
        mass = -lp.logsumexp(-1)
        margins = lp[:, 0] - lp[:, 1]
        for r, f, m, z in zip(batch, full.tolist(), mass.tolist(), margins.tolist()):
            if not all(math.isfinite(x) for x in [f,m,z]): raise ArithmeticError('Nonfinite prediction')
            result.append(dict(entity=r['entity'], q=r['q'], full=f, mass=m,
                               relation=f-m, margin=z))
    return result


def mean_loss(rows, key='full'): return float(np.mean([x[key] for x in rows]))


def relation_kl(q, margin):
    q = np.asarray(q, dtype=np.float64); margin = np.asarray(margin, dtype=np.float64)
    return np.logaddexp(0., margin) - q[:, 0]*margin + (q*np.log(q)).sum(axis=1)


def bias_accounting(before, after):
    if [r['entity'] for r in before] != [r['entity'] for r in after]:
        raise ValueError('Mismatched entity ordering')
    z0 = np.array([r['margin'] for r in before]); z1 = np.array([r['margin'] for r in after])
    q = np.array([r['q'] for r in before]); shift = float(np.mean(z1-z0))
    old = float(relation_kl(q,z0).mean()); actual = float(relation_kl(q,z1).mean()-old)
    common = float(relation_kl(q,z0+shift).mean()-old)
    residual = float(relation_kl(q,z1-shift).mean()-old)
    return dict(mean_shift=shift, shift_sd=float(np.std(z1-z0)),
                actual_relation_change=actual, common_shift_only=common,
                residual_only=residual, interaction=actual-common-residual,
                interpretation='Output counterfactual; not internal causal mediation or fraction of memory erased.')


def metrics(rows):
    """Ordinary and equal-label-weight loss; missing classes remain explicit."""
    classes = [[r for r in rows if (r['q'][0] > .5) == label] for label in [False, True]]
    return dict(full=mean_loss(rows), relation=mean_loss(rows,'relation'), mass=mean_loss(rows,'mass'),
                balanced_full=float(np.mean([mean_loss(x) for x in classes])) if all(classes) else None,
                class_counts=[len(x) for x in classes],
                accuracy=float(np.mean([(r['margin'] > 0) == (r['q'][0] > .5) for r in rows])))


def write_csv(rows, path):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


class Progress:
    """Heartbeat is separate from progress: a live process may still be stuck."""
    def __init__(self, out):
        self.out = Path(out); self.started = time.time(); self.lock = threading.Lock()
        self.state = dict(status='running', phase='initializing', pid=os.getpid(), started=self.started)
        self.stop = threading.Event(); self.thread = threading.Thread(target=self._heartbeat, daemon=True)
    def __enter__(self): self.thread.start(); return self
    def update(self, **kw):
        with self.lock:
            self.state.update(kw); self.state['last_progress_time'] = time.time(); self._write()
    def _write(self):
        self.state['heartbeat'] = time.time(); self.state['elapsed_seconds'] = time.time()-self.started
        atomic_json(self.state, self.out/'status.json')
    def _heartbeat(self):
        while not self.stop.is_set():
            with self.lock: self._write()
            self.stop.wait(5)
    def __exit__(self, kind, error, tb):
        self.stop.set(); self.thread.join()
        if error:
            self.update(status='failed', error=str(error), traceback=''.join(traceback.format_exception(kind,error,tb)))
        elif self.state['status']=='running': self.update(status='complete')


class VectorBank:
    """Disk-backed normalized full-parameter directions. No P-by-k GPU allocation."""
    def __init__(self, root): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.names=[]
    def put(self,name,v):
        n=vnorm(v)
        if n<=1e-20: return False
        torch.save(scaled(v,1/n),self.root/(name+'.pt')); self.names.append(name); return True
    def get(self,name): return torch.load(self.root/(name+'.pt'),map_location='cpu',weights_only=True)
    def gram(self, progress):
        g=np.zeros((len(self.names),len(self.names)))
        for i,n in enumerate(self.names):
            progress.update(phase='direction Gram matrix', direction=n)
            a=self.get(n)
            for j in range(i+1):
                b=a if i==j else self.get(self.names[j]); g[i,j]=g[j,i]=vdot(a,b)
                if i!=j: del b
        return g
    def combine(self,weights):
        out=None
        for name,w in zip(self.names,weights):
            if abs(w)<1e-15: continue
            v=self.get(name)
            if out is None: out=scaled(v,w)
            else:
                for k in out: out[k].add_(v[k],alpha=float(w))
            del v
        if out is None: out=scaled(self.get(self.names[0]),0.)
        return out


def gradient_vector(e, rows):
    e.opt.zero_grad(set_to_none=True)
    for i in range(0,len(rows),e.c.microbatch):
        (e.loss_rows(rows[i:i+e.c.microbatch]).sum()/len(rows)).backward()
    out={n: (-p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu'))
         for n,p in e.params.items()}
    e.opt.zero_grad(set_to_none=True)
    return out


def margin_probe_gradient(e, rows, weights):
    """Independent reverse-mode check of two scalar projections of each response column."""
    e.opt.zero_grad(set_to_none=True)
    for i in range(0,len(rows),e.c.microbatch):
        batch=rows[i:i+e.c.microbatch]
        inputs,_,labels=collate(batch,e.tok.pad_token_id,e.c.device)
        h=e.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
        z=e.model.lm_head(h)[:,labels].double()
        w=torch.as_tensor(weights[i:i+len(batch)],device=z.device,dtype=torch.float64)
        ((z[:,0]-z[:,1])*w).sum().backward()
    result={n:(p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p,device='cpu'))
            for n,p in e.params.items()}
    e.opt.zero_grad(set_to_none=True)
    return result


def optimizer_channels(e):
    """After clipped gradient, before Adam.step. Split only the first-moment numerator.
    Both components use the SAME updated variance and bias corrections as ordinary Adam.
    This isolates numerator history; it is not a full optimizer reset or refreshed history.
    """
    fresh={}; history={}; sgd={}
    groups={id(p):g for g in e.opt.param_groups for p in g['params']}
    for n,p in e.params.items():
        group=groups[id(p)]; state=e.opt.state[p]
        if group.get('weight_decay',0) or group.get('amsgrad',False) or group.get('maximize',False):
            raise ValueError('Channel formula supports ordinary Adam without decay/AMSGrad/maximize only')
        g=p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device='cpu')
        b1,b2=group['betas']; t=int(state.get('step',0))+1
        m=state.get('exp_avg',torch.zeros_like(p)).detach().cpu()
        v=state.get('exp_avg_sq',torch.zeros_like(p)).detach().cpu()
        den=(v*b2+(1-b2)*g.square()).sqrt()/math.sqrt(1-b2**t)+group['eps']
        factor=-group['lr']/(1-b1**t)
        fresh[n]=factor*(1-b1)*g/den; history[n]=factor*b1*m/den; sgd[n]=-g.clone()
    return fresh,history,sgd


def spectral_decomposition(jacobian, gram, rtol=1e-6):
    """Whiten the parameter metric before naming directions loud/quiet.
    Singular values are of the CALIBRATION margin Jacobian restricted to this span.
    """
    vals,vecs=np.linalg.eigh((gram+gram.T)/2)
    keep=vals>max(float(vals.max())*rtol,1e-12)
    if not keep.any(): raise ArithmeticError('Empty resolved parameter span')
    whitening=vecs[:,keep]/np.sqrt(vals[keep])[None,:]
    j=jacobian@whitening
    _,s,vt=np.linalg.svd(j,full_matrices=True)
    s=np.pad(s,(0,vt.shape[0]-len(s)))
    coeff=whitening@vt.T
    energy=s*s
    rank90=int(np.searchsorted(np.cumsum(energy)/max(energy.sum(),1e-30),.9)+1)
    return dict(singular_values=s.tolist(),rank90=min(rank90,len(s)),parameter_span_rank=int(keep.sum()),
                parameter_gram_eigenvalues=vals.tolist()),coeff


def select_matched(rows, relative=.10, absolute=1e-4):
    """Uses VALIDATION only. An empty match is a result, never silently relaxed."""
    base=next(r for r in rows if r['variant']=='actual' and r['dose']==1.)
    target=-base['B_valid_delta']; tol=max(absolute,relative*abs(target))
    if target<=absolute: return dict(status='no_positive_nominal_validation_gain',target_gain=target,selected=None)
    candidates=[r for r in rows if r['variant']!='actual' and abs(-r['B_valid_delta']-target)<=tol
                and r['injection_error']<=.05]
    if not candidates: return dict(status='no_B_gain_match',target_gain=target,tolerance=tol,selected=None)
    choice=min(candidates,key=lambda r:(r['A_valid_delta'],r['variant'],r['dose']))
    return dict(status='matched',target_gain=target,tolerance=tol,
                selected=dict(variant=choice['variant'],dose=choice['dose']),
                validation_A_delta=choice['A_valid_delta'])


@torch.no_grad()
def token_trace(e,rows,sites):
    """Capture projection output RMS at EVERY prompt token, not just the answer position.
    These are descriptive traces; no attention-map or causal-mediation claim.
    """
    result=[]; active={}; handles=[]
    def hook(name):
        def f(mod,args,y): active[name]=y[0].detach().float().cpu()
        return f
    for n in sites: handles.append(e.mods[n].register_forward_hook(hook(n)))
    try:
        for r in rows:
            active.clear(); predictions(e,[r])
            if set(active)!=set(sites): raise RuntimeError('Incomplete token trace')
            result.append(dict(entity=r['entity'],prompt=r['prompt'],token_ids=r['ids'],
                               rms={n:active[n].double().square().mean(-1).sqrt().tolist() for n in sites}))
    finally:
        for h in handles:h.remove()
    return result


def preflight(e,out):
    import psutil
    # Disk-backed basis + poststep optimizer checkpoint + temporary vectors.
    need_host=10*e.pb+2*1024**3; need_disk=18*e.pb+1024**3
    if psutil.virtual_memory().available<need_host:
        raise MemoryError(f'Estimated free host RAM needed: {need_host/1024**3:.1f} GiB')
    if shutil.disk_usage(out).free<need_disk:
        raise OSError(f'Estimated free disk needed: {need_disk/1024**3:.1f} GiB')


def run_case(settings,seed,event,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    with Progress(out) as progress:
        c=Config(**read(Path(settings['source'])/'config.json')); c.device=settings['device']; c.threads=settings['threads']; c.validate()
        progress.update(phase='loading model',seed=seed,event=event)
        e=Engine(c); preflight(e,out)
        data=make_data(e.tok,c,seed); src=Path(settings['source'])/f'seed{seed}'
        if digest(data)!=read(src/'data.json')['hash']:raise ValueError('Source data hash mismatch')
        panels={sp:data[sp] for sp in ['A_valid','B_valid','A_test','B_test']}
        e.restore(load(src/'anchor.pt'))
        anchor={sp:predictions(e,rows) for sp,rows in panels.items()}
        forks=[p for p in src.glob('fork*/fork.pt') if int(p.parent.name[4:])<=event]
        if not forks:raise ValueError('No saved pre-event fork')
        fork=max(forks,key=lambda p:int(p.parent.name[4:])); z=load(fork)
        if int(z['step'])!=int(fork.parent.name[4:]):raise ValueError('Fork step mismatch')
        e.restore(z['engine']); start=int(z['step']); del z
        for t in range(start,event):
            progress.update(phase='replaying source',step=t)
            e.gradient(minibatches(data,'B',seed,t,c)); e.opt.step(); e.opt.zero_grad(set_to_none=True)
        base={n:p.detach().cpu().clone() for n,p in e.params.items()}
        before={sp:predictions(e,rows) for sp,rows in panels.items()}
        batch=minibatches(data,'B',seed,event,c); train_rows=[r for micro in batch for r in micro]
        before_train=predictions(e,train_rows)
        progress.update(phase='splitting optimizer numerator')
        stats=e.gradient(batch); stats.pop('factors',None)
        fresh,history,sgd=optimizer_channels(e)
        e.opt.step();e.opt.zero_grad(set_to_none=True)
        actual={n:p.detach().cpu()-base[n] for n,p in e.params.items()}; size=vnorm(actual)
        if size<=1e-15:raise ArithmeticError('Zero actual displacement')
        predicted=add_vectors(fresh,history)
        channel_error=vnorm({n:actual[n]-predicted[n] for n in actual})/size
        del predicted
        torch.save(e.pack(),out/'poststep.pt')
        after={sp:predictions(e,rows) for sp,rows in panels.items()}
        atomic_json(dict(seed=seed,event=event,source_fork=start,anchor=anchor,before=before,after=after,
                         actual_norm=size,gradient_stats=stats,optimizer_sum_relative_error=channel_error),out/'outputs.json')
        if channel_error>.05:raise ArithmeticError('Optimizer channel reconstruction failed')
        bank=VectorBank(out/'vectors'); native_norms={}
        for name,v in [('actual',actual),('current_gradient',fresh),('history',history),('sgd',sgd)]:
            native_norms[name]=vnorm(v); bank.put(name,v)
        del fresh,history,sgd
        layer0={n:v if n.startswith('model.layers.0.') else torch.zeros_like(v) for n,v in actual.items()}
        bank.put('layer0',layer0); layer0_norm=vnorm(layer0); del layer0
        inject(e,base)
        for domain in ['A','B']:
            progress.update(phase='full training-panel gradient',domain=domain)
            v=gradient_vector(e,data[domain+'_train']); bank.put(domain+'_train_descent',v); del v
        # Null orientation control preserves coordinate magnitudes, using a fixed generator.
        gen=torch.Generator().manual_seed(910000+seed*100+event)
        v={n:x*torch.randint(0,2,x.shape,generator=gen,dtype=torch.int8).float().mul_(2).sub_(1) for n,x in actual.items()}
        bank.put('sign_scrambled',v);del v
        # Output basis is learned on training prompts only; validation selects dose; test evaluates it.
        ids=list(range(c.entities)); rng=np.random.default_rng(80000+seed); rng.shuffle(ids)
        ids=sorted(ids[:min(settings['calibration_entities'],c.entities)])
        calibration=[next(r for r in data[d+'_train'] if r['entity']==ent) for d in ['A','B'] for ent in ids]
        atomic_json(dict(entities=ids,prompts=[r['prompt'] for r in calibration],basis=bank.names,
                         interpretation='Training-prompt calibration; entities also occur in original training. Test prompts never fit the spectrum.'),out/'calibration.json')
        gram=bank.gram(progress); jac=[]; audits=[]
        probes=np.stack([np.ones(len(calibration)),np.random.default_rng(61000+seed).normal(size=len(calibration))])
        probes/=np.linalg.norm(probes,axis=1,keepdims=True)
        exact=np.zeros((2,len(bank.names)))
        for pi,probe in enumerate(probes):
            progress.update(phase='independent autodiff derivative check',probe=pi)
            g=margin_probe_gradient(e,calibration,probe)
            for j,name in enumerate(bank.names):
                v=bank.get(name);exact[pi,j]=vdot(g,v);del v
            del g
        h=settings['derivative_fraction']*size
        for name in bank.names:
            progress.update(phase='calibration response derivatives',direction=name)
            v=bank.get(name); columns=[]; errors=[]
            for hh in [h,h/2]:
                errors.append(inject(e,base,v,hh)); plus=np.array([r['margin'] for r in predictions(e,calibration)])
                errors.append(inject(e,base,v,-hh)); minus=np.array([r['margin'] for r in predictions(e,calibration)])
                columns.append((plus-minus)/(2*hh))
            rel=float(np.linalg.norm(columns[0]-columns[1])/max(np.linalg.norm(columns[1]),1e-12))
            observed=probes@columns[-1];target=exact[:,len(audits)]
            ad_error=float(np.linalg.norm(observed-target))
            ad_relative=ad_error/max(float(np.linalg.norm(target)),1e-12)
            # Numerically flat columns are unresolved, not proof of an exact null direction.
            signal=float(np.linalg.norm(columns[-1]))
            resolved=(rel<=settings['derivative_tolerance'] and max(errors)<=.05 and signal>1e-8
                      and (ad_relative<=settings['derivative_tolerance'] or ad_error<=1e-7))
            audits.append(dict(direction=name,relative_spacing_difference=rel,max_injection_error=max(errors),
                               response_norm=signal,autodiff_probe_error=ad_error,autodiff_probe_relative_error=ad_relative,
                               resolved=resolved))
            jac.append(columns[-1]/math.sqrt(len(calibration)));del v
        inject(e,base); jac=np.stack(jac,axis=1)
        atomic_json(audits,out/'derivative_audits.json')
        np.savez_compressed(out/'calibration_response.npz',jacobian=jac,parameter_gram=gram)
        # Failed columns are excluded, not interpreted as quiet or zero sensitivity.
        good=np.array([i for i,x in enumerate(audits) if x['resolved']],dtype=int)
        spectrum=dict(status='unresolved',resolved_columns=int(len(good)),total_columns=len(audits))
        spectral_names=[]
        if len(good)>=2:
            sub,coef=spectral_decomposition(jac[:,good],gram[np.ix_(good,good)])
            spectrum.update(sub); spectrum['status']='resolved_restricted_span'
            coeff=np.zeros((len(bank.names),coef.shape[1]));coeff[good,:]=coef
            # Split by response energy, reserving at least one lower-gain direction.
            cut=min(sub['rank90'],coef.shape[1]-1); cut=max(1,cut)
            spectrum['loud_dimension']=cut
            projection=coeff.T@gram[:,bank.names.index('actual')]
            bcoeff=coeff.T@gram[:,bank.names.index('B_train_descent')]
            initial_names=list(bank.names)
            spectrum['candidate_audits']=[]
            def audited_candidate(name,weights):
                # Whitening/cancellation can amplify error even if primitive columns pass.
                # Independently check every derived direction before finite comparisons.
                v=bank.combine(np.pad(weights,(0,len(bank.names)-len(weights))))
                norm=vnorm(v)
                if norm<=1e-7:
                    spectrum['candidate_audits'].append(dict(direction=name,resolved=False,reason='vanishing projection'))
                    return
                v=scaled(v,1/norm);predicted=jac@weights/norm;observed=[];errors=[]
                progress.update(phase='derived-direction audit',direction=name)
                for hh in [h,h/2]:
                    errors.append(inject(e,base,v,hh));plus=np.array([r['margin'] for r in predictions(e,calibration)])
                    errors.append(inject(e,base,v,-hh));minus=np.array([r['margin'] for r in predictions(e,calibration)])
                    observed.append((plus-minus)/(2*hh*math.sqrt(len(calibration))))
                signal=float(np.linalg.norm(observed[-1]));den=max(signal,1e-12)
                prediction_error=float(np.linalg.norm(predicted-observed[-1])/den)
                spacing_error=float(np.linalg.norm(observed[0]-observed[1])/den)
                passed=max(prediction_error,spacing_error)<=settings['derivative_tolerance'] and max(errors)<=.05 and signal>1e-8
                spectrum['candidate_audits'].append(dict(direction=name,resolved=passed,
                    prediction_relative_error=prediction_error,spacing_relative_error=spacing_error,
                    max_injection_error=max(errors)))
                if passed:bank.put(name,v);spectral_names.append(name)
                inject(e,base)
            for name,weights in [('loud_actual',coeff[:,:cut]@projection[:cut]),
                                 ('quiet_actual',coeff[:,cut:]@projection[cut:]),
                                 ('quiet_B_descent',coeff[:,cut:]@bcoeff[cut:])]:
                audited_candidate(name,weights)
            spectrum['actual_loud_parameter_energy']=float(np.sum(projection[:cut]**2))
            spectrum['actual_quiet_parameter_energy']=float(np.sum(projection[cut:]**2))
            spectrum['coefficient_basis']=initial_names
            spectrum['mode_coefficients']=coeff.tolist()
            responses=jac@coeff
            spectrum['modes']=[]
            for index in range(responses.shape[1]):
                response=responses[:,index];energy=float(response@response)
                spectrum['modes'].append(dict(mode=index+1,squared_sensitivity=energy,
                    common_margin_fraction=float(response.sum()**2/(len(response)*max(energy,1e-30))),
                    registry_contrast_fraction=float(np.sum((response[:len(ids)]-response[len(ids):])**2)/(2*max(energy,1e-30)))))
            inject(e,base)
            z0=np.array([r['margin'] for r in predictions(e,calibration)])
            q0=np.array([r['q'][0] for r in calibration])
            target=(np.log(q0/(1-q0))-z0)/math.sqrt(len(calibration))
            singular=np.array(sub['singular_values']);nonzero=singular>max(singular.max()*1e-6,1e-10)
            amplitudes=np.zeros(len(singular));amplitudes[nonzero]=(responses[:,nonzero].T@target)/singular[nonzero]
            coordinates=np.zeros(len(singular));coordinates[nonzero]=amplitudes[nonzero]/singular[nonzero]
            predicted_norm=float(np.linalg.norm(coordinates))
            spectrum['joint_margin_target']=dict(
                loud_target_energy=float(np.sum(amplitudes[:cut]**2)),
                quiet_target_energy=float(np.sum(amplitudes[cut:]**2)),
                outside_resolved_response_span_energy=max(0.,float(target@target-amplitudes@amplitudes)),
                predicted_minimum_parameter_norm=predicted_norm,
                predicted_norm_over_actual=predicted_norm/size,
                interpretation='Training-label oracle linearized margin fit, not full-KL fit or a global capacity bound.')
            weights=coeff@coordinates
            audited_candidate('joint_target_fit',weights)
            spectrum['interpretation']='Sensitivity within a chosen parameter span and training-prompt margin panel; not full Fisher/Hessian/NTK spectrum, information capacity, or a global impossibility test.'
        atomic_json(spectrum,out/'spectrum.json')
        # Actual dose and layer-0 dose both use the actual native displacement.
        # Comparator directions use the FULL actual norm, and validation checks B gain.
        variants=['actual','layer0_scaled','current_gradient','history','sgd','B_train_descent','sign_scrambled']+spectral_names
        records=[]; prediction_records={}
        def direction_for(name,dose):
            if name=='layer0_scaled':
                return {n:(dose*v if n.startswith('model.layers.0.') else v.clone()) for n,v in actual.items()}
            return scaled(bank.get(name),size*dose)
        for name in variants:
            for dose in settings['doses']:
                progress.update(phase='finite interventions',variant=name,dose=dose,completed=len(records),total=len(variants)*len(settings['doses']))
                d=direction_for(name,dose); err=inject(e,base,d); norm=vnorm(d);del d
                vals={sp:predictions(e,rows) for sp,rows in panels.items()}; train=predictions(e,train_rows)
                rec=dict(seed=seed,event=event,variant=name,dose=dose,norm=norm,injection_error=err,
                         train_minibatch_delta=mean_loss(train)-mean_loss(before_train))
                for sp in panels:
                    m=metrics(vals[sp]); bm=metrics(before[sp]); rec[sp+'_delta']=m['full']-bm['full']
                    rec[sp+'_from_anchor']=m['full']-mean_loss(anchor[sp])
                    rec[sp+'_balanced_delta']=None if m['balanced_full'] is None else m['balanced_full']-bm['balanced_full']
                    rec[sp+'_margin_shift']=bias_accounting(before[sp],vals[sp])['mean_shift']
                records.append(rec); prediction_records[name+'@'+str(dose)]=vals
                atomic_json(records,out/'interventions.json');write_csv(records,out/'interventions.csv')
        # Native first-moment contributions, separate from matched-norm comparisons.
        native=[]
        for name in ['current_gradient','history']:
            d=scaled(bank.get(name),native_norms[name]);err=inject(e,base,d);del d
            vals={sp:predictions(e,rows) for sp,rows in panels.items()}
            native.append(dict(variant=name,norm=native_norms[name],injection_error=err,
                               **{sp+'_delta':mean_loss(vals[sp])-mean_loss(before[sp]) for sp in panels}))
        atomic_json(native,out/'native_optimizer_channels.json')
        atomic_json(prediction_records,out/'intervention_predictions.json')
        selected=select_matched(records,settings['match_relative'],settings['match_absolute'])
        atomic_json(selected,out/'selection.json')
        # Replay measured full step exactly before interpreting intervention results.
        replay=prediction_records['actual@1.0']
        replay_error=max(abs(x['margin']-y['margin']) for sp in panels for x,y in zip(replay[sp],after[sp]))
        atomic_json(dict(max_margin_error=replay_error,passed=replay_error<=2e-5),out/'replay_audit.json')
        if replay_error>2e-5:raise ArithmeticError('Full-step endpoint replay failed')
        # Fine finite-dose curve resolves practical turn-over without trusting a Taylor fit.
        fine=[]
        for alpha in settings['fine_doses']:
            progress.update(phase='fine actual dose curve',dose=alpha)
            err=inject(e,base,actual,alpha)
            fine.append(dict(dose=alpha,injection_error=err,**{sp+'_delta':mean_loss(predictions(e,rows))-mean_loss(before[sp]) for sp,rows in panels.items()}))
        atomic_json(fine,out/'fine_dose.json')
        # Descriptive all-token traces at pre/full/quarter states on training calibration prompts.
        sites=[n for n in e.mods if n in ['model.layers.0.self_attn.q_proj','model.layers.0.self_attn.k_proj',
                  'model.layers.0.self_attn.v_proj','model.layers.0.self_attn.o_proj','model.layers.0.mlp.down_proj']]
        trace={}
        for alpha in [0.,.25,1.]:
            progress.update(phase='all-token layer-0 traces',dose=alpha);inject(e,base,actual,alpha)
            trace[str(alpha)]=token_trace(e,calibration[:min(4,len(ids))]+calibration[len(ids):len(ids)+min(4,len(ids))],sites)
        atomic_json(trace,out/'token_traces.json')
        # Weight-only continuation: all branches inherit the nominal POST-step Adam state.
        continuation=[]; branches=[dict(variant='actual',dose=1.)]
        if selected.get('selected'):branches.append(selected['selected'])
        for branch in branches:
            e.restore(load(out/'poststep.pt'));d=direction_for(branch['variant'],branch['dose']);inject(e,base,d);del d
            for step in range(settings['continuation_steps']+1):
                progress.update(phase='weight-only continuation',branch=branch,step=step)
                continuation.append(dict(**branch,step=step,**{sp+'_from_pre_event':mean_loss(predictions(e,rows))-mean_loss(before[sp]) for sp,rows in panels.items()}))
                if step<settings['continuation_steps']:
                    e.gradient(minibatches(data,'B',seed,event+1+step,c));e.opt.step();e.opt.zero_grad(set_to_none=True)
            atomic_json(continuation,out/'continuation.json')
        atomic_json(dict(kind='weight-only',optimizer='Identical nominal post-event Adam moments in every branch',
                         selection='Validation-only gain matching; no test-based dose selection',
                         native_norms=native_norms,layer0_native_norm=layer0_norm),out/'protocol.json')
        plot_case(out)
        if not settings['keep_vectors']:
            shutil.rmtree(out/'vectors');(out/'poststep.pt').unlink()
        atomic_json(dict(complete=True,seed=seed,event=event,seconds=time.time()-progress.started),out/'done.json')
        progress.update(status='complete',phase='finished')


def plot_case(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out); rows=read(out/'interventions.json');s=read(out/'spectrum.json')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    for variant in ['actual','layer0_scaled']:
        r=[x for x in rows if x['variant']==variant]
        for sp,style in [('A_test','-'),('B_test','--')]:
            ax.plot([x['dose'] for x in r],[x[sp+'_delta'] for x in r],style,marker='o',label=variant+' '+sp)
    ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='Dose (layer0_scaled keeps other blocks at full step)',ylabel='Full KL change from pre-event',title='Finite dose response');ax.legend(fontsize=7)
    ax=axes[0,1]
    for name in sorted(set(x['variant'] for x in rows)):
        r=[x for x in rows if x['variant']==name]
        ax.plot([-x['B_test_delta'] for x in r],[x['A_test_delta'] for x in r],'.-',label=name,alpha=.8)
    ax.axhline(0,color='gray',lw=.7);ax.axvline(0,color='gray',lw=.7)
    ax.set(xlabel='B test improvement',ylabel='A test deterioration',title='Held-out prompt trade-off');ax.legend(fontsize=6)
    ax=axes[1,0]
    if s['status']=='resolved_restricted_span':
        values=np.array(s['singular_values']);ax.bar(np.arange(1,len(values)+1),values**2)
        ax.set_yscale('symlog',linthresh=max(1e-12,float(np.max(values**2))*1e-10))
        ax.set(xlabel='Mode in audited parameter span',ylabel='Squared margin sensitivity',title='Restricted spectrum (not capacity)')
    else:ax.text(.5,.5,'Derivative audits unresolved\nNo loud/quiet inference',ha='center',va='center',transform=ax.transAxes)
    ax=axes[1,1]
    o=read(out/'outputs.json')
    for sp in ['A_test','B_test']:
        d=np.array([v['margin']-u['margin'] for u,v in zip(o['before'][sp],o['after'][sp])])
        ax.plot(range(len(d)),d,'.-',label=sp)
    ax.set(xlabel='Entity',ylabel='Actual-step answer log-odds shift',title='Shared bias versus entity variation');ax.legend()
    fig.suptitle(f"Seed {o['seed']} · update {o['event']} · exploratory local tests")
    fig.savefig(out/'overview.png',dpi=160);plt.close(fig)


def analyze_measurements(source,out):
    """Reanalyse supplied ZIP without loading model checkpoints. No fabricated experiments."""
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    records=[];curves=[]
    archive=zipfile.ZipFile(source) if source.is_file() else None
    try:
        paths=sorted(n for n in archive.namelist() if n.endswith('/outputs.json') and '/event' in n) if archive else sorted(source.glob('seed*/event*/outputs.json'))
        for path in paths:
            def get(name):
                return json.loads(archive.read(str(Path(path).parent/name))) if archive else read(Path(path).parent/name)
            o=get('outputs.json');dose=get('dose_path.json');r=dict(seed=o['seed'],event=o['event'])
            for sp in ['A_test','B_test']:
                r[sp+'_delta']=mean_loss(o['after'][sp])-mean_loss(o['before'][sp])
                r[sp+'_from_anchor']=mean_loss(o['after'][sp])-mean_loss(o['anchor'][sp])
                for k,v in bias_accounting(o['before'][sp],o['after'][sp]).items():
                    if k!='interpretation':r[sp+'_'+k]=v
            records.append(r)
            for alpha,x in dose.items():
                curves.append(dict(seed=o['seed'],event=o['event'],dose=float(alpha),**{sp+'_delta':mean_loss(x[sp])-mean_loss(o['before'][sp]) for sp in ['A_test','B_test']}))
    finally:
        if archive:archive.close()
    if not records:raise ValueError('No seed/event outputs found in measurement input')
    atomic_json(records,out/'archive_analysis.json');write_csv(records,out/'archive_analysis.csv');write_csv(curves,out/'archive_dose_curves.csv')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed in sorted(set(r['seed'] for r in records)):
        rr=[r for r in records if r['seed']==seed]
        axes[0].scatter([-r['B_test_delta'] for r in rr],[r['A_test_delta'] for r in rr],label=f'Seed {seed}')
    axes[0].axhline(0,color='gray',lw=.7);axes[0].axvline(0,color='gray',lw=.7);axes[0].legend()
    axes[0].set(xlabel='B improvement',ylabel='A deterioration',title='Observed updates')
    rr=sorted([r for r in curves if r['seed']==1 and r['event']==21],key=lambda r:r['dose'])
    if rr:
        for sp in ['A_test','B_test']:axes[1].plot([r['dose'] for r in rr],[r[sp+'_delta'] for r in rr],'o-',label=sp)
        axes[1].legend()
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set(xlabel='Fraction of actual update',ylabel='KL change',title='Seed 1, update 21: saved finite-dose data')
    fig.savefig(out/'archive_overview.png',dpi=170);plt.close(fig)
    lines=['SAVED MEASUREMENT REANALYSIS — no new model experiments', 'seed event       dA          dB      mean A shift   shift SD']
    lines += [f"{r['seed']:4d} {r['event']:5d} {r['A_test_delta']:+11.7f} {r['B_test_delta']:+11.7f} {r['A_test_mean_shift']:+12.6f} {r['A_test_shift_sd']:10.6f}" for r in records]
    lines += [f"Both worsen: {sum(r['A_test_delta']>0 and r['B_test_delta']>0 for r in records)}/{len(records)}.",
              'These are correlated, previously inspected events. Output bias counterfactuals are descriptive.']
    (out/'archive_report.txt').write_text('\n'.join(lines)+'\n')
    return records


def report(output,print_report=True):
    output=Path(output);lines=['FORGETTING MECHANISM EXPERIMENT — available completed cases'];summary=[]
    for case in sorted(output.glob('seed*/event*')):
        if not (case/'done.json').exists():continue
        rows=read(case/'interventions.json');sel=read(case/'selection.json');spec=read(case/'spectrum.json')
        actual=next(r for r in rows if r['variant']=='actual' and r['dose']==1.)
        lines.append(f"\nSeed {actual['seed']}, update {actual['event']}: A {actual['A_test_delta']:+.7f}; B {actual['B_test_delta']:+.7f}")
        lines.append(f"Spectrum: {spec['status']}; audited columns {spec['resolved_columns']}/{spec['total_columns']}. Validation match: {sel['status']}.")
        lines.append('variant                    dose       dA test      dB test      dB batch')
        for r in rows:
            lines.append(f"{r['variant']:26s} {r['dose']:5.2f} {r['A_test_delta']:+12.7f} {r['B_test_delta']:+12.7f} {r['train_minibatch_delta']:+12.7f}")
        summary.extend(rows)
    lines += ['\nDirections/spectra are local and restricted. Events within a seed are correlated.',
              'A/B deltas are from pre-event; *_from_anchor fields retain accumulated forgetting.',
              'Matched continuation changes weights only and uses nominal post-event optimizer state.',
              'A failed restricted search does not prove an information-capacity limit.']
    text='\n'.join(lines)+'\n';(output/'report.txt').write_text(text);write_csv(summary,output/'all_interventions.csv')
    if print_report:print(text)
    return text


def default_settings(source,output):
    return dict(source=str(Path(source).resolve()),output=str(Path(output).resolve()),
                seeds=[1,2,3],events=[21,47],device='cuda:0',threads=8,
                doses=[0.,.25,.5,.75,1.],fine_doses=[0.,.0625,.125,.25,.375,.5,.625,.75,.875,1.],
                calibration_entities=8,derivative_fraction=.02,derivative_tolerance=.15,
                match_relative=.10,match_absolute=1e-4,continuation_steps=16,keep_vectors=False)


def validate_settings(s):
    required=set(default_settings('.', './run'))
    if set(s)!=required:raise ValueError('Settings fields mismatch: '+str(set(s)^required))
    source=Path(s['source']).resolve();output=Path(s['output']).resolve()
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('Source and output must be separate, non-nested directories')
    for key in ['seeds','events']:
        if not s[key] or len(set(s[key]))!=len(s[key]) or any(type(x)!=int or x<1 for x in s[key]):raise ValueError('Invalid '+key)
    for key in ['doses','fine_doses']:
        if not s[key] or 1. not in s[key] or 0. not in s[key] or len(set(s[key]))!=len(s[key]) or any(not math.isfinite(x) or x<0 for x in s[key]):raise ValueError('Invalid '+key)
    if s['calibration_entities']<1 or s['threads']<1 or s['continuation_steps']<0:raise ValueError('Invalid size settings')
    for k in ['derivative_fraction','derivative_tolerance','match_relative','match_absolute']:
        if not math.isfinite(s[k]) or s[k]<=0:raise ValueError('Invalid '+k)
    if not (source/'config.json').exists():raise FileNotFoundError(f'Missing original association source: {source}/config.json')
    c=Config(**read(source/'config.json'));c.validate()
    if any(x not in c.seeds for x in s['seeds']):raise ValueError('Seed absent from source config')
    if any(t>c.b_steps or t+s['continuation_steps']>c.b_steps for t in s['events']):raise ValueError('Event plus continuation exceeds source B schedule')
    if not c.smoke and (c.model!='allenai/OLMo-1B-hf' or len(c.revision)!=40):
        raise ValueError('Production requires original OLMo-1B and pinned 40-character revision in source config')
    return s


def source_manifest(settings):
    src=Path(settings['source']);files=[src/'config.json']
    for seed in settings['seeds']:
        p=src/f'seed{seed}';files += [p/'data.json',p/'anchor.pt']
        for t in settings['events']:
            fs=[f for f in p.glob('fork*/fork.pt') if int(f.parent.name[4:])<=t]
            if not fs:raise FileNotFoundError(f'No preceding fork for seed {seed}, event {t}')
            files.append(max(fs,key=lambda f:int(f.parent.name[4:])))
    return {str(p.relative_to(src)):sha256_file(p) for p in sorted(set(files))}


def run_all(settings):
    import fcntl
    validate_settings(settings);out=Path(settings['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'queue.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with Progress(out) as progress:
            progress.update(phase='fingerprinting inputs')
            import transformers
            design=dict(settings=settings,source=source_manifest(settings),code=sha256_file(__file__),
                        versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__))
            if (out/'design.json').exists() and read(out/'design.json')!=design:
                raise ValueError('Source, settings, code or versions changed. Choose a NEW output directory.')
            atomic_json(design,out/'design.json')
            atomic_json(settings,out/'settings.json')
            jobs=[(seed,t) for seed in settings['seeds'] for t in settings['events']]
            for index,(seed,event) in enumerate(jobs):
                case=out/f'seed{seed}'/f'event{event:03d}';case.mkdir(parents=True,exist_ok=True)
                if (case/'done.json').exists():continue
                progress.update(phase='case worker',seed=seed,event=event,completed=index,total=len(jobs))
                with open(case/'worker.log','w') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker',
                                              '--settings',str(out/'settings.json'),'--seed',str(seed),'--event',str(event)],
                                             stdout=log,stderr=subprocess.STDOUT)
                    code=process.wait()
                if code or not (case/'done.json').exists():
                    raise RuntimeError(f'Case failed ({code}); inspect {case}/worker.log. Completed cases retained.')
                report(out,False)
            report(out,False);progress.update(status='complete',phase='finished',completed=len(jobs),total=len(jobs))


def launch(settings):
    """Returns immediately. Notebook kernel does not own the detached worker process."""
    import fcntl
    validate_settings(settings);out=Path(settings['output']);out.mkdir(parents=True,exist_ok=True)
    # Guard launches and reject changed settings even before fingerprinting.
    with open(out/'launch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=status(out)
        if state.get('process_alive') or state.get('launch_pending'):raise RuntimeError('A run is already active in this output directory')
        if (out/'settings.json').exists() and read(out/'settings.json')!=settings:raise ValueError('Settings changed; choose a new output directory')
        atomic_json(settings,out/'settings.json')
        atomic_json(dict(status='launching',launch_time=time.time()),out/'status.json')
        with open(out/'launcher.log','a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--settings',str(out/'settings.json')],
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                     cwd=str(Path(__file__).resolve().parent))
        atomic_json(dict(pid=process.pid,launched=time.time()),out/'process.json')
    return dict(pid=process.pid,output=str(out),message='Started. Rerun the status cell to refresh; logs are saved to disk.')


def status(output):
    out=Path(output);s=read(out/'status.json') if (out/'status.json').exists() else dict(status='not_started')
    alive=False
    if (out/'process.json').exists():
        pid=read(out/'process.json')['pid']
        try:
            import psutil
            p=psutil.Process(pid)
            alive=p.status()!=psutil.STATUS_ZOMBIE and any('forgetting_mechanisms.py' in x for x in p.cmdline())
        except (psutil.Error,OSError):pass
    s['process_alive']=alive;s['launch_pending']=s.get('status')=='launching' and time.time()-s.get('launch_time',0)<30
    if s.get('status') in ['running','launching'] and not alive and not s['launch_pending']:s['status']='interrupted_or_failed'
    s['heartbeat_age_seconds']=round(time.time()-s['heartbeat'],1) if 'heartbeat' in s else None
    if s.get('phase')=='case worker':
        case=out/f"seed{s['seed']}"/f"event{s['event']:03d}"
        if (case/'status.json').exists():
            x=read(case/'status.json');s['worker']={k:v for k,v in x.items() if k not in ['traceback']}
    return s


def print_status(output):
    s=status(output)
    print(f"Status: {s['status']} | process alive: {s['process_alive']} | elapsed: {s.get('elapsed_seconds',0)/60:.1f} min")
    print(f"Cases: {s.get('completed',0)}/{s.get('total','?')} | phase: {s.get('phase','—')} | heartbeat age: {s.get('heartbeat_age_seconds')} s")
    if 'worker' in s:
        w=s['worker'];print('Current case:',w.get('seed'),w.get('event'),'|',w.get('phase'),
                           '|', {k:w[k] for k in ['variant','dose','direction','step'] if k in w})
    if s.get('error'):print('Error:',s['error'])
    print('A fresh heartbeat means the process is alive; it does not guarantee that the current operation is making progress.')


def stop_run(output):
    """Explicit stop cell: terminates this detached process group, preserves completed cases."""
    s=status(output)
    if not s.get('process_alive'):return 'No active process.'
    pid=read(Path(output)/'process.json')['pid'];os.killpg(pid,signal.SIGTERM)
    return 'Stop requested. Completed cases are retained; rerun Launch to resume unfinished cases.'


def show_results(output,full=False):
    out=Path(output)
    if (out/'archive_report.txt').exists():print((out/'archive_report.txt').read_text())
    if full:report(out,True)
    else:
        for p in sorted(out.glob('seed*/event*/done.json')):
            case=p.parent;rows=read(case/'interventions.json');r=next(x for x in rows if x['variant']=='actual' and x['dose']==1.)
            print(f"Seed {r['seed']}, event {r['event']}: ΔA={r['A_test_delta']:+.6f}, ΔB={r['B_test_delta']:+.6f}; {read(case/'selection.json')['status']}")
    try:
        from IPython.display import display,Image
        for p in ([out/'archive_overview.png']+sorted(out.glob('seed*/event*/overview.png'))):
            if p.exists():display(Image(filename=str(p)))
    except ImportError:print('Plot files:',[str(p) for p in out.rglob('*.png')])


def create_smoke_source(output):
    """Fresh tiny random OLMo fixture. Implementation validation, never scientific evidence."""
    output=Path(output)
    if output.exists():raise FileExistsError('Smoke source already exists; choose a fresh directory')
    c=Config(smoke=True,device='cpu',seeds=[1],entities=4,eval_entities=4,microbatch=2,accumulation=1,
             a_steps=4,b_steps=12,forks=[8],horizon=2,washout=0,threads=1)
    c.validate();atomic_json(asdict(c),output/'config.json');e=Engine(c);data=make_data(e.tok,c,1)
    atomic_json(dict(hash=digest(data)),output/'seed1'/'data.json')
    for t in range(c.a_steps):e.gradient(minibatches(data,'A',1,t,c));e.opt.step()
    save(e.pack(),output/'seed1'/'anchor.pt')
    for t in range(8):e.gradient(minibatches(data,'B',1,t,c));e.opt.step()
    save(dict(step=8,engine=e.pack()),output/'seed1'/'fork008'/'fork.pt')
    return output


def self_test():
    """Analytical tests: metric whitening, output counterfactuals, validation selection."""
    g=np.array([[4.,0.],[0.,1.]])
    j=np.array([[6.,0.],[0.,1.]])
    s,coeff=spectral_decomposition(j,g)
    np.testing.assert_allclose(s['singular_values'],[3.,1.],atol=1e-12)
    np.testing.assert_allclose(coeff.T@g@coeff,np.eye(2),atol=1e-12)
    q=np.array([[.1,.9],[.9,.1]]);z=np.log(q[:,0]/q[:,1])
    np.testing.assert_allclose(relation_kl(q,z),0,atol=1e-14)
    a=[dict(entity=i,q=q[i].tolist(),margin=float(z[i])) for i in range(2)]
    b=[dict(**{k:v for k,v in r.items() if k!='margin'},margin=r['margin']+.7) for r in a]
    result=bias_accounting(a,b)
    assert abs(result['actual_relation_change']-result['common_shift_only'])<1e-12
    assert abs(result['residual_only'])<1e-12
    rows=[dict(variant='actual',dose=1.,B_valid_delta=-.1,A_valid_delta=.2,injection_error=0),
          dict(variant='safe',dose=.5,B_valid_delta=-.105,A_valid_delta=.01,injection_error=0),
          dict(variant='not_matched',dose=1.,B_valid_delta=-.2,A_valid_delta=-1.,injection_error=0)]
    assert select_matched(rows)['selected']['variant']=='safe'
    # Test values cannot affect validation selection.
    for r in rows:r['A_test_delta']=-999 if r['variant']=='not_matched' else 999
    assert select_matched(rows)['selected']['variant']=='safe'
    rows[0]['B_valid_delta']=.1
    assert select_matched(rows)['selected'] is None
    from types import SimpleNamespace
    model=torch.nn.Module();model.left=torch.nn.Parameter(torch.tensor([.2,-.4]));model.right=model.left
    e=SimpleNamespace(params=dict(model.named_parameters()),opt=torch.optim.Adam(model.parameters(),lr=.01,foreach=False))
    assert len(e.params)==1
    base={n:p.detach().clone() for n,p in e.params.items()};d={n:torch.ones_like(p)*.02 for n,p in e.params.items()}
    assert inject(e,base,d)<1e-5
    torch.testing.assert_close(model.left,model.right)
    inject(e,base);model.left.grad=torch.tensor([.3,-.1])
    fresh,history,_=optimizer_channels(e);e.opt.step()
    np.testing.assert_allclose(model.left.detach().numpy()-base['left'].numpy(),
                               (fresh['left']+history['left']).numpy(),rtol=1e-5,atol=1e-7)
    # A second step covers nonempty optimizer history.
    base={n:p.detach().clone() for n,p in e.params.items()};model.left.grad=torch.tensor([-.2,.5])
    fresh,history,_=optimizer_channels(e);e.opt.step()
    np.testing.assert_allclose(model.left.detach().numpy()-base['left'].numpy(),
                               (fresh['left']+history['left']).numpy(),rtol=1e-5,atol=1e-7)
    print('7 analytical/optimizer tests passed, including tied parameters and nonempty Adam history.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['run','worker']:
        q=sub.add_parser(name);q.add_argument('--settings',required=True)
        if name=='worker':q.add_argument('--seed',type=int,required=True);q.add_argument('--event',type=int,required=True)
    q=sub.add_parser('analyze');q.add_argument('--measurement',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('report');q.add_argument('--output',required=True)
    q=sub.add_parser('smoke');q.add_argument('--output',required=True)
    sub.add_parser('self-test');a=p.parse_args()
    if a.command=='self-test':self_test()
    elif a.command=='analyze':analyze_measurements(a.measurement,a.output);print((Path(a.output)/'archive_report.txt').read_text())
    elif a.command=='report':report(a.output)
    elif a.command=='smoke':
        root=Path(a.output).resolve();create_smoke_source(root/'source')
        s=default_settings(root/'source',root/'run');s.update(seeds=[1],events=[9],device='cpu',threads=1,calibration_entities=2,
                doses=[0.,.25,1.],fine_doses=[0.,.25,.5,1.],continuation_steps=1,derivative_fraction=.1)
        atomic_json(s,root/'run'/'settings.json');run_all(s)
    else:
        s=read(a.settings)
        if a.command=='run':run_all(s)
        else:run_case(s,a.seed,a.event,Path(s['output'])/f'seed{a.seed}'/f'event{a.event:03d}')


if __name__=='__main__':main()
