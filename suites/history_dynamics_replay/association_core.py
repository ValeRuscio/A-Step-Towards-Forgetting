"""Preserved registry data, deterministic minibatches and Adam engine primitives.
The reviewer suite subclasses this engine without changing registry semantics.
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

