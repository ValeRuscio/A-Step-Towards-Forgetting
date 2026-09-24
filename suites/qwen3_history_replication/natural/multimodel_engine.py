"""Full-parameter FP32 adapters: Qwen2, Llama/SmolLM2, OLMo, GPT-NeoX and Qwen3.
No Q/K/V unpacking assumptions; bias, norm and tied parameters participate.
"""
import math,re
import torch
from association_core import Engine, ToyTokenizer, collate

class MultiEngine(Engine):
    def __init__(self,c):
        self.c=c;torch.set_num_threads(c.threads);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        if c.device.startswith('cuda'):
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        from transformers import AutoTokenizer,AutoModelForCausalLM
        if c.smoke:
            self.tok=ToyTokenizer();kind=c.model.removeprefix('tiny-')
            common=dict(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,max_position_embeddings=128,pad_token_id=0,eos_token_id=63)
            if kind=='gpt_neox':
                from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
                cfg=GPTNeoXConfig(**common,rotary_pct=.5,attention_dropout=0.,hidden_dropout=0.);cls=GPTNeoXForCausalLM
            elif kind=='qwen2':
                from transformers import Qwen2Config,Qwen2ForCausalLM
                cfg=Qwen2Config(**common,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.);cls=Qwen2ForCausalLM
            elif kind=='llama':
                from transformers import LlamaConfig,LlamaForCausalLM
                cfg=LlamaConfig(**common,num_key_value_heads=2,tie_word_embeddings=True,attention_dropout=0.);cls=LlamaForCausalLM
            elif kind=='qwen3':
                from transformers import Qwen3Config,Qwen3ForCausalLM
                cfg=Qwen3Config(**common,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True,attention_dropout=0.);cls=Qwen3ForCausalLM
            else:
                from transformers import OlmoConfig,OlmoForCausalLM
                cfg=OlmoConfig(**common,num_key_value_heads=2);cls=OlmoForCausalLM
            cfg._attn_implementation='eager';self.model=cls(cfg)
        else:
            self.tok=AutoTokenizer.from_pretrained(c.model,revision=c.revision,trust_remote_code=False)
            self.model=AutoModelForCausalLM.from_pretrained(c.model,revision=c.revision,torch_dtype=torch.float32,attn_implementation='eager',trust_remote_code=False)
        self.kind=self.model.config.model_type
        if self.kind not in {'olmo','gpt_neox','qwen3','qwen2','llama'}:raise ValueError('Unsupported model architecture: '+self.kind)
        if self.tok.pad_token_id is None:self.tok.pad_token_id=self.tok.eos_token_id
        if self.tok.pad_token_id is None:raise ValueError('Missing padding token')
        self.model.to(c.device,dtype=torch.float32).eval();self.model.config.use_cache=False
        if hasattr(self.model,'gradient_checkpointing_disable'):self.model.gradient_checkpointing_disable()
        self.params=dict(self.model.named_parameters());self.mods=dict(self.model.named_modules())
        # The inherited training-gradient method needs no activation hooks for this study.
        self.names=[];self.selected={};self.reset_optimizer();self.pb=sum(v.numel()*v.element_size() for v in self.params.values())
    def loss_rows(self,rows,details=False):
        inputs,q,labels=collate(rows,self.tok.pad_token_id,self.c.device)
        backbone=self.model.gpt_neox if self.kind=='gpt_neox' else self.model.model
        head=self.model.get_output_embeddings()
        h=backbone(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
        logits=head(h);lp=logits.double().log_softmax(-1);qq=q.double()
        loss=(qq*(qq.log()-lp[:,labels])).sum(-1)
        if not torch.isfinite(loss).all():raise FloatingPointError('Nonfinite full-vocabulary KL')
        mass=-lp[:,labels].logsumexp(-1);relation=loss-mass
        return (loss,relation,mass) if details else loss
