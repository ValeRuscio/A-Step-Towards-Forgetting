import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,platform,subprocess,sys,importlib.metadata
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the new environment. Check nvidia-smi and setup.log.')
torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
torch.use_deterministic_algorithms(True)
c=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,attention_dropout=0.,tie_word_embeddings=True)
c._attn_implementation='eager'
m=Qwen3ForCausalLM(c).float().cuda().eval();x=torch.tensor([[1,2,3,4]],device='cuda');y=m(x).logits[:,-1].double().log_softmax(-1)[0,7];y.backward()
assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
# Second derivative support is needed by the final path stage.
m.zero_grad(set_to_none=True);loss=-m(x).logits[:,-1].double().log_softmax(-1)[0,7];p=m.get_input_embeddings().weight
g=torch.autograd.grad(loss,p,create_graph=True)[0];h=torch.autograd.grad(g.square().sum(),p)[0];assert torch.isfinite(h).all()
torch.cuda.synchronize();prop=torch.cuda.get_device_properties(0)
r=dict(python=sys.version,architecture=platform.machine(),gpu=prop.name,vram_gib=prop.total_memory/2**30,torch=torch.__version__,cuda=torch.version.cuda,transformers=importlib.metadata.version('transformers'),tiny_qwen3_forward_backward_and_second_derivative='passed',note='Full-size pretrained model and numerical finite-path audits run during the experiment.')
Path(__file__).with_name('gpu_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
