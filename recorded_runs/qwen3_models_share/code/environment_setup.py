"""Create isolated Qwen environment while reusing the existing CUDA Torch install."""
from pathlib import Path
import subprocess,sys,json,sysconfig

def setup_qwen(path):
 path=Path(path).resolve()
 if path==Path(sys.prefix).resolve():raise ValueError('Choose a NEW environment directory, not the running environment')
 subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(path)],check=True)
 python=path/'bin/python'
 # A nested venv inherits the BASE interpreter site, not the current CUDA venv.
 target_site=subprocess.check_output([str(python),'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip()
 (Path(target_site)/'existing_cuda_environment.pth').write_text(sysconfig.get_paths()['purelib']+'\n')
 subprocess.run([str(python),'-c','import torch; print(torch.__version__,torch.cuda.is_available())'],check=True)
 subprocess.run([str(python),'-m','pip','install','transformers==4.57.6','safetensors','numpy','matplotlib'],check=True)
 subprocess.run([str(python),'-c','from transformers import Qwen3Config,Qwen3ForCausalLM; import torch; print("Qwen3 ready; Torch",torch.__version__)'],check=True)
 return str(python)

def check_native(python):
 code='import json;from open_pilot import readiness;print(json.dumps(readiness(),indent=2))'
 r=subprocess.run([str(python),'-c',code],cwd=Path(__file__).parent,text=True,capture_output=True)
 print(r.stdout or r.stderr);return r.returncode
