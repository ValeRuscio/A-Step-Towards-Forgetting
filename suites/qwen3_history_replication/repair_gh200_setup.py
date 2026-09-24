"""Resume GH200 setup after a narrowly verified cuSPARSELt SBSA wheel-tag mismatch.
No packages, vendor metadata, model settings or result records are modified.
Run with the notebook Python: %run repair_gh200_setup.py
"""
import ctypes,importlib.metadata as metadata,json,platform,struct,subprocess,sys
from pathlib import Path
EXPECTED='nvidia-cusparselt-cu12 0.7.1 is not supported on this platform'

def validate_exception(output,code,system,machine,dist,loader):
    if code!=1 or output.strip()!=EXPECTED:
        raise RuntimeError('Dependency check failed for another reason; not bypassed:\n'+output)
    if system!='Linux' or machine.lower() not in ('aarch64','arm64'):
        raise RuntimeError('This exception applies only to Linux ARM64.')
    if dist.version!='0.7.1':raise RuntimeError('Unexpected cuSPARSELt version.')
    wheel=dist.read_text('WHEEL') or ''
    tags=[l.strip() for l in wheel.splitlines() if l.startswith('Tag:')]
    if tags!=['Tag: py3-none-manylinux2014_sbsa']:
        raise RuntimeError('Unexpected wheel tags; compatibility mismatch not established: '+repr(tags))
    libs=[Path(dist.locate_file(f)) for f in (dist.files or []) if str(f).replace('\\','/').endswith('nvidia/cusparselt/lib/libcusparseLt.so.0')]
    if len(libs)!=1 or not libs[0].is_file():raise RuntimeError('Expected cuSPARSELt library not found.')
    lib=libs[0]
    with lib.open('rb') as f:header=f.read(20)
    if len(header)<20 or header[:4]!=b'\x7fELF' or header[4:6]!=b'\x02\x01' or struct.unpack('<H',header[18:20])[0]!=183:
        raise RuntimeError('cuSPARSELt binary is not a 64-bit little-endian AArch64 ELF.')
    loader(str(lib))
    return dict(accepted_metadata_exception=True,package='nvidia-cusparselt-cu12',version=dist.version,wheel_tags=tags,library=str(lib),elf_machine='AArch64',library_load='passed',note='Only this exact SBSA tag mismatch accepted. Vendor metadata unchanged. GPU probe still required.')

def check():
    result=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    output=result.stdout+result.stderr
    if result.returncode==0:return dict(pip_check='passed',accepted_metadata_exception=False)
    # Import loads PyTorch's CUDA dependencies before testing the cuSPARSELt library.
    # Any torch import failure is fatal, not ignored.
    import torch
    return validate_exception(output,result.returncode,platform.system(),platform.machine(),metadata.distribution('nvidia-cusparselt-cu12'),ctypes.CDLL)

def resume():
    root=Path(__file__).resolve().parent;py=root/'.venv/bin/python'
    if not py.is_file():raise FileNotFoundError('Existing .venv not found. Put this file beside setup_gh200.py and rerun the original setup first.')
    proc=subprocess.run([str(py),str(Path(__file__).resolve()),'--check'],capture_output=True,text=True)
    if proc.returncode:raise RuntimeError('Environment verification failed:\n'+proc.stdout+proc.stderr)
    evidence=json.loads(proc.stdout);(root/'dependency_check.json').write_text(json.dumps(evidence,indent=2))
    print('Dependency verification:', 'verified ARM64/SBSA metadata exception' if evidence.get('accepted_metadata_exception') else 'passed')
    subprocess.run([str(py),str(root/'verify_gpu.py')],check=True)
    with (root/'installed_packages.txt').open('w') as f:subprocess.run([str(py),'-m','pip','freeze'],stdout=f,check=True)
    print('Setup complete. Run the existing configuration and Launch/resume cells.')
    return str(py)

if __name__=='__main__':
    if '--check' in sys.argv:print(json.dumps(check()))
    else:resume()
