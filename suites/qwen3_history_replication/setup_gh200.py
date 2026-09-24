"""Run once with the existing notebook Python. All installs go into package/.venv."""
import json,os,platform,subprocess,sys,venv
from pathlib import Path
HERE=Path(__file__).resolve().parent

def setup():
    if platform.system()!='Linux':raise RuntimeError('Run this setup on the Linux GPU server.')
    if not (3,10)<=sys.version_info[:2]<(3,14):raise RuntimeError('Use Python 3.10–3.13 to create this environment.')
    env=HERE/'.venv';py=env/'bin/python';log=HERE/'setup.log'
    if not py.exists():venv.EnvBuilder(with_pip=True,system_site_packages=False).create(env)
    def run(args):
        with log.open('a') as f:
            p=subprocess.run([str(py),*args],stdout=f,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Setup failed. Last log lines:\n'+ '\n'.join(log.read_text().splitlines()[-35:]))
    print('Installing into',env,'— progress log:',log,flush=True)
    run(['-m','pip','install','--upgrade','pip'])
    run(['-m','pip','install','--no-cache-dir','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu128'])
    run(['-m','pip','install','--no-cache-dir','-r',str(HERE/'requirements.txt')])
    run([str(HERE/'repair_gh200_setup.py')])
    with (HERE/'installed_packages.txt').open('w') as f:subprocess.run([str(py),'-m','pip','freeze'],stdout=f,check=True)
    print((HERE/'gpu_check.json').read_text())
    print('Ready. The notebook can keep its current kernel; workers use',py)
    return str(py)

if __name__=='__main__':setup()
