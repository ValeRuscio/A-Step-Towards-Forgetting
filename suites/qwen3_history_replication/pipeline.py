import argparse,fcntl,os,subprocess,time,traceback
from pathlib import Path
from replication import HERE,read,write,code_hashes

def run(s):
    out=Path(s['output']);deadline=time.monotonic()+s['hours']*3600;stage=None
    try:
        if code_hashes()!=read(out/'code_hashes.json'):raise RuntimeError('Package changed after launch')
        subprocess.run([s['python'],str(HERE/'prepare.py'),'--settings',str(out/'settings.json')],check=True)
        stages=[('natural','natural','run_study.py'),('history','history','run_replay.py')]
        if s['run_paths']:stages.append(('paths','natural','run_paths.py'))
        for stage,sub,script in stages:
            if (out/(stage+'_finished.json')).exists():continue
            if (out/'STOP').exists() or time.monotonic()>=deadline:
                write(out/'status.json',dict(status='paused',stage=stage,message='Stopped or session budget reached; launch to resume',last_progress=time.time()));return
            config=read(out/sub/'settings.json');config['hours']=max(.001,(deadline-time.monotonic())/3600);write(out/sub/'settings.json',config)
            (out/sub/'STOP').unlink(missing_ok=True)
            write(out/'status.json',dict(status='running',stage=stage,last_progress=time.time()))
            with open(out/sub/'worker.log','a') as log:
                proc=subprocess.Popen([s['python'],str(HERE/sub/script),'--settings',str(out/sub/'settings.json')],cwd=HERE/sub,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:
                    if (out/'STOP').exists() or time.monotonic()>=deadline:(out/sub/'STOP').touch()
                    time.sleep(1)
            st=read(out/sub/'status.json',{})
            if proc.returncode!=0 or st.get('status') not in ['complete','complete_with_limitations']:
                write(out/'status.json',dict(status='paused' if st.get('status')=='paused' else 'failed',stage=stage,message=st.get('message','Worker exited; inspect '+sub+'/worker.log'),last_progress=time.time()));return
            write(out/(stage+'_finished.json'),dict(completed=True,worker_status=st))
        hs=read(out/'history/summary.json',{});ns=read(out/'natural/summary.json',{})
        limits=ns.get('limitations',[])
        if hs.get('age_audits_pass')!=hs.get('updates'):limits=list(limits)+['Some age audits failed']
        write(out/'status.json',dict(status='complete_with_limitations' if limits else 'complete',stage=stage,last_progress=time.time(),limitations=limits,message='Finished scheduled work. Inspect acquisition and derivative audits before interpretation.'))
    except Exception:write(out/'status.json',dict(status='failed',stage=stage,message=traceback.format_exc(),last_progress=time.time()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--settings',required=True);s=read(ap.parse_args().settings)
    with open(Path(s['output'])/'pipeline.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);run(s)
