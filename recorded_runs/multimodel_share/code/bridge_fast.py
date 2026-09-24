"""Performance-only overlay for mechanism_bridge v1. Original scientific settings/signature stay intact."""
from __future__ import annotations
import os,sys,time,json,hashlib,math,signal,subprocess,traceback,shutil
from pathlib import Path
import numpy as np
import torch
import mechanism_bridge as b
from bridge_replay import save,load

FAST_VERSION='1.0'
_original_capture=b.capture_state
DEFAULT_RUNTIME=dict(checkpoint_every=4,cleanup_completed_checkpoints=True,report_every_jobs=4,verify_all_source_hashes=False)

def runtime(s):
    f=Path(s['output'])/'performance_options.json'
    options=dict(DEFAULT_RUNTIME,**(b.read(f) if f.exists() else {}))
    for key in ['checkpoint_every','report_every_jobs']:
        if not isinstance(options[key],int) or isinstance(options[key],bool) or options[key]<1:raise ValueError(key+' must be a positive integer.')
    return options

def attention_from_trace(e,raw,layers):
    result={}
    for l in layers:
        q,k,v=[raw[f'{l}_{a}'].astype(np.float64) for a in ['q','k','v']];length=len(q)
        heads=e.model.config.num_attention_heads;kv=e.model.config.num_key_value_heads;dim=q.shape[-1]//heads
        clip=getattr(e.model.config,'clip_qkv',None)
        if clip is not None:q,k,v=[np.clip(a,-clip,clip) for a in [q,k,v]]
        q=q.reshape(length,heads,dim).transpose(1,0,2);k=np.repeat(k.reshape(length,kv,dim).transpose(1,0,2),heads//kv,axis=0)
        v=np.repeat(v.reshape(length,kv,dim).transpose(1,0,2),heads//kv,axis=0)
        rotary=getattr(e.model.model,'rotary_emb',None) or getattr(e.mods[f'model.layers.{l}.self_attn'],'rotary_emb',None)
        inv=rotary.inv_freq.detach().float().cpu().numpy() if rotary is not None else 1/(e.model.config.rope_theta**(np.arange(0,dim,2)/dim))
        phase=np.arange(length)[:,None]*inv[None,:];co=np.cos(phase)[None];si=np.sin(phase)[None];half=dim//2
        def rotate(a):return np.concatenate([a[:,:,:half]*co-a[:,:,half:]*si,a[:,:,:half]*si+a[:,:,half:]*co],-1)
        q,k=rotate(q),rotate(k);scores=np.einsum('hid,hjd->hij',q,k)/math.sqrt(dim)
        scores=np.where(np.tri(length,dtype=bool),scores,-np.inf);P=np.exp(scores-scores.max(-1,keepdims=True));P/=P.sum(-1,keepdims=True)
        Y=np.einsum('hij,hjd->hid',P,v).transpose(1,0,2).reshape(length,-1);native=raw[f'{l}_Y']
        err=float(np.linalg.norm(Y-native)/max(np.linalg.norm(native),1e-30))
        result[l]=dict(P=P.astype(np.float32),V=v.astype(np.float32),native=native,error=err,
            X_normalized=raw[f'{l}_input'],Q_projected=raw[f'{l}_q'],K_projected=raw[f'{l}_k'])
    return result


def fused_trace(e,row,layers):
    values={};extra={};handles=[]
    def pre(target,name):
        def hook(m,args):target[name]=args[0]
        return hook
    def post(target,name):
        def hook(m,args,out):target[name]=out[0] if isinstance(out,tuple) else out
        return hook
    for l in layers:
        root=f'model.layers.{l}'
        handles.append(e.mods[root+'.self_attn.o_proj'].register_forward_pre_hook(pre(values,f'{l}_Y')))
        handles.append(e.mods[root+'.mlp.gate_proj'].register_forward_hook(post(values,f'{l}_gate')))
        handles.append(e.mods[root].register_forward_hook(post(values,f'{l}_residual')))
        for a in ['q','k','v']:
            handles.append(e.mods[root+f'.self_attn.{a}_proj'].register_forward_hook(post(extra,f'{l}_{a}')))
        handles.append(e.mods[root+'.self_attn.q_proj'].register_forward_pre_hook(pre(extra,f'{l}_input')))
    try:
        inputs,q,labels=b.collate([row],e.tok.pad_token_id,e.c.device)
        h=e.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1]
        lp=e.model.lm_head(h).double().log_softmax(-1);p=lp[:,labels]
        loss=(q.double()*(q.double().log()-p)).sum();margin=p[0,0]-p[0,1];support=-p.logsumexp(-1).item()
        names=list(values);targets=[values[n] for n in names]
        gm=torch.autograd.grad(margin,targets,retain_graph=True,allow_unused=False)
        gl=torch.autograd.grad(loss,targets,allow_unused=False)
        arrays={}
        for name,x,a,c in zip(names,targets,gm,gl):
            arrays[name]=x[0].detach().float().cpu().numpy()
            arrays[name+'_margin_grad']=a[0].detach().float().cpu().numpy()
            arrays[name+'_loss_grad']=c[0].detach().float().cpu().numpy()
        raw={k:x[0].detach().float().cpu().numpy() for k,x in extra.items()};raw.update(arrays)
        attention=attention_from_trace(e,raw,layers)
        for l,v in attention.items():
            for k,value in v.items():arrays[f'{l}_{k}']=value
        arrays['final_hidden']=h[0].detach().float().cpu().numpy()
        r=dict(loss=loss.item(),margin=margin.item(),target_margin=math.log(row['q'][0]/row['q'][1]),support=support,relation=loss.item()-support)
        return arrays,r
    finally:
        for h in handles:h.remove()
        e.opt.zero_grad(set_to_none=True)


def audit_fused(e,row,layers,arrays,r):
    reference=b.trace(e,row,layers)
    for l,v in b.core.night_attention_arrays(e,row,layers).items():
        for k,value in v.items():reference[f'{l}_{k}']=value
    rr=b.eval_row(e,row,return_hidden=True);reference['final_hidden']=rr.pop('_final_hidden')
    if set(reference)!=set(arrays):raise ArithmeticError('Fused capture changed the array inventory.')
    maximum=0.
    for k,value in reference.items():
        a=np.asarray(value);c=np.asarray(arrays[k]);np.testing.assert_allclose(c,a,rtol=1e-6,atol=1e-7,err_msg=k)
        maximum=max(maximum,float(np.max(np.abs(c-a))))
    for k,value in rr.items():
        if abs(r[k]-value)>1e-8:raise ArithmeticError('Fused capture changed evaluation metrics.')
    return dict(array_count=len(reference),maximum_absolute_difference=maximum,passed=True)


def capture_state(e,s,data,folder,layers):
    folder.mkdir(parents=True,exist_ok=True)
    for split in b.SPLITS:
        for row in data[split]:
            name=b.key(split,row);f=folder/(name+'.json');a=folder/(name+'.npz')
            if f.exists() and a.exists():continue
            b.progress_check(s,phase='single-forward capture and exact local derivatives',split=split,entity=row['entity'])
            arrays,r=fused_trace(e,row,layers)
            for l in layers:
                if float(arrays[f'{l}_error'])>s['attention_tolerance']:raise ArithmeticError('Attention reconstruction audit failed.')
            if not getattr(e,'_fused_verified',False):
                audit=audit_fused(e,row,layers,arrays,r);audit.update(split=split,entity=row['entity'])
                b.atomic_json(audit,folder/'fused_capture_audit.json');e._fused_verified=True
            if any(not np.isfinite(x).all() for x in arrays.values()):raise FloatingPointError('Nonfinite capture array.')
            if any(not math.isfinite(x) for x in r.values()):raise FloatingPointError('Nonfinite captured metric.')
            r.update(entity=row['entity'],split=split,q=row['q'])
            b.core.night_npz(a,**arrays);b.atomic_json(r,f)


def cleanup_completed(s):
    """Remove only replayable optimizer snapshots of marked-complete branches; keep measurements."""
    if not runtime(s)['cleanup_completed_checkpoints']:return []
    out=Path(s['output']);removed=[]
    for folder in out.glob('branch_*'):
        if folder.is_symlink() or not folder.is_dir():continue
        done=folder/'DONE.json';curve=folder/'curve.json'
        if not done.exists() or not curve.exists():continue
        if not b.read(curve).get('complete'):continue
        if b.read(done).get('job',{}).get('kind')!='branch':continue
        for name in ['resume.pt','resume.pt.tmp']:
            file=folder/name
            if not file.is_file() or file.is_symlink():continue
            size=file.stat().st_size;file.unlink();removed.append(dict(path=str(file.relative_to(out)),bytes=size,time=time.time()))
    if removed:
        file=out/'checkpoint_cleanup.json';old=b.read(file) if file.exists() else []
        b.atomic_json(old+removed,file)
    return removed


def hash_progress(s,**kw):
    # Hashing is read-only; permit it below the disk reserve so verified cleanup can run.
    out=Path(s['output'])
    if (out/'STOP').exists():raise b.Pause('Stop requested.')
    budget=b.read(out/'budget.json') if (out/'budget.json').exists() else {}
    if time.time()>budget.get('deadline',float('inf')):raise b.Pause('Session time budget reached.')
    state=b.read(out/'status.json') if (out/'status.json').exists() else {}
    state.update(kw,status='running',heartbeat=time.time());b.atomic_json(state,out/'status.json')


def cleanup_staging(out):
    # Called only while this worker exclusively owns worker.lock.
    out=Path(out).resolve();candidates=list(out.glob('branch_*/resume.pt.tmp'))+list(out.glob('results_share*.zip.tmp'));removed=[]
    for file in candidates:
        if file.is_file() and not file.is_symlink() and out in file.resolve().parents:
            size=file.stat().st_size;file.unlink();removed.append(dict(path=str(file.relative_to(out)),bytes=size,time=time.time()))
    if removed:
        log=out/'staging_cleanup.json';previous=b.read(log) if log.exists() else [];b.atomic_json(previous+removed,log)
    return removed


def fingerprint(s):
    out=Path(s['output']);f=out/'source_fingerprints.json';previous=b.read(f) if f.exists() else None
    cachefile=out/'source_hash_cache.json';cache=b.read(cachefile) if cachefile.exists() else {};new={};records=[]
    paths=[Path(s['source'])/'config.json']
    for seed in s['seeds']:
        paths.append(Path(s['source'])/f'seed{seed}'/'data.json');paths.extend(sorted((Path(s['source'])/f'seed{seed}').glob('fork*/fork.pt')))
    def metadata(p):
        v=p.stat();return dict(device=v.st_dev,inode=v.st_ino,size=v.st_size,mtime_ns=v.st_mtime_ns,ctime_ns=v.st_ctime_ns)
    for p in paths:
        stat=metadata(p);saved=cache.get(str(p));hash_progress(s,phase='verify source identity (cached hashes where unchanged)',file=str(p))
        if not runtime(s)['verify_all_source_hashes'] and saved and saved['metadata']==stat:sha=saved['sha256']
        else:
            h=hashlib.sha256()
            with open(p,'rb') as stream:
                while True:
                    block=stream.read(64*1024**2)
                    if not block:break
                    h.update(block);hash_progress(s,phase='hash source checkpoint',file=str(p),hashed_bytes=stream.tell())
            if metadata(p)!=stat:raise RuntimeError('Source changed while hashing.')
            sha=h.hexdigest()
        new[str(p)]=dict(metadata=stat,sha256=sha);records.append(dict(path=str(p),sha256=sha,bytes=stat['size']))
    if previous is not None and previous!=records:raise ValueError('Source content changed; refusing to mix experiments.')
    b.atomic_json(records,f);b.atomic_json(new,cachefile)


def branch_job(s,job):
    folder=Path(s['output'])/job['id'];folder.mkdir(parents=True,exist_ok=True)
    curve=folder/'curve.json'
    # An interruption after committing a complete curve needs no replay.
    if curve.exists() and b.read(curve).get('complete'):return
    j=dict(seed=job['seed'],event=job['event'],id=job['id'])
    e,data=b.core.night_prepare(b.settings_for_core(s),j,b.Progress(s),do_step=False)
    cal=b.partition(data,s);layers=b.layers_for(e,s,True);baseline=folder/'baseline'
    capture_state(e,s,data,baseline,layers)
    checkpoint=folder/'resume.pt';rows=[];start=job['event']
    if checkpoint.exists():
        saved=load(checkpoint);e.restore(saved['engine']);rows=saved['rows'];start=saved['step']+1;del saved
    elif job['branch']=='reset_first_moment':
        for state in e.opt.state.values():
            if 'exp_avg' in state:state['exp_avg'].zero_()
    if job['branch']=='all_steps_lr_025':
        for g in e.opt.param_groups:g['lr']=e.c.lr*.25
    recorded=b.read(curve)['rows'] if curve.exists() else []
    # Recover any measured steps since the less-frequent optimizer snapshot.
    for record in recorded:
        step=record['step']
        if step<start:continue
        if step!=start:raise ArithmeticError('Continuation curve has a gap.')
        b.progress_check(s,phase='replay saved updates after last optimizer snapshot',step=step,condition=job['branch'])
        e.gradient(b.minibatches(data,'B',job['seed'],step,e.c));e.opt.step();e.opt.zero_grad(set_to_none=True)
        rows.append(record);start=step+1
    if recorded and start>job['event']:
        actual=b.core.night_eval(e,data)
        for split in b.SPLITS:
            if abs(actual[split]['mean']-rows[-1]['losses'][split]['mean'])>1e-8:
                raise ArithmeticError('Replayed continuation differs from saved losses; refusing to continue.')
    stop=min(e.c.b_steps,job['event']+s['branch_steps']-1);interval=int(runtime(s)['checkpoint_every'])
    for step in range(start,stop+1):
        b.progress_check(s,phase='optimizer continuation with computational traces',step=step,condition=job['branch'])
        update,after,delta=b.core.measured_adam_step(e,data,job['seed'],step,b.Progress(s));del after,delta
        losses=b.core.night_eval(e,data);stepfolder=folder/f'step{step:03d}';stepfolder.mkdir(exist_ok=True)
        b.atomic_json(update,stepfolder/'update.json')
        if (step-job['event'])%s['branch_trace_every']==0 or step==stop:
            capture_state(e,s,data,stepfolder/'capture',layers)
            b.geometry(s,data,cal,baseline,stepfolder/'capture',layers,stepfolder/'geometry')
        rows.append(dict(step=step,losses=losses,optimizer_channels=update['totals']))
        b.atomic_json(dict(branch=job['branch'],complete=step==stop,rows=rows),curve)
        # All measurements are committed at every step; only bulky restart snapshots are less frequent.
        if (step-job['event']+1)%interval==0 and step!=stop:
            tmp=checkpoint.with_name(checkpoint.name+'.tmp')
            if tmp.is_file() and not tmp.is_symlink():tmp.unlink()  # worker owns the lock; uncommitted stale snapshot
            estimate=max(checkpoint.stat().st_size if checkpoint.exists() else 0,3*e.pb)+1024**3
            free=shutil.disk_usage(folder).free;required=estimate+s['minimum_free_gb']*1024**3
            if free<required:
                # Measurements plus deterministic replay are sufficient. Skip a convenience snapshot.
                b.atomic_json(dict(step=step,free_bytes=free,required_bytes=required,recovery='Replay from last committed snapshot/source; all measured steps saved.'),folder/'snapshot_skipped.json')
            else:
                b.progress_check(s,phase='save rolling optimizer snapshot',step=step)
                save(dict(engine=e.pack(),rows=rows,step=step),checkpoint)
    del e
    if torch.cuda.is_available():torch.cuda.empty_cache()


def process(output):
    import psutil
    f=Path(output)/'process.json'
    if not f.exists():return None
    r=b.read(f)
    try:
        p=psutil.Process(r['pid'])
        if abs(p.create_time()-r['created'])>.1 or p.status()==psutil.STATUS_ZOMBIE:return None
        cmd=p.cmdline()
        if not any(Path(x).name in ['mechanism_bridge.py','bridge_fast.py'] for x in cmd):return None
        if str((Path(output)/'settings.json').resolve()) not in cmd:return None
        return p
    except psutil.Error:return None


def run(s):
    import fcntl
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with open(out/'worker.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(b.Pause('Stop requested.')))
        jobs=b.plan(s);completed=sum((out/j['id']/'DONE.json').exists() for j in jobs)
        b.atomic_json(dict(status='running',started=time.time(),heartbeat=time.time(),completed=completed,total=len(jobs)),out/'status.json')
        state='complete';error=None
        try:
            # Discard only uncommitted staging files left by a stopped worker, then verify source.
            cleanup_staging(out)
            fingerprint(s);cleanup_completed(s)
            import transformers
            hardware=b.core.night_hardware();hardware.update(numpy=np.__version__,transformers=transformers.__version__,package_version=b.VERSION,performance_version=FAST_VERSION)
            b.atomic_json(hardware,out/'hardware.json')
            for job in jobs:
                folder=out/job['id'];folder.mkdir(exist_ok=True)
                if (folder/'DONE.json').exists():continue
                b.progress_check(s,job=job['id'],phase='starting job',completed=completed,total=len(jobs))
                {'factorial':b.factorial_job,'directions':b.matched_directions_job,'branch':branch_job}[job['kind']](s,job)
                b.atomic_json(dict(finished=time.time(),job=job),folder/'DONE.json');completed+=1
                cleanup_completed(s);b.progress_check(s,completed=completed)
                if completed%runtime(s)['report_every_jobs']==0:b.report(out)
        except b.Pause as exc:
            error=str(exc);state='stopped' if (out/'STOP').exists() else ('disk_paused' if 'disk' in error.lower() or 'space' in error.lower() else 'budget_paused')
        except Exception as exc:
            state='failed';error=str(exc);b.atomic_json(dict(error=error,traceback=traceback.format_exc()),out/'failure.json')
        finally:
            b.atomic_json(dict(status=state,error=error,heartbeat=time.time(),completed=completed,total=len(jobs)),out/'status.json')
            b.report(out);b.export(out,s['export_arrays'])


def launch(s):
    """Resume in-place only when the original signature matches; never silently restart a campaign."""
    import fcntl,psutil
    b.validate(s);out=Path(s['output']).expanduser().resolve();s['output']=str(out);out.mkdir(parents=True,exist_ok=True)
    with open(out/'launch.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if process(out) is not None:return out
        sig=b.signature(s);old=b.read(out/'signature.json') if (out/'signature.json').exists() else None
        if old and old['signature']!=sig:
            raise ValueError('Original settings/code signature differs. Keep your existing three Python modules and load settings.json from this run; this performance overlay will not start a new campaign or bypass the mismatch.')
        with open(out/'worker.lock','a') as worker:
            try:fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return out
        (out/'STOP').unlink(missing_ok=True)
        b.atomic_json(s,out/'settings.json');b.atomic_json(dict(signature=sig),out/'signature.json')
        b.atomic_json(runtime(s),out/'performance_options.json')
        file=out/'performance_history.json';history=b.read(file) if file.exists() else []
        history.append(dict(version=FAST_VERSION,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time(),options=runtime(s)))
        b.atomic_json(history,file)
        b.atomic_json(dict(deadline=time.time()+(s['hours']*60-s['reserve_minutes'])*60),out/'budget.json')
        b.atomic_json(dict(status='launching',heartbeat=time.time()),out/'status.json')
        with open(out/'worker.log','a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'run','--settings',str(out/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:created=psutil.Process(child.pid).create_time()
        except psutil.Error:created=0
        b.atomic_json(dict(pid=child.pid,created=created),out/'process.json')
    return out


def resume_existing(output,hours=11.5):
    """Stop verified old/new worker, load exact stored settings, and resume the same directory."""
    output=Path(output).expanduser().resolve()
    if not (output/'settings.json').exists():raise FileNotFoundError('Choose the existing results directory containing settings.json.')
    print(b.stop(output))
    s=b.read(output/'settings.json');s['output']=str(output);s['hours']=hours
    out=launch(s);print('Resuming existing results:',out);return s


def refresh(output):
    r=b.status(output);free=shutil.disk_usage(output).free/2**30
    print(f"Status: {r['status']} | alive: {r['alive']} | jobs: {r.get('completed',0)}/{r.get('total','?')} | free: {free:.1f} GiB")
    for k in ['job','phase','step','layer','condition','error']:
        if r.get(k) is not None:print(k+':',r[k])
    if r.get('heartbeat'):print('Last progress:',round(time.time()-r['heartbeat'],1),'seconds ago.')
    if r.get('phase')=='hash source checkpoint':print('Hashed this file:',round(r.get('hashed_bytes',0)/2**30,2),'GiB')
    print('Results:',Path(output).resolve());return r

# Patch globals used by the original scientific routines without editing their source files/signature.
b.capture_state=capture_state;b.branch_job=branch_job;b.fingerprint=fingerprint;b.process=process;b.run=run;b.launch=launch
stop=b.stop;status=b.status;report=b.report;export=b.export;defaults=b.defaults;validate=b.validate;read=b.read;atomic_json=b.atomic_json
restart=b.restart;plan=b.plan;discover_source=b.discover_source;VERSION=b.VERSION
b.refresh=refresh

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run']);parser.add_argument('--settings',required=True)
    args=parser.parse_args();run(b.read(args.settings))



def deduplicate(output):
    """Optional stopped-run lossless compaction: byte-identical NPZ files share one inode."""
    import uuid
    out=Path(output).resolve()
    if process(out) is not None:raise RuntimeError('Stop the worker before deduplicating.')
    import fcntl
    with open(out/'worker.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Worker still owns this run; wait for it to stop.')
        groups={}
        for file in out.rglob('*.npz'):
            if file.is_symlink() or not file.is_file() or out not in file.resolve().parents:continue
            groups.setdefault(file.stat().st_size,[]).append(file)
        canonical={};seen=set();linked=0;saved=0
        for size,files in groups.items():
            if len(files)<2:continue
            for file in files:
                stat=file.stat();identity=(stat.st_dev,stat.st_ino)
                if identity in seen:continue
                seen.add(identity);h=hashlib.sha256()
                with open(file,'rb') as stream:
                    for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
                key=(size,h.digest(),stat.st_dev)
                if key not in canonical:canonical[key]=file;continue
                reference=canonical[key]
                # Verify byte equality as well as hash equality before replacing a physical copy.
                with open(reference,'rb') as a,open(file,'rb') as c:
                    while True:
                        aa=a.read(8*1024**2);cc=c.read(8*1024**2)
                        if aa!=cc:raise ArithmeticError('Hash collision or file changed during deduplication.')
                        if not aa:break
                tmp=file.with_name(file.name+'.hardlink-'+uuid.uuid4().hex)
                try:os.link(reference,tmp);os.replace(tmp,file)
                finally:tmp.unlink(missing_ok=True)
                linked+=1;saved+=size if stat.st_nlink==1 else 0
        result=dict(linked_files=linked,estimated_reclaimed_bytes=saved,free_bytes=shutil.disk_usage(out).free,time=time.time())
        b.atomic_json(result,out/'deduplication.json')
    print(f"Shared {linked} identical NPZ files; approximately {saved/2**30:.2f} GiB reclaimed.")
    print(f"Free space: {result['free_bytes']/2**30:.2f} GiB")
    return result



def storage_report(output):
    import stat
    out=Path(output);groups={};seen=set()
    for root,dirs,files in os.walk(out,followlinks=False):
        for name in files:
            file=Path(root)/name
            try:st=file.lstat()
            except FileNotFoundError:continue
            if not stat.S_ISREG(st.st_mode):continue
            category=('optimizer snapshots' if name.endswith('.pt') else 'raw arrays' if name.endswith('.npz') else
                      'share ZIP' if name.endswith('.zip') else 'uncommitted temporary files' if name.endswith('.tmp') else 'tables, logs and plots')
            r=groups.setdefault(category,dict(files=0,logical_bytes=0,unique_file_bytes=0))
            r['files']+=1;r['logical_bytes']+=st.st_size;identity=(st.st_dev,st.st_ino)
            if identity not in seen:r['unique_file_bytes']+=st.st_size;seen.add(identity)
    for name,r in sorted(groups.items(),key=lambda x:-x[1]['unique_file_bytes']):
        print(f"{name}: {r['unique_file_bytes']/2**30:.2f} GiB in unique files; {r['files']} paths")
    print(f"Free filesystem space: {shutil.disk_usage(out).free/2**30:.2f} GiB")
    b.atomic_json(groups,out/'storage_report.json');return groups

if __name__=='__main__':main()
