"""Held-out activation-frame correction × native Adam channel factorial.
Original OLMo checkpoints only; no new model acquisition or optimizer branches.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess,sys,time,traceback,uuid
import numpy as np
import torch
import four_vector_study as f
from association_core import Config,Engine,make_data,digest,collate
from geometry_math import persistent_homology

def defaults(source,output=None):
    return dict(source=str(Path(source).resolve()),output=str(Path(output or 'runs/frame_history_v1').resolve()),
        jobs=[dict(seed=1,step=21),dict(seed=1,step=20),dict(seed=1,step=48),dict(seed=3,step=7)],
        layers=[0,1,9],alphas=[.25,.625,1.],random_controls=4,partition_seed=8041,
        device='cuda:0',threads=8,batch=2,hours=11.5,reserve_minutes=10,minimum_free_gb=25.,max_output_gb=5.,
        topology=True,topology_points=16,version='frame-history-1')

class Control(f.p.Control):
    storage_checks=0
    def storage(self):
        self.storage_checks+=1
        if self.storage_checks%32==0:super().storage()


def fit_frame(x,y):
    """Uncentered proper orthogonal Procrustes in the joint row span, identity elsewhere."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    _,sv,vt=np.linalg.svd(np.concatenate([x,y]),full_matrices=False)
    rank=int(np.sum(sv>max(sv[0],1e-30)*1e-10));basis=vt[:rank].T
    a,b=np.einsum("ij,jk->ik",x,basis),np.einsum("ij,jk->ik",y,basis)
    u,_,vt=np.linalg.svd(np.einsum("ji,jk->ik",a,b));sg=np.ones(rank)
    if np.linalg.det(np.einsum("ij,jk->ik",u,vt))<0:sg[-1]=-1
    rot=np.einsum("ij,jk->ik",u*sg,vt)
    if rank==0 or not np.isfinite(rot).all():raise ValueError("Degenerate/nonfinite calibration frame")
    return dict(basis=basis,rot=rot,shift=y.mean(0)-x.mean(0),scale=np.array(np.linalg.norm(y)/max(np.linalg.norm(x),1e-30)))

def rotate(x,basis,rot):return x+np.einsum("ij,jk,kl,ml->im",x,basis,rot-np.eye(rot.shape[0]),basis)

def distance(x):
    d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=-1);np.fill_diagonal(d,0);return d

def geometry(x,reference,topology):
    d0=distance(reference);scale=np.median(d0[np.triu_indices(len(d0),1)]) if len(d0)>1 else 1.
    scale=max(float(scale),1e-12);d=distance(x)
    sv=np.linalg.svd(x-x.mean(0),compute_uv=False);eig=sv**2/max(len(x)-1,1)
    out=dict(centered_covariance_eigenvalues=eig.tolist(),mean_norm=float(np.linalg.norm(x,axis=1).mean()),
        distance_relative_change=float(np.linalg.norm(d-d0)/max(np.linalg.norm(d0),1e-30)),reference_distance_scale=scale)
    if topology:out['ph']=persistent_homology(d/scale,np.linspace(0,3,31).tolist())
    return out

def output_points(rows):
    # Fixed token identities, coarse probabilities; sqrt embedding gives Hellinger geometry up to a constant.
    return np.sqrt(np.maximum([[r['label0_probability'],r['label1_probability'],1-r['answer_mass']] for r in rows],0))

def apply_patch(x,frame,condition,seed):
    if condition=='sham':return x
    basis=torch.as_tensor(frame['basis'],device=x.device,dtype=x.dtype);rot=torch.as_tensor(frame['rot'],device=x.device,dtype=x.dtype)
    correction=(x@basis)@(rot-torch.eye(rot.shape[0],device=x.device,dtype=x.dtype))@basis.T
    if condition=='rotation':return x+correction
    if condition=='inverse':return x+(x@basis)@(rot.T-torch.eye(rot.shape[0],device=x.device,dtype=x.dtype))@basis.T
    if condition=='mean':return x+torch.as_tensor(frame['shift'],device=x.device,dtype=x.dtype)
    if condition=='scale':return x*float(frame['scale'])
    if condition.startswith('orthogonal_'):
        b=frame[condition];b=torch.as_tensor(b,device=x.device,dtype=x.dtype)
        return x+(x@b)@(rot-torch.eye(rot.shape[0],device=x.device,dtype=x.dtype))@b.T
    if condition.startswith('additive_'):
        # Per-entity deterministic vector; matched norm to fitted rotation at THIS corner.
        rng=np.random.default_rng(seed);noise=torch.as_tensor(rng.normal(size=x.shape),device=x.device,dtype=x.dtype)
        return x+noise/noise.norm(dim=-1,keepdim=True).clamp_min(1e-30)*correction.norm(dim=-1,keepdim=True)
    raise ValueError(condition)

def probe(e,rows,layer,frame=None,condition='unpatched',seed=0,ctl=None):
    """One prompt per forward keeps random controls stable across batching/resume."""
    metrics=[];acts=[];patched=[]
    module=e.model.model.layers[layer]
    for row in rows:
        if ctl:ctl.check()
        captured={}
        def hook(mod,args,out):
            h=out[0] if isinstance(out,tuple) else out
            x=h[:,-1,:];y=x if condition=='unpatched' else apply_patch(x,frame,condition,seed+int(row['entity'])*1009)
            captured['x']=x.detach().cpu().double().numpy()[0];captured['y']=y.detach().cpu().double().numpy()[0]
            if condition=='unpatched':return out
            new=h.clone();new[:,-1,:]=y
            return (new,)+out[1:] if isinstance(out,tuple) else new
        handle=module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                inputs,q,labels=collate([row],e.tok.pad_token_id,e.c.device)
                h=e.model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[:,-1,:]
                logits=e.model.lm_head(h).double();lp=logits.log_softmax(-1);qq=q.double()
                loss=float((qq*(qq.log()-lp[:,labels])).sum());major=labels[int(row['q'][1]>row['q'][0])]
                margin=float(lp[0,major]-lp[0,labels[1] if major==labels[0] else labels[0]])
                metrics.append(dict(entity=row['entity'],loss=loss,majority_probability=float(lp[0,major].exp()),
                    answer_mass=float(lp[0,labels].exp().sum()),two_answer_accuracy=.5 if margin==0 else float(margin>0),
                    label0_probability=float(lp[0,labels[0]].exp()),label1_probability=float(lp[0,labels[1]].exp()),full_vocabulary_accuracy=float(int(logits.argmax(-1))==major),signed_margin=margin,
                    patch_norm=float(np.linalg.norm(captured['y']-captured['x']))))
                acts.append(captured['x']);patched.append(captured['y'])
        finally:handle.remove()
    return metrics,np.array(acts),np.array(patched)

def set_corner(e,before,vec,residual,alpha,u,v):
    with torch.no_grad():
        for n,q in e.params.items():q.copy_((before[n]+alpha*(u*vec['history'][n]+v*vec['current'][n]+u*v*residual[n])).to(q.device))

def summarize(root):
    root=Path(root);groups={};rows=[]
    for path in sorted((root/'units').glob('*.json')):
        r=f.read(path);key=(r['job'],r['layer'],r['alpha'],r['split'],r['condition']);groups.setdefault(key,{})[r['corner']]=r
    for key,corners in groups.items():
        if not all(k in corners for k in ['00','10','01','11']):continue
        means={k:np.mean([r['loss'] for r in v['rows']]) for k,v in corners.items()}
        interaction=means['11']-means['10']-means['01']+means['00']
        entity_corners={k:{r['entity']:r['loss'] for r in v['rows']} for k,v in corners.items()}
        if any(set(v)!=set(entity_corners['00']) for v in entity_corners.values()):raise ValueError('Factorial entities do not match')
        entity_interactions={str(entity):entity_corners['11'][entity]-entity_corners['10'][entity]-entity_corners['01'][entity]+entity_corners['00'][entity] for entity in entity_corners['00']}
        rows.append(dict(zip(['job','layer','alpha','split','condition'],key),interaction=float(interaction),entity_interactions=entity_interactions,corner_losses={k:float(v) for k,v in means.items()}))
    index={(r['job'],r['layer'],r['alpha'],r['split'],r['condition']):r for r in rows}
    for r in rows:
        ref=index.get((r['job'],r['layer'],r['alpha'],r['split'],'unpatched'))
        if ref:
            r['interaction_change']=r['interaction']-ref['interaction']
            r['endpoint_loss_change']=r['corner_losses']['11']-ref['corner_losses']['11']
            r['baseline_loss_change']=r['corner_losses']['00']-ref['corner_losses']['00']
    f.atomic_json(rows,root/'factorial_summary.json')
    return rows

def case(e,s,job,ctl):
    root=Path(s['output']);jid=f.p.job_id(job);data=make_data(e.tok,e.c,job['seed'])
    if digest(data)!=f.read(Path(s['source'])/f"seed{job['seed']}"/'data.json')['hash']:raise ValueError('Original data hash mismatch')
    f.restore_before(e,s,data,job['seed'],job['step'],ctl)
    before,after,delta,vec,train=f.capture_step(e,data,job['seed'],job['step'])
    vec.pop('raw_m',None);vec.pop('history_scale',None)
    residual={n:delta[n]-vec['history'][n]-vec['current'][n] for n in delta}
    audit=root/(jid+'_audit.json')
    if not audit.exists():
        record=f.measure_vectors(e,dict(s,gradient_splits=['A_test','B_test'],gradient_batch=s['batch']),data,before,after,delta,vec,0.,ctl,jid)
        f.atomic_json(dict(pre=record,training=train,weights_before_sha256=f.p.tensor_hash(before),weights_after_sha256=f.p.tensor_hash(after),
            residual_convention='alpha*u*v*r; 11 equals natural interpolation up to float rounding; factorial includes this audited residual'),audit)
        vec.pop('gA',None);vec.pop('gB',None)
    ids=sorted({r['entity'] for r in data['A_test']});order=np.random.default_rng(s['partition_seed']).permutation(ids);cal=set(map(int,order[:len(ids)//2]));test=set(ids)-cal
    f.atomic_json(dict(calibration_entities=sorted(cal),test_entities=sorted(test)),root/(jid+'_partition.json'))
    calrows=[r for split in ['A_valid','B_valid'] for r in data[split] if r['entity'] in cal]
    for layer in s['layers']:
        if layer>=len(e.model.model.layers):raise ValueError(f'Layer {layer} unavailable')
        ctl.pulse(phase='fitting reference frame',job=jid,layer=layer)
        f.p.assign(e,before,after,0);_,target,_=probe(e,calrows,layer,ctl=ctl)
        refs={};reference_outputs={}
        for split in ['A_test','B_test']:
            rr=[r for r in data[split] if r['entity'] in test]
            reference_outputs[split],refs[split],_=probe(e,rr,layer,ctl=ctl)
        f.p.assign(e,before,after,1);_,source,_=probe(e,calrows,layer,ctl=ctl)
        frame=fit_frame(source,target)
        for k in range(s['random_controls']):
            rng=np.random.default_rng(s['partition_seed']+layer*997+k)
            frame[f'orthogonal_{k}']=np.linalg.qr(rng.normal(size=frame['basis'].shape),mode='reduced')[0]
        np.savez_compressed(root/'maps'/f'{jid}_layer{layer}.npz',**frame)
        fitted=rotate(source,frame['basis'],frame['rot'])
        f.atomic_json(dict(rank=int(frame['rot'].shape[0]),dimension=int(source.shape[1]),calibration_mse_before=float(np.mean((source-target)**2)),calibration_mse_after=float(np.mean((fitted-target)**2)),orthogonality_error=float(np.linalg.norm(np.einsum('ji,jk->ik',frame['rot'],frame['rot'])-np.eye(frame['rot'].shape[0]))),fit='uncentered proper Procrustes; identity outside joint calibration row span'),root/'maps'/f'{jid}_layer{layer}_fit.json')
        conditions=['unpatched','sham','rotation','inverse','mean','scale']+[f'{name}_{k}' for k in range(s['random_controls']) for name in ['orthogonal','additive']]
        for alpha in s['alphas']:
            for corner in ['00','10','01','11']:
                set_corner(e,before,vec,residual,alpha,int(corner[0]),int(corner[1]))
                for split in ['A_test','B_test']:
                    rr=[r for r in data[split] if r['entity'] in test]
                    for condition in conditions:
                        label=f'{jid}_l{layer}_a{alpha}_{corner}_{split}_{condition}';dest=root/'units'/(label+'.json')
                        if dest.exists():continue
                        ctl.pulse(phase='held-out frame/channel factorial',job=jid,layer=layer,alpha=alpha,corner=corner,split=split,condition=condition)
                        rows,x,y=probe(e,rr,layer,frame,condition,s['partition_seed']+sum(map(ord,condition))*7919,ctl)
                        k=min(len(x),s['topology_points']);ref=refs[split]
                        out=dict(job=jid,layer=layer,alpha=alpha,corner=corner,split=split,condition=condition,rows=rows,
                            alignment_mse=float(np.mean((y-ref)**2)),unpatched_alignment_mse=float(np.mean((x-ref)**2)),
                            geometry=geometry(y[:k],ref[:k],s['topology']),
                            coarse_output_geometry=geometry(output_points(rows[:k]),output_points(reference_outputs[split][:k]),s['topology']),landmark_entities=[r['entity'] for r in rr[:k]],
                            norm_preservation_max_error=float(np.max(np.abs(np.linalg.norm(y,axis=1)-np.linalg.norm(x,axis=1)))))
                        f.atomic_json(out,dest);ctl.storage()
        summarize(root)
    f.p.assign(e,before,after,1)
    f.atomic_json(dict(done=True),root/(jid+'_DONE.json'))

def validate(s):
    if s['version']!='frame-history-1':raise ValueError('Settings version mismatch')
    c=Config(**f.read(Path(s['source'])/'config.json'));c.device=s['device'];c.threads=s['threads']
    if c.model!='allenai/OLMo-1B-hf' and not c.smoke and not Path(c.model).exists():raise ValueError('This release targets original OLMo checkpoints')
    if not c.smoke and not Path(c.model).exists() and (len(c.revision)!=40 or any(x not in '0123456789abcdef' for x in c.revision.lower())):raise ValueError('Original immutable model revision required')
    if c.entities<4 or not s['jobs'] or not s['layers']:raise ValueError('Need jobs, layers and >=4 entities')
    if len(set(s['layers']))!=len(s['layers']) or min(s['layers'])<0:raise ValueError('Invalid layers')
    if not s['alphas'] or any(not 0<a<=1 for a in s['alphas']):raise ValueError('Use alpha in (0,1]')
    if not 2<=s['topology_points']<=32 or s['random_controls']<1:raise ValueError('Invalid controls/landmarks')
    if s['hours']*60<=s['reserve_minutes']:raise ValueError('Invalid time budget')
    for j in s['jobs']:
        if j['seed'] not in c.seeds or not 1<=j['step']<=c.b_steps:raise ValueError('Unavailable seed/update')
        if not (Path(s['source'])/f"seed{j['seed']}"/'anchor.pt').exists():raise FileNotFoundError('Original anchor missing')
    return c

def alive(output):
    import fcntl
    p=Path(output)/'worker.lock'
    if not p.exists():return False
    with p.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
        except BlockingIOError:return True

def run(s):
    import fcntl
    root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        for folder in ['units','maps']:(root/folder).mkdir(exist_ok=True)
        try:
            c=validate(s);ctl=Control(s);ctl.pulse(phase='loading original pinned OLMo');e=Engine(c)
            for j in s['jobs']:
                if not (root/(f.p.job_id(j)+'_DONE.json')).exists():case(e,s,j,ctl)
            state='complete';message='All configured probes finished. Inspect controls and audits; no universal mechanism inferred.'
        except f.ng.Pause as exc:state='paused';message=str(exc)
        except Exception:state='failed';message=traceback.format_exc()
        summarize(root);f.atomic_json(dict(status=state,message=message,last_progress=time.time()),root/'status.json')

def launch(s):
    import fcntl
    validate(s);root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if alive(root):return str(root)
        previous=f.read(root/'settings.json');runtime={'hours','reserve_minutes','minimum_free_gb','max_output_gb'}
        if previous and {k:v for k,v in previous.items() if k not in runtime}!={k:v for k,v in s.items() if k not in runtime}:
            root=root.with_name(root.name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);root.mkdir();s=dict(s,output=str(root))
        code=root/'code';code.mkdir(exist_ok=True)
        for p in Path(__file__).parent.glob('*.py'):
            if p.resolve()!=(code/p.name).resolve():shutil.copy2(p,code/p.name)
        f.atomic_json(s,root/'settings.json');(root/'STOP').unlink(missing_ok=True)
        f.atomic_json(dict(code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')},source_config=f.read(Path(s['source'])/'config.json'),torch=torch.__version__),root/'provenance.json')
        f.atomic_json(dict(status='launching',last_progress=time.time()),root/'status.json')
        with (root/'worker.log').open('a') as log:subprocess.Popen([sys.executable,str(code/'frame_history.py'),'--worker',str(root/'settings.json')],stdout=log,stderr=log,start_new_session=True)
    return str(root)

def restart(s):
    if alive(s['output']):raise RuntimeError('Request Stop, then wait until alive=False before starting a fresh run.')
    root=Path(s['output']);new=root.with_name(root.name+'_fresh_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4])
    return launch(dict(s,output=str(new)))

def status(output):
    root=Path(output);r=f.read(root/'status.json',{'status':'not_started'});r.update(alive=alive(root),units=len(list((root/'units').glob('*.json'))),completed_jobs=len(list(root.glob('*_DONE.json'))))
    if 'last_progress' in r:r['seconds_since_progress']=round(time.time()-r['last_progress'],1)
    return r

def stop(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);(root/'STOP').touch();return 'Stop requested; refresh until alive=False. Launch resumes saved units.'

def export(output):
    import zipfile
    root=Path(output);summarize(root);dest=root.parent/(root.name+'_share.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for p in root.rglob('*'):
            if p.is_file() and p.suffix in {'.json','.py','.npz','.txt','.png'}:z.write(p,p.relative_to(root))
    return str(dest)

def show_results(output,job=None,layer=None,alpha=1.,plot=True):
    root=Path(output);rows=summarize(root)
    selected=[r for r in rows if r['condition']=='rotation' and r['split']=='A_test']
    lines=['FRAME × HISTORY — held-out entities; evaluation patches, not continued training']
    for r in selected:
        if r['alpha']!=alpha:continue
        b=next((b for b in rows if b['job']==r['job'] and b['layer']==r['layer'] and b['alpha']==alpha and b['condition']=='rotation' and b['split']=='B_test'),{})
        lines.append(f"{r['job']} layer {r['layer']} alpha {alpha}: fitted-rotation B endpoint change {b.get('endpoint_loss_change',float('nan')):+.6f}; A endpoint change {r.get('endpoint_loss_change',float('nan')):+.6f}; interaction change {r.get('interaction_change',float('nan')):+.6f}; baseline change {r.get('baseline_loss_change',float('nan')):+.6f}")
    lines.append('Negative endpoint change is an immediate rescue. Negative interaction change alone is not a rescue. Compare B and matched controls in factorial_summary.json.')
    report='\n'.join(lines);(root/'REPORT.txt').write_text(report);print(report)
    if plot and selected:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        job=job or selected[0]['job'];layer=selected[0]['layer'] if layer is None else layer
        a=[r for r in rows if r['job']==job and r['layer']==layer and r['alpha']==alpha and r['split']=='A_test' and 'endpoint_loss_change' in r]
        if not a:return rows
        fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
        for ax,key,title in zip(axes,['endpoint_loss_change','interaction_change'],['A endpoint loss change','A factorial interaction change']):
            ax.barh([r['condition'] for r in a],[r[key] for r in a],color=['#b55a30' if r['condition']=='rotation' else '#32758a' for r in a]);ax.axvline(0,color='black',lw=.7);ax.set_title(title);ax.set_xlabel('Difference from unpatched (nats)')
        fig.suptitle(f'{job}, layer {layer}, alpha={alpha}; completed four-corner groups')
        fig.savefig(root/'effects.png',dpi=160);plt.close(fig)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();run(f.read(a.worker))
