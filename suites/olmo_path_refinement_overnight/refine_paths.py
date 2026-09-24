"""Adaptive refinement of four unresolved natural updates; input results read-only."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, shutil, sqlite3, subprocess, sys, time, traceback, uuid
from pathlib import Path
import numpy as np
import torch
import precise_forgetting as p
ng=p.ng
read=p.read
atomic_json=p.atomic_json
VERSION='path-refinement-1.0'


def discover_previous():
    options=[Path('/home/ubuntu/4/runs/olmo_precise_forgetting_v1'),Path.cwd()/'runs/olmo_precise_forgetting_v1',Path.cwd().parent/'runs/olmo_precise_forgetting_v1']
    return next((str(x.resolve()) for x in options if (x/'measurements.sqlite').exists() and (x/'settings.json').exists()),None)


def defaults(previous=None,output=None):
    previous=previous or discover_previous()
    if previous is None:raise FileNotFoundError('Set PREVIOUS to the extracted original precise-results directory, containing settings.json and measurements.sqlite.')
    old=read(Path(previous)/'settings.json');s=dict(old)
    s.update(version=VERSION,previous=str(Path(previous).resolve()),output=str(Path(output or Path.cwd()/'runs/olmo_path_refinement_v1').resolve()),
        jobs=[dict(seed=1,step=21),dict(seed=2,step=11),dict(seed=2,step=10),dict(seed=1,step=20)],
        hours=11.5,reserve_minutes=10,max_output_gb=30.,minimum_free_gb=25.,
        max_points_per_prompt=257,max_depth=12,fd_widths=[.005,.0025,.00125,.000625,.0003125,.00015625],
        interior_geometry_points=2,geometry_splits=['A_valid','A_test'])
    return s


def validate(s):
    if s['version']!=VERSION:raise ValueError('Use refinement defaults().')
    q=dict(s,version=p.VERSION);c=p.validate(q)
    if s['max_points_per_prompt']<33 or s['max_depth']<4:raise ValueError('Allow >=33 points and depth >=4.')
    if not s['fd_widths'] or min(s['fd_widths'])<=0 or s['fd_widths']!=sorted(set(s['fd_widths']),reverse=True):raise ValueError('Use distinct decreasing positive finite-difference widths.')
    if not 0<=s['interior_geometry_points']<=8 or set(s['geometry_splits'])-set(s['splits']):raise ValueError('Invalid geometry configuration.')
    prev=Path(s['previous']);old=read(prev/'settings.json')
    if not old or not (prev/'measurements.sqlite').exists():raise FileNotFoundError('Previous results require settings.json and measurements.sqlite.')
    for key in ['source','splits','partition_seed','topology_landmarks','layers']:
        if old[key]!=s[key]:raise ValueError(f'Preserve prior {key} for comparable cached measurements.')
    available={(j['seed'],j['step']) for j in old['jobs']}
    if any((j['seed'],j['step']) not in available for j in s['jobs']):raise ValueError('Selected event missing from previous results.')
    if Path(s['output']).resolve()==prev.resolve():raise ValueError('Output must differ from the previous results directory.')
    return c


def connect(out):
    con=p.connect(out)
    con.execute('CREATE TABLE IF NOT EXISTS refined(job TEXT, split TEXT, entity INTEGER, record TEXT, PRIMARY KEY(job,split,entity))')
    con.execute('CREATE TABLE IF NOT EXISTS slice_geometry(job TEXT, a REAL, b REAL, split TEXT, record TEXT, PRIMARY KEY(job,a,b,split))')
    con.commit();return con


def initialize(s):
    out=Path(s['output']);prev=Path(s['previous']);out.mkdir(parents=True,exist_ok=True)
    if read(out/'IMPORTED.json'):return
    if ng.alive(prev):raise RuntimeError('Previous worker is still active. Stop it before importing its measurements.')
    target=out/'measurements.sqlite'
    with sqlite3.connect(f'file:{prev/"measurements.sqlite"}?mode=ro',uri=True) as src,sqlite3.connect(target) as dst:src.backup(dst)
    ids=[p.job_id(j) for j in s['jobs']]
    with sqlite3.connect(target) as con:
        for table in ['probes','geometry']:con.execute(f'DELETE FROM {table} WHERE job NOT IN ({",".join("?" for _ in ids)})',ids)
        con.commit()
    for jid in ids:
        folder=out/jid;folder.mkdir(exist_ok=True)
        for name in ['update.json','finite_difference.json']:
            shutil.copy2(prev/jid/name,folder/name)
        shutil.copy2(prev/jid/'path_summary.json',folder/'previous_path_summary.json')
    atomic_json(dict(previous=str(prev),jobs=ids,imported=time.time(),note='Consistent SQLite copy; original results never modified.'),out/'IMPORTED.json')


class PointLimit(Exception):pass


def adaptive_integral(evaluate,initial_points,atol,rtol,max_points=257,max_depth=12):
    """Per-entity adaptive Simpson with local exact endpoint-loss closure.
    A returned cap flag never implies convergence. Function values are persistent upstream.
    """
    cache=dict(initial_points)
    def f(x):
        x=float(x)
        if x not in cache:
            if len(cache)>=max_points:raise PointLimit()
            cache[x]=evaluate(x)
        return cache[x]
    def assess(a,b,depth):
        m=(a+b)/2;q1=(a+m)/2;q3=(m+b)/2;xx=[a,q1,m,q3,b];rr=[f(x) for x in xx]
        def integral(key):
            yy=[r['projections'][key] for r in rr]
            return (b-a)/12*(yy[0]+4*yy[1]+2*yy[2]+4*yy[3]+yy[4])
        vals={key:integral(key) for key in ['delta','history','current','roundoff']}
        coarse=(b-a)/6*(rr[0]['projections']['delta']+4*rr[2]['projections']['delta']+rr[4]['projections']['delta'])
        change=rr[-1]['loss']-rr[0]['loss'];tol=atol*(b-a)+rtol*max(abs(change),abs(vals['delta']))
        closure=vals['delta']-change;difference=abs(vals['delta']-coarse)
        return dict(a=a,b=b,depth=depth,contributions=vals,loss_change=change,closure_error=closure,nested_difference=difference,
            tolerance=tol,passed=abs(closure)<=tol and difference<=tol,score=max(abs(closure),difference)/max(tol,1e-30))
    leaves=[];capped=False
    # Seed all eight equal subintervals, retaining the previous 17-point information.
    try:
        for a in np.linspace(0,1,9)[:-1]:leaves.append(assess(float(a),float(a+.125),3))
        while True:
            bad=[(i,l) for i,l in enumerate(leaves) if not l['passed'] and l['depth']<max_depth]
            if not bad:break
            i,l=max(bad,key=lambda z:z[1]['score']);m=(l['a']+l['b'])/2
            children=[assess(l['a'],m,l['depth']+1),assess(m,l['b'],l['depth']+1)]
            leaves[i:i+1]=children
    except PointLimit:capped=True
    complete_cover=bool(leaves) and abs(sum(l['b']-l['a'] for l in leaves)-1)<1e-12
    total={k:sum(l['contributions'][k] for l in leaves) for k in ['delta','history','current','roundoff']}
    actual=f(1.)['loss']-f(0.)['loss'];tol=atol+rtol*max(abs(actual),abs(total['delta']))
    passed=complete_cover and all(l['passed'] for l in leaves) and abs(total['delta']-actual)<=tol
    return dict(points=len(cache),alphas=sorted(cache),leaves=sorted(leaves,key=lambda l:l['a']),quadrature_passed=passed,
        cap_reached=capped or any(not l['passed'] and l['depth']>=max_depth for l in leaves),complete_cover=complete_cover,
        loss_change=actual,initial_slope=f(0.)['projections']['delta'],finite_remainder=actual-f(0.)['projections']['delta'],
        integral=total['delta'] if complete_cover else None,closure_error=total['delta']-actual if complete_cover else None,
        contributions=total if complete_cover else None)


def finite_difference_audit(e,before,after,row,alpha,slope,s,ctl,jid,split):
    attempts=[];last=None;passed=False
    for width in s['fd_widths']:
        ctl.pulse(job=jid,phase='shrinking finite-difference audit',split=split,entity=row['entity'],alpha=alpha,width=width)
        vals=[]
        for a in [alpha-width,alpha+width]:
            p.assign(e,before,after,a)
            with torch.no_grad():vals.append(float(e.loss_rows([row])[0]))
        fd=(vals[1]-vals[0])/(2*width);tol=s['fd_atol']+s['fd_rtol']*max(abs(slope),abs(fd))
        good=last is not None and abs(fd-slope)<=tol and abs(fd-last)<=tol
        attempts.append(dict(width=width,estimate=fd,agreement_error=abs(fd-slope),adjacent_width_difference=None if last is None else abs(fd-last),tolerance=tol,passed=good))
        if good:passed=True;break
        last=fd
    return dict(alpha=alpha,autograd=slope,passed=passed,attempts=attempts,
        note='Two adjacent FD widths must agree with each other and autograd. Failure may reflect truncation, FP32 effects or an implementation issue; it is retained.')


def refine_job(e,s,job,con,ctl):
    jid=p.job_id(job);folder=Path(s['output'])/jid
    if read(folder/'REFINED_DONE.json'):return
    data,before,after,dirs=p.prepare(e,s,job,ctl)
    prior=read(folder/'previous_path_summary.json');oldfd=read(folder/'finite_difference.json')
    try:
        for split in s['splits']:
            oldrows={r['entity']:r for r in prior['populations'][split]['rows']}
            # Previously failing entities first, without excluding the others from audits.
            ordered=sorted(data[split],key=lambda r:oldrows[r['entity']]['integration_passed'])
            for row in ordered:
                entity=row['entity']
                if con.execute('SELECT 1 FROM refined WHERE job=? AND split=? AND entity=?',(jid,split,entity)).fetchone():continue
                initial={float(a):json.loads(rec) for a,rec in con.execute('SELECT alpha,record FROM probes WHERE job=? AND split=? AND entity=?',(jid,split,entity))}
                def evaluate(alpha):
                    cached=con.execute('SELECT record FROM probes WHERE job=? AND split=? AND entity=? AND alpha=?',(jid,split,entity,alpha)).fetchone()
                    if cached:return json.loads(cached[0])
                    ctl.pulse(job=jid,phase='adaptive exact directional derivative',split=split,entity=entity,alpha=alpha)
                    p.assign(e,before,after,alpha);rec,_=p.probe(e,row,dirs,[],False)
                    con.execute('INSERT INTO probes VALUES (?,?,?,?,?)',(jid,alpha,split,entity,json.dumps(rec,allow_nan=False)));con.commit();return rec
                old=oldrows[entity];tol=s['integration_atol']+s['integration_rtol']*max(abs(old['loss_change']),abs(old['integrated_slope']))
                if abs(old['integration_closure_error'])<=tol and old['nested_grid_difference'] is not None and old['nested_grid_difference']<=tol:
                    result=dict(points=len(initial),alphas=sorted(initial),quadrature_passed=True,cap_reached=False,complete_cover=True,
                        loss_change=old['loss_change'],initial_slope=old['initial_slope'],finite_remainder=old['exact_finite_remainder'],integral=old['integrated_slope'],closure_error=old['integration_closure_error'],
                        contributions=dict(delta=old['integrated_slope'],**old['path_contributions']),leaves=[],reused_previous_quadrature=True)
                else:result=adaptive_integral(evaluate,initial,s['integration_atol'],s['integration_rtol'],s['max_points_per_prompt'],s['max_depth'])
                cache={float(a):json.loads(rec) for a,rec in con.execute('SELECT alpha,record FROM probes WHERE job=? AND split=? AND entity=?',(jid,split,entity))}
                peak=max(cache,key=lambda a:abs(cache[a]['projections']['delta']))
                targets={peak}|{float(a['alpha']) for a in oldfd if a['split']==split and a['entity']==entity and not a['passed']}
                audits=[finite_difference_audit(e,before,after,row,a,cache[a]['projections']['delta'],s,ctl,jid,split) for a in sorted(targets)]
                result.update(entity=entity,q0=row['q'][0],split=split,peak_sampled_slope_alpha=peak,fd_audits=audits,
                    resolved=result['quadrature_passed'] and all(a['passed'] for a in audits))
                con.execute('INSERT INTO refined VALUES (?,?,?,?)',(jid,split,entity,json.dumps(result,allow_nan=False)));con.commit()
        atomic_json(dict(done=True,completed=time.time()),folder/'REFINED_DONE.json')
    finally:
        p.assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True)
        del before,after,dirs
        if torch.cuda.is_available():torch.cuda.empty_cache()


def choose_geometry_points(con,jid,count):
    votes=collections.Counter()
    for split,rec in con.execute('SELECT split,record FROM refined WHERE job=?',(jid,)):
        if not split.startswith('A_'):continue
        r=json.loads(rec);a=r['peak_sampled_slope_alpha']
        if 0<a<1:votes[a]+=1
    chosen=[]
    for a,n in sorted(votes.items(),key=lambda v:(-v[1],v[0])):
        if all(abs(a-b)>=.025 for b in chosen):chosen.append(a)
        if len(chosen)>=count:break
    return sorted(chosen) if count else []


def geometry_job(e,s,job,con,ctl):
    jid=p.job_id(job);folder=Path(s['output'])/jid
    if read(folder/'SLICES_DONE.json'):return
    selected=choose_geometry_points(con,jid,s['interior_geometry_points']);alphas=[0.]+selected+[1.]
    atomic_json(dict(alphas=alphas,selection='Most frequent per-entity sampled absolute-slope peak among A-valid/A-test; retrospective diagnostic selection, minimum separation 0.025.'),folder/'geometry_points.json')
    data,before,after,dirs=p.prepare(e,s,job,ctl);layers=s['layers'] or list(range(e.model.config.num_hidden_layers))
    try:
        for a,b in zip(alphas,alphas[1:]):
            slice_folder=folder/'slices'/f'{a:.12f}_{b:.12f}';scratch=slice_folder/'scratch';scratch.mkdir(parents=True,exist_ok=True)
            for split in s['geometry_splits']:
                if con.execute('SELECT 1 FROM slice_geometry WHERE job=? AND a=? AND b=? AND split=?',(jid,a,b,split)).fetchone():continue
                for index,alpha in enumerate([a,b]):
                    p.assign(e,before,after,alpha)
                    for row in data[split]:
                        original=folder/'full_token_scratch'/f'{alpha:.12f}'/f'{split}_{row["entity"]:03d}.npz'
                        if not original.exists():
                            ctl.pulse(job=jid,phase='interior all-token layer capture',alpha=alpha,split=split,entity=row['entity'])
                            _,arrays=p.probe(e,row,dirs,layers,True);ng.npz_save(original,arrays)
                        link=scratch/f'{index}_{split}_{row["entity"]:03d}.npz'
                        if not link.exists():os.link(original,link)
                g=p.geometry_split(e,s,job,split,data,slice_folder,ctl)
                g.update(reference_alpha=a,current_alpha=b)
                for site in g['sites'].values():site['reference']='Previous selected fraction of this actual update; scale fixed within this pair, not across pairs.'
                con.execute('INSERT INTO slice_geometry VALUES (?,?,?,?,?)',(jid,a,b,split,json.dumps(g,allow_nan=False)));con.commit()
                for f in scratch.glob(f'*_{split}_*.npz'):f.unlink()
        atomic_json(dict(done=True,completed=time.time()),folder/'SLICES_DONE.json')
        # Delete only reconstructible temporary tensors generated by this follow-up.
        if (folder/'full_token_scratch').exists():shutil.rmtree(folder/'full_token_scratch')
    finally:
        p.assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True);del before,after,dirs
        if torch.cuda.is_available():torch.cuda.empty_cache()


def report(output):
    out=Path(output);s=read(out/'settings.json')
    if not s or not (out/'measurements.sqlite').exists():return
    con=connect(out);lines=['ADAPTIVE NATURAL-PATH REFINEMENT — original results unchanged'];table=[]
    try:
        for job in s['jobs']:
            jid=p.job_id(job)
            for split in s['splits']:
                rows=[json.loads(r[0]) for r in con.execute('SELECT record FROM refined WHERE job=? AND split=? ORDER BY entity',(jid,split))]
                if not rows:continue
                integrated=[r for r in rows if r['integral'] is not None]
                row=dict(job=jid,split=split,n=len(rows),n_integrated=len(integrated),resolved=sum(r['resolved'] for r in rows),capped=sum(r['cap_reached'] for r in rows),
                    max_points=max(r['points'] for r in rows),mean_loss_change=float(np.mean([r['loss_change'] for r in rows])),
                    mean_closure_error=float(np.mean([r['closure_error'] for r in integrated])) if integrated else None,
                    max_abs_closure_error=max((abs(r['closure_error']) for r in integrated),default=None))
                table.append(row);lines.append(f'{jid} {split}: {row["resolved"]}/{len(rows)} finished prompt checks resolved; capped={row["capped"]}; max points={row["max_points"]}; mean closure={row["mean_closure_error"]}')
        lines += ['Counts cover completed prompt checks; compare n with the original entity count before treating a population as complete.',
            'Resolved requires local and global integration checks plus shrinking-width derivative audits. Caps/failures remain unresolved.',
            'Channel integrals are diagnostic contributions along a fixed actual-displacement path, not optimizer-reset effects.',
            'Intermediate geometry fractions are retrospectively chosen; no universal mechanism is assumed.']
        (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
        if table:
            import csv
            with (out/'refinement_audits.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
        plot_paths(out,s,con)
    finally:con.close()


def plot_paths(out,s,con):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(s['jobs']),2,figsize=(12,3*len(s['jobs'])),squeeze=False)
    for i,job in enumerate(s['jobs']):
        jid=p.job_id(job)
        for split,color in [('A_valid','#3779ae'),('A_test','#c24d39')]:
            # Use a single worst endpoint-change entity per prompt form, never average unmatched alpha samples.
            endpoints=p.get_rows(con,jid,1.,split);initial={r['entity']:r for r in p.get_rows(con,jid,0.,split)}
            if not endpoints:continue
            chosen=max(endpoints,key=lambda r:abs(r['loss']-initial[r['entity']]['loss']))['entity']
            points=[(a,json.loads(r)) for a,r in con.execute('SELECT alpha,record FROM probes WHERE job=? AND split=? AND entity=? ORDER BY alpha',(jid,split,chosen))]
            xx=[a for a,r in points];yy=[r['loss']-initial[chosen]['loss'] for a,r in points];dd=[r['projections']['delta'] for a,r in points]
            axes[i,0].plot(xx,yy,'.-',ms=2,color=color,label=f'{split}, entity {chosen}');axes[i,1].plot(xx,dd,'.-',ms=2,color=color,label=f'{split}, entity {chosen}')
        for ax in axes[i]:ax.set_title(jid);ax.set_xlabel('Fraction of actual displacement');ax.legend(fontsize=8);ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.5)
        axes[i,0].set_ylabel('Single-prompt loss change');axes[i,1].set_ylabel('Single-prompt directional derivative')
    fig.tight_layout();fig.savefig(out/'refined_paths.png',dpi=150);plt.close(fig)


def run(s):
    import fcntl
    out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        con=None
        try:
            c=validate(s);ctl=p.Control(s);ctl.pulse(phase='importing completed measurements');initialize(s);con=connect(out)
            ctl.pulse(phase='loading original model');e=p.Engine(c)
            for job in s['jobs']:refine_job(e,s,job,con,ctl);report(out);ctl.storage()
            for job in s['jobs']:geometry_job(e,s,job,con,ctl);report(out);ctl.storage()
            final='complete';message='Requested refinement finished. Check unresolved/capped prompt counts.'
        except ng.Pause as exc:final='paused';message=str(exc)
        except Exception:final='failed';message=traceback.format_exc();(out/'error.txt').write_text(message)
        finally:
            if con:con.close()
        try:report(out)
        except Exception:(out/'report_error.txt').write_text(traceback.format_exc())
        atomic_json(dict(status=final,message=message,last_progress=time.time()),out/'status.json')
        try:export(out)
        except Exception:(out/'export_error.txt').write_text(traceback.format_exc())


def export(output):
    dest=Path(p.export(output));new=dest.with_name('path_refinement_share.zip');os.replace(dest,new);return str(new)


def signature(s):return hashlib.sha256(json.dumps({k:v for k,v in s.items() if k not in ['hours','reserve_minutes','minimum_free_gb','max_output_gb','output']},sort_keys=True).encode()).hexdigest()


def launch(s):
    import fcntl
    validate(s);out=Path(s['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if ng.alive(out):print('Already running. Refresh status.');return str(out)
        old=read(out/'settings.json')
        if old and signature(old)!=signature(s):
            out=out.with_name(out.name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);out.mkdir();s['output']=str(out);print('Scientific settings changed; new output:',out)
        atomic_json(s,out/'settings.json');(out/'STOP').unlink(missing_ok=True)
        for f in Path(__file__).parent.glob('*.py'):
            q=out/'code'/f.name;q.parent.mkdir(exist_ok=True);shutil.copy2(f,q)
        atomic_json(dict(files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},previous=s['previous']),out/'provenance.json')
        atomic_json(dict(status='launching',last_progress=time.time()),out/'status.json')
        with (out/'worker.log').open('a') as log:proc=subprocess.Popen([sys.executable,str(out/'code/refine_paths.py'),'--worker',str(out/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            if ng.alive(out) or proc.poll() is not None:break
            time.sleep(.1)
        print('Started/resumed:',out);return str(out)


def status(output):
    out=Path(output);s=read(out/'settings.json',{});st=read(out/'status.json',{'status':'not_started'});st['alive']=ng.alive(out)
    st['seconds_since_progress']=round(max(0,time.time()-st.get('last_progress',time.time())),1)
    st['refined_jobs']=sum(bool(read(out/p.job_id(j)/'REFINED_DONE.json')) for j in s.get('jobs',[]));st['geometry_jobs']=sum(bool(read(out/p.job_id(j)/'SLICES_DONE.json')) for j in s.get('jobs',[]))
    return st

def refresh(output):st=status(output);print(json.dumps(st,indent=2));return st

def stop(output):p.stop(output)

def show(output):
    report(output);f=Path(output)/'REPORT.txt';print(f.read_text() if f.exists() else 'No completed refinements yet.')
    from IPython.display import display,Image
    f=Path(output)/'refined_paths.png'
    if f.exists():display(Image(filename=str(f)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker');a=ap.parse_args()
    if a.worker:run(read(a.worker))
    else:ap.print_help()
