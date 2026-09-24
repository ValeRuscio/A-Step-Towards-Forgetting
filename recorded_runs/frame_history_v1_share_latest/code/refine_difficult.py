"""Targeted refinement in a separate directory; original results stay read-only."""
import json,sqlite3,zipfile,gc,math
from pathlib import Path
import numpy as np
import four_vector_study as f
p=f.p

JOBS=[dict(seed=1,step=21),dict(seed=2,step=11)]

def import_original(source,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);dest=out/'measurements.sqlite'
    if (out/'IMPORTED.json').exists():return
    source=Path(source)
    if source.is_dir():
        with sqlite3.connect(f'file:{source/"measurements.sqlite"}?mode=ro',uri=True) as a,sqlite3.connect(dest) as b:a.backup(b)
        read=lambda n:(source/n).read_bytes()
    else:
        with zipfile.ZipFile(source) as z:
            # Only explicitly named data files are imported; archive code is never executed.
            dest.write_bytes(z.read('measurements.sqlite'));blobs={n:z.read(n) for n in z.namelist() if n=='settings.json' or n.endswith(('/update.json','/curvature_points.json'))}
        read=lambda n:blobs[n]
    (out/'original_settings.json').write_bytes(read('settings.json'))
    (out/'settings.json').write_bytes(read('settings.json'))
    with sqlite3.connect(dest) as con:con.execute('DELETE FROM curvature');con.commit()
    for j in JOBS:
        folder=out/p.job_id(j);folder.mkdir(exist_ok=True)
        for name in ['update.json','curvature_points.json']:
            try:(folder/('original_'+name)).write_bytes(read(p.job_id(j)+'/'+name))
            except (KeyError,FileNotFoundError):pass
        if (folder/'original_update.json').exists():(folder/'update.json').write_bytes((folder/'original_update.json').read_bytes())
    f.atomic_json(dict(source=str(source.resolve())),out/'IMPORTED.json')


def settings(original,source,out,max_points=1025):
    s=dict(original,source=str(Path(source).resolve()),output=str(Path(out).resolve()),max_path_points=max_points,max_path_depth=16,
        diagnostic_jobs=JOBS,slope_fd_widths=[.01,.005,.0025,.00125,.000625,.0003125,.00015625])
    return s


def curvature_audit(e,s,j,con,ctl):
    jid=p.job_id(j);folder=Path(s['output'])/jid
    con.execute('CREATE TABLE IF NOT EXISTS curvature_refined(job TEXT,alpha REAL,split TEXT,record TEXT,PRIMARY KEY(job,alpha,split))');con.commit()
    saved=f.read(folder/'audit_sites.json')
    if saved:sites=saved['alphas']
    else:
        old=f.read(folder/'original_curvature_points.json',{}).get('alphas',[])
        pts=[(a,json.loads(r)) for a,r in con.execute('SELECT alpha,record FROM path WHERE job=? ORDER BY alpha',(jid,))]
        candidates=[(max(abs(b['functional'][sp]['slope']-a['functional'][sp]['slope'])/(y-x) for sp in s['gradient_splits']),(x+y)/2) for (x,a),(y,b) in zip(pts,pts[1:])]
        sites=list(old[:1])
        for _,alpha in sorted(candidates,reverse=True):
            if all(abs(alpha-x)>1e-5 for x in sites):sites.append(alpha)
            if len(sites)>=2:break
        f.atomic_json(dict(alphas=sites,selection='Original failed-audit site, plus largest refined sampled directional-derivative secant. Retrospective.'),folder/'audit_sites.json')
    data,before,after,dirs=p.prepare(e,s,j,ctl)
    try:
        for alpha in sites:
            for split in s['gradient_splits']:
                old=con.execute('SELECT record FROM curvature_refined WHERE job=? AND alpha=? AND split=?',(jid,alpha,split)).fetchone()
                r=json.loads(old[0]) if old else None
                if r and r.get('finished'):continue
                p.assign(e,before,after,alpha)
                if r is None:
                    r=f.curvature_measure(e,data[split],dirs,s['curvature_batch'],ctl,jid,alpha,split)
                    r['mixed_symmetry_error']=abs(r['hHc']-r['cHh']);r['mixed_symmetry_passed']=r['mixed_symmetry_error']<=s['curvature_symmetry_atol']+s['curvature_symmetry_rtol']*max(abs(r['hHc']),abs(r['cHh']))
                    r['fd_attempts']=[]
                def commit():
                    con.execute('INSERT OR REPLACE INTO curvature_refined VALUES (?,?,?,?)',(jid,alpha,split,json.dumps(r,allow_nan=False)));con.commit()
                commit()
                for width in s['slope_fd_widths']:
                    if any(x['width']==width for x in r['fd_attempts']):continue
                    a=max(0.,alpha-width);b=min(1.,alpha+width);slopes=[]
                    for x in [a,b]:
                        ctl.pulse(phase='shrinking-width curvature audit',job=jid,alpha=alpha,split=split,width=width)
                        p.assign(e,before,after,x);g,_=f.population_gradient(e,data[split],s['gradient_batch'],ctl)
                        slopes.append(sum(f.ng.dot(g[n],after[n]-before[n]) for n in g));del g
                    fd=(slopes[1]-slopes[0])/(b-a);tol=s['curvature_symmetry_atol']+s['fd_rtol']*max(abs(fd),abs(r['dHd']))
                    prev=r['fd_attempts'][-1] if r['fd_attempts'] else None
                    item=dict(width=width,estimate=fd,autograd=r['dHd'],error=abs(fd-r['dHd']),tolerance=tol,agreement_passed=abs(fd-r['dHd'])<=tol)
                    item['adjacent_stable']=bool(prev and prev['agreement_passed'] and item['agreement_passed'] and abs(fd-prev['estimate'])<=max(tol,prev['tolerance']))
                    r['fd_attempts'].append(item);commit()
                    if item['adjacent_stable']:break
                r['fd_passed']=any(x['adjacent_stable'] for x in r['fd_attempts']);r['resolved']=r['fd_passed'] and r['mixed_symmetry_passed'];r['finished']=True
                r['note']='All widths use original FP32 computation. A failed audit remains unresolved; narrowing width can amplify roundoff. No automatic precision/model change.';commit()
        f.atomic_json(dict(done=True),folder/'AUDITS_FINISHED.json')
    finally:p.assign(e,before,after,1.);e.opt.zero_grad(set_to_none=True);del before,after,dirs;gc.collect()


def run(original_input,source,out,ctl,max_points=1025):
    import_original(original_input,out);out=Path(out)
    s=settings(f.read(out/'original_settings.json'),source,out,max_points);s.update(device=ctl.device,threads=ctl.threads,minimum_free_gb=ctl.minimum_free_gb)
    f.validate(s);f.atomic_json(s,out/'settings.json')
    if all((out/p.job_id(j)/'AUDITS_FINISHED.json').exists() for j in JOBS):return
    con=f.connect(out);e=p.Engine(f.validate(s))
    try:
        # Preserve cached positions. Each path is refined to the larger point/depth budget.
        for j in JOBS:f.path_event(e,s,j,con,ctl);ctl.storage()
        for j in JOBS:curvature_audit(e,s,j,con,ctl);ctl.storage()
    finally:
        con.close();del e;gc.collect()
        if f.torch.cuda.is_available():f.torch.cuda.empty_cache()
        f.report(out)
        with sqlite3.connect(out/'measurements.sqlite') as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='curvature_refined'").fetchone():
                lines=['\nREFINED CURVATURE AUDITS (original single-width results retained in the original archive)']
                for jid,a,sp,raw in db.execute('SELECT * FROM curvature_refined'):
                    r=json.loads(raw);lines.append(f'{jid} {sp} alpha={a:.8f}: resolved={r.get("resolved",False)}, dHd={r["dHd"]:.6g}, widths={len(r["fd_attempts"])}')
                with (out/'REPORT.txt').open('a') as file:file.write('\n'.join(lines)+'\n')
