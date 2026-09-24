"""Adaptive straight-path integrals, component/channel Hessians, multi-width audits."""
import numpy as np
from study_math import assign,evaluate
from components import component_gradients,projections,curvature,COMPONENTS,digest
from storage import get,put

def simpson(y):
    y=list(y);n=len(y)-1
    if n<2 or n%2:raise ValueError('Simpson needs an even number of intervals')
    return float((y[0]+y[-1]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2]))/(3*n))

def close(a,b,s):return bool(abs(a-b)<=s['audit_atol']+s['audit_rtol']*max(abs(a),abs(b)))

def path(e,before,after,dirs,rows,s,ctl,con,job,population):
    spec=dict(before=digest(before),after=digest(after),population=population,example_ids=[r['example_id'] for r in rows],dtype=str(next(e.model.parameters()).dtype))
    old=get(con,job,'path_spec')
    if old and old[0]!=spec:raise ArithmeticError('Cached path endpoint/population mismatch')
    put(con,job,0,'path_spec',spec)
    done=get(con,job,'path_done')
    if done:return done[0]
    cache={r['alpha']:r for r in get(con,job,'path_point')};hcache={r['alpha']:r for r in get(con,job,'hessian_point')}
    def move(a):
        # Endpoints assigned exactly; off-grid points rounded once into native parameters.
        if a==0:assign(e,before)
        elif a==1:assign(e,after)
        else:assign(e,{n:before[n].double()+a*dirs['d'][n] for n in before})
    def point(a):
        a=float(a)
        if a not in cache:
            ctl.check();ctl.pulse(phase='finite path exact component gradients',alpha=a,population=population)
            move(a);gs,loss=component_gradients(e,rows,s['derivative_batch'],ctl)
            rec=dict(alpha=a,loss=loss,geometry=projections(gs,dirs,with_layers=True));del gs
            cache[a]=rec;put(con,job,round(a*1e10),'path_point',rec)
        return cache[a]
    def hpoint(a):
        a=float(a)
        if a not in hcache:
            ctl.check();ctl.pulse(phase='history/current Hessian bilinear forms',alpha=a,population=population)
            move(a);rec=dict(alpha=a,components=curvature(e,rows,dirs,s['derivative_batch'],ctl));hcache[a]=rec
            put(con,job,round(a*1e10),'hessian_point',rec)
        return hcache[a]['components']
    try:
        n=s['path_initial_points'];levels=[]
        while True:
            grid=np.linspace(0,1,n);pts=[point(a) for a in grid];hs=[hpoint(a) for a in grid];results={}
            for k in COMPONENTS:
                observed=pts[-1]['loss'][k]-pts[0]['loss'][k];initial=pts[0]['geometry'][k]['slopes']['d'];rem=observed-initial
                integ={v:simpson(p['geometry'][k]['slopes'][v] for p in pts) for v in dirs}
                curv={v:simpson((1-a)*h[k][v] for a,h in zip(grid,hs)) for v in hs[0][k]}
                qs=abs(integ['d']-simpson(p['geometry'][k]['slopes']['d'] for p in pts[::2]))/15
                qh=abs(curv['dd']-simpson((1-a)*h[k]['dd'] for a,h in zip(grid[::2],hs[::2])))/15
                # Individual terms must converge too; cancellation cannot hide unresolved channels.
                termerrors={v:abs(curv[v]-simpson((1-a)*h[k][v] for a,h in zip(grid[::2],hs[::2])))/15 for v in curv}
                passed=close(integ['d'],integ['h']+integ['c']+integ['r'],s) and close(curv['direction_sum_residual'],0,s) and all(close(h[k]['hc'],h[k]['ch'],s) for h in hs) and close(integ['d'],observed,s) and close(curv['dd'],rem,s) and close(qs,0,s) and close(qh,0,s) and all(close(v,0,s) for v in termerrors.values())
                results[k]=dict(observed=observed,initial_projection=initial,remainder=rem,channel_integrals=integ,curvature_integrals=curv,slope_closure=integ['d']-observed,hessian_closure=curv['dd']-rem,quadrature_error_slope=qs,quadrature_error_hessian=qh,term_quadrature_errors=termerrors,integration_pass=passed)
            levels.append(dict(points=n,components=results))
            if all(v['integration_pass'] for v in results.values()) or n>=s['path_max_points']:break
            n=2*n-1
        # One location selected by maximal total slope change, plus midpoint: selection is documented.
        slopes=[p['geometry']['total']['slopes']['d'] for p in pts]
        ix=int(np.argmax(np.abs(np.diff(slopes))));locations=sorted(set([.5,float((grid[ix]+grid[ix+1])/2)]));audits=[]
        for a in locations:
            mid=point(a);hh=hpoint(a)
            for eps in s['audit_eps']:
                # d probes may extend beyond [0,1]; they are numerical derivative probes only.
                left,right=point(a-eps),point(a+eps)
                # Mixed derivative: change c-projection when moving in h, independent of Hd.
                mixed=[]
                for sign in [-1,1]:
                    assign(e,{n:before[n].double()+a*dirs['d'][n]+sign*eps*dirs['h'][n] for n in before})
                    gs,_=component_gradients(e,rows,s['derivative_batch'],ctl)
                    mixed.append(projections(gs,{'c':dirs['c']},with_layers=False));del gs
                for k in COMPONENTS:
                    slopefd=(right['loss'][k]-left['loss'][k])/(2*eps)
                    hfd=(right['geometry'][k]['slopes']['d']-left['geometry'][k]['slopes']['d'])/(2*eps)
                    crossfd=(mixed[1][k]['slopes']['c']-mixed[0][k]['slopes']['c'])/(2*eps)
                    audits.append(dict(alpha=a,epsilon=eps,component=k,slope_fd=slopefd,slope=mid['geometry'][k]['slopes']['d'],hessian_fd=hfd,hessian=hh[k]['dd'],mixed_fd=crossfd,mixed=hh[k]['hc'],slope_pass=close(slopefd,mid['geometry'][k]['slopes']['d'],s),hessian_pass=close(hfd,hh[k]['dd'],s),mixed_pass=close(crossfd,hh[k]['hc'],s),symmetry_pass=close(hh[k]['hc'],hh[k]['ch'],s)))
        for k,res in results.items():
            checks=[]
            for a in locations:
                rr=[r for r in audits if r['alpha']==a and r['component']==k][-2:]
                checks.append(all(r[z] for r in rr for z in ['slope_pass','hessian_pass','mixed_pass','symmetry_pass']) and all(close(rr[0][z],rr[1][z],s) for z in ['slope_fd','hessian_fd','mixed_fd']))
            res['derivative_pass']=all(checks);res['fully_audited']=res['integration_pass'] and res['derivative_pass']
            v=res['curvature_integrals'];res['mixed_contribution']=v['hc']+v['ch'];res['residual_contribution']=2*v['hr']+2*v['cr']+v['rr']
            res['mixed_material']=abs(res['mixed_contribution'])>=s['material_atol'] and abs(res['mixed_contribution'])>=s['material_fraction']*abs(res['observed'])
            res['classification']=classify(res,s)
        # Fixed first K examples only, not selected by damage; loss and local derivatives at 0,.5,1.
        for a in [0.,.5,1.]:
            kind='prompt_derivatives';step=round(a*1000)
            if any(r['alpha']==a for r in get(con,job,kind)):continue
            move(a);records=[]
            for row in rows[:s['prompt_derivative_count']]:
                gs,loss=component_gradients(e,[row],1,ctl);pr=projections(gs,dirs,with_layers=False);del gs
                records.append(dict(example_id=row['example_id'],class_id=row['class_id'],loss=loss,geometry=pr))
            put(con,job,step,kind,dict(alpha=a,rows=records,selection='fixed population order, first K'))
        result=dict(population=population,n_prompts=len(rows),components=results,levels=levels,derivative_audits=audits,fully_audited=all(v['fully_audited'] for v in results.values()),grid_points=n,capped=not all(v['integration_pass'] for v in results.values()),semantics='Native FP32 endpoint displacement, real-arithmetic autograd audited against finite differences. Component residual and mixed terms are path-dependent attributions, not independent causal interventions.')
        if not result['fully_audited'] and s.get('fp64_recheck_failed',False):
            ctl.pulse(phase='separate FP64 derivative recheck',population=population)
            result['floating_alternative']=floating_recheck(e,before,after,dirs,rows,s,ctl,locations[-1])
        put(con,job,0,'path_done',result);return result
    finally:assign(e,after)

def classify(r,s):
    if not r['fully_audited']:return 'unresolved'
    y,p,n=r['observed'],r['initial_projection'],r['remainder'];tol=s['material_atol']
    if y<=s['event_threshold']:return 'ordinary_or_nonlarge'
    if p < -tol:return 'favorable_to_harmful_reversal'
    if n>tol and n>=s['material_fraction']*abs(y):return 'nonlinear_amplification'
    if p>0 and abs(p)>=abs(n):return 'first_order_dominated'
    return 'other_large_increase'

def floating_recheck(e,before,after,dirs,rows,s,ctl,alpha):
    """Optional parameter-FP64 alternative, never relabels the native FP32 audit."""
    import torch
    original=next(e.model.parameters()).dtype
    try:
        e.model.to(dtype=torch.float64);e.params=dict(e.model.named_parameters())
        def p(a):
            assign(e,{n:before[n].double()+a*dirs['d'][n] for n in before})
            g,l=component_gradients(e,rows,s['derivative_batch'],ctl);v=projections(g,dirs,with_layers=False);del g
            return l,v
        loss,g=p(alpha);h=curvature(e,rows,dirs,s['derivative_batch'],ctl);checks=[]
        for eps in s['audit_eps']:
            left,lg=p(alpha-eps);right,rg=p(alpha+eps)
            for k in COMPONENTS:
                sf=(right[k]-left[k])/(2*eps);hf=(rg[k]['slopes']['d']-lg[k]['slopes']['d'])/(2*eps)
                checks.append(dict(epsilon=eps,component=k,slope_fd=sf,slope=g[k]['slopes']['d'],hessian_fd=hf,hessian=h[k]['dd'],passed=close(sf,g[k]['slopes']['d'],s) and close(hf,h[k]['dd'],s)))
        return dict(alpha=alpha,parameter_dtype='float64',component_loss=loss,curvature=h,checks=checks,semantics='Floating-point alternative at native endpoints. Architecture internals may retain FP32 operations. Does not validate or replace the native-path result; optimizer is not stepped.')
    finally:
        e.model.to(dtype=original);e.params=dict(e.model.named_parameters());assign(e,after)
