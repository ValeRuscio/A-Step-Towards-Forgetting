"""Resumable scalar path records; actual FP32 displacement, exact autograd derivatives.
No interpolation of optimizer states. Model weights move on a straight segment.
"""
import numpy as np
from study_math import assign,difference,grad,dot,evaluate,directional_hessian
from storage import get,put

def simpson(y):
    y=list(y);n=len(y)-1
    if n<2 or n%2:raise ValueError('Even number of intervals required')
    return float((y[0]+y[-1]+4*sum(y[1:-1:2])+2*sum(y[2:-1:2]))/(3*n))
def close(a,b,s):return abs(a-b)<=s['audit_atol']+s['audit_rtol']*max(abs(a),abs(b))

def measure(e,before,after,rows,s,ctl,con,job,ch=None):
    done=get(con,job,'path_done')
    if done:return done[0]
    delta=difference(after,before)
    assign(e,after);actual_endpoint_loss=evaluate(e,rows,s['eval_batch'],ctl)['mean']
    cache={r['alpha']:r for r in get(con,job,'path_point')}
    def point(a):
        a=float(a)
        if a not in cache:
            ctl.check();ctl.pulse(phase='path loss and exact slope',alpha=a,path_job=job)
            assign(e,{n:v+a*delta[n] for n,v in before.items()});g=grad(e,rows,s['eval_batch'],ctl)
            r=dict(alpha=a,loss=evaluate(e,rows,s['eval_batch'],ctl)['mean'],slope=dot(g,delta))
            if ch is not None:r.update(history_slope=dot(g,ch[0]),current_slope=dot(g,ch[1]))
            cache[a]=r;put(con,job,round(a*10000000),'path_point',r)
        return cache[a]
    try:
        n=s['path_initial_points'];levels=[]
        while True:
            pts=[point(a) for a in np.linspace(0,1,n)];observed=pts[-1]['loss']-pts[0]['loss'];initial=pts[0]['slope']
            integral=simpson(p['slope'] for p in pts);coarse=simpson(p['slope'] for p in pts[::2]);qerr=abs(integral-coarse)/15
            tol=s['audit_atol']+s['audit_rtol']*abs(observed)
            passed=abs(integral-observed)<=tol and qerr<=tol
            levels.append(dict(points=n,closure=integral-observed,error_estimate=qerr,passed=passed))
            if passed or n>=s['path_max_points']:break
            n=2*n-1
        slope_remainder=simpson(p['slope']-initial for p in pts)
        # Independently integrate directional Hessian; do not relabel a loss residual as curvature.
        hcache={r['alpha']:r for r in get(con,job,'hessian_point')}
        def curvature(a):
            a=float(a)
            if a not in hcache:
                ctl.check();ctl.pulse(phase='exact directional Hessian',alpha=a,path_job=job)
                assign(e,{n:v+a*delta[n] for n,v in before.items()})
                h=directional_hessian(e,rows,delta,s['eval_batch'],ctl)
                hcache[a]=dict(alpha=a,hessian=h);put(con,job,round(a*10000000),'hessian_point',hcache[a])
            return hcache[a]['hessian']
        hn=s['hessian_initial_points'];hlevels=[];remainder=observed-initial
        while True:
            grid=np.linspace(0,1,hn);weighted=[(1-a)*curvature(a) for a in grid]
            hi=simpson(weighted);he=abs(hi-simpson(weighted[::2]))/15
            htol=s['audit_atol']+s['audit_rtol']*abs(remainder)
            hp=abs(hi-remainder)<=htol and he<=htol
            hlevels.append(dict(points=hn,weighted_integral=hi,remainder_closure=hi-remainder,error_estimate=he,passed=hp))
            if hp or hn>=s['hessian_max_points']:break
            hn=2*hn-1
        # Audit where the path slope changes most, not wherever finite differences look easiest.
        k=int(np.argmax(np.abs(np.diff([p['slope'] for p in pts]))));a=(pts[k]['alpha']+pts[k+1]['alpha'])/2
        h=curvature(a);checks=[]
        for eps in s['audit_eps']:
            left,mid,right=point(a-eps),point(a),point(a+eps)
            fd=(right['loss']-left['loss'])/(2*eps);hfd=(right['slope']-left['slope'])/(2*eps)
            checks.append(dict(epsilon=eps,slope=mid['slope'],slope_fd=fd,slope_pass=close(fd,mid['slope'],s),hessian=h,hessian_fd=hfd,hessian_pass=close(hfd,h,s)))
        stable=close(checks[-1]['hessian_fd'],checks[-2]['hessian_fd'],s)
        result=dict(population='A_valid',n_prompts=len(rows),actual_endpoint_loss=actual_endpoint_loss,interpolated_endpoint_loss=pts[-1]['loss'],endpoint_roundoff_loss=pts[-1]['loss']-actual_endpoint_loss,observed=observed,initial_projection=initial,
            finite_remainder=remainder,slope_integral=integral,slope_remainder_integral=slope_remainder,
            weighted_hessian_integral=hi,path_levels=levels,hessian_levels=hlevels,
            closure_pass=passed and close(pts[-1]['loss'],actual_endpoint_loss,s),curvature_integral_pass=hp,derivative_pass=stable and all(c['slope_pass'] and c['hessian_pass'] for c in checks[-2:]),
            audit_alpha=a,derivative_checks=checks,initial_favorable_endpoint_harmful=initial<0 and observed>0,
            capped=not passed or not hp,semantics='All terms use the same full A_valid population. Independent weighted-Hessian quadrature; unresolved audits are retained, not evidence of an exact decomposition.')
        if ch is not None:result.update(history_integral=simpson(p['history_slope'] for p in pts),current_integral=simpson(p['current_slope'] for p in pts),channel_note='First-step block matching scales both reset channels by each block factor. FP32 assigned displacement can retain a small sum residual.')
        put(con,job,0,'path_done',result);return result
    finally:assign(e,after)
