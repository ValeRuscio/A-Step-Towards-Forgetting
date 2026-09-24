import itertools
import numpy as np

def points_distance(x):
    x=np.asarray(x,dtype=np.float64)
    sq=(x*x).sum(1);d=np.sqrt(np.maximum(sq[:,None]+sq[None,:]-2*x@x.T,0))
    np.fill_diagonal(d,0);return (d+d.T)/2

def persistent_homology(d,thresholds):
    """Exact H0/H1 of a finite flag filtration over F2, including triangles.
    Works for symmetric dissimilarities too, but then it is not a metric-space claim.
    Pure Python fallback avoids extra dependencies; cap at 32 landmarks.
    """
    d=np.asarray(d,dtype=np.float64);n=len(d)
    if d.shape!=(n,n) or not np.isfinite(d).all() or np.min(d)<-1e-10 or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0):
        raise ValueError('Topology input must be finite, nonnegative, symmetric, zero diagonal')
    if n>32:raise ValueError('Built-in H1 supports at most 32 fixed landmarks; use fewer points')
    simplices=[(0.,(i,)) for i in range(n)]
    simplices += [(float(d[i,j]),(i,j)) for i,j in itertools.combinations(range(n),2)]
    simplices += [(max(float(d[i,j]),float(d[i,k]),float(d[j,k])),(i,j,k)) for i,j,k in itertools.combinations(range(n),3)]
    simplices.sort(key=lambda z:(z[0],len(z[1]),z[1]))
    ids={s:i for i,(_,s) in enumerate(simplices)};pivots={};births=set();pairs={}
    for j,(value,s) in enumerate(simplices):
        col=set() if len(s)==1 else {ids[s[:i]+s[i+1:]] for i in range(len(s))}
        while col and max(col) in pivots:col ^= pivots[max(col)]
        if col:
            low=max(col);pivots[low]=col;pairs[low]=j
        else:births.add(j)
    out={}
    for dim in [0,1]:
        intervals=[]
        for b in sorted(births):
            if len(simplices[b][1])-1!=dim:continue
            death=simplices[pairs[b]][0] if b in pairs else None
            if death is not None and death<=simplices[b][0]+1e-12:continue
            intervals.append([simplices[b][0],death])
        finite=[death-birth for birth,death in intervals if death is not None]
        out['H'+str(dim)]=dict(intervals=intervals,finite_total_persistence=float(sum(finite)),
            finite_max_persistence=float(max(finite,default=0)),
            betti=[sum(b<=t and (end is None or t<end) for b,end in intervals) for t in thresholds])
    out['thresholds']=list(thresholds);out['n_points']=n
    return out

def entropy_rank(e):
    e=np.maximum(np.asarray(e,dtype=float),0);total=e.sum()
    if total<=1e-30:return 0.
    p=e[e>0]/total;return float(np.exp(-np.sum(p*np.log(p))))
