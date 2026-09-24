"""Fixed ridge logistic utilities shared by all follow-up feature sets.
C=.1; train-only scaling; no target-seed tuning. Fits are bundled, not run on targets.
"""
import numpy as np
PROTOCOL={"regularization_C":.1}

def fit(rows,cols):
    x=np.asarray([[r['x'][k] for k in cols] for r in rows]);
    if not np.isfinite(x).all():raise FloatingPointError('Nonfinite predictor feature')
    y=np.asarray([r['y'] for r in rows]);mu=x.mean(0);sd=x.std(0);sd[sd<1e-12]=1.
    if len(set(y))<2:return dict(columns=cols,mean=mu.tolist(),scale=sd.tolist(),constant=float(y.mean()))
    design=np.column_stack([(x-mu)/sd,np.ones(len(x))]);w=np.zeros(design.shape[1]);pen=np.full(len(w),1/PROTOCOL['regularization_C']);pen[-1]=0.
    objective=lambda w:float(np.logaddexp(0,np.einsum('ij,j->i',design,w)).sum()-np.sum(y*np.einsum('ij,j->i',design,w))+.5*np.sum(pen*w*w))
    converged=False
    for iteration in range(200):
        z=np.einsum('ij,j->i',design,w);prob=1/(1+np.exp(-np.clip(z,-700,700)));grad=np.einsum('ij,i->j',design,prob-y)+pen*w
        hess=np.einsum('ni,n,nj->ij',design,prob*(1-prob),design)+np.diag(pen+1e-10)
        delta=np.linalg.solve(hess,grad);step=1.;old=objective(w)
        for _ in range(30):
            if objective(w-step*delta)<=old:break
            step*=.5
        w-=step*delta
        if np.max(np.abs(grad))/len(rows)<1e-8:converged=True;break
    if not converged:raise ArithmeticError('Frozen logistic fit did not converge; refusing unvalidated predictions')
    return dict(columns=cols,mean=mu.tolist(),scale=sd.tolist(),coef=w[:-1].tolist(),intercept=float(w[-1]),iterations=iteration+1,
        normalized_gradient_tolerance=1e-8,objective='Sum binary log loss + 0.5/C times squared coefficients; intercept unpenalized. Damped Newton solver.')


def predict(model,rows):
    if 'constant' in model:return np.full(len(rows),model['constant'])
    x=np.asarray([[r['x'][k] for k in model['columns']] for r in rows]);z=np.einsum('ij,j->i',(x-np.asarray(model['mean']))/np.asarray(model['scale']),np.asarray(model['coef']))+model['intercept']
    return 1/(1+np.exp(-np.clip(z,-700,700)))

def scores(rows,prob):
    y=np.array([r['y'] for r in rows]);prob=np.asarray(prob);two=len(set(y))==2
    if two:
        pos=prob[y==1];neg=prob[y==0];auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))
        order=np.argsort(-prob,kind='stable');yy=y[order];pp=prob[order];tp=0;seen=0;ap=0.;i=0
        while i<len(y):
            j=i+1
            while j<len(y) and pp[j]==pp[i]:j+=1
            added=int(yy[i:j].sum());tp+=added;seen+=j-i;ap+=(added/y.sum())*(tp/seen);i=j
    else:auc=None;ap=None
    return dict(n=len(rows),positives=int(y.sum()),prevalence=float(y.mean()),auroc=auc,average_precision=float(ap) if ap is not None else None,brier=float(np.mean((prob-y)**2)))


