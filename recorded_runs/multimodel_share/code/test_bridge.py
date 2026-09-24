"""Numerical tests plus a tiny original-OLMo end-to-end test (no downloads)."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import mechanism_bridge as b


def numerical_tests():
    rng=np.random.default_rng(10);x=rng.normal(size=(32,6));q,_=np.linalg.qr(rng.normal(size=(6,6)))
    if np.linalg.det(q)<0:q[:,-1]*=-1
    y=x@q;f=b.fit_frame(x[:20],y[:20],6);z=b.rotate(y[20:],f['basis'],f['rotation'])
    np.testing.assert_allclose(z,x[20:],atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(z,axis=1),np.linalg.norm(y[20:],axis=1),atol=1e-10)
    p0=rng.random((2,4,4));p1=rng.random((2,4,4));v0=rng.normal(size=(2,4,3));v1=rng.normal(size=(2,4,3))
    np.testing.assert_allclose((p1-p0)@v0+p0@(v1-v0)+(p1-p0)@(v1-v0),p1@v1-p0@v0)
    result=b.cloud_stats(x[:8],x[:8],np.eye(6),2,21)
    rotated=b.cloud_stats(x[:8]@q,x[:8],np.eye(6),0,21)
    np.testing.assert_allclose(result['PH']['H1']['finite_total_persistence'],rotated['PH']['H1']['finite_total_persistence'],atol=1e-10)
    print('Numerical rotation, channel decomposition and PH invariance tests passed.')


def smoke(root):
    from bridge_replay import create_smoke_source
    root=Path(root);root.mkdir(parents=True,exist_ok=True);src=root/'source'
    if not src.exists():create_smoke_source(src)
    s=b.defaults(src,root/'results');s.update(device='cpu',threads=1,seeds=[1],events=[8],fractions=[.625],
        hours=1,minimum_free_gb=0,patch_layers=[0],capture_layers=[0],fit_max_rank=8,fit_positions=2,
        random_repeats=1,topology_nulls=1,branch_steps=2,branch_event=8)
    b.validate(s);out=Path(s['output']);out.mkdir(exist_ok=True)
    b.atomic_json(s,out/'settings.json');b.run(s)
    state=b.read(out/'status.json')
    assert state['status']=='complete',state
    interactions=b.read(out/'interaction_summary.json');assert interactions
    assert len(list(out.glob('*/DONE.json')))==5
    for f in out.glob('factorial_*/[01][01]/layer00_sham_0/summary.json'):
        assert max(abs(r['loss_change']) for r in b.read(f)['rows'])<s['sham_atol']
    for f in out.glob('branch_*/curve.json'):
        r=b.read(f);assert r['complete'] and len(r['rows'])==2
    for f in out.glob('factorial_*/[01][01]/layer00_rotation_0/summary.json'):
        rr=[r for r in b.read(f)['rows'] if 'margin_fd_pass' in r]
        assert rr and all(r['margin_fd_pass'] for r in rr),rr
    assert (out/'results_share.zip').exists()
    # Resume must leave completed experimental units untouched.
    markers={str(f):f.stat().st_mtime_ns for f in out.glob('*/DONE.json')};b.run(s)
    assert markers=={str(f):f.stat().st_mtime_ns for f in out.glob('*/DONE.json')}
    print('Tiny OLMo full factorial, patches, geometry, optimizer directions, continuations and resume passed:',out)
    return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke-output');a=p.parse_args();numerical_tests()
    if a.smoke_output:smoke(a.smoke_output)
