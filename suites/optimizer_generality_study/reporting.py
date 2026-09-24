"""All outcomes retained: calibration gates, numerical failures, and matched probes."""
import json,math
from pathlib import Path
import numpy as np
from storage import get,write

def report(s,con):
 out=Path(s['output']);summary=[];selection=[];probes=[];paths=[];targets=[];failures=[]
 jobs=[r[0] for r in con.execute("SELECT DISTINCT job FROM records WHERE kind='spec' ORDER BY job")]
 for job in jobs:
  spec=get(con,job,'spec')[0];anchors=get(con,job,'anchor');rows=get(con,job,'step');done=get(con,job,'done')
  r=dict(job=job,model=spec['model']['name'],optimizer=spec['optimizer'],lr=spec['lr'],seed=spec['seed'],role=spec['role'],variant=spec['variant'],complete=bool(done),updates=rows[-1]['step'] if rows else 0)
  if anchors:r['acquisition']=anchors[0]['acquisition']
  if anchors and rows:
   anchor=anchors[0]['evaluations'];r['populations']={}
   for sp in rows[-1]['post']:
    vals=[x['post'][sp]['mean'] for x in rows];base=anchor[sp]['mean'];last=rows[-1]['post'][sp]
    r['populations'][sp]=dict(initial=base,final=vals[-1],mean=float(np.mean(vals)),peak=max(vals),mean_positive_excess=float(np.mean(np.maximum(np.asarray(vals)-base,0))),restricted_accuracy=last['restricted_accuracy'],full_accuracy=last['full_accuracy'],confusion=last['confusion'],leakage=last['leakage'])
   if spec['role']=='confirmation':
    for threshold in [.75,1.,1.5,2.]:
     for sustained in [1,4]:
      run=0;hit=None
      for x in rows:
       run=run+1 if x['post']['B_valid']['mean']<=threshold else 0
       if run>=sustained:hit=dict(step=x['step'],A_test=x['post']['A_test']['mean'],B_test=x['post']['B_test']['mean']);break
      targets.append(dict(job=job,B_validation_target=threshold,consecutive_endpoints=sustained,hit=hit,acquisition_pass=r['acquisition']['passed']))
  summary.append(r)
  for t,raw in con.execute("SELECT step,record FROM records WHERE job=? AND kind='probe' ORDER BY step",(job,)):
   x=json.loads(raw)
   for mode,b in x['branches'].items():
    z=dict(job=job,step=t,condition=mode,state_audit=b['state_audit'],norm_audit=b['norm_audit'],history_tensors_reset=b['history_tensors_reset'])
    for sp in ['A_test','B_test','A_valid','B_valid']:
     z[sp]=dict(preserved_change=x['preserved_post'][sp]['mean']-x['pre'][sp]['mean'],counterfactual_change=b['post'][sp]['mean']-x['pre'][sp]['mean'],effect_vs_preserved=b['post'][sp]['mean']-x['preserved_post'][sp]['mean'])
    probes.append(z)
  for t,raw in con.execute("SELECT step,record FROM records WHERE job=? AND kind='path' ORDER BY step",(job,)):
   x=json.loads(raw);paths.append(dict(job=job,step=t,**{k:v for k,v in x.items() if k!='points'}))
 for job,kind,raw in con.execute("SELECT job,kind,record FROM records WHERE kind IN ('selection','failed','skipped') ORDER BY job"):
  x=dict(job=job,**json.loads(raw));(selection if kind=='selection' else failures).append(x)
 write(out/'summary.json',summary);write(out/'calibration_selection.json',selection);write(out/'history_probes.json',probes);write(out/'path_audits.json',paths);write(out/'acquisition_targets.json',targets);write(out/'failures_and_skips.json',failures)
 lines=['OPTIMIZER GENERALITY — validation-only calibration; native A history retained into B',f'{sum(x["complete"] for x in summary)} completed trajectories; {len(failures)} failures/skips recorded.',
 'Models start from pinned pretrained weights. A and B are both trained with the chosen optimizer. Different optimizers therefore have different A anchors; this is not a same-weight optimizer swap.',
 'All decay is zero. Same minibatch recipe and clipping. Learning rates are optimizer/model-specific, selected on disjoint development pools. Appendix matrix optimizers use auxiliary Adam on nonhidden parameters.',
 'Main optimizer ranking must consider acquisition gates, matched B-validation targets and all seeds. A lower loss increase with failed learning is not evidence of reduced forgetting.']
 for x in summary:
  if x['role']=='confirmation' and 'populations' in x:
   a=x['populations']['A_test'];b=x['populations']['B_test'];lines.append(f"{x['job']}: complete={x['complete']}; acquired={x['acquisition']['passed']}; A {a['initial']:.6f}->{a['final']:.6f}; B {b['final']:.6f}; lr={x['lr']}")
 for x in selection:lines.append(f"CALIBRATION {x['job']}: selected={x['lr']}; eligible={x['valid']}")
 lines += [f'Immediate history-probe condition records: {len(probes)}. First-step norm matching is audited; no continuation benefit is asserted.',f'Path audits: {len(paths)}; closure passed {sum(x["closure_pass"] for x in paths)}; selected-point Hessian passed {sum(x["hessian_pass"] for x in paths)}. Separate criteria, unresolved cases retained.',
 'Timing: see timing_existing/REPORT.txt and timing_new/REPORT.txt. Horizon0 uses only geometry of the previous update and the latest already-completed validation change.']
 (out/'REPORT.txt').write_text('\n'.join(lines))
 return summary

def plot(s,con):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(s['output']);rows=json.loads((out/'summary.json').read_text());rows=[x for x in rows if x['role']=='confirmation' and x['variant']=='standard' and 'populations' in x]
 models=sorted({r['model'] for r in rows})
 if not models:return
 fig,axes=plt.subplots(len(models),2,figsize=(12,4*len(models)),squeeze=False,layout='constrained')
 colors=dict(sgd='#195e83',momentum_sgd='#d48b22',adam_beta1_0='#7b4c94',adam='#3e8254',shampoo_block='#777777',muon_fp32='#be5353')
 for i,m in enumerate(models):
  for r in rows:
   if r['model']!=m:continue
   records=get(con,r['job'],'step');a0=r['populations']['A_test']['initial'];color=colors[r['optimizer']];label=f"{r['optimizer']} s{r['seed']}"+(' [acquisition failed]' if not r['acquisition']['passed'] else '')
   axes[i,0].plot([x['step'] for x in records],[x['post']['A_test']['mean']-a0 for x in records],color=color,alpha=.65,label=label)
   axes[i,1].plot([x['post']['B_test']['mean'] for x in records],[x['post']['A_test']['mean'] for x in records],color=color,alpha=.65)
  axes[i,0].set(title=m+' — old-task loss change',xlabel='B update',ylabel='A-test NLL minus A anchor');axes[i,0].legend(fontsize=7)
  axes[i,1].set(title=m+' — acquisition / retention trajectory',xlabel='B-test NLL',ylabel='A-test NLL')
 fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
