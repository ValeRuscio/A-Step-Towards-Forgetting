"""Paired effects, seed-level summaries, acquisition targets, and explicit audit limits."""
import json
from pathlib import Path
import numpy as np
from storage import get,write

def corr(v):
    return float(np.corrcoef(v[:-1],v[1:])[0,1]) if len(v)>2 and np.std(v[:-1])>0 and np.std(v[1:])>0 else None

def report(con,out,predictions=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);branches=[];paired=[];naturals=[];paths=[]
    for job,raw in con.execute("SELECT job,record FROM records WHERE kind='natural_done' ORDER BY job"):
        done=json.loads(raw);anchor=get(con,job,'anchor')[0];rows=get(con,job,'step')
        naturals.append(dict(job=job,acquisition=anchor['acquisition'],A_anchor=anchor['evaluations']['A_test']['mean'],A_final=done['final']['A_test']['mean'],B_final=done['final']['B_test']['mean'],
            A_valid_increment_lag1=corr([r['post']['A_valid']['mean']-r['pre']['A_valid']['mean'] for r in rows]),
            A_test_increment_lag1=corr([r['post']['A_test']['mean']-r['pre']['A_test']['mean'] for r in rows])))
    for job,raw in con.execute("SELECT job,record FROM records WHERE kind='branch_done' ORDER BY job"):
        spec=get(con,job,'branch_spec')[0];rows=get(con,job,'branch_step');first=rows[0];final=json.loads(raw)['final'];base=spec['pre'];parent=spec['parent'];gate=get(con,parent,'anchor')[0]['acquisition']
        item=dict(job=job,parent=parent,condition=spec['condition'],fork=spec['fork'],anchor_sha256=spec['anchor_sha256'],acquisition_passed=gate['passed'],n_steps=len(rows),state_audit=get(con,job,'state_audit')[0],first_norm_audit=first['norm_audit'],outcomes={})
        for domain in ['A','B']:
            sp=domain+'_test';loss=np.array([r['post'][sp]['mean'] for r in rows]);d=loss-base[sp]['mean']
            item['outcomes'][domain]=dict(pre=base[sp]['mean'],first=loss[0].item(),final=loss[-1].item(),final_change=d[-1].item(),mean_excess=float(d.mean()),mean_positive_excess=float(np.maximum(d,0).mean()),peak_excess=float(d.max()),
                full_accuracy=final[sp]['full_accuracy'],restricted_accuracy=final[sp]['restricted_accuracy'],confusion=final[sp]['confusion'],leakage=final[sp]['leakage'],
                increment_lag1=corr([r['post'][sp]['mean']-r['pre'][sp]['mean'] for r in rows]))
        item['B_valid_targets']=[]
        for target in [.5,.75,1.,1.5]:
            ids=[i for i in range(len(rows)-3) if all(r['post']['B_valid']['mean']<=target for r in rows[i:i+4])]
            # End of first four qualifying endpoints, not look-ahead selection of its start.
            k=ids[0]+3 if ids else None
            item['B_valid_targets'].append(dict(target=target,sustained_end_offset=k+1 if k is not None else None,A_test=rows[k]['post']['A_test']['mean'] if k is not None else None,B_test=rows[k]['post']['B_test']['mean'] if k is not None else None))
        branches.append(item)
    lookup={(x['parent'],x['fork'],x['condition']):x for x in branches}
    for r in branches:
        if r['condition']=='preserve':continue
        ref=lookup.get((r['parent'],r['fork'],'preserve'))
        if ref is None:continue
        if r['anchor_sha256']!=ref['anchor_sha256']:raise ArithmeticError('Paired branches do not share an anchor')
        row=dict(parent=r['parent'],fork=r['fork'],condition=r['condition'],acquisition_passed=r['acquisition_passed'],effects={})
        for domain in ['A','B']:
            row['effects'][domain]={k:r['outcomes'][domain][k]-ref['outcomes'][domain][k] for k in ['first','final','mean_excess','mean_positive_excess','peak_excess','full_accuracy','restricted_accuracy','confusion','leakage']}
        row['matched_B_targets']=[]
        for a,b in zip(r['B_valid_targets'],ref['B_valid_targets']):
            row['matched_B_targets'].append(dict(target=a['target'],both_reached=a['A_test'] is not None and b['A_test'] is not None,
                A_difference=a['A_test']-b['A_test'] if a['A_test'] is not None and b['A_test'] is not None else None,
                intervention_offset=a['sustained_end_offset'],preserve_offset=b['sustained_end_offset']))
        paired.append(row)
    for job,raw in con.execute("SELECT job,record FROM records WHERE kind='path_done' ORDER BY job"):paths.append(dict(job=job,**json.loads(raw)))
    path_lookup={p['job']:p for p in paths};path_effects=[]
    for p in paths:
        if '/preserve/' in p['job']:continue
        ref=path_lookup.get(p['job'].rsplit('/',2)[0]+'/preserve/path')
        if ref is None:continue
        path_effects.append(dict(job=p['job'],observed_difference=p['observed']-ref['observed'],initial_projection_difference=p['initial_projection']-ref['initial_projection'],remainder_difference=p['finite_remainder']-ref['finite_remainder'],
            weighted_hessian_integral_difference=p['weighted_hessian_integral']-ref['weighted_hessian_integral'],
            both_fully_audited=all(x[k] for x in [p,ref] for k in ['closure_pass','curvature_integral_pass','derivative_pass'])))
    seed_rows=[]
    for parent in sorted({p['parent'] for p in paired}):
        for mode in ['reset','reset_blocks']:
            rr=[p for p in paired if p['parent']==parent and p['condition']==mode]
            if not rr:continue
            seed_rows.append(dict(parent=parent,condition=mode,n_forks=len(rr),acquisition_passed=rr[0]['acquisition_passed'],
                mean_A_final_effect=float(np.mean([x['effects']['A']['final'] for x in rr])),mean_A_cumulative_effect=float(np.mean([x['effects']['A']['mean_excess'] for x in rr])),mean_B_final_effect=float(np.mean([x['effects']['B']['final'] for x in rr]))))
    for name,value in [('natural_summary',naturals),('branch_summary',branches),('paired_branch_effects',paired),('seed_level_branch_effects',seed_rows),('path_audits',paths),('paired_path_decomposition',path_effects)]:write(out/(name+'.json'),value)
    if predictions:
        from prediction import analyze
        analyze(con,out)
    text=['HISTORY, PERSISTENCE AND FINITE-UPDATE GEOMETRY',f'Completed natural trajectories: {len(naturals)}; branches: {len(branches)}; audited paths: {len(paths)}.',
        'Negative paired loss effects favor the intervention. One-time history intervention; subsequent updates are native. Fixed checkpoints, not test-selected events.',
        'Gate-failed runs are retained but excluded from primary acquired-task inference. Seeds are replication units; forks and prompts are dependent.']
    for r in naturals:text.append(f"{r['job']}: acquired={r['acquisition']['passed']}; A {r['A_anchor']:.6f} -> {r['A_final']:.6f}; B final {r['B_final']:.6f}")
    for r in paired:text.append(f"{r['parent']} fork{r['fork']} {r['condition']}: A immediate effect {r['effects']['A']['first']:+.6f}; A final effect {r['effects']['A']['final']:+.6f}; B final effect {r['effects']['B']['final']:+.6f}; acquired={r['acquisition_passed']}")
    for p in path_effects:text.append(f"{p['job']}: observed effect {p['observed_difference']:+.6f} = initial projection {p['initial_projection_difference']:+.6f} + finite remainder {p['remainder_difference']:+.6f}; independent Hessian effect {p['weighted_hessian_integral_difference']:+.6f}; audits={p['both_fully_audited']}")
    text.append('Prediction: preceding completed update only. Quiet/onset subset uses retrospective past-test labels for stratification, never as predictor inputs. Inspect per-seed combined-minus-loss differences; no significance claim from three seeds.')
    (out/'REPORT.txt').write_text('\n'.join(text)+'\n')
    if predictions and paired:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(11,4))
        for mode,color in [('reset','#b65b36'),('reset_blocks','#286f9b')]:
            rr=[p for p in paired if p['condition']==mode and p['acquisition_passed']]
            ax[0].scatter([r['effects']['B']['final'] for r in rr],[r['effects']['A']['final'] for r in rr],label=mode,color=color,alpha=.7)
            ss=[r for r in seed_rows if r['condition']==mode and r['acquisition_passed']]
            ax[1].scatter([r['mean_B_final_effect'] for r in ss],[r['mean_A_final_effect'] for r in ss],label=mode,color=color)
        for a,title in zip(ax,['Individual fixed forks (dependent)','Means within each natural trajectory']):
            a.axhline(0,color='grey',lw=.8);a.axvline(0,color='grey',lw=.8);a.set(title=title,xlabel='B final loss: intervention − preserve',ylabel='A final loss: intervention − preserve');a.legend()
        fig.tight_layout();fig.savefig(out/'overview.png',dpi=160);plt.close(fig)
        plotdir=out/'trajectory_plots';plotdir.mkdir(exist_ok=True)
        for parent in sorted({r['parent'] for r in branches}):
            forks=sorted({r['fork'] for r in branches if r['parent']==parent})
            fig,axes=plt.subplots(len(forks),2,figsize=(11,3*len(forks)),squeeze=False)
            for i,t in enumerate(forks):
                for mode in ['preserve','reset','reset_blocks']:
                    rr=get(con,f'{parent}/fork{t:03d}/{mode}','branch_step')
                    if not rr:continue
                    for j,domain in enumerate(['A','B']):
                        base=rr[0]['pre'][domain+'_test']['mean']
                        axes[i,j].plot([0]+[r['branch_offset'] for r in rr],[0]+[r['post'][domain+'_test']['mean']-base for r in rr],label=mode)
                for j,domain in enumerate(['A','B']):
                    axes[i,j].axhline(0,color='grey',lw=.7);axes[i,j].set(title=f'Fork before update {t}: task {domain}',xlabel='Branch updates (first includes intervention)',ylabel='Test NLL minus common pre-fork NLL');axes[i,j].legend(fontsize=8)
            fig.suptitle(parent);fig.tight_layout();fig.savefig(plotdir/(parent.replace('/','_')+'.png'),dpi=140);plt.close(fig)

