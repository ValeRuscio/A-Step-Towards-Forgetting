"""Completed and incomplete denominators stay separate; no prompt-level p-values."""
import json,statistics,collections
from pathlib import Path
from storage import get,write

def mean(x):return statistics.mean(x) if x else None

def report(con,out,s):
    out=Path(out);jobs=[r[0] for r in con.execute("SELECT DISTINCT job FROM records WHERE kind='spec' ORDER BY job")];natural=[];events=[];paths=[];uniform=[];paired=[]
    for job in jobs:
        spec=get(con,job,'spec')[0];anchor=get(con,job,'anchor');steps=get(con,job,'step');done=bool(get(con,job,'natural_done'));sel=get(con,job,'selection');passed=anchor[0]['acquisition']['passed'] if anchor else None
        row=dict(job=job,model=spec['model']['name'],seed=spec['seed'],condition=spec['variant'],acquired=passed,natural_complete=done,case_complete=bool(get(con,job,'case_done')),updates=len(steps),anchor=anchor[0] if anchor else None,final=steps[-1]['post'] if steps else None)
        natural.append(row)
        for r in steps:
            for sp in ['A_valid','A_test','B_valid','B_test']:
                pre={x['entity']:x for x in r['pre'][sp]['rows']};changes=[x['loss']-pre[x['entity']]['loss'] for x in r['post'][sp]['rows']]
                event=dict(job=job,step=r['step'],population=sp,acquired=passed,natural_complete=done,n_examples=len(changes),change=r['post'][sp]['mean']-r['pre'][sp]['mean'],confusion_change=r['post'][sp]['confusion']-r['pre'][sp]['confusion'],leakage_change=r['post'][sp]['leakage']-r['pre'][sp]['leakage'],harmed=sum(x>0 for x in changes),large_harmed=sum(x>s['event_threshold'] for x in changes),median_change=statistics.median(changes),mean_positive_change=mean([max(x,0) for x in changes]),mean_negative_change=mean([min(x,0) for x in changes]),full_accuracy_change=r['post'][sp]['full_accuracy']-r['pre'][sp]['full_accuracy'],restricted_accuracy_change=r['post'][sp]['restricted_accuracy']-r['pre'][sp]['restricted_accuracy'])
                event['component_closure']=event['change']-event['confusion_change']-event['leakage_change']
                if sp=='A_valid':event['geometry']=r['geometry']
                events.append(event)
        for t in sel[0]['selected'] if sel else []:
            for sp in ['A_valid','A_test']:
                pj=f"{job}/update{t['step']:03d}/{sp}";r=get(con,pj,'path_done')
                if r:paths.append(dict(job=job,step=t['step'],reasons=t['reasons'],acquired=passed,**r[0]))
        if sel:
            for sp in ['A_valid','A_test']:
                selected=sel[0]['uniform'];rr=[r for r in paths if r['job']==job and r['population']==sp and 'uniform' in r['reasons']]
                for comp in ['total','confusion','leakage']:
                    good=[r['components'][comp] for r in rr if r['components'][comp]['fully_audited']]
                    counts=dict(collections.Counter(x['classification'] for x in good))
                    uniform.append(dict(job=job,population=sp,component=comp,acquired=passed,planned=len(selected),completed=len(rr),audited=len(good),unresolved_completed=len(rr)-len(good),class_counts=counts,mixed_material_count=sum(x['mixed_material'] for x in good),denominator_note='Uniform sample only; report audited/planned and missing/unresolved explicitly. No enriched events in prevalence.'))
    lookup={(r['model'],r['seed'],r['condition']):r for r in natural}
    for r in natural:
        if r['condition']!='counterbalanced_shared':continue
        q=lookup.get((r['model'],r['seed'],'counterbalanced_disjoint'))
        if not q or not r['natural_complete'] or not q['natural_complete']:continue
        da=r['anchor'];db=q['anchor'];a_same=da['weight_hash']==db['weight_hash']
        specs=[get(con,x['job'],'spec')[0] for x in [r,q]]
        data=[]
        for x in [r,q]:data.append(json.loads((Path(s['output'])/'data'/(x['job'].replace('/','_')+'.json')).read_text())['data'])
        matched=all([(z['example_id'],z['class_id']) for z in data[0][sp]]==[(z['example_id'],z['class_id']) for z in data[1][sp]] for sp in data[0])
        effects={k:q['final']['A_test'][k]-db['evaluations']['A_test'][k]-(r['final']['A_test'][k]-da['evaluations']['A_test'][k]) for k in ['mean','confusion','leakage','full_accuracy','restricted_accuracy']}
        paired.append(dict(model=r['model'],seed=r['seed'],both_acquired=r['acquired'] and q['acquired'],identical_A_weights=a_same,identical_example_selection=matched,disjoint_minus_shared_A_change=effects,interpretation='Paired answer-codebook manipulation; B codebook/instruction and output-token identities change. Not a pure parameter intervention.'))
    limitations=[]
    if any(r['acquired'] is False for r in natural):limitations.append('Acquisition-failed trajectories retained, excluded from primary interpretation.')
    if any(not r['fully_audited'] for r in paths):limitations.append('Some finite paths have unresolved numerical audits.')
    if any(not r['identical_A_weights'] or not r['identical_example_selection'] for r in paired):limitations.append('Paired answer-condition matching failed; inspect pair records.')
    summary=dict(completed_natural=sum(r['natural_complete'] for r in natural),completed_cases=sum(r['case_complete'] for r in natural),updates=len([r for r in events if r['population']=='A_valid']),completed_paths=len(paths),fully_audited_paths=sum(r['fully_audited'] for r in paths),acquired_trajectories=sum(r['acquired'] is True for r in natural),limitations=limitations)
    for n,v in [('summary',summary),('natural_summary',natural),('event_components',events),('path_summary',paths),('uniform_prevalence',uniform),('answer_condition_pairs',paired)]:write(out/(n+'.json'),v)
    lines=['NATURAL FORGETTING COMPONENTS — observational paths, native Adam',json.dumps(summary),'Paths use fixed A-valid/A-test subpopulations. Primary acquired-task inference excludes failed gates.','Uniform samples estimate prevalence; enriched validation-selected events do not. Mixed terms are path-dependent attributions.']
    for r in natural:
        if not r['final']:continue
        a=r['anchor']['evaluations']['A_test'];b=r['final']['A_test']
        lines.append(f"{r['job']}: acquired={r['acquired']}; steps={r['updates']}; A change={b['mean']-a['mean']:+.6f}, confusion={b['confusion']-a['confusion']:+.6f}, leakage={b['leakage']-a['leakage']:+.6f}")
    for r in paths:
        z=r['components']['total'];lines.append(f"{r['job']} step{r['step']} {r['population']} {r['reasons']}: change={z['observed']:+.6f}, linear={z['initial_projection']:+.6f}, remainder={z['remainder']:+.6f}, mixed={z['mixed_contribution']:+.6f}, residual={z['residual_contribution']:+.6g}; audited={z['fully_audited']}; {z['classification']}")
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    if paths:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for sp,color in [('A_valid','#4477aa'),('A_test','#cc6677')]:
            rr=[r['components']['total'] for r in paths if r['population']==sp and r['acquired'] and r['fully_audited']]
            axes[0].scatter([r['initial_projection'] for r in rr],[r['observed'] for r in rr],label=sp,color=color,s=20)
            axes[1].scatter([r['mixed_contribution'] for r in rr],[r['remainder'] for r in rr],label=sp,color=color,s=20)
        for ax in axes:ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6);ax.legend()
        axes[0].set(xlabel='Initial projection',ylabel='Observed loss change',title='Audited acquired-task paths')
        axes[1].set(xlabel='Integrated mixed history/current term',ylabel='Finite remainder',title='Uniform and enriched samples shown separately in JSON')
        fig.tight_layout();fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
