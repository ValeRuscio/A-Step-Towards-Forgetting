"""Per-trajectory descriptive summaries. Signed and absolute contributions are separate."""
import json,statistics,collections
from pathlib import Path
from storage import get,write

def report(con,out,s):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);jobs=[r[0] for r in con.execute("SELECT job FROM records WHERE kind='spec' ORDER BY job")];totals=[];age_totals=[];audits=[];events=[];timeline=[]
    for job in jobs:
        spec=get(con,job,'spec')[0];rr=get(con,job,'update');done=bool(get(con,job,'done'))
        audits.append(dict(job=job,acquired=spec['acquired'],complete=done,updates=len(rr),all_hashes_pass=all(r['replay_audit']['passed'] for r in rr),all_algebra_pass=all(r['algebra_audit']['passed'] for r in rr),age_audits_pass=sum(r['age_audit']['passed'] for r in rr),max_relative_moment_residual=max((r['age_audit']['relative_moment_residual'] for r in rr),default=None)))
        groups=collections.defaultdict(list);ag=collections.defaultdict(list)
        for r in rr:
            for v in r['geometry']:
                groups[(v['population'],v['component'],v['channel'],v['layer'])].append(v)
                if v['layer']=='GLOBAL':timeline.append(dict(job=job,step=r['step'],acquired=spec['acquired'],complete=done,outcome=r['outcomes'][v['population']]['change'][v['component']],**v))
            for v in r['age_contributions']:ag[(v['population'],v['component'],v['layer'])].append(v['projections'])
        for (pop,comp,ch,layer),values in groups.items():
            fields=['projection','gradient_change','channel_change','numerator_change','adaptive_scaling_change','clock_change','roundoff_change','orientation_effect','gradient_magnitude_effect','channel_magnitude_effect']
            totals.append(dict(job=job,population=pop,component=comp,channel=ch,layer=layer,acquired=spec['acquired'],complete=done,n=len(values),signed_sums={k:sum(v[k] for v in values if v.get(k) is not None) for k in fields},absolute_sums={k:sum(abs(v[k]) for v in values if v.get(k) is not None) for k in fields},available={k:sum(v.get(k) is not None for v in values) for k in fields},mean_cosine=statistics.mean(v['cosine'] for v in values if v['cosine'] is not None) if any(v['cosine'] is not None for v in values) else None))
        for (pop,comp,layer),values in ag.items():age_totals.append(dict(job=job,population=pop,component=comp,layer=layer,acquired=spec['acquired'],complete=done,n=len(values),signed_sums={k:sum(v[k] for v in values) for k in values[0]},absolute_sums={k:sum(abs(v[k]) for v in values) for k in values[0]}))
        # Event windows selected by validation total change only. Mark overlaps, do not invent independent samples.
        selected=[r['step'] for r in rr if r['source_full_population_changes']['A_valid']['total']>s['event_threshold']]
        by={r['step']:r for r in rr}
        for t in selected:
            for lag in range(-s['event_window'],s['event_window']+1):
                r=by.get(t+lag)
                if r is None:continue
                events.append(dict(job=job,event_step=t,lag=lag,measured_step=t+lag,acquired=spec['acquired'],complete=done,validation_selected=True,geometry=[v for v in r['geometry'] if v['layer']=='GLOBAL'],outcomes=r['outcomes']))
    summary=dict(trajectories=len(jobs),completed=sum(a['complete'] for a in audits),updates=sum(a['updates'] for a in audits),replay_hashes_pass=all(a['all_hashes_pass'] for a in audits),algebra_pass=all(a['all_algebra_pass'] for a in audits),age_audits_pass=sum(a['age_audits_pass'] for a in audits),notes=['Cross-time sums telescope; use absolute contributions and signed event windows to describe dynamics.','Age bins describe gradient contributions under the current denominator, not causal deletion effects.','Events selected on validation; held-out derivatives use their own fixed population. Overlapping events are dependent.'])
    for name,value in [('summary',summary),('replay_audits',audits),('geometry_summary',totals),('age_summary',age_totals),('global_timeline',timeline),('event_windows',events)]:write(out/(name+'.json'),value)
    lines=['HISTORY DYNAMICS — EXACT SOURCE REPLAY',json.dumps(summary),'Age-bin and adaptive-scaling analysis: descriptive channel attribution, not continued-training interventions.']
    for r in age_totals:
        if r['layer']=='GLOBAL':lines.append(f"{r['job']} {r['population']} {r['component']}: acquired={r['acquired']}, complete={r['complete']}; history-age signed sums={r['signed_sums']}")
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    if timeline:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        # Last available case, two behavioral components, clear populations.
        job=jobs[-1];fig,axes=plt.subplots(2,2,figsize=(12,7))
        for i,comp in enumerate(['confusion','leakage']):
            for pop,style in [('A_valid','-'),('A_test','--')]:
                for ch,color in [('h','#4477aa'),('c','#cc6677')]:
                    v=[r for r in timeline if r['job']==job and r['component']==comp and r['population']==pop and r['channel']==ch]
                    axes[i,0].plot([r['step'] for r in v],[r['projection'] for r in v],style,color=color,label=pop+' '+ch)
                    axes[i,1].plot([r['step'] for r in v],[r['cosine'] for r in v],style,color=color,label=pop+' '+ch)
            axes[i,0].set_ylabel(comp);axes[i,0].set_title('Loss projection');axes[i,1].set_title('Task-relative cosine')
        for ax in axes.flat:ax.axhline(0,color='gray',lw=.7);ax.set_xlabel('B update');ax.legend(fontsize=7)
        fig.suptitle(job);fig.tight_layout();fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
