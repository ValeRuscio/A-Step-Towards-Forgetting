"""Read-only persistence and entity-level analysis of four-vector results.
Usage: python analyze_retention.py four_vector_share.zip --output retention_analysis
Standard library only. Never edits the input archive/database.
"""
import argparse,csv,json,math,sqlite3,tempfile,zipfile,statistics
from pathlib import Path

def read_records(source):
    source=Path(source)
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)/'measurements.sqlite'
        if source.suffix=='.zip':
            with zipfile.ZipFile(source) as z:
                candidates=[n for n in z.namelist() if n.endswith('measurements.sqlite')]
                if len(candidates)!=1:raise ValueError('Expected one four-vector measurement database')
                dest.write_bytes(z.read(candidates[0]))
        else:
            db=source/'measurements.sqlite' if source.is_dir() else source
            with sqlite3.connect(f'file:{db.resolve()}?mode=ro',uri=True) as src,sqlite3.connect(dest) as dst:src.backup(dst)
        with sqlite3.connect(dest) as con:
            return [json.loads(x[0]) for x in con.execute('SELECT record FROM trajectory ORDER BY seed,step')]

def mean(xs):return statistics.mean(xs) if xs else None

def functional(rows):
    # margin = log p(label0) - log p(label1); support = -log total answer mass.
    answers=[];soft=[];correct=[];margins=[]
    for r in rows:
        z=r['margin'];p0=math.exp(-max(-z,0))/(1+math.exp(-abs(z)))
        majority0=r['q0']>.5;tie=z==0
        correct.append(.5 if tie else float((z>0)==majority0))
        support=math.exp(-r['support']);answers.append(support)
        soft.append(support*(p0 if majority0 else 1-p0));margins.append(z if majority0 else -z)
    return dict(two_answer_accuracy=mean(correct),mean_majority_token_probability=mean(soft),mean_answer_mass=mean(answers),mean_target_signed_margin=mean(margins))

def analyze(records,horizons=(1,4,8,16,32),tolerance=1e-4,sustain=8):
    byseed={seed:{r['step']:r for r in records if r['seed']==seed} for seed in sorted({r['seed'] for r in records})}
    events=[];entities=[]
    for seed,steps in byseed.items():
        anchor=steps[min(steps)]['pre']['loss']['A_test']['mean']
        for step,r in sorted(steps.items()):
            pre=r['pre']['loss']['A_test']['mean'];post=r['post_losses']['A_test']['mean'];delta=post-pre
            before={x['entity']:x['loss'] for x in r['pre']['loss']['A_test']['entities']}
            changes=[x['loss']-before[x['entity']] for x in r['post_losses']['A_test']['rows']]
            for x in r['post_losses']['A_test']['rows']:
                entities.append(dict(seed=seed,step=step,entity=x['entity'],pre_loss=before[x['entity']],post_loss=x['loss'],loss_change=x['loss']-before[x['entity']],q0=x['q0']))
            e=dict(seed=seed,step=step,pre_A=pre,post_A=post,delta_A=delta,delta_B=r['functional_consequence']['B_test']['actual_loss_change'],linear_A=r['functional_consequence']['A_test']['linear'],remainder_A=r['functional_consequence']['A_test']['finite_remainder'],
                post_minus_anchor=post-anchor,harmed_entities=sum(x>tolerance for x in changes),entities=len(changes),median_entity_change=statistics.median(changes),**{'post_'+k:v for k,v in functional(r['post_losses']['A_test']['rows']).items()})
            if step-1 in steps:
                prefun=functional(steps[step-1]['post_losses']['A_test']['rows'])
                e.update({'pre_'+k:v for k,v in prefun.items()})
            # Post-event checkpoints include the event endpoint; horizons count additional updates.
            future={k:steps[k]['post_losses']['A_test']['mean'] for k in range(step,max(steps)+1) if k in steps}
            recovery=None
            if delta>tolerance:
                for k in future:
                    window=[future.get(j) for j in range(k,k+sustain)]
                    if all(v is not None and v<=pre+tolerance for v in window):recovery=k-step;break
            e['first_sustained_recovery_lag']=recovery
            e['recovery_status']='not_applicable' if delta<=tolerance else ('observed' if recovery is not None else 'not_observed_within_saved_followup')
            for h in horizons:
                seq=[future.get(step+j) for j in range(h+1)]
                if any(v is None for v in seq):
                    e[f'available_h{h}']=False;continue
                excess=[v-pre for v in seq]
                e.update({f'available_h{h}':True,f'excess_at_h{h}':excess[-1],f'mean_excess_h{h}':mean(excess),f'mean_positive_excess_h{h}':mean([max(x,0) for x in excess]),f'fraction_above_pre_h{h}':mean([float(x>tolerance) for x in excess])})
            events.append(e)
    return events,entities

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def run(source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    events,entities=analyze(read_records(source));write_csv(output/'all_updates.csv',events);write_csv(output/'entity_changes.csv',entities)
    selected=[x for x in events if (x['seed'],x['step']) in [(1,21),(1,48),(2,11),(3,7)]]
    (output/'selected_events.json').write_text(json.dumps(selected,indent=2))
    lines=['RETENTION AND FUNCTIONAL BREADTH — existing natural trajectories','Positive excess compares to the state immediately before the selected update.','Later damage is trajectory persistence, not causal attribution to that one update.','Recovery requires eight consecutive saved endpoints within 1e-4 nats of the pre-event loss.','Accuracy is restricted to the two designated answer tokens, not full-vocabulary greedy accuracy.','']
    for x in selected:
        lines.append(f"Seed {x['seed']} update {x['step']}: A {x['pre_A']:.6f} -> {x['post_A']:.6f}; harmed {x['harmed_entities']}/{x['entities']}; median entity change {x['median_entity_change']:+.6f}")
        lines.append(f"  two-answer accuracy {x.get('pre_two_answer_accuracy')} -> {x['post_two_answer_accuracy']}; majority-token probability {x.get('pre_mean_majority_token_probability')} -> {x['post_mean_majority_token_probability']}")
        lines.append(f"  excess after 8/16/32 further updates: {x.get('excess_at_h8')} / {x.get('excess_at_h16')} / {x.get('excess_at_h32')}; sustained recovery lag: {x['first_sustained_recovery_lag']} ({x['recovery_status']})")
    report='\n'.join(lines)+'\n';(output/'REPORT.txt').write_text(report);print(report)
    return events

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--output',default='retention_analysis');a=p.parse_args();run(a.source,a.output)
