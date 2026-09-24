"""Archive-reader fix. Does not modify the experiment runner or its resume fingerprint.
Place beside forgetting_mechanisms.py and import analyze_measurements from this file.
Supports older measurement ZIPs, new mechanism-result ZIPs, nested directories,
and individual case directories. Does not load checkpoints or run training.
"""
from pathlib import Path, PurePosixPath
import json
import zipfile
import numpy as np
from forgetting_mechanisms import read, atomic_json, write_csv, mean_loss, bias_accounting


def analyze_measurements(source, out):
    source = Path(source).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}. Set MEASUREMENT_ZIP to an existing ZIP or extracted results directory.")
    if source.is_file() and not zipfile.is_zipfile(source):
        raise ValueError(f"Input is not a ZIP archive: {source}. Choose the results ZIP or its extracted directory, not a text report.")
    archive = zipfile.ZipFile(source) if source.is_file() else None
    records, curves, notes = [], [], []
    seen = set()
    try:
        if archive:
            members=set(archive.namelist())
            paths=sorted(n for n in members if PurePosixPath(n).name == 'outputs.json' and '__MACOSX' not in PurePosixPath(n).parts)
        else:
            paths=sorted(source.rglob('outputs.json'))
        if not paths:
            examples=sorted(archive.namelist())[:12] if archive else sorted(p.name for p in source.iterdir())[:12]
            raise ValueError(f"No outputs.json files found inside {source}. Expected saved predictions from a measurement or mechanism run, including files below nested folders. This may be a code ZIP, a report-only ZIP, an empty directory, or an archive containing another ZIP. First entries: {examples}")
        for path in paths:
            def get(name, optional=False):
                if archive:
                    member=str(PurePosixPath(path).parent / name)
                    if member not in members:
                        if optional: return None
                        raise ValueError(f"Missing {member} in {source}")
                    return json.loads(archive.read(member))
                p=Path(path).parent/name
                if not p.exists() and optional: return None
                return read(p)
            o=get('outputs.json')
            required={'seed','event','before','after','anchor'}
            if not isinstance(o,dict) or not required.issubset(o):
                notes.append(f"Skipped unrelated outputs.json: {path}")
                continue
            key=(o['seed'],o['event'])
            if key in seen:
                raise ValueError(f"Multiple cases with seed/event {key} found. Choose a single campaign directory/ZIP instead of a parent containing several runs.")
            seen.add(key)
            r=dict(seed=o['seed'],event=o['event'])
            for sp in ['A_test','B_test']:
                for state in ['anchor','before','after']:
                    if not o[state].get(sp): raise ValueError(f"Missing {state}/{sp} predictions in {path}")
                r[sp+'_delta']=mean_loss(o['after'][sp])-mean_loss(o['before'][sp])
                r[sp+'_from_anchor']=mean_loss(o['after'][sp])-mean_loss(o['anchor'][sp])
                for k,v in bias_accounting(o['before'][sp],o['after'][sp]).items():
                    if k!='interpretation':r[sp+'_'+k]=v
            records.append(r)
            dose=get('dose_path.json',optional=True)
            fine=get('fine_dose.json',optional=True) if dose is None else None
            if dose is not None:
                r['dose_source']='dose_path.json'
                for alpha,x in dose.items():
                    curves.append(dict(seed=o['seed'],event=o['event'],dose=float(alpha),**{sp+'_delta':mean_loss(x[sp])-mean_loss(o['before'][sp]) for sp in ['A_test','B_test']}))
            elif fine is not None:
                r['dose_source']='fine_dose.json'
                for x in fine:
                    curves.append(dict(seed=o['seed'],event=o['event'],dose=float(x['dose']),**{sp+'_delta':x[sp+'_delta'] for sp in ['A_test','B_test']}))
            else:
                interventions=get('interventions.json',optional=True)
                actual=[x for x in (interventions or []) if x.get('variant')=='actual']
                r['dose_source']='interventions.json' if actual else 'unavailable'
                for x in actual:
                    curves.append(dict(seed=o['seed'],event=o['event'],dose=float(x['dose']),**{sp+'_delta':x[sp+'_delta'] for sp in ['A_test','B_test']}))
                if not actual:notes.append(f"Seed {key[0]}, event {key[1]}: before/after available, but no saved dose curve (possibly an unfinished run).")
    finally:
        if archive:archive.close()
    if not records:
        raise ValueError(f"Found outputs.json files in {source}, but none have the expected seed/event/anchor/before/after schema. Choose the actual results archive, not a code or research-record bundle.")
    records.sort(key=lambda r:(r['seed'],r['event']))
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(dict(source=str(source),cases=len(records),notes=notes),out/'reader_diagnostics.json')
    atomic_json(records,out/'archive_analysis.json');write_csv(records,out/'archive_analysis.csv');write_csv(curves,out/'archive_dose_curves.csv')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed in sorted(set(r['seed'] for r in records)):
        rr=[r for r in records if r['seed']==seed]
        axes[0].scatter([-r['B_test_delta'] for r in rr],[r['A_test_delta'] for r in rr],label=f'Seed {seed}')
    axes[0].axhline(0,color='gray',lw=.7);axes[0].axvline(0,color='gray',lw=.7);axes[0].legend()
    axes[0].set(xlabel='B improvement',ylabel='A deterioration',title='Observed updates')
    rr=sorted([r for r in curves if r['seed']==1 and r['event']==21],key=lambda r:r['dose'])
    if not rr and curves:
        first=min((x['seed'],x['event']) for x in curves)
        rr=sorted([x for x in curves if (x['seed'],x['event'])==first],key=lambda x:x['dose'])
    if rr:
        for sp in ['A_test','B_test']:axes[1].plot([r['dose'] for r in rr],[r[sp+'_delta'] for r in rr],'o-',label=sp)
        axes[1].legend()
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set(xlabel='Fraction of actual update',ylabel='KL change',title=(f"Seed {rr[0]['seed']}, update {rr[0]['event']}: saved dose data" if rr else 'No saved dose curve available'))
    fig.savefig(out/'archive_overview.png',dpi=170);plt.close(fig)
    lines=['SAVED MEASUREMENT REANALYSIS — no new model experiments', 'seed event       dA          dB      mean A shift   shift SD']
    lines += [f"{r['seed']:4d} {r['event']:5d} {r['A_test_delta']:+11.7f} {r['B_test_delta']:+11.7f} {r['A_test_mean_shift']:+12.6f} {r['A_test_shift_sd']:10.6f}" for r in records]
    lines += [f"Both worsen: {sum(r['A_test_delta']>0 and r['B_test_delta']>0 for r in records)}/{len(records)}.",
              'These are correlated, previously inspected events. Output bias counterfactuals are descriptive.']
    lines.extend(notes)
    (out/'archive_report.txt').write_text('\n'.join(lines)+'\n')
    return records
