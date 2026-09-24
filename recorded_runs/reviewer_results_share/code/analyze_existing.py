"""Additional frozen-source baselines on EXISTING trajectory DBs/ZIPs, read-only.
Usage: python analyze_existing.py results.zip --output existing_reanalysis
No classifier is fitted on these inputs. Raw projection from historical records
uses h+c; actual Adam displacement differs by recorded floating-point residual.
"""
import argparse,json,sqlite3,tempfile,zipfile
from pathlib import Path
from study_prediction import convert_original,analyze,decompose_original

def analyze_archive(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True);frozen=json.loads((Path(__file__).parent/'frozen_baselines.json').read_text());metrics=[];errors=[];decomposition=[]
    def process(path,label,immutable=False):
        try:
            records=convert_original(path,immutable=immutable)
            decomposition.extend(dict(source=label,**r) for r in decompose_original(path,immutable=immutable))
            for trajectory in records:
                seed=trajectory[0]['seed'];metrics.extend(dict(source=label,seed=seed,**r) for r in analyze(trajectory,frozen))
        except (sqlite3.Error,KeyError,ValueError) as exc:errors.append(dict(source=label,reason=str(exc)))
    if source.suffix=='.zip':
        with zipfile.ZipFile(source) as z,tempfile.TemporaryDirectory() as temp:
            for i,name in enumerate(z.namelist()):
                if not name.endswith(('.sqlite','.db')):continue
                # Do not extract untrusted archive paths or checkpoint pickle files.
                target=Path(temp)/f'{i}.sqlite';target.write_bytes(z.read(name));process(target,name,immutable=True)
    elif source.is_dir():
        for path in source.rglob('*.sqlite'):process(path,str(path.relative_to(source)))
    else:process(source,str(source))
    (out/'answer_decomposition.json').write_text(json.dumps(decomposition,allow_nan=False))
    (out/'prediction_metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));(out/'skipped_inputs.json').write_text(json.dumps(errors,indent=2))
    if not metrics:raise ValueError('No compatible four-vector trajectory tables found; see skipped_inputs.json. No training was launched.')
    return dict(results=len(metrics),skipped=len(errors),output=str(out.resolve()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--output',default='existing_reanalysis');a=p.parse_args();print(analyze_archive(a.source,a.output))
