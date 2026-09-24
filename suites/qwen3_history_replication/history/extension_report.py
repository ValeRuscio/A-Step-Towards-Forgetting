from collections import defaultdict
from pathlib import Path
import json
from storage import write

def report_extensions(con,out,s):
    ages=[];cohorts=[];groups=defaultdict(list)
    for job,step,raw in con.execute("SELECT job,step,record FROM records WHERE kind='update' ORDER BY job,step"):
        r=json.loads(raw)
        for v in r['age_contributions']:
            if v['layer']=='GLOBAL':ages.append(dict(job=job,step=step,**v))
        for v in r.get('cohorts',[]):
            if v['layer']=='GLOBAL':
                cohorts.append(dict(job=job,**v));groups[(job,v['origin_step'],v['population'],v['component'],v['denominator'])].append(v)
    write(Path(out)/'age_geometry_timeline.json',ages)
    write(Path(out)/'cohort_timeline.json',cohorts)
    summary=[]
    for key,rr in groups.items():
        rr.sort(key=lambda x:x['age']);first=rr[0]
        summary.append(dict(job=key[0],origin_step=key[1],population=key[2],component=key[3],denominator=key[4],ages_saved=len(rr),first_projection=first['projection'],last_projection=rr[-1]['projection'],first_cosine=first['cosine'],last_cosine=rr[-1]['cosine'],sign_transitions=[dict(from_age=a['age'],to_age=b['age'],before=a['projection'],after=b['projection']) for a,b in zip(rr,rr[1:]) if a['projection']*b['projection']<0],note='Descriptive sign crossings; inspect magnitude, floating-point noise and persistence. No causal deletion or significance claim.'))
    write(Path(out)/'cohort_summary.json',summary)
