import argparse,sys,json
from pathlib import Path
from replication import read,write,HERE

def prepare(s):
    out=Path(s['output']);sys.path.insert(0,str(HERE/'natural'))
    import study
    n=study.defaults(out/'natural')
    n.update(models=[dict(name='qwen3_06b_base',repo='Qwen/Qwen3-0.6B-Base',revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd')],seeds=s['seeds'],learning_rates={'qwen3_06b_base':s['learning_rate']},defer_paths=True,selection_large_only=True,uniform_updates=1,enriched_per_stratum=1,path_max_points=9,fp64_recheck_failed=True,prompt_derivative_count=4)
    if s['smoke']:
        n=study.smoke_settings(out/'natural');n.update(models=[dict(name='tiny_qwen3',repo='tiny-qwen3',revision='smoke')],learning_rates={'tiny_qwen3':.001},seeds=s['seeds'],defer_paths=True,selection_large_only=True)
    for k in ['python','hours','device','threads','minimum_free_gib','max_output_gib']:n[k]=s[k]
    h=dict(version='history-dynamics-1',source=n['output'],output=str(out/'history'),python=s['python'],device=s['device'],hours=s['hours'],threads=s['threads'],minimum_free_gib=s['minimum_free_gib'],max_output_gib=s['max_output_gib'],checkpoint_every=32,populations=['A_valid','A_test'],test_population=2 if s['smoke'] else 32,derivative_batch=1,norm_floor=1e-12,closure_atol=1e-8,closure_rtol=1e-6,age_residual_rtol=1e-4,cohort_origins=[1,8,32,64],cohort_max_age=32,event_threshold=.05,event_window=4)
    for sub,config in [('natural',n),('history',h)]:
        p=out/sub;p.mkdir(exist_ok=True);old=read(p/'settings.json')
        ignore={'hours','threads','minimum_free_gib','max_output_gib','python'}
        if old and {k:v for k,v in old.items() if k not in ignore}!={k:v for k,v in config.items() if k not in ignore}:raise ValueError('Stage settings changed: '+sub)
        write(p/'settings.json',config)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--settings',required=True);prepare(read(ap.parse_args().settings))
