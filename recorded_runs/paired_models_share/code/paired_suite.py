"""One resumable GPU worker: OLMo audits, new-model replication, frozen prediction."""
from pathlib import Path
from dataclasses import asdict
import argparse,gc,hashlib,json,os,shutil,sqlite3,subprocess,sys,time,traceback,uuid,zipfile
import torch
import four_vector_study as f
import refine_difficult as refine
import predict_forgetting as pred
from association_core import Config,Engine as OriginalEngine,make_data,minibatches,save,load,dump,digest
from multimodel_engine import MultiEngine,install,unavailable_models
VERSION='paired-base-instruct-1'


def defaults(olmo_source,previous,output=None):
    return dict(version=VERSION,olmo_source=str(Path(olmo_source).resolve()),previous=str(Path(previous).resolve()),
        output=str(Path(output or Path.cwd()/'runs/paired_base_instruct_v1').resolve()),device='cuda:0',threads=8,hours=11.5,reserve_minutes=10,
        minimum_free_gb=30.,max_output_gb=150.,replication_fraction=.8,
        seeds=[1,2,3],path_seeds=[1],chunk_steps=16,checkpoint_every=50,acquisition_max_valid_kl=.25,
        prompt_protocol='identical_plain_text',
        models=[dict(name='qwen25_05b_base',repo='Qwen/Qwen2.5-0.5B',revision='main',enabled=True,family='qwen25_05b',variant='base',architecture='qwen2'),
                dict(name='qwen25_05b_instruct',repo='Qwen/Qwen2.5-0.5B-Instruct',revision='main',enabled=True,family='qwen25_05b',variant='instruct',architecture='qwen2'),
                dict(name='smollm2_360m_base',repo='HuggingFaceTB/SmolLM2-360M',revision='main',enabled=True,family='smollm2_360m',variant='base',architecture='llama'),
                dict(name='smollm2_360m_instruct',repo='HuggingFaceTB/SmolLM2-360M-Instruct',revision='main',enabled=True,family='smollm2_360m',variant='instruct',architecture='llama')])


def import_frozen(s):
    source=Path(s['previous']);root=Path(s['output']);relative='prediction_protocol/frozen_predictors.json'
    if source.is_dir():raw=(source/relative).read_bytes()
    elif source.suffix=='.zip':
        with zipfile.ZipFile(source) as z:
            names=[n for n in z.namelist() if n==relative or n.endswith('/'+relative)]
            if len(names)!=1:raise ValueError('Expected exactly one existing frozen predictor artifact')
            raw=z.read(names[0])
    else:raw=source.read_bytes()
    frozen=json.loads(raw)
    if frozen['protocol']!=pred.PROTOCOL:raise ValueError('Existing frozen predictor protocol differs from this runner')
    if set(frozen['models'])!={'0','1'}:raise ValueError('Missing frozen predictor horizons')
    dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=raw:raise ValueError('Frozen predictors changed; use a new output directory')
    dest.write_bytes(raw)
    f.atomic_json(dict(source=str(source),sha256=hashlib.sha256(raw).hexdigest(),refitted=False),root/'prediction_protocol/import.json')
    return frozen


class Control:
    def __init__(self,s,case='suite'):
        self.s=s;self.root=Path(s['output']);self.out=self.root;self.case=case;self.device=s['device'];self.threads=s['threads'];self.minimum_free_gb=s['minimum_free_gb']
        self.deadline=time.time()+s['hours']*3600-s['reserve_minutes']*60;self.stage_deadline=self.deadline
    def check(self):
        if (self.root/'STOP').exists():raise f.ng.Pause('Stopped; saved units resume on Launch.')
        if time.time()>=self.deadline:raise f.ng.Pause('Session time budget reached.')
        if time.time()>=self.stage_deadline:raise f.StagePause()
        if shutil.disk_usage(self.root).free<self.minimum_free_gb*1024**3:raise f.ng.Pause('Disk reserve reached.')
    def pulse(self,**fields):
        self.check();f.atomic_json(dict(status='running',case=self.case,last_progress=time.time(),**fields),self.root/'status.json')
    def storage(self):
        if sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())>self.s['max_output_gb']*1024**3:raise f.ng.Pause('Suite output limit reached.')
    def checkpoint(self,e,payload,path):
        # Keep room for one atomic replacement; completed anchors are not duplicated.
        if shutil.disk_usage(self.root).free<3*e.pb+self.minimum_free_gb*1024**3:raise f.ng.Pause('Insufficient reserve for atomic A-training checkpoint.')
        save(payload,path)


def pin_model(item,root):
    folder=Path(root)/'sources'/item['name'];folder.mkdir(parents=True,exist_ok=True);path=folder/'model_pin.json'
    if path.exists():return f.read(path)
    if item['repo'].startswith('tiny-'):sha='0'*40
    else:
        from huggingface_hub import HfApi
        sha=HfApi().model_info(item['repo'],revision=item['revision']).sha
        if not sha or len(sha)!=40:raise ValueError('Could not resolve immutable model revision')
    pin=dict(repo=item['repo'],requested_revision=item['revision'],revision=sha);f.atomic_json(pin,path);return pin


def prepare_config(s,item):
    root=Path(s['output']);pin=pin_model(item,root);src=root/'sources'/item['name']
    c=Config(**f.read(Path(s['olmo_source'])/'config.json'));c.model=pin['repo'];c.revision=pin['revision'];c.device=s['device'];c.threads=s['threads'];c.seeds=s['seeds'];c.attn='eager';c.smoke=c.model.startswith('tiny-')
    cfg=asdict(c);old=f.read(src/'config.json')
    if old and old!=cfg:raise ValueError('Stored source configuration changed; use a new suite output.')
    dump(cfg,src/'config.json');return c,src


def acquire(e,c,src,seed,s,ctl):
    folder=src/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True);anchor=folder/'anchor.pt';data=make_data(e.tok,c,seed)
    # Fix answer semantics across architectures instead of silently selecting a different label pair.
    if not c.smoke:
        ids=[e.tok.encode(x,add_special_tokens=False) for x in [' red',' blue']]
        if any(len(x)!=1 for x in ids) or data['A_train'][0]['labels']!=[x[0] for x in ids]:raise ValueError('This tokenizer does not support the fixed single-token red/blue protocol.')
    old_data=f.read(folder/'data.json')
    if old_data and old_data['hash']!=digest(data):raise ValueError('Stored training/evaluation data changed')
    dump(dict(hash=digest(data),labels=data['A_train'][0]['labels'],semantic_prompts_hash=digest({k:[dict(prompt=r['prompt'],entity=r['entity'],q=r['q']) for r in rows] for k,rows in data.items()})),folder/'data.json')
    if 'models' in s:
        item=next((x for x in s['models'] if x['name']==src.name),None)
        if item:check_pair_data(s,item,seed)
    if anchor.exists():
        meta=f.read(folder/'acquisition.json',{})
        if 'after' not in meta:
            e.restore(load(anchor));meta['after']=f.ng.evaluate(e,data,c.microbatch);meta['criterion']=dict(max_A_valid_kl=s['acquisition_max_valid_kl'],passed=meta['after']['A_valid']['mean']<=s['acquisition_max_valid_kl']);dump(meta,folder/'acquisition.json')
        return data
    progress=folder/'A_progress.pt';start=0
    if progress.exists():
        z=load(progress);e.restore(z['engine']);start=z['step'];del z
    initial=f.read(folder/'acquisition.json')
    if initial is None:
        ctl.pulse(phase='pretraining-checkpoint evaluation',seed=seed)
        initial=dict(before=f.ng.evaluate(e,data,c.microbatch));dump(initial,folder/'acquisition.json')
    step=start;completed=start
    try:
        for step in range(start+1,c.a_steps+1):
            ctl.pulse(phase='learning original A task',seed=seed,step=step,total=c.a_steps)
            e.gradient(minibatches(data,'A',seed,step,c));e.opt.step();e.opt.zero_grad(set_to_none=True);completed=step
            if step%s['checkpoint_every']==0:ctl.checkpoint(e,dict(step=step,engine=e.pack()),progress)
    except (f.ng.Pause,f.StagePause):
        # A pulse may fail before executing this iteration: use the number actually completed.
        ctl.checkpoint(e,dict(step=completed,engine=e.pack()),progress);raise
    ctl.checkpoint(e,e.pack(),anchor);progress.unlink(missing_ok=True)
    initial['after']=f.ng.evaluate(e,data,c.microbatch);initial['criterion']=dict(max_A_valid_kl=s['acquisition_max_valid_kl'],passed=initial['after']['A_valid']['mean']<=s['acquisition_max_valid_kl'])
    initial['note']='Fixed original A-training budget; no tuning on B or test outcomes. Weak acquisition is reported, never hidden.';dump(initial,folder/'acquisition.json');return data


def adapter_audit(e,item,src,seed,ctl):
    dest=src/'adapter_audit.json'
    if dest.exists():return
    ctl.pulse(phase='checking full-model forward and gradients',model=item['name'])
    data=make_data(e.tok,e.c,seed);rows=data['A_valid'][:2]
    if not e.c.smoke:
        labels=[e.tok.encode(x,add_special_tokens=False) for x in [' red',' blue']]
        if any(len(x)!=1 for x in labels) or rows[0]['labels']!=[x[0] for x in labels]:raise ValueError('Fixed red/blue single-token protocol unsupported')
    from association_core import collate
    inputs,q,labels=collate(rows,e.tok.pad_token_id,e.c.device)
    with torch.no_grad():
        lp=e.model(**inputs,use_cache=False,return_dict=True).logits[:,-1,:].double().log_softmax(-1)
        target=q.double();reference=(target*(target.log()-lp[:,labels])).sum(-1)
        actual=e.loss_rows(rows);err=float((reference-actual).abs().max())
        if not torch.allclose(reference,actual,atol=1e-6,rtol=1e-5):raise ArithmeticError('Adapter differs from native model forward')
    e.opt.zero_grad(set_to_none=True);e.loss_rows(rows).mean().backward()
    missing=[n for n,p in e.params.items() if p.grad is None]
    finite=all(bool(torch.isfinite(p.grad).all()) for p in e.params.values() if p.grad is not None)
    e.opt.zero_grad(set_to_none=True)
    if missing or not finite:raise ArithmeticError('Missing or nonfinite full-parameter gradients: '+str(missing))
    if len({p.data_ptr() for p in e.params.values()})!=len(e.params):raise ValueError('Aliased parameter counted twice')
    import transformers
    f.atomic_json(dict(passed=True,model_type=e.kind,unique_parameters=len(e.params),forward_max_error=err,all_parameter_gradients_finite=True,torch=torch.__version__,transformers=transformers.__version__,prompt_protocol='identical_plain_text'),dest)


def case_settings(s,item,seed,src,out):
    q=f.defaults(src,out);q.update(device=s['device'],threads=s['threads'],seeds=[seed],diagnostic_jobs=[dict(seed=seed,step=1)],
        end_step=Config(**f.read(src/'config.json')).b_steps,topology_landmarks=min(16,Config(**f.read(src/'config.json')).entities//2),chunk_steps=s['chunk_steps'],minimum_free_gb=s['minimum_free_gb'])
    return q


def case(s,item,seed,ctl,stage):
    c,src=prepare_config(s,item);out=Path(s['output'])/'runs'/item['name']/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    ctl.case=item['name']+f'/seed{seed}';ctl.out=out
    q=case_settings(s,item,seed,src,out);con=f.connect(out)
    existing=con.execute('SELECT MAX(step) FROM trajectory').fetchone()[0] or 0
    if stage=='trajectory' and existing>=c.b_steps:con.close();return True
    if stage=='paths' and (existing<c.b_steps or (out/'REPLICATION_PATHS_DONE.json').exists()):con.close();return existing>=c.b_steps
    e=None
    try:
        ctl.pulse(phase='loading pinned checkpoint');e=MultiEngine(c)
        if item.get('architecture') and e.kind!=item['architecture']:raise ValueError('Checkpoint architecture mismatch')
        adapter_audit(e,item,src,seed,ctl)
        acquire(e,c,src,seed,s,ctl);check_pair_data(s,item,seed);f.validate(q)
        f.atomic_json(q,out/'settings.json')
        if stage=='trajectory':
            short=dict(q,end_step=min(c.b_steps,existing+s['chunk_steps']));f.trajectory(e,short,con,ctl)
        else:
            selected=f.read(out/'path_selection.json')
            if not selected:
                rows=[json.loads(r[0]) for r in con.execute('SELECT record FROM trajectory ORDER BY step')]
                # Selection uses held-out prompt form A_valid, not A_test predictor labels.
                before=f.read(src/f'seed{seed}'/'acquisition.json')['after']['A_valid']['mean'];changes=[]
                for r in rows:
                    after=r['post_losses']['A_valid']['mean'];changes.append((r['step'],after-before));before=after
                largest=max(changes,key=lambda x:x[1]);ordinary=min((x for x in changes if x[0]!=largest[0]),key=lambda x:abs(x[1]))
                selected=dict(jobs=[dict(seed=seed,step=x[0]) for x in [largest,ordinary]],valid_changes=[x[1] for x in [largest,ordinary]],
                    note='Retrospective within-model diagnostics: largest A_valid increase and nearest-zero A_valid change. No assumption that A_test worsens. Never used to fit predictors.')
                f.atomic_json(selected,out/'path_selection.json')
            q['diagnostic_jobs']=selected['jobs'];f.atomic_json(q,out/'settings.json')
            for j in q['diagnostic_jobs']:f.path_event(e,q,j,con,ctl)
            for j in q['diagnostic_jobs']:f.curvature_event(e,q,j,con,ctl)
            f.atomic_json(dict(done=True),out/'REPLICATION_PATHS_DONE.json')
        return (con.execute('SELECT MAX(step) FROM trajectory').fetchone()[0] or 0)>=c.b_steps
    finally:
        con.close()
        if e is not None:del e
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def check_pair_data(s,item,seed):
    root=Path(s['output']);own=f.read(root/'sources'/item['name']/f'seed{seed}'/'data.json')
    if own is None:return
    for other in s['models']:
        if other.get('family')!=item.get('family') or other['name']==item['name']:continue
        peer=f.read(root/'sources'/other['name']/f'seed{seed}'/'data.json')
        if peer is not None and (peer['semantic_prompts_hash']!=own['semantic_prompts_hash'] or peer['hash']!=own['hash']):
            raise ValueError('Base/instruct pair has mismatched plain-text tokenization or task data: '+item['name'])


def paired_summary(s):
    root=Path(s['output']);rows=[]
    def accuracy(values):
        return sum(.5 if x['margin']==0 else float((x['margin']>0)==(x['q0']>.5)) for x in values)/len(values)
    for item in s['models']:
        for seed in s['seeds']:
            folder=root/'runs'/item['name']/f'seed{seed}';db=folder/'measurements.sqlite'
            if not db.exists():continue
            with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as con:
                records=[json.loads(x[0]) for x in con.execute('SELECT record FROM trajectory ORDER BY step')]
            if not records:continue
            anchor=f.read(root/'sources'/item['name']/f'seed{seed}'/'acquisition.json')['after']
            row=dict(name=item['name'],family=item.get('family'),variant=item.get('variant'),seed=seed,updates=len(records),
                anchor_A_valid=anchor['A_valid']['mean'],anchor_A_test=anchor['A_test']['mean'],anchor_B_test=anchor['B_test']['mean'],
                final_A_test=records[-1]['post_losses']['A_test']['mean'],final_B_test=records[-1]['post_losses']['B_test']['mean'],
                anchor_A_two_answer_accuracy=accuracy(anchor['A_test']['rows']),final_A_two_answer_accuracy=accuracy(records[-1]['post_losses']['A_test']['rows']),
                large_A_increases=sum(r['functional_consequence']['A_test']['actual_loss_change']>.05 for r in records),
                large_A_sign_reversals=sum(r['functional_consequence']['A_test']['actual_loss_change']>.05 and r['functional_consequence']['A_test']['linear']<0 for r in records),
                path_diagnostics_requested=seed in s['path_seeds'],path_computation_finished=(folder/'REPLICATION_PATHS_DONE.json').exists())
            row['A_anchor_to_final_change']=row['final_A_test']-row['anchor_A_test'];rows.append(row)
    comparisons=[]
    for base in rows:
        if base['variant']!='base':continue
        other=next((r for r in rows if r['family']==base['family'] and r['seed']==base['seed'] and r['variant']=='instruct'),None)
        if other and other['updates']==base['updates']:
            comparisons.append(dict(family=base['family'],seed=base['seed'],updates=base['updates'],
                instruct_minus_base_A_retention_change=other['A_anchor_to_final_change']-base['A_anchor_to_final_change'],
                instruct_minus_base_anchor_A=other['anchor_A_test']-base['anchor_A_test'],
                note='Descriptive fixed-budget comparison, not matched acquisition or causal isolation of instruction tuning.'))
    f.atomic_json(dict(cases=rows,comparisons=comparisons),root/'paired_comparison.json')
    return rows


def analyze_cases(s,frozen):
    root=Path(s['output']);summary=[]
    for item in s['models']:
        if not item['enabled']:continue
        merged=root/'predictions'/item['name'];merged.mkdir(parents=True,exist_ok=True);db=merged/'joined.sqlite'
        with sqlite3.connect(db) as dst:
            dst.execute('CREATE TABLE IF NOT EXISTS trajectory(seed INTEGER,step INTEGER,record TEXT,PRIMARY KEY(seed,step))')
            for seed in s['seeds']:
                source=root/'runs'/item['name']/f'seed{seed}'/'measurements.sqlite'
                if not source.exists():continue
                with sqlite3.connect(f'file:{source}?mode=ro',uri=True) as src:
                    rows=src.execute('SELECT seed,step,record FROM trajectory').fetchall();dst.executemany('INSERT OR REPLACE INTO trajectory VALUES (?,?,?)',rows)
            dst.commit()
        results=pred.evaluate(db,frozen,merged,item['name'])
        for result in results:
            acquisition=f.read(root/'sources'/item['name']/f'seed{result["seed"]}'/'acquisition.json',{})
            result['acquisition_passed']=acquisition.get('criterion',{}).get('passed')
        f.atomic_json(results,merged/'prediction_metrics.json');summary.extend(results)
    f.atomic_json(summary,root/'transfer_metrics.json')
    pairs=paired_summary(s)
    lines=['BASE/INSTRUCT PAIRS — identical plain-text prompts; original OLMo predictors reused without fitting','Lead 1 = advance warning from preceding pre-update vectors. Lead 0 = current-gradient-informed diagnostic.','Prediction threshold: A_test increase > 0.05 KL; all steps retained. Metrics are per seed, not independent-update confidence intervals.']
    for case_row in pairs:
        lines.append(f"{case_row['name']} seed{case_row['seed']}: {case_row['updates']} updates; A_test {case_row['anchor_A_test']:.6f} -> {case_row['final_A_test']:.6f}; answer accuracy {case_row['anchor_A_two_answer_accuracy']:.3f} -> {case_row['final_A_two_answer_accuracy']:.3f}; large increases={case_row['large_A_increases']}; sign reversals={case_row['large_A_sign_reversals']}")
    for r in summary:
        if r['lead']==1:lines.append(f'{r["architecture"]} seed{r["seed"]} {r["predictor"]}: n={r["n"]}, positives={r["positives"]}, acquisition={r["acquisition_passed"]}, AUROC={r["auroc"]}, AP={r["average_precision"]}, Brier={r["brier"]:.4f}')
    for item in s['models']:
        if not item['enabled']:continue
        for seed in s['seeds']:
            acq=f.read(root/'sources'/item['name']/f'seed{seed}'/'acquisition.json',{})
            if 'after' in acq:lines.append(f'{item["name"]} seed{seed}: A_valid after acquisition={acq["after"]["A_valid"]["mean"]:.6g}; criterion_passed={acq["criterion"]["passed"]}')
    for b in f.read(root/'deferred_models.json',[]):lines.append('PENDING '+b['name']+': '+b['action'])
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    plot_predictions(root,s)
    return summary


def plot_predictions(root,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    items=[]
    for item in s['models']:
        data=f.read(root/'predictions'/item['name']/'predictions.json',[])
        for seed in s['seeds']:
            rows=[r for r in data if r['seed']==seed and r['lead']==1 and r['predictor']=='geometry_plus_changes']
            if rows:items.append((item['name'],seed,rows))
    if not items:return
    fig,axes=plt.subplots(len(items),2,figsize=(12,2.5*len(items)),squeeze=False)
    for i,(name,seed,rows) in enumerate(items):
        xx=[r['step'] for r in rows];axes[i,0].plot(xx,[r['actual_delta'] for r in rows]);axes[i,0].axhline(pred.PROTOCOL['threshold'],color='red',ls='--',label='Large-increase threshold')
        axes[i,1].plot(xx,[r['probability'] for r in rows],label='Frozen advance-warning probability');axes[i,1].scatter(xx,[r['target'] for r in rows],s=7,alpha=.3,label='Observed event');axes[i,1].set_ylim(-.05,1.05)
        for ax in axes[i]:ax.set_title(f'{name}, seed {seed}');ax.set_xlabel('Target update');ax.legend(fontsize=7)
        axes[i,0].set_ylabel('Actual A-test loss change');axes[i,1].set_ylabel('Probability / binary outcome')
    fig.tight_layout();fig.savefig(root/'prediction_trajectories.png',dpi=120);plt.close(fig)


def run(s):
    import fcntl
    root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        ctl=Control(s);start=time.time();duration=ctl.deadline-start;frozen=None;blocked=[]
        try:
            blocked=unavailable_models(s['models']);f.atomic_json(blocked,root/'deferred_models.json')
            ctl.pulse(phase='importing unchanged frozen OLMo predictors')
            frozen=import_frozen(s)
            install();ctl.stage_deadline=start+duration*s['replication_fraction']
            items=[(item,seed) for seed in s['seeds'] for item in s['models'] if item['enabled'] and item['name'] not in {b['name'] for b in blocked}]
            try:
                while True:
                    done=[]
                    for item,seed in items:ctl.check();done.append(case(s,item,seed,ctl,'trajectory'));ctl.storage()
                    if all(done):break
            except f.StagePause:pass
            ctl.stage_deadline=ctl.deadline
            # One seed's pair of diagnostic paths per architecture before extending coverage.
            for item,seed in items:
                if seed in s['path_seeds']:ctl.check();case(s,item,seed,ctl,'paths');ctl.storage()
            analyze_cases(s,frozen)
            complete=all((root/'runs'/item['name']/f'seed{seed}'/'REPLICATION_PATHS_DONE.json').exists() for item,seed in items if seed in s['path_seeds'])
            for item,seed in items:
                db=root/'runs'/item['name']/f'seed{seed}'/'measurements.sqlite'
                if not db.exists():complete=False;continue
                with sqlite3.connect(db) as c:n=c.execute('SELECT COUNT(*) FROM trajectory').fetchone()[0]
                complete=complete and n==Config(**f.read(root/'sources'/item['name']/'config.json')).b_steps
            final=('dependency_blocked' if blocked else 'complete') if complete else 'paused';message='Configured work finished; inspect unresolved numerical audits.' if complete else 'Session allocation ended; Launch resumes completed units.'
            if complete and blocked:message='Available-model work finished; unavailable architectures remain pending. See deferred_models.json.'
        except (f.ng.Pause,f.StagePause) as exc:final='paused';message=str(exc) or 'Session allocation ended.'
        except Exception:final='failed';message=traceback.format_exc();(root/'error.txt').write_text(message)
        if frozen:
            try:analyze_cases(s,frozen)
            except Exception:(root/'analysis_error.txt').write_text(traceback.format_exc())
        f.atomic_json(dict(status=final,message=message,deferred_models=blocked,last_progress=time.time()),root/'status.json')
        try:export(root)
        except Exception:(root/'export_error.txt').write_text(traceback.format_exc())


def validate(s):
    if s.get('version')!=VERSION:raise ValueError('Use defaults from this release')
    if not Path(s['previous']).exists() or not (Path(s['olmo_source'])/'config.json').exists():raise FileNotFoundError('Set previous results and original OLMo source paths')
    if s['hours']*60<=s['reserve_minutes'] or s['minimum_free_gb']<0:raise ValueError('Invalid time/disk budget')
    if not 0<s['replication_fraction']<1:raise ValueError('Leave time for trajectories and path diagnostics')
    if not set(s['path_seeds']).issubset(s['seeds']):raise ValueError('Path seeds must be among trajectory seeds')
    if s['prompt_protocol']!='identical_plain_text':raise ValueError('This release implements the matched plain-text comparison only')
    if len({x['name'] for x in s['models']})!=len(s['models']):raise ValueError('Model names must be unique')
    if not s['seeds'] or len(set(s['seeds']))!=len(s['seeds']) or s['chunk_steps']<1:raise ValueError('Invalid seeds or chunks')
    if not any(x['enabled'] for x in s['models']):raise ValueError('Enable at least one model')


def launch(s):
    import fcntl
    validate(s);root=Path(s['output']);root.mkdir(parents=True,exist_ok=True)
    runtime={'hours','reserve_minutes','minimum_free_gb','max_output_gb','audit_fraction','replication_fraction','output','chunk_steps'}
    sig=lambda x:hashlib.sha256(json.dumps({k:v for k,v in x.items() if k not in runtime},sort_keys=True).encode()).hexdigest()
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if f.ng.alive(root):print('Already running. Refresh status.');return str(root)
        old=f.read(root/'settings.json')
        if old and sig(old)!=sig(s):
            root=root.with_name(root.name+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]);root.mkdir();s['output']=str(root)
        f.atomic_json(s,root/'settings.json');(root/'STOP').unlink(missing_ok=True);code=root/'code';code.mkdir(exist_ok=True)
        for file in Path(__file__).parent.glob('*.py'):
            if file.resolve()!=(code/file.name).resolve():shutil.copy2(file,code/file.name)
        f.atomic_json(dict(files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in code.glob('*.py')},protocol=pred.PROTOCOL),root/'provenance.json')
        f.atomic_json(dict(status='launching',last_progress=time.time()),root/'status.json')
        with (root/'worker.log').open('a') as log:proc=subprocess.Popen([sys.executable,str(code/'paired_suite.py'),'--worker',str(root/'settings.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(100):
            if f.ng.alive(root) or proc.poll() is not None:break
            time.sleep(.1)
        print('Started/resumed:',root);return str(root)


def restart(s):
    if f.ng.alive(s['output']):raise RuntimeError('Stop the worker and wait for alive=False before a fresh run.')
    root=Path(s['output']);return launch(dict(s,output=str(root.with_name(root.name+'_fresh_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]))))


def refresh(output):
    root=Path(output);st=f.read(root/'status.json',{'status':'not_started'});st['alive']=f.ng.alive(root);st['seconds_since_progress']=round(max(0,time.time()-st.get('last_progress',time.time())),1)
    st['free_gib']=round(shutil.disk_usage(root).free/1024**3,1);st['cases']={};st['configured_path_seeds']=f.read(root/'settings.json',{}).get('path_seeds',[]);st['deferred_models']=f.read(root/'deferred_models.json',[])
    for db in root.glob('runs/*/seed*/measurements.sqlite'):
        with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as c:
            try:st['cases'][str(db.parent.relative_to(root))]=dict(updates=c.execute('SELECT COUNT(*) FROM trajectory').fetchone()[0],path_points=c.execute('SELECT COUNT(*) FROM path').fetchone()[0])
            except sqlite3.OperationalError:pass
    print(json.dumps(st,indent=2));return st


def stop(output):f.ng.stop(output)


def show(output):
    root=Path(output);report=root/'REPORT.txt';print(report.read_text() if report.exists() else 'No prediction results yet.')
    from IPython.display import display,Image
    for img in root.glob('runs/*/seed1/trajectory.png'):display(Image(filename=str(img)))
    if (root/'prediction_trajectories.png').exists():display(Image(filename=str(root/'prediction_trajectories.png')))


def export(output):
    import fcntl
    root=Path(output)
    with (root/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);return _export(output)


def _export(output):
    root=Path(output);dest=root/'paired_models_share.zip';tmp=dest.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for pth in root.rglob('*'):
            if not pth.is_file():continue
            if pth.suffix in {'.json','.csv','.txt','.png','.py'}:z.write(pth,pth.relative_to(root))
            elif pth.name=='measurements.sqlite':
                snapshot=root/'export_snapshot.sqlite'
                with sqlite3.connect(pth) as a,sqlite3.connect(snapshot) as b:a.backup(b)
                z.write(snapshot,pth.relative_to(root));snapshot.unlink()
        z.writestr('EXPORT_SCOPE.txt','All saved measurements, predictions, code, settings and pinned revisions. Model/A-training checkpoints excluded; retained in sources for reproducibility/resume.\n')
    os.replace(tmp,dest);return str(dest)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker');args=ap.parse_args()
    if args.worker:run(f.read(args.worker))
    else:ap.print_help()
