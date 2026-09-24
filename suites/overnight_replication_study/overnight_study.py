"""Public notebook API. A separate output directory is required."""
from pathlib import Path
import reviewer_suite as core
from reviewer_suite import stop,status

def defaults(output='runs/overnight_replication_v1'):
    s=core.defaults(output)
    s.update(overnight_version='replication-1',models=[dict(name='pythia410m',repo='EleutherAI/pythia-410m',revision='9879c9b5f8bea9051dcb0e68dff21493d67e9d4f')],
        datasets={'sst2':'8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb','ag_news':'eb185aade064a813bc0b7f42de02595523103ca4'},
        seeds=[12,13],tasks=['text'],experiments=[],hours=11.5,max_output_gib=100.,
        acquisition_targets=[.75,1.,1.5,2.],random_replicates=5,event_threshold=.05,event_count=2,event_fixed_steps=[20],event_horizon=8,
        event_modes=['preserve','reset_m','reset_m_global','reset_m_layer'],event_checkpoint_every=2,
        replay_atol=2e-6,replay_rtol=2e-5,norm_match_rtol=1e-3,
        deferred_frames=True,curvature_reaudit_source=None,curvature_reaudit_seed=11,curvature_reaudit_step=5,
        curvature_alphas=[.30,.328125,.36],curvature_eps=[.04,.02,.01,.005,.0025,.00125])
    return s

def smoke_settings(output):
    s=core.smoke_settings(output)
    s.update(overnight_version='replication-1',seeds=[12,13],experiments=[],b_steps=11,
        random_replicates=2,event_threshold=.05,event_count=1,event_fixed_steps=[2],event_horizon=2,
        event_modes=['preserve','reset_m','reset_m_global','reset_m_layer'],event_checkpoint_every=1,
        replay_atol=2e-6,replay_rtol=2e-5,norm_match_rtol=1e-3,deferred_frames=False,
        curvature_reaudit_source=None,curvature_reaudit_seed=11,curvature_reaudit_step=5,
        curvature_alphas=[.3,.328125,.36],curvature_eps=[.04,.02,.01,.005,.0025,.00125])
    return s

def report(s):
    from overnight_engine import report
    return report(s)

def show_results(s):
    core.show_results(s);report(s)
    print((Path(s['output'])/'OVERNIGHT_REPORT.txt').read_text())

def export(s):
    report(s);p=Path(core.export(s));target=p.with_name('overnight_replication_share.zip');p.replace(target);return str(target)


def launch(s):
    from overnight_engine import validation
    validation(s)
    return core.launch(s)


def restart(s):
    from overnight_engine import validation
    validation(s)
    return core.restart(s)
