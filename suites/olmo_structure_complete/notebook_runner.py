"""Quiet notebook entry points for the complete OLMo structure study."""
from pathlib import Path
import json
import structure_study as study

SESSION = Path.cwd() / '.olmo_structure_session.json'


def start(source=None, action='resume', **overrides):
    """Start/resume, or explicitly restart/stop. Remember the actual output folder."""
    if action not in ('resume', 'restart', 'stop'):
        raise ValueError('ACTION must be resume, restart, or stop')
    saved = json.loads(SESSION.read_text()) if SESSION.exists() else None
    if source is None and saved:
        source = saved['source']
    if source is None:
        roots = [Path.cwd(), Path.cwd().parent, Path.home() / '1', Path('/home/ubuntu/1')]
        candidates = list(dict.fromkeys((r/'runs'/'olmo_association_v1').resolve() for r in roots))
        found = [p for p in candidates if (p/'config.json').is_file()]
        if len(found) != 1:
            raise FileNotFoundError('Set SOURCE in the first cell to your original association campaign folder containing config.json and seed1/anchor.pt. Candidates: '+str(found))
        source = found[0]
    source = Path(source).expanduser().resolve()
    output = source.parent / 'olmo_structure_v1'
    if saved and Path(saved['source']).resolve() == source:
        output = Path(saved['output'])
    settings = study.default_settings(source, output)
    settings.update(overrides)
    if action == 'stop':
        print(study.stop(output))
        return settings
    if action == 'restart':
        study.restart(settings)
    else:
        study.launch(settings)
    study.atomic_json(dict(source=str(source), output=str(settings['output'])), SESSION)
    return settings


def refresh(full=False):
    """One manual refresh: status, available numbers/plots, failure log if needed."""
    if not SESSION.exists():
        print('Run the Start cell first.');return
    output = Path(json.loads(SESSION.read_text())['output'])
    print('Results:', output)
    study.print_status(output)
    state = study.status(output)
    if (output/'report.txt').exists():
        study.show(output, full=full)
    elif state.get('status') not in ('failed', 'interrupted_or_failed'):
        print('Measurements are being prepared. Rerun this cell for an update.')
    if state.get('status') in ('failed', 'interrupted_or_failed'):
        log = output/state['job']/'worker.log' if state.get('job') else output/'launcher.log'
        if log.exists():
            print('\nRecent error log:\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-25:]))
